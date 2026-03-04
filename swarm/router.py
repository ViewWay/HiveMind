"""
混合路由层模块

实现自适应混合路由策略，结合软路由和稀疏激活。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass


@dataclass
class RouterConfig:
    """路由器配置"""

    num_experts: int = 8
    hidden_size: int = 128
    task_feature_dim: int = 128
    temperature: float = 0.5
    min_k: int = 2
    max_k: int = 6
    load_balance_weight: float = 0.1
    dropout: float = 0.1


class HybridRouter(nn.Module):
    """
    混合路由层

    结合任务分析、软路由门控和 Top-K 稀疏选择。
    """

    def __init__(self, config: RouterConfig):
        super().__init__()
        self.config = config

        # 任务分析器
        self.task_analyzer = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.task_feature_dim),
            nn.LayerNorm(config.task_feature_dim),
        )

        # 复杂度估计器
        self.complexity_estimator = nn.Sequential(
            nn.Linear(config.task_feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # 软路由门控
        self.gate_network = nn.Sequential(
            nn.Linear(config.task_feature_dim, config.num_experts),
        )

        # 领域偏好 (可学习的专家领域倾向)
        self.domain_bias = nn.Parameter(torch.zeros(config.num_experts))

        # 温度参数 (可学习)
        self.temperature = nn.Parameter(torch.tensor(config.temperature))

        # 统计
        self.register_buffer("selection_history", torch.zeros(config.num_experts))
        self.register_buffer("total_selections", torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        路由前向传播

        Args:
            x: 输入隐藏状态 [batch, seq_len, hidden_size]
            training: 是否训练模式

        Returns:
            路由结果字典，包含:
            - expert_weights: 专家权重 [batch, num_experts]
            - selected_experts: 选中的专家索引 [batch, k]
            - k: 使用的专家数量
            - complexity: 任务复杂度
        """
        batch_size = x.shape[0]

        # 1. 分析任务 (对序列取平均)
        task_features = self.task_analyzer(x.mean(dim=1))  # [batch, task_feature_dim]

        # 2. 估计复杂度
        complexity = self.complexity_estimator(task_features)  # [batch, 1]

        # 3. 计算自适应 k 值
        k = self._compute_adaptive_k(complexity, batch_size)

        # 4. 计算门控权重
        gate_logits = self.gate_network(task_features)  # [batch, num_experts]

        # 添加领域偏好
        gate_logits = gate_logits + self.domain_bias.unsqueeze(0)

        # 温度缩放
        gate_logits = gate_logits / (self.temperature.abs() + 0.1)

        # 5. Softmax 得到权重
        expert_weights = F.softmax(gate_logits, dim=-1)  # [batch, num_experts]

        # 6. Top-K 选择
        top_k_weights, top_k_indices = torch.topk(
            expert_weights,
            k=k,
            dim=-1,
        )

        if training:
            # 训练模式: 返回软权重用于负载均衡
            selected_experts = top_k_indices
            final_weights = expert_weights
        else:
            # 推理模式: 只使用 Top-K 权重，重新归一化
            final_weights = torch.zeros_like(expert_weights)
            normalized_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)
            final_weights.scatter_(-1, top_k_indices, normalized_weights)
            selected_experts = top_k_indices

        # 更新统计
        with torch.no_grad():
            for idx in selected_experts.flatten():
                self.selection_history[idx] += 1
            self.total_selections += batch_size * k

        return {
            "expert_weights": final_weights,
            "selected_experts": selected_experts,
            "k": k,
            "complexity": complexity,
            "task_features": task_features,
        }

    def _compute_adaptive_k(self, complexity: torch.Tensor, batch_size: int) -> int:
        """
        根据复杂度计算自适应 k 值

        Args:
            complexity: 复杂度分数 [batch, 1]
            batch_size: 批次大小

        Returns:
            k 值
        """
        # 对批次内的复杂度取平均
        avg_complexity = complexity.mean().item()

        # 映射到 [min_k, max_k]
        k = int(
            self.config.min_k +
            avg_complexity * (self.config.max_k - self.config.min_k)
        )

        # 确保不超出范围
        k = max(self.config.min_k, min(k, self.config.max_k))
        k = min(k, self.config.num_experts)

        return k

    def load_balance_loss(self, expert_weights: torch.Tensor) -> torch.Tensor:
        """
        计算负载均衡损失

        鼓励均匀使用所有专家。

        Args:
            expert_weights: 专家权重 [batch, num_experts]

        Returns:
            负载均衡损失
        """
        # 平均专家使用概率
        mean_probs = expert_weights.mean(dim=0)  # [num_experts]

        # 目标是均匀分布
        target = torch.ones_like(mean_probs) / self.config.num_experts

        # KL 散度损失
        loss = F.kl_div(
            mean_probs.log() + 1e-8,
            target,
            reduction="batchmean",
        )

        return loss

    def diversity_loss(self, expert_weights: torch.Tensor) -> torch.Tensor:
        """
        计算多样性损失

        鼓励不同样本选择不同的专家。

        Args:
            expert_weights: 专家权重 [batch, num_experts]

        Returns:
            多样性损失
        """
        # 计算批次内专家权重的标准差
        std = expert_weights.std(dim=0).mean()

        # 我们希望标准差较大（表示选择多样化）
        # 所以用负标准差作为损失
        return -std

    def get_routing_stats(self) -> Dict[str, any]:
        """获取路由统计信息"""
        total = self.total_selections.item()
        if total == 0:
            return {
                "expert_usage": [0.0] * self.config.num_experts,
                "avg_k": 0.0,
                "temperature": self.temperature.item(),
            }

        usage = (self.selection_history / total).tolist()
        avg_k = total / self.selection_history.sum().item() if self.selection_history.sum() > 0 else 0

        return {
            "expert_usage": usage,
            "avg_k": avg_k,
            "temperature": self.temperature.item(),
        }

    def reset_stats(self):
        """重置统计"""
        self.selection_history.zero_()
        self.total_selections.zero_()


class AdaptiveRouter(nn.Module):
    """
    自适应路由器

    根据任务类型自动选择最优的路由策略。
    """

    def __init__(self, config: RouterConfig):
        super().__init__()
        self.config = config

        # 任务分类器 (决定使用哪种路由策略)
        self.task_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 4),  # 4 种路由策略
        )

        # 不同策略的路由器
        self.routers = nn.ModuleDict({
            "soft": SoftRouter(config),
            "sparse": SparseRouter(config),
            "hybrid": HybridRouter(config),
            "adaptive": HybridRouter(config),
        })

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        自适应路由

        Args:
            x: 输入
            training: 是否训练模式

        Returns:
            路由结果
        """
        # 预测任务类型
        task_type = self.task_classifier(x.mean(dim=1))
        strategy_idx = task_type.argmax(dim=-1)

        # 统计最常用的策略
        strategy = ["soft", "sparse", "hybrid", "adaptive"][strategy_idx.mode().values[0].item()]

        # 使用对应策略的路由器
        router = self.routers[strategy]
        return router(x, training)


class SoftRouter(nn.Module):
    """软路由器 - 所有专家都参与，按权重加权"""

    def __init__(self, config: RouterConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts)

    def forward(self, x: torch.Tensor, training: bool = True) -> Dict:
        gate_logits = self.gate(x.mean(dim=1))
        weights = F.softmax(gate_logits / 0.5, dim=-1)

        return {
            "expert_weights": weights,
            "selected_experts": torch.arange(self.config.num_experts).expand(x.shape[0], -1),
            "k": self.config.num_experts,
        }


class SparseRouter(nn.Module):
    """稀疏路由器 - 固定 Top-K"""

    def __init__(self, config: RouterConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts)
        self.k = config.min_k

    def forward(self, x: torch.Tensor, training: bool = True) -> Dict:
        gate_logits = self.gate(x.mean(dim=1))
        weights = F.softmax(gate_logits / 0.5, dim=-1)

        top_k_weights, top_k_indices = torch.topk(weights, self.k, dim=-1)

        # 稀疏化权重
        sparse_weights = torch.zeros_like(weights)
        sparse_weights.scatter_(-1, top_k_indices, top_k_weights)

        return {
            "expert_weights": sparse_weights,
            "selected_experts": top_k_indices,
            "k": self.k,
        }
