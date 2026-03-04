"""
蜂群专家模块

实现单个专家和专家池。
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ExpertConfig:
    """专家配置"""

    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    intermediate_size: int = 512
    dropout: float = 0.1
    max_seq_length: int = 2048


class SwarmExpert(nn.Module):
    """
    单个蜂群专家

    轻量级 Transformer 编码器，专注于特定领域。
    """

    def __init__(
        self,
        expert_id: int,
        config: ExpertConfig,
        domain: Optional[str] = None,
    ):
        super().__init__()
        self.expert_id = expert_id
        self.config = config
        self.domain = domain or f"expert_{expert_id}"

        # 嵌入层
        self.embedding = nn.Embedding(
            config.max_seq_length,
            config.hidden_size,
        )

        # Transformer 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # 输出投影
        self.output_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
        )

        # 层归一化
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch, seq_len, hidden_size]
            attention_mask: 注意力掩码

        Returns:
            输出张量 [batch, seq_len, hidden_size]
        """
        # 如果输入是 token ids，先嵌入
        if x.dim() == 2 and x.dtype == torch.long:
            x = self.embedding(x)

        # Transformer 处理
        if attention_mask is not None:
            # 转换掩码格式
            attention_mask = self._convert_attention_mask(attention_mask)

        hidden_states = self.transformer(x, src_key_padding_mask=attention_mask)

        # 输出投影
        output = self.output_proj(hidden_states)
        output = self.layer_norm(output)

        return output

    def _convert_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """转换注意力掩码格式"""
        # 从 [batch, seq_len] 转换为 TransformerEncoder 需要的格式
        return attention_mask == 0  # Padding 位置为 True

    def get_num_params(self) -> int:
        """获取参数数量"""
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str):
        """保存专家"""
        torch.save({
            "expert_id": self.expert_id,
            "domain": self.domain,
            "config": self.config,
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SwarmExpert":
        """加载专家"""
        checkpoint = torch.load(path, map_location=device)
        expert = cls(
            expert_id=checkpoint["expert_id"],
            config=checkpoint["config"],
            domain=checkpoint["domain"],
        )
        expert.load_state_dict(checkpoint["state_dict"])
        return expert


class SwarmExpertPool(nn.Module):
    """
    蜂群专家池

    管理多个专家，支持加权输出和并行推理。
    """

    def __init__(
        self,
        num_experts: int,
        config: ExpertConfig,
        domains: Optional[List[str]] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.config = config

        # 创建专家
        self.experts = nn.ModuleList([
            SwarmExpert(
                expert_id=i,
                config=config,
                domain=domains[i] if domains else None,
            )
            for i in range(num_experts)
        ])

        # 专家统计
        self.register_buffer(
            "expert_usage_count",
            torch.zeros(num_experts),
        )

    def forward(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        加权专家输出

        Args:
            x: 输入 [batch, seq_len, hidden_size]
            expert_weights: 专家权重 [batch, num_experts]
            attention_mask: 注意力掩码

        Returns:
            加权输出 [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.shape

        # 每个专家的输出
        expert_outputs = []
        for expert in self.experts:
            output = expert(x, attention_mask)
            expert_outputs.append(output)

        # 堆叠专家输出
        all_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, seq_len, hidden_size]

        # 加权组合
        weights = expert_weights.unsqueeze(-1).unsqueeze(-1)  # [batch, num_experts, 1, 1]
        weighted_output = (all_outputs * weights).sum(dim=1)  # [batch, seq_len, hidden_size]

        return weighted_output

    def forward_with_weights(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        top_k_indices: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        使用指定权重前向传播（支持稀疏激活）

        Args:
            x: 输入
            expert_weights: 专家权重
            top_k_indices: Top-K 专家索引（稀疏模式）
            attention_mask: 注意力掩码

        Returns:
            包含输出和元数据的字典
        """
        batch_size = x.shape[0]

        if top_k_indices is not None:
            # 稀疏模式: 只计算选中的专家
            return self._sparse_forward(x, expert_weights, top_k_indices, attention_mask)
        else:
            # 密集模式: 计算所有专家
            return self._dense_forward(x, expert_weights, attention_mask)

    def _sparse_forward(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        top_k_indices: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """稀疏前向传播"""
        batch_size, seq_len, hidden_size = x.shape

        # 初始化输出
        output = torch.zeros_like(x)
        total_weight = 0

        # 更新使用统计
        with torch.no_grad():
            for idx in top_k_indices.flatten():
                self.expert_usage_count[idx] += 1

        # 只计算选中的专家
        for b in range(batch_size):
            selected_indices = top_k_indices[b]
            selected_weights = expert_weights[b, selected_indices]

            for idx, weight in zip(selected_indices, selected_weights):
                if weight > 0:
                    expert_output = self.experts[idx](
                        x[b:b+1],
                        attention_mask[b:b+1] if attention_mask is not None else None,
                    )
                    output[b:b+1] += weight * expert_output
                    total_weight += weight

        # 归一化
        if total_weight > 0:
            output = output / (total_weight + 1e-8)

        return {
            "output": output,
            "num_active_experts": top_k_indices.shape[1],
        }

    def _dense_forward(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """密集前向传播"""
        output = self.forward(x, expert_weights, attention_mask)

        return {
            "output": output,
            "num_active_experts": self.num_experts,
        }

    def get_expert_stats(self) -> Dict[str, Any]:
        """获取专家使用统计"""
        total = self.expert_usage_count.sum().item()
        if total == 0:
            return {
                "usage_counts": [0] * self.num_experts,
                "usage_ratios": [0.0] * self.num_experts,
            }

        ratios = (self.expert_usage_count / total).tolist()

        return {
            "usage_counts": self.expert_usage_count.tolist(),
            "usage_ratios": ratios,
            "total_activations": int(total),
        }

    def reset_stats(self):
        """重置统计"""
        self.expert_usage_count.zero_()

    def get_total_params(self) -> int:
        """获取总参数量"""
        return sum(expert.get_num_params() for expert in self.experts)

    def save_pool(self, directory: str):
        """保存专家池"""
        import os
        os.makedirs(directory, exist_ok=True)

        for expert in self.experts:
            expert.save(os.path.join(directory, f"expert_{expert.expert_id}.pt"))

        # 保存配置
        torch.save({
            "num_experts": self.num_experts,
            "config": self.config,
            "expert_usage_count": self.expert_usage_count,
        }, os.path.join(directory, "pool_config.pt"))

    @classmethod
    def load_pool(cls, directory: str, device: str = "cpu") -> "SwarmExpertPool":
        """加载专家池"""
        config_path = f"{directory}/pool_config.pt"
        checkpoint = torch.load(config_path, map_location=device)

        pool = cls(
            num_experts=checkpoint["num_experts"],
            config=checkpoint["config"],
        )

        # 加载每个专家
        import os
        for i in range(pool.num_experts):
            expert_path = f"{directory}/expert_{i}.pt"
            if os.path.exists(expert_path):
                pool.experts[i] = SwarmExpert.load(expert_path, device)

        pool.expert_usage_count = checkpoint["expert_usage_count"].to(device)

        return pool
