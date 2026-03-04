"""
蜂群模型主模块

整合专家池和路由层，实现完整的蜂群专家模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass

from .experts import SwarmExpertPool, ExpertConfig
from .router import HybridRouter, RouterConfig


@dataclass
class SwarmModelConfig:
    """蜂群模型配置"""

    # 专家配置
    num_experts: int = 8
    expert_hidden_size: int = 128
    expert_num_layers: int = 2
    expert_num_heads: int = 4
    expert_intermediate_size: int = 512

    # 路由配置
    router_task_feature_dim: int = 128
    router_temperature: float = 0.5
    router_min_k: int = 2
    router_max_k: int = 6

    # 模型配置
    vocab_size: int = 151936
    max_seq_length: int = 2048
    dropout: float = 0.1

    def to_expert_config(self) -> ExpertConfig:
        """转换为专家配置"""
        return ExpertConfig(
            hidden_size=self.expert_hidden_size,
            num_layers=self.expert_num_layers,
            num_heads=self.expert_num_heads,
            intermediate_size=self.expert_intermediate_size,
            dropout=self.dropout,
            max_seq_length=self.max_seq_length,
        )

    def to_router_config(self) -> RouterConfig:
        """转换为路由器配置"""
        return RouterConfig(
            num_experts=self.num_experts,
            hidden_size=self.expert_hidden_size,
            task_feature_dim=self.router_task_feature_dim,
            temperature=self.router_temperature,
            min_k=self.router_min_k,
            max_k=self.router_max_k,
            dropout=self.dropout,
        )


class SwarmModel(nn.Module):
    """蜂群专家模型"""

    def __init__(self, config: SwarmModelConfig):
        super().__init__()
        self.config = config

        # 输入嵌入
        self.token_embedding = nn.Embedding(config.vocab_size, config.expert_hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.expert_hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # 路由层
        self.router = HybridRouter(config.to_router_config())

        # 专家池
        self.expert_pool = SwarmExpertPool(
            num_experts=config.num_experts,
            config=config.to_expert_config(),
        )

        # 输出投影
        self.output_proj = nn.Linear(config.expert_hidden_size, config.vocab_size)

        # 层归一化
        self.input_norm = nn.LayerNorm(config.expert_hidden_size)
        self.output_norm = nn.LayerNorm(config.expert_hidden_size)

        # 初始化
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
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """前向传播"""
        batch_size, seq_len = input_ids.shape

        # 1. 嵌入
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)

        hidden_states = self.dropout(self.input_norm(token_emb + pos_emb))

        # 2. 路由
        router_output = self.router(hidden_states, training=training)
        expert_weights = router_output["expert_weights"]
        selected_experts = router_output["selected_experts"]

        # 3. 专家处理
        expert_output = self.expert_pool.forward_with_weights(
            hidden_states,
            expert_weights,
            selected_experts if not training else None,
            attention_mask,
        )

        hidden_states = expert_output["output"]
        num_active_experts = expert_output.get("num_active_experts", self.config.num_experts)

        # 4. 输出投影
        hidden_states = self.output_norm(hidden_states)
        logits = self.output_proj(hidden_states)

        # 5. 计算损失
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            if training:
                balance_loss = self.router.load_balance_loss(expert_weights)
                loss = loss + 0.1 * balance_loss

        return {
            "logits": logits,
            "loss": loss,
            "expert_weights": expert_weights,
            "selected_experts": selected_experts,
            "num_active_experts": num_active_experts,
            "complexity": router_output.get("complexity"),
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """生成文本"""
        self.eval()
        with torch.no_grad():
            batch_size = input_ids.shape[0]
            current_ids = input_ids.clone()

            for _ in range(max_new_tokens):
                outputs = self.forward(current_ids, training=False)
                logits = outputs["logits"]
                next_token_logits = logits[:, -1, :]

                next_token_logits = next_token_logits / temperature

                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                probs = F.softmax(next_token_logits, dim=-1)
                if do_sample:
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)

                current_ids = torch.cat([current_ids, next_token], dim=-1)

                if (next_token == 0).all():
                    break

        return current_ids

    def get_num_params(self, non_embedding: bool = False) -> int:
        """获取参数数量"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            n_params -= self.position_embedding.weight.numel()
        return n_params

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = self.get_num_params()
        expert_params = self.expert_pool.get_total_params()
        router_params = sum(p.numel() for p in self.router.parameters())

        return {
            "total_params": total_params,
            "expert_params": expert_params,
            "router_params": router_params,
            "num_experts": self.config.num_experts,
            "router_stats": self.router.get_routing_stats(),
            "expert_stats": self.expert_pool.get_expert_stats(),
        }

    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            "config": self.config,
            "state_dict": self.state_dict(),
            "model_info": self.get_model_info(),
        }, path)

    @classmethod
    def load_model(cls, path: str, device: str = "cpu") -> "SwarmModel":
        """加载模型"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model

    def freeze_experts(self):
        """冻结专家参数"""
        for expert in self.expert_pool.experts:
            for param in expert.parameters():
                param.requires_grad = False

    def unfreeze_experts(self):
        """解冻专家参数"""
        for expert in self.expert_pool.experts:
            for param in expert.parameters():
                param.requires_grad = True

    def freeze_router(self):
        """冻结路由参数"""
        for param in self.router.parameters():
            param.requires_grad = False

    def unfreeze_router(self):
        """解冻路由参数"""
        for param in self.router.parameters():
            param.requires_grad = True


def create_swarm_model(
    num_experts: int = 8,
    expert_size: str = "small",
    **kwargs
) -> SwarmModel:
    """创建蜂群模型的便捷函数"""
    size_configs = {
        "small": {
            "expert_hidden_size": 128,
            "expert_num_layers": 2,
            "expert_num_heads": 4,
            "expert_intermediate_size": 512,
        },
        "medium": {
            "expert_hidden_size": 256,
            "expert_num_layers": 4,
            "expert_num_heads": 8,
            "expert_intermediate_size": 1024,
        },
        "large": {
            "expert_hidden_size": 512,
            "expert_num_layers": 6,
            "expert_num_heads": 16,
            "expert_intermediate_size": 2048,
        },
    }

    config = SwarmModelConfig(
        num_experts=num_experts,
        **size_configs.get(expert_size, size_configs["small"]),
        **kwargs,
    )

    return SwarmModel(config)
