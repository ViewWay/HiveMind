"""
蜂群专家 (Swarm Experts) 模块

在 M4 Pro 上实现轻量级 MoE 架构，通过自适应专家选择
实现与大型全参数模型竞争的性能。
"""

from .experts import SwarmExpert, SwarmExpertPool
from .router import HybridRouter
from .swarm_model import SwarmModel, create_swarm_model

__version__ = "0.1.0"

__all__ = [
    "SwarmExpert",
    "SwarmExpertPool",
    "HybridRouter",
    "SwarmModel",
    "create_swarm_model",
]
