"""
蜂群专家模型测试脚本
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from swarm import create_swarm_model

def test_swarm_model():
    """测试蜂群模型"""
    print("初始化蜂群模型...")
    model = create_swarm_model(
        num_experts=8,
        expert_size="small",
    )

    # 获取设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    print(f"使用设备: {device}")

    # 打印模型信息
    info = model.get_model_info()
    print(f"\n模型信息:")
    print(f"  总参数量: {info['total_params']:,}")
    print(f"  专家参数: {info['expert_params']:,}")
    print(f"  路由参数: {info['router_params']:,}")
    print(f"  专家数量: {info['num_experts']}")

    # 测试前向传播
    print("\n测试前向传播...")
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask, training=True)

    print(f"  logits shape: {outputs['logits'].shape}")
    print(f"  num_active_experts: {outputs['num_active_experts']}")
    print(f"  expert_weights shape: {outputs['expert_weights'].shape}")

    # 测试生成
    print("\n测试文本生成...")
    input_ids = torch.randint(0, 1000, (1, 10)).to(device)
    generated = model.generate(input_ids, max_new_tokens=20, temperature=0.7)
    print(f"  生成长度: {generated.shape[1]}")

    print("\n✓ 所有测试通过!")

if __name__ == "__main__":
    test_swarm_model()
