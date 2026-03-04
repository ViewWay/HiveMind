# HiveMind - 蜂巢智能专家系统

在 Apple M4 Pro (24GB) 上实现的轻量级自适应 MoE (Mixture of Experts) 架构，通过"蜂群专家"协作机制，实现小模型的大能力。

## 核心特性

| 特性 | 描述 |
|------|------|
| **蜂群专家** | 多个轻量专家协同工作，每个专家专注特定领域 |
| **自适应路由** | 根据任务复杂度动态选择激活的专家数量 |
| **混合路由策略** | 结合软路由和 Top-K 稀疏激活，平衡精度与效率 |
| **负载均衡** | 智能调度避免专家过度使用 |
| **分阶段训练** | 渐进式训练策略，在有限算力下达到最优效果 |

## 架构概览

```
输入 → 任务分析器 → 混合路由层 → 蜂群专家池 → 加权融合 → 输出
         ↓                                          ↑
    复杂度评估                                  自适应规模
         ↓                                          ↑
    路由策略决策                                  动态扩缩
```

## 参数规模

| 组件 | 参数量 | 激活量 | 说明 |
|------|--------|--------|------|
| 任务分析器 | 2M | 2M | 小型 Transformer |
| 混合路由层 | 4M | 4M | Gate + Top-K |
| 蜂群专家池 | 50-200M | 10-25M | 8-16 专家，稀疏激活 |
| 加权融合层 | 1M | 1M | 线性组合 |
| **总计** | **57-207M** | **17-32M** | **仅 20-30% 参数激活** |

## 项目结构

```
hivemind/
├── swarm/                    # 蜂群专家核心模块
│   ├── __init__.py
│   ├── experts.py            # 专家池实现
│   ├── router.py             # 混合路由层
│   └── swarm_model.py        # 完整模型
├── scripts/                  # 训练脚本
│   ├── train_stage1.py       # 阶段1: 专家初始化
│   ├── train_stage2.py       # 阶段2: 专业分化
│   ├── train_stage3.py       # 阶段3: 路由训练
│   └── train_stage4.py       # 阶段4: 端到端精调
├── data/                     # 训练数据
├── checkpoints/              # 模型检查点
├── docs/                     # 设计文档
└── test_swarm.py            # 测试脚本
```

## 快速开始

### 1. 安装依赖

```bash
uv add torch transformers datasets peft accelerate rich
```

### 2. 测试蜂群模型

```bash
uv run python test_swarm.py
```

### 3. 运行训练 (分阶段)

```bash
# 阶段 1: 专家初始化 (2-3小时)
uv run python scripts/train_stage1.py

# 阶段 2: 专业分化 (4-6小时/领域)
uv run python scripts/train_stage2.py --domain code

# 阶段 3: 路由训练 (6-8小时)
uv run python scripts/train_stage3.py

# 阶段 4: 端到端精调 (8-12小时)
uv run python scripts/train_stage4.py
```

## LoRA 训练 (传统模式)

项目同时保留了传统 LoRA 微调功能：

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 uv run python train.py
```

## 模型使用

```python
from swarm import create_swarm_model

# 创建模型
model = create_swarm_model(
    num_experts=8,
    expert_size="small",  # small, medium, large
)

# 生成文本
input_ids = tokenizer("什么是人工智能？", return_tensors="pt").input_ids
output = model.generate(input_ids, max_new_tokens=100)
```

## 分阶段训练策略

| 阶段 | 目标 | 数据 | 时长 | 输出 |
|------|------|------|------|------|
| 阶段0 | 环境验证 | - | 30min | 环境就绪 |
| 阶段1 | 专家初始化 | 通用数据 50MB | 2-3h | 8个基础专家 |
| 阶段2 | 专业分化 | 领域数据 400MB | 16-24h | 专业化专家 |
| 阶段3 | 路由训练 | 混合任务数据 | 6-8h | 智能路由层 |
| 阶段4 | 端到端精调 | 困难样本 | 8-12h | 最终模型 |

## 目标性能

| 指标 | 目标 | 对比模型 |
|------|------|----------|
| MMLU | 70% | Qwen2-1.5B: 42.5%, Phi-2: 55.8% |
| HumanEval | 50% | Qwen2-1.5B: 18.3%, Phi-2: 28.5% |
| GSM8K | 65% | Qwen2-1.5B: 38.2%, Phi-2: 55.4% |

## 系统要求

- **硬件**: Apple M4 Pro (或类似 M 系列)
- **内存**: 24GB+
- **系统**: macOS 15.0+

## 设计文档

详细设计请参阅: [docs/2025-03-04-swarm-experts-design.md](docs/2025-03-04-swarm-experts-design.md)

## 核心概念

### 蜂群智能 (Swarm Intelligence)

灵感来自自然界中的蜂群行为：
- **分布式智能**: 每个专家独立运作
- **协作增效**: 通过路由层协同输出
- **自适应扩展**: 根据任务调整参与专家数量

### 混合路由 (Hybrid Routing)

结合两种路由策略的优势：
- **软路由**: 所有关注，平滑决策 (训练阶段)
- **稀疏路由**: Top-K 选择，高效计算 (推理阶段)

## 许可证

MIT License

---

**HiveMind** - 小模型，大智慧 🧠
