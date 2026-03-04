# HiveMind - 蜂巢智能专家系统

[English](./README.en.md) | **中文**

在 Apple M4 Pro (24GB) 上实现的轻量级自适应 MoE (Mixture of Experts) 架构，通过"蜂群专家"协作机制，实现小模型的大能力。

## 核心特性

| 特性 | 描述 |
|------|------|
| **蜂群专家** | 多个轻量专家协同工作，每个专家专注特定领域 |
| **自适应路由** | 根据任务复杂度动态选择激活的专家数量 |
| **混合路由策略** | 结合软路由和 Top-K 稀疏激活，平衡精度与效率 |
| **负载均衡** | 智能调度避免专家过度使用 |
| **分阶段训练** | 渐进式训练策略，在有限算力下达到最优效果 |

## 项目结构

```
HiveMind/
├── README.md                  # 项目说明
├── pyproject.toml             # 项目配置
│
├── swarm/                     # 蜂群专家核心模块
│   ├── __init__.py
│   ├── experts.py             # 专家池实现
│   ├── router.py              # 混合路由层
│   └── swarm_model.py         # 完整模型
│
├── training/                  # 训练相关
│   ├── configs/               # 训练配置
│   ├── lora/                  # LoRA 训练脚本
│   │   ├── train.py           # 主训练脚本
│   │   └── train_v2.py        # 优化日志版本
│   └── utils/                 # 训练工具
│       └── logger.py          # 日志工具
│
├── inference/                 # 推理相关
│   ├── configs/               # 推理配置
│   ├── generate.py            # 生成脚本
│   └── compare_lora.py        # LoRA 对比
│
├── scripts/                   # 工具脚本
│   ├── verify_env.py          # 环境验证
│   ├── train_stage1.py        # 阶段1训练
│   └── data_crawler.py        # 数据爬虫
│
├── tests/                     # 测试文件
│   ├── test_swarm.py          # 蜂群模型测试
│   └── test_lora.py           # LoRA 测试
│
├── data/                      # 数据目录
├── checkpoints/               # 模型检查点
├── docs/                      # 设计文档
└── archive/                   # 归档文件
```

## 快速开始

### 1. 安装依赖

```bash
uv add torch transformers datasets peft accelerate rich
```

### 2. 环境验证

```bash
uv run python scripts/verify_env.py
```

### 3. 测试蜂群模型

```bash
uv run python tests/test_swarm.py
```

### 4. LoRA 训练

```bash
# 传统 LoRA 训练
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 uv run python training/lora/train.py
```

### 5. 推理

```bash
# 单次推理
uv run python inference/generate.py --mode single --prompt "什么是AI？"

# 交互式推理
uv run python inference/generate.py --mode chat
```

## 分阶段训练

| 阶段 | 脚本 | 目标 | 时长 |
|------|------|------|------|
| 阶段0 | `scripts/verify_env.py` | 环境验证 | 30min |
| 阶段1 | `scripts/train_stage1.py` | 专家初始化 | 2-3h |
| 阶段2 | `scripts/train_stage2.py` | 专业分化 | 16-24h |
| 阶段3 | `scripts/train_stage3.py` | 路由训练 | 6-8h |
| 阶段4 | `scripts/train_stage4.py` | 端到端精调 | 8-12h |

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

## 参数规模

| 组件 | 参数量 | 激活量 |
|------|--------|--------|
| 任务分析器 | 2M | 2M |
| 混合路由层 | 4M | 4M |
| 蜂群专家池 | 50-200M | 10-25M |
| 加权融合层 | 1M | 1M |
| **总计** | **57-207M** | **17-32M** |

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
**更新日期**: 2026-03-04

## 许可证

MIT License

---

**HiveMind** - 小模型，大智慧 🧠
