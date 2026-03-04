# M4 Pro LoRA 训练项目

在 Apple M4 Pro (24GB) 上进行 LLM LoRA 微调的完整项目。

## 项目结构

```
qwen-training/
├── train.py              # 主训练脚本
├── inference.py           # 推理脚本（支持多种模式）
├── evaluate.py            # 评估指标脚本
├── compare_lora.py        # LoRA 配置对比实验
├── download_model.sh      # 模型下载脚本
├── MODEL_GUIDE.md         # 模型选择指南
├── data/
│   ├── train.txt         # 原始测试数据 (10条)
│   └── train_large.txt   # 扩展训练数据 (152条)
└── output/
    ├── lora-adapter/     # 训练输出
    └── comparison/       # 对比实验输出
```

## 快速开始

### 1. 安装依赖

```bash
uv add torch transformers datasets peft accelerate
```

### 2. 准备数据

编辑 `data/train_large.txt`，每行一条训练文本。

### 3. 开始训练

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 uv run python train.py
```

### 4. 推理测试

```bash
# 单次推理
uv run python inference.py --mode single --prompt "什么是AI？"

# 交互式聊天
uv run python inference.py --mode chat

# 批量测试
uv run python inference.py --mode batch --data data/train_large.txt
```

## 高级功能

### 对比实验

对比不同 LoRA 配置的训练效果：

```bash
uv run python compare_lora.py
```

对比三种配置：
- **仅注意力层** - 关注语义理解
- **仅MLP层** - 关注知识存储  
- **全部层** - 效果通常最好

### 评估指标

```bash
uv run python evaluate.py ../Qwen3.5-4B ./output/lora-adapter ./data/train_large.txt
```

支持指标：
- 困惑度 (Perplexity)
- BLEU 分数
- ROUGE 分数

### 切换模型

```bash
# 使用环境变量
MODEL_PATH="microsoft/Phi-3.5-mini-instruct" uv run python train.py

# 或直接从 HuggingFace 下载
MODEL_PATH="Qwen/Qwen2.5-3B" uv run python train.py
```

## 配置说明

### 训练参数 (train.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_path` | ../Qwen3.5-4B | 基础模型路径 |
| `lora_r` | 8 | LoRA rank |
| `lora_alpha` | 16 | LoRA alpha |
| `batch_size` | 1 | 批次大小 |
| `gradient_accumulation` | 8 | 梯度累积步数 |
| `num_train_epochs` | 3 | 训练轮数 |
| `max_seq_length` | 2048 | 序列长度 |

### 优化技巧

- **内存不足？** → 降低 `batch_size` 到 1，增加 `gradient_accumulation`
- **训练太慢？** → 启用 gradient_checkpointing (已默认启用)
- **效果不好？** → 增加 `lora_r` 或添加更多训练数据
- **想更快训练？** → 使用更小的模型（如 Phi-3.5-mini）

## 推理模式

```bash
# 单次推理
uv run python inference.py --mode single \
  --prompt "你好" \
  --temp 0.7 \
  --max-tokens 512

# 交互聊天
uv run python inference.py --mode chat

# 批量测试
uv run python inference.py --mode batch \
  --data data/prompts.txt \
  --output output/results.json

# 参数对比
uv run python inference.py --mode compare \
  --prompt "解释机器学习"
```

## 训练数据格式

纯文本文件，每行一条样本：

```
人工智能是计算机科学的一个分支。
机器学习使计算机能够从数据中学习。
深度学习使用多层神经网络模拟人脑。
```

## 输出文件

训练完成后，LoRA adapter 保存在：

```
output/lora-adapter/
├── adapter_model.safetensors  # LoRA 权重 (~6MB)
├── adapter_config.json        # LoRA 配置
├── tokenizer.json             # 分词器
└── training_args.bin          # 训练参数
```

## 系统要求

- **硬件**: Apple M4 Pro (或类似 M 系列)
- **内存**: 建议 16GB+
- **系统**: macOS 15.0+

## 常见问题

**Q: 如何使用其他模型？**
A: 设置 `MODEL_PATH` 环境变量或修改 `train.py` 中的配置。

**Q: 内存不足怎么办？**
A: 
1. 降低 `batch_size` 
2. 启用 `gradient_checkpointing`
3. 使用更小的模型

**Q: 如何提高训练效果？**
A:
1. 增加训练数据量
2. 调整 LoRA rank
3. 训练更多 epochs
4. 尝试不同的 target_modules

## 许可证

MIT License
