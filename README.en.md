# HiveMind - Swarm Intelligence Expert System

**[中文](./README.zh-CN.md)** | English

A lightweight adaptive MoE (Mixture of Experts) architecture implemented on Apple M4 Pro (24GB), achieving large-model capabilities through collaborative "Swarm Expert" mechanisms.

## Key Features

| Feature | Description |
|---------|-------------|
| **Swarm Experts** | Multiple lightweight experts working collaboratively, each specializing in specific domains |
| **Adaptive Routing** | Dynamically selects the number of activated experts based on task complexity |
| **Hybrid Routing** | Combines soft routing and Top-K sparse activation for optimal efficiency |
| **Load Balancing** | Intelligent scheduling to prevent expert overuse |
| **Staged Training** | Progressive training strategy achieving optimal results with limited compute |

## Project Structure

```
HiveMind/
├── README.zh-CN.md         # Chinese documentation
├── README.en.md            # English documentation (this file)
├── pyproject.toml           # Project configuration
│
├── swarm/                   # Swarm Expert core module
│   ├── experts.py           # Expert pool implementation
│   ├── router.py            # Hybrid routing layer
│   └── swarm_model.py       # Complete model
│
├── training/                # Training modules
│   ├── configs/             # Training configurations
│   ├── lora/                # LoRA training scripts
│   └── utils/              # Training utilities
│
├── inference/               # Inference modules
│   ├── configs/             # Inference configurations
│   ├── generate.py          # Generation scripts
│   └── compare_lora.py      # LoRA comparison
│
├── scripts/                 # Utility scripts
│   ├── verify_env.py        # Environment verification
│   ├── train_stage1.py      # Stage 1 training
│   └── data_crawler.py       # Data crawler
│
├── tests/                   # Test files
│   ├── test_swarm.py         # Swarm model tests
│   └── test_lora.py          # LoRA tests
│
├── data/                    # Data directory
├── checkpoints/             # Model checkpoints
├── docs/                   # Documentation
└── archive/                # Archived files
```

## Quick Start

### 1. Install Dependencies

```bash
uv add torch transformers datasets peft accelerate rich
```

### 2. Environment Verification

```bash
uv run python scripts/verify_env.py
```

### 3. Test Swarm Model

```bash
uv run python tests/test_swarm.py
```

### 4. LoRA Training

```bash
# Traditional LoRA training
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 uv run python training/lora/train.py
```

### 5. Inference

```bash
# Single inference
uv run python inference/generate.py --mode single --prompt "What is AI?"

# Interactive inference
uv run python inference/generate.py --mode chat
```

## Staged Training

| Stage | Script | Goal | Duration |
|-------|--------|------|----------|
| Stage 0 | `scripts/verify_env.py` | Environment verification | 30min |
| Stage 1 | `scripts/train_stage1.py` | Expert initialization | 2-3h |
| Stage 2 | `scripts/train_stage2.py` | Domain specialization | 16-24h |
| Stage 3 | `scripts/train_stage3.py` | Router training | 6-8h |
| Stage 4 | `scripts/train_stage4.py` | End-to-end fine-tuning | 8-12h |

## Model Usage

```python
from swarm import create_swarm_model

# Create model
model = create_swarm_model(
    num_experts=8,
    expert_size="small",  # small, medium, large
)

# Generate text
input_ids = tokenizer("What is artificial intelligence?", return_tensors="pt").input_ids
output = model.generate(input_ids, max_new_tokens=100)
```

## Parameter Scale

| Component | Parameters | Activated |
|-----------|-----------|-----------|
| Task Analyzer | 2M | 2M |
| Hybrid Router | 4M | 4M |
| Swarm Expert Pool | 50-200M | 10-25M |
| Weighted Fusion | 1M | 1M |
| **Total** | **57-207M** | **17-32M** |

## Performance Targets

| Metric | Target | Comparison Models |
|--------|--------|-------------------|
| MMLU | 70% | Qwen2-1.5B: 42.5%, Phi-2: 55.8% |
| HumanEval | 50% | Qwen2-1.5B: 18.3%, Phi-2: 28.5% |
| GSM8K | 65% | Qwen2-1.5B: 38.2%, Phi-2: 55.4% |

## System Requirements

- **Hardware**: Apple M4 Pro (or similar M-series)
- **Memory**: 24GB+
- **System**: macOS 15.0+

## Documentation

Detailed design: [docs/2025-03-04-swarm-experts-design.md](docs/2025-03-04-swarm-experts-design.md)

Model comparison: [docs/model-comparison.md](docs/model-comparison.md)

## License

MIT License

---

**HiveMind** - Small Model, Big Intelligence 🧠
