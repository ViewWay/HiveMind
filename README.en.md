# HiveMind - Swarm Intelligence Expert System

**[中文](./README.zh-CN.md)** | English

> **Small Model, Big Intelligence, Low Energy** - Building a lightweight AI on Apple M4 Pro that can compete with Claude Opus

---

## 🎯 Project Vision

**Bringing AI to everyone's device with 100x less energy consumption**

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   Current: Claude/GPT require massive cloud clusters, 1000W+     │
│   Target: HiveMind runs on local devices, 10-30W                │
│                                                                  │
│   Energy Efficiency: 100x │ Deployment Cost: $0 │ Privacy: 100% │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Project?

| Problem | Traditional Approach | HiveMind Approach |
|---------|---------------------|-------------------|
| **Energy** | Data centers 1000W+ | Local device 10-30W |
| **Privacy** | Data uploaded to cloud | Data never leaves device |
| **Cost** | Pay-per-use API | One-time training, unlimited use |
| **Dependency** | Requires internet | Completely offline |
| **Customization** | Black box model | Open source & customizable |

### Is It Possible?

```
AI history proves "impossible" is often proven wrong:

2022: "Small models can never match ChatGPT"
2023: Llama, Mistral, Phi prove small models can be strong
2024: DeepSeek-V3 achieves efficiency with MoE architecture
2025+: Can HiveMind match large models with 1/1000 energy?
```

**Core Belief**: Through MoE (Mixture of Experts) + Sparse Activation + Domain Specialization, small models can absolutely match large model performance in specific scenarios.

---

## 📊 Key Features

| Feature | Description |
|---------|-------------|
| **Swarm Experts** | Multiple lightweight experts collaborate, each specializing in specific domains |
| **Adaptive Routing** | Dynamically selects number of activated experts (2-6) based on task complexity |
| **Hybrid Routing** | Combines soft routing and Top-K sparse activation for optimal efficiency |
| **Load Balancing** | Intelligent scheduling prevents expert overuse |
| **Staged Training** | Progressive training strategy achieving optimal results with limited compute |
| **Local Deployment** | Completely offline operation with 100% data privacy |

---

## 🗺️ Roadmap

### Phase 1: Foundation (2026 Q1)

```
✓ Core architecture complete
⚠ Four-stage training pipeline
⚠ Benchmark testing framework
```

**Goal**: Surpass Qwen2-1.5B, Phi-2 and other small models

### Phase 2: Domain Breakthrough (2026 Q2-Q4)

Focus on vertical domains with small data achieving big results:

- **Chinese Coding Assistant**: 100-500K high-quality code samples
- **Chinese Writing Assistant**: 500K-1M high-quality text samples
- **Chinese Reasoning Assistant**: 50-200K chain-of-thought samples

**Goal**: Approach Claude performance in specific domains

### Phase 3: Architecture Evolution (2027)

| Dimension | Current | Advanced | Expert |
|-----------|---------|----------|--------|
| Experts | 8 | 32 | 128 |
| Total Params | 200M | 1.6B | 12.8B |
| Activation Ratio | 20% | 10% | 5% |
| Context Length | 2K-4K | 16K-32K | 64K-128K |

New technologies: MLA, Sparse Attention, RL, Knowledge Distillation

### Phase 4: Continuous Learning (2028-2030)

**Ultimate Goal**: Match Claude level in 1-2 vertical domains

| Dimension | Claude Opus | HiveMind Target |
|-----------|-------------|-----------------|
| Coding (Chinese) | ~90% | ~85% |
| Math (Chinese) | ~95% | ~80% |
| Writing (Chinese) | ~95% | ~85% |
| **Energy** | **1000W+** | **10-30W** |
| **Deployment** | **Cloud** | **Local** |

Detailed roadmap: [docs/2026-03-05-roadmap-to-claude-level.md](docs/2026-03-05-roadmap-to-claude-level.md)

---

## 🚀 Quick Start

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

Output:
```
Initializing swarm model...
Using device: mps

Model info:
  Total parameters: 44,789,906
  Expert parameters: 5,403,648
  Router parameters: 76,050
  Number of experts: 8

✓ All tests passed!
```

### 4. Staged Training

```bash
# Stage 1: Expert initialization (2-3 hours)
uv run python scripts/train_stage1.py

# Stage 2: Domain specialization (16-24 hours)
uv run python scripts/train_stage2.py

# Stage 3: Router training (6-8 hours)
uv run python scripts/train_stage3.py

# Stage 4: End-to-end fine-tuning (8-12 hours)
uv run python scripts/train_stage4.py
```

### 5. Inference

```bash
# Single inference
uv run python inference/generate.py --mode single --prompt "What is AI?"

# Interactive chat
uv run python inference/generate.py --mode chat
```

---

## 📁 Project Structure

```
HiveMind/
├── README.zh-CN.md         # Chinese documentation
├── README.en.md            # English documentation (this file)
├── pyproject.toml          # Project configuration
│
├── swarm/                  # Swarm Expert core module
│   ├── __init__.py
│   ├── experts.py          # Expert pool implementation
│   ├── router.py           # Hybrid routing layer
│   └── swarm_model.py      # Complete model
│
├── training/               # Training modules
│   ├── configs/            # Training configurations
│   ├── lora/               # LoRA training scripts
│   └── utils/              # Training utilities
│
├── inference/              # Inference modules
│   ├── configs/            # Inference configurations
│   ├── generate.py         # Generation scripts
│   └── compare_lora.py     # LoRA comparison
│
├── scripts/                # Utility scripts
│   ├── verify_env.py       # Environment verification
│   ├── train_stage*.py     # Stage-wise training
│   └── data_crawler.py     # Data crawler
│
├── tests/                  # Test files
│   ├── test_swarm.py       # Swarm model tests
│   └── test_lora.py        # LoRA tests
│
├── data/                   # Data directory
├── checkpoints/            # Model checkpoints
└── docs/                   # Documentation
```

---

## 📐 Parameter Scale

| Component | Parameters | Activated |
|-----------|-----------|-----------|
| Task Analyzer | 2M | 2M |
| Hybrid Router | 4M | 4M |
| Swarm Expert Pool | 50-200M | 10-25M |
| Weighted Fusion | 1M | 1M |
| **Total** | **57-207M** | **17-32M** |

**Sparse Activation**: Only 20% parameters participate in inference, significantly reducing energy consumption

---

## 🎯 Performance Targets

| Metric | Phase 1 | Phase 2 | Phase 3 | Comparison |
|--------|---------|---------|---------|------------|
| MMLU | 50% | 70% | 85% | Qwen2-1.5B: 42.5% |
| CMMLU | 55% | 75% | 88% | Chinese general knowledge |
| HumanEval | 30% | 50% | 70% | Qwen2-1.5B: 18.3% |
| GSM8K | 45% | 65% | 80% | Qwen2-1.5B: 38.2% |

---

## 💻 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Hardware | Apple M1 | Apple M4 Pro |
| Memory | 16GB | 24GB+ |
| System | macOS 14.0 | macOS 15.0+ |

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Roadmap](docs/2026-03-05-roadmap-to-claude-level.md) | Long-term development plan |
| [Design Doc](docs/2026-03-04-swarm-experts-design.md) | Architecture design details |
| [Model Comparison](docs/model-comparison.md) | Comparison with other models |
| [Capability Analysis](docs/MODEL_CAPABILITY.md) | Current model capabilities |

---

## 🤝 Contributing

This is a long-term research project. We welcome contributions in all forms:

- **Code**: PRs to fix bugs or add features
- **Data**: Provide high-quality training data
- **Experiments**: Try different configs and share results
- **Documentation**: Improve docs and tutorials
- **Feedback**: Report bugs or suggest improvements

---

## 📄 License

MIT License

---

## 💬 Conclusion

> "The best time to plant a tree was 20 years ago. The second best time is now."

This project's goal is not to replace Claude, but to prove:

- **Small Model + Good Architecture = Big Capability**
- **Low Energy + Local Deployment = Privacy Protection**
- **Open Source + Community = Sustainable Development**

The future of AI shouldn't be controlled by a few big companies. Let's bring AI to everyone's device together!

---

**HiveMind** - Small Model, Big Intelligence 🧠

*Updated: 2026-03-05*
