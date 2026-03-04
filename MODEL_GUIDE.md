# 模型选择指南 - M4 Pro (24GB)

## 推荐模型 (按大小排序)

| 模型 | 大小 | 内存需求 | 特点 | 下载命令 |
|------|------|----------|------|----------|
| **Phi-3.5-mini** | 3.8B | ~8GB | Microsoft，小而强 | `./download_model.sh` |
| **Qwen2.5-3B** | 3B | ~7GB | 中文优秀 | `MODEL_PATH=hf-mirror.com/Qwen/Qwen2.5-3B uv run python train.py` |
| **Gemma-2-2B** | 2B | ~5GB | Google，极小 | `MODEL_PATH=hf-mirror.com/google/gemma-2-2b-it uv run python train.py` |
| **Llama-3.2-1B** | 1B | ~3GB | Meta，最快 | `MODEL_PATH=hf-mirror.com/meta-llama/Llama-3.2-1B-Instruct uv run python train.py` |

## 使用方式

### 方法 1: 修改配置
```bash
# 编辑 train.py 中的 model_path
nano train.py
# 改为: "model_path": "microsoft/Phi-3.5-mini-instruct"
```

### 方法 2: 环境变量
```bash
MODEL_PATH="microsoft/Phi-3.5-mini-instruct" uv run python train.py
```

### 方法 3: 先下载再使用 (推荐)
```bash
# 下载模型到本地
./download_model.sh

# 使用本地模型
MODEL_PATH="./models/Phi-3.5-mini" uv run python train.py
```

## 下载其他模型

```bash
# Phi-3.5-mini (推荐)
HF_ENDPOINT=https://hf-mirror.com uv run --with huggingface_hub python -c "
from huggingface_hub import snapshot_download
snapshot_download('microsoft/Phi-3.5-mini-instruct', local_dir='./models/Phi-3.5-mini')
"

# Qwen2.5-3B
HF_ENDPOINT=https://hf-mirror.com uv run --with huggingface_hub python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-3B', local_dir='./models/Qwen2.5-3B')
"

# Gemma-2-2B
HF_ENDPOINT=https://hf-mirror.com uv run --with huggingface_hub python -c "
from huggingface_hub import snapshot_download
snapshot_download('google/gemma-2-2b-it', local_dir='./models/Gemma-2-2B')
"
```

## 模型对比建议

- **中文任务**: Qwen 系列 > Phi-3.5 > Llama > Gemma
- **英文任务**: Llama > Gemma > Phi-3.5 > Qwen
- **代码任务**: Phi-3.5 > Qwen > Llama
- **速度优先**: Llama-3.2-1B > Gemma-2-2B > Qwen2.5-3B
