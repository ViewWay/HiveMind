#!/bin/bash
# 下载 Phi-3.5-mini 模型 (推荐用于 M4 Pro)

MODEL="microsoft/Phi-3.5-mini-instruct"
OUTPUT_DIR="./models/Phi-3.5-mini"

echo "下载 $MODEL ..."

HF_ENDPOINT=https://hf-mirror.com uv run --with huggingface_hub python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$MODEL',
    local_dir='$OUTPUT_DIR',
    local_dir_use_symlinks=False
)
print('下载完成!')
"
