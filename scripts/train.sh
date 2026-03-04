#!/bin/bash
# HiveMind Qwen3.5-4B MoE 训练启动脚本 (Linux/macOS)

set -e

echo "===================================="
echo "  HiveMind - Qwen3.5-4B MoE 训练"
echo "===================================="
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3"
    echo "请安装 Python 3.10+"
    exit 1
fi

# 检查 uv
if command -v uv &> /dev/null; then
    CMD="uv run python"
    echo "使用 uv 运行"
else
    CMD="python3"
    echo "使用 Python 直接运行"
fi

# 设置环境变量
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# 运行训练脚本
echo "开始训练..."
echo ""
$CMD scripts/train_qwen_full.py "$@"
