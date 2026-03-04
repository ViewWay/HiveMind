#!/bin/bash
# HiveMind Qwen3.5-4B MoE 训练启动脚本 (Linux/macOS)
# 兼容 bash 和 zsh

set -e

# 颜色定义 (跨平台兼容)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    CYAN='\033[0;36m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    CYAN=''
    NC=''
fi

echo ""
echo "===================================="
echo "  HiveMind - Qwen3.5-4B MoE 训练"
echo "===================================="
echo ""

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 检查 Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}[错误] 未找到 Python${NC}"
    echo "请安装 Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo -e "${GREEN}[信息] Python: $PYTHON_VERSION${NC}"

# 检查训练脚本
TRAIN_SCRIPT="$PROJECT_ROOT/scripts/train_qwen_full.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo -e "${RED}[错误] 未找到训练脚本: $TRAIN_SCRIPT${NC}"
    echo "请确保在项目根目录运行此脚本"
    exit 1
fi

# 检查 uv
if command -v uv &> /dev/null; then
    PYTHON_EXEC="uv run python"
    echo -e "${GREEN}[信息] 使用 uv 运行${NC}"
else
    PYTHON_EXEC="$PYTHON_CMD"
    echo -e "${YELLOW}[警告] 未找到 uv，使用 Python 直接运行${NC}"
fi

# 设置环境变量
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.0"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 运行训练脚本
echo ""
echo -e "${CYAN}[信息] 开始训练...${NC}"
echo ""

$PYTHON_EXEC scripts/train_qwen_full.py "$@"

# 检查结果
if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}[错误] 训练失败，退出码: $?${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}[成功] 训练完成！${NC}"
