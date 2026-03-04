@echo off
REM HiveMind Qwen3.5-4B MoE 训练启动脚本 (Windows)

echo ====================================
echo   HiveMind - Qwen3.5-4B MoE 训练
echo ====================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Python
    echo 请安装 Python 3.10+
    pause
    exit /b 1
)

REM 检查 uv
uv --version >nul 2>&1
if errorlevel 1 (
    echo 警告: 未找到 uv，使用 Python 直接运行
    set CMD=python
) else (
    set CMD=uv run python
)

REM 设置环境变量
set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

REM 运行训练脚本
echo 开始训练...
echo.
%CMD% scripts\train_qwen_full.py %*

pause
