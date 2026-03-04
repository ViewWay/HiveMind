@echo off
REM HiveMind Qwen3.5-4B MoE 训练启动脚本 (Windows)

chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

echo ====================================
echo   HiveMind - Qwen3.5-4B MoE 训练
echo ====================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python
    echo 请安装 Python 3.10+ 或确保 python 在 PATH 中
    pause
    exit /b 1
)

REM 检查脚本目录
if not exist "scripts\train_qwen_full.py" (
    echo [错误] 未找到训练脚本
    echo 请确保在项目根目录运行此脚本
    pause
    exit /b 1
)

REM 检查 uv
uv --version >nul 2>&1
if errorlevel 1 (
    echo [警告] 未找到 uv，使用 Python 直接运行
    set "PYTHON_CMD=python"
) else (
    set "PYTHON_CMD=uv run python"
    echo [信息] 使用 uv 运行
)

REM 设置环境变量
set "PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0"

REM 检查是否传入了参数
set "ARGS=%*"

REM 运行训练脚本
echo.
echo [信息] 开始训练...
echo.
!PYTHON_CMD! scripts\train_qwen_full.py %ARGS%

REM 检查执行结果
if errorlevel 1 (
    echo.
    echo [错误] 训练失败，退出码: !errorlevel!
    pause
    exit /b !errorlevel!
)

echo.
echo [成功] 训练完成！
pause
