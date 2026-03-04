# HiveMind Qwen3.5-4B MoE 训练启动脚本 (PowerShell)
# 兼容 Windows PowerShell 5.1+ 和 PowerShell Core 6+

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "  HiveMind - Qwen3.5-4B MoE 训练" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Python
$pythonCmd = Get-Command "python" -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    $pythonCmd = Get-Command "python3" -ErrorAction SilentlyContinue
}

if (-not $pythonCmd) {
    Write-Host "[错误] 未找到 Python" -ForegroundColor Red
    Write-Host "请安装 Python 3.10+ 或确保 python 在 PATH 中" -ForegroundColor Yellow
    Read-Host "按 Enter 退出"
    exit 1
}

Write-Host "[信息] Python: $($pythonCmd.Source)" -ForegroundColor Green

# 检查脚本
$scriptPath = Join-Path $PSScriptRoot "train_qwen_full.py"
if (-not (Test-Path $scriptPath)) {
    Write-Host "[错误] 未找到训练脚本: $scriptPath" -ForegroundColor Red
    Write-Host "请确保在项目目录运行此脚本" -ForegroundColor Yellow
    Read-Host "按 Enter 退出"
    exit 1
}

# 检查 uv
$uvCmd = Get-Command "uv" -ErrorAction SilentlyContinue
if ($uvCmd) {
    $pythonExec = "uv", "run", "python"
    Write-Host "[信息] 使用 uv 运行" -ForegroundColor Green
} else {
    $pythonExec = "python"
    Write-Host "[信息] 使用 Python 直接运行" -ForegroundColor Yellow
}

# 设置环境变量
$env:PYTORCH_MPS_HIGH_WATERMARK_RATIO = "0.0"

# 传递参数
$argsList = $args

# 运行训练脚本
Write-Host ""
Write-Host "[信息] 开始训练..." -ForegroundColor Green
Write-Host ""

& $pythonExec $scriptPath @argsList

# 检查结果
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[错误] 训练失败，退出码: $LASTEXITCODE" -ForegroundColor Red
    Read-Host "按 Enter 退出"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "[成功] 训练完成！" -ForegroundColor Green
