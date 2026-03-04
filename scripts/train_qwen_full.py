#!/usr/bin/env python3
"""
HiveMind - Qwen3.5-4B MoE 完整训练管线

一键运行全部 4 个阶段的训练。

支持平台: Linux, macOS, Windows
"""

import os
import sys
import time
import subprocess
import argparse
import platform
import shutil
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

if RICH_AVAILABLE:
    console = Console()
else:
    console = None


# 平台检测
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MAC = platform.system() == "Darwin"


# 训练阶段配置
STAGES = [
    {
        "name": "Stage 1: 基础模型",
        "script": "train_qwen_stage1.py",
        "output": "checkpoints/qwen_stage1",
        "estimated_time": "2-3 小时",
        "description": "加载 Qwen3.5-4B 并添加 LoRA",
    },
    {
        "name": "Stage 2: 领域分化",
        "script": "train_qwen_stage2.py",
        "output": "checkpoints/qwen_stage2",
        "estimated_time": "3-4 小时",
        "description": "代码/写作/数学/知识 四个领域微调",
    },
    {
        "name": "Stage 3: 路由训练",
        "script": "train_qwen_stage3.py",
        "output": "checkpoints/qwen_stage3",
        "estimated_time": "2-3 小时",
        "description": "训练智能路由层",
    },
    {
        "name": "Stage 4: 端到端精调",
        "script": "train_qwen_stage4.py",
        "output": "checkpoints/qwen_final",
        "estimated_time": "3-4 小时",
        "description": "最终整体优化",
    },
]


def get_platform_info() -> str:
    """获取平台信息"""
    system = platform.system()
    machine = platform.machine()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return f"{system} {machine} | Python {python_version}"


def find_python_executable() -> str:
    """查找 Python 可执行文件"""
    # 首先尝试当前 Python
    current_python = sys.executable
    if current_python:
        return current_python

    # 根据平台查找
    python_names = ["python3", "python", "py"] if IS_WINDOWS else ["python3", "python"]

    for name in python_names:
        exe = shutil.which(name)
        if exe:
            return exe

    return "python"


def find_uv_executable() -> Optional[str]:
    """查找 uv 可执行文件"""
    uv_names = ["uv.exe", "uv"] if IS_WINDOWS else ["uv"]

    for name in uv_names:
        exe = shutil.which(name)
        if exe:
            return exe

    return None


def print_header(title: str, subtitle: str = ""):
    """打印标题"""
    if RICH_AVAILABLE:
        console.print(Panel(
            f"[bold cyan]{title}[/bold cyan]",
            title="HiveMind",
            subtitle=subtitle,
            box=box.DOUBLE_EDGE,
            border_style="bright_blue"
        ))
    else:
        print(f"\n{'='*50}")
        print(f"  HiveMind - {title}")
        if subtitle:
            print(f"  {subtitle}")
        print(f"{'='*50}\n")


def print_stage_table(start_from: int = 0):
    """打印阶段表格"""
    if not RICH_AVAILABLE:
        return

    table = Table(title="[bold yellow]训练管线[/bold yellow]", box=box.ROUNDED)
    table.add_column("[cyan]阶段[/cyan]", style="cyan", width=22)
    table.add_column("[yellow]说明[/yellow]", style="yellow")
    table.add_column("[green]预计时间[/green]", style="green")

    for i, stage in enumerate(STAGES):
        prefix = "➤ " if i == start_from else "  "
        if i < start_from:
            prefix = "✓ "
            table.add_row(
                f"{prefix}[dim]{stage['name']}[/dim]",
                f"[dim]{stage['description']}[/dim]",
                "[dim]已完成[/dim]"
            )
        else:
            table.add_row(
                f"{prefix}{stage['name']}",
                stage['description'],
                stage['estimated_time']
            )

    console.print(table)


def check_dependencies() -> bool:
    """检查依赖"""
    if RICH_AVAILABLE:
        console.print("[bold yellow]检查依赖...[/bold yellow]")

    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("peft", "PEFT"),
        ("bitsandbytes", "BitsAndBytes"),
        ("accelerate", "Accelerate"),
        ("datasets", "Datasets"),
        ("rich", "Rich"),
    ]

    missing = []
    for package, display_name in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(display_name)

    if missing:
        if RICH_AVAILABLE:
            console.print(f"[red]✗ 缺少依赖: {', '.join(missing)}[/red]")
            if IS_WINDOWS:
                console.print(f"[dim]运行: uv sync[/dim]")
            else:
                console.print(f"[dim]运行: uv sync[/dim]")
        else:
            print(f"✗ 缺少依赖: {', '.join(missing)}")
            print("运行: uv sync")
        return False

    # 检查 uv
    uv_exe = find_uv_executable()
    if not uv_exe:
        if RICH_AVAILABLE:
            console.print(f"[yellow]⚠ 未找到 uv，将直接使用 Python[/yellow]")
        else:
            print("⚠ 未找到 uv，将直接使用 Python")

    if RICH_AVAILABLE:
        console.print("[green]✓ 所有依赖已安装[/green]")
    return True


def get_default_model_path() -> Path:
    """获取默认模型路径"""
    # 尝试多个可能的路径
    possible_paths = [
        Path("/Users/yimiliya/github/Qwen3.5-4B"),  # macOS
        Path.home() / "models" / "Qwen3.5-4B",  # Linux 用户目录
        Path("C:\\models\\Qwen3.5-4B"),  # Windows
        Path("./Qwen3.5-4B"),  # 相对路径
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # 返回第一个路径作为默认值（用于显示）
    return possible_paths[0]


def check_model_path(custom_path: Optional[str] = None) -> bool:
    """检查模型路径"""
    if custom_path:
        model_path = Path(custom_path)
    else:
        model_path = get_default_model_path()

    if RICH_AVAILABLE:
        console.print(f"[bold yellow]检查 Qwen3.5-4B 模型...[/bold yellow]")
        console.print(f"[dim]路径: {model_path}[/dim]")

    if not model_path.exists():
        if RICH_AVAILABLE:
            console.print(f"[red]✗ 模型目录不存在: {model_path}[/red]")
            console.print(f"[dim]使用 --model-path 指定路径[/dim]")
        else:
            print(f"✗ 模型目录不存在: {model_path}")
            print("使用 --model-path 指定路径")
        return False

    # 检查关键文件
    required_files = [
        "config.json",
        "tokenizer.json",
    ]

    for filename in required_files:
        if not (model_path / filename).exists():
            if RICH_AVAILABLE:
                console.print(f"[red]✗ 缺少文件: {filename}[/red]")
            else:
                print(f"✗ 缺少文件: {filename}")
            return False

    # 检查 safetensors 文件 (支持分片)
    safetensors_files = list(model_path.glob("*.safetensors"))
    # 排除 index.json
    safetensors_files = [f for f in safetensors_files if not f.name.endswith(".index.json")]

    if not safetensors_files:
        if RICH_AVAILABLE:
            console.print(f"[red]✗ 缺少 safetensors 模型文件[/red]")
        else:
            print(f"✗ 缺少 safetensors 模型文件")
        return False

    if RICH_AVAILABLE:
        console.print(f"[dim]  找到 {len(safetensors_files)} 个 safetensors 文件[/dim]")

    if RICH_AVAILABLE:
        console.print(f"[green]✓ Qwen3.5-4B 模型就绪[/green]")

    return True


def check_data() -> bool:
    """检查训练数据"""
    data_path = Path("data/stage2")

    if RICH_AVAILABLE:
        console.print(f"[bold yellow]检查训练数据...[/bold yellow]")

    if not data_path.exists():
        if RICH_AVAILABLE:
            console.print(f"[red]✗ 数据目录不存在: {data_path}[/red]")
        else:
            print(f"✗ 数据目录不存在: {data_path}")
        return False

    required_files = [
        "code_sample.txt",
        "writing_sample.txt",
        "math_sample.txt",
        "knowledge_sample.txt",
    ]

    all_exist = True
    for file in required_files:
        file_path = data_path / file
        if not file_path.exists():
            if RICH_AVAILABLE:
                console.print(f"[yellow]⚠ 缺少数据: {file}[/yellow]")
            else:
                print(f"⚠ 缺少数据: {file}")
            all_exist = False
        elif file_path.stat().st_size == 0:
            if RICH_AVAILABLE:
                console.print(f"[yellow]⚠ 数据文件为空: {file}[/yellow]")
            else:
                print(f"⚠ 数据文件为空: {file}")
            all_exist = False

    if all_exist and RICH_AVAILABLE:
        console.print(f"[green]✓ 所有数据文件就绪[/green]")
    elif not all_exist and RICH_AVAILABLE:
        console.print(f"[yellow]⚠ 部分数据文件缺失或为空[/yellow]")

    return True


def run_stage(stage_index: int, model_path: Optional[str] = None, dry_run: bool = False) -> bool:
    """运行单个阶段"""
    stage = STAGES[stage_index]

    if RICH_AVAILABLE:
        console.print(f"\n[bold cyan]{'='*50}[/bold cyan]")
        console.print(f"[bold cyan]开始: {stage['name']}[/bold cyan]")
        console.print(f"[bold cyan]{'='*50}[/bold cyan]\n")
        console.print(f"[dim]{stage['description']}[/dim]")
    else:
        print(f"\n{'='*50}")
        print(f"开始: {stage['name']}")
        print(f"{'='*50}\n")
        print(stage['description'])

    if dry_run:
        if RICH_AVAILABLE:
            console.print(f"[dim][DRY RUN] 跳过实际训练[/dim]")
        return True

    script_path = Path(__file__).parent / stage["script"]
    if not script_path.exists():
        if RICH_AVAILABLE:
            console.print(f"[red]✗ 脚本不存在: {script_path}[/red]")
        else:
            print(f"✗ 脚本不存在: {script_path}")
        return False

    # 设置环境变量
    env = os.environ.copy()

    # 设置 MPS 内存比例 (macOS)
    if IS_MAC:
        env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    # 设置模型路径
    if model_path:
        env["QWEN_MODEL_PATH"] = model_path

    # 构建命令
    uv_exe = find_uv_executable()
    python_exe = find_python_executable()

    if uv_exe:
        # 使用 uv
        if IS_WINDOWS:
            cmd = [uv_exe, "run", "python", str(script_path)]
        else:
            cmd = [uv_exe, "run", "python", str(script_path)]
    else:
        # 直接使用 Python
        cmd = [python_exe, str(script_path)]

    # 运行脚本
    start_time = time.time()

    try:
        if IS_WINDOWS:
            # Windows 需要特殊处理
            process = subprocess.Popen(
                cmd,
                cwd=Path(__file__).parent.parent,
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0,
            )
            returncode = process.wait()
        else:
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent.parent,
                env=env,
                check=False,
            )
            returncode = result.returncode

        elapsed = time.time() - start_time

        if returncode == 0:
            if RICH_AVAILABLE:
                console.print(f"\n[green]✓ {stage['name']} 完成！[/green]")
                console.print(f"[dim]耗时: {elapsed/60:.1f} 分钟[/dim]")
            else:
                print(f"\n✓ {stage['name']} 完成！")
                print(f"耗时: {elapsed/60:.1f} 分钟")
            return True
        else:
            if RICH_AVAILABLE:
                console.print(f"\n[red]✗ {stage['name']} 失败 (退出码: {returncode})[/red]")
            else:
                print(f"\n✗ {stage['name']} 失败 (退出码: {returncode})")
            return False

    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            console.print(f"\n[yellow]⚠ 训练被用户中断[/yellow]")
        else:
            print(f"\n⚠ 训练被用户中断")
        return False

    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"\n[red]✗ 错误: {e}[/red]")
        else:
            print(f"\n✗ 错误: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="HiveMind Qwen3.5-4B MoE 完整训练管线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行全部阶段
  python scripts/train_qwen_full.py

  # 从指定阶段开始
  python scripts/train_qwen_full.py --start 2

  # 只运行指定阶段
  python scripts/train_qwen_full.py --stage 2

  # 指定模型路径
  python scripts/train_qwen_full.py --model-path /path/to/Qwen3.5-4B

  # 试运行 (不实际训练)
  python scripts/train_qwen_full.py --dry-run

支持平台: Linux, macOS, Windows
        """
    )

    parser.add_argument(
        "--start", "-s",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="从指定阶段开始 (0-3)",
    )

    parser.add_argument(
        "--stage", "-st",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help="只运行指定阶段 (0-3)",
    )

    parser.add_argument(
        "--model-path", "-m",
        type=str,
        default=None,
        help="Qwen3.5-4B 模型路径",
    )

    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="试运行，不实际执行训练",
    )

    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="跳过依赖和模型检查",
    )

    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="跳过确认，直接开始训练",
    )

    args = parser.parse_args()

    # 打印欢迎信息
    print_header("Qwen3.5-4B MoE 训练", "完整管线")

    # 打印平台信息
    if RICH_AVAILABLE:
        console.print(f"[dim]平台: {get_platform_info()}[/dim]\n")

    # 打印训练阶段
    print_stage_table(args.start if args.stage is None else args.stage)

    # 检查依赖
    if not args.skip_checks:
        if not check_dependencies():
            sys.exit(1)

        if not check_model_path(args.model_path):
            sys.exit(1)

        if not check_data():
            pass  # 数据警告但不退出

    # 确定模型路径
    model_path = args.model_path
    if not model_path:
        default_path = get_default_model_path()
        if default_path.exists():
            model_path = str(default_path)

    # 计算总预计时间
    stages_to_run = [args.stage] if args.stage is not None else list(range(args.start, len(STAGES)))
    total_estimated = "10-14 小时"

    if RICH_AVAILABLE:
        console.print(f"\n[bold yellow]即将运行 {len(stages_to_run)} 个阶段，预计总耗时: {total_estimated}[/bold yellow]")
        if model_path:
            console.print(f"[dim]模型路径: {model_path}[/dim]")
        console.print(f"[dim]按 Ctrl+C 可随时中断训练[/dim]")
    else:
        print(f"\n即将运行 {len(stages_to_run)} 个阶段，预计总耗时: {total_estimated}")
        if model_path:
            print(f"模型路径: {model_path}")
        print("按 Ctrl+C 可随时中断训练")

    # 确认
    if not args.yes and not args.dry_run:
        try:
            input("\n按 Enter 开始训练...")
        except (KeyboardInterrupt, EOFError):
            if RICH_AVAILABLE:
                console.print("\n[yellow]已取消[/yellow]")
            else:
                print("\n已取消")
            sys.exit(0)

    # 开始训练
    start_time = time.time()

    if args.stage is not None:
        # 只运行一个阶段
        success = run_stage(args.stage, model_path, args.dry_run)
    else:
        # 运行多个阶段
        success = True
        for i in range(args.start, len(STAGES)):
            if not run_stage(i, model_path, args.dry_run):
                success = False
                if RICH_AVAILABLE:
                    console.print(f"\n[red]训练在 Stage {i+1} 中止[/red]")
                else:
                    print(f"\n训练在 Stage {i+1} 中止")

                # 保存断点信息
                if RICH_AVAILABLE:
                    console.print(f"[dim]要继续训练，运行: --start {i}[/dim]")
                break

    total_elapsed = time.time() - start_time

    # 打印总结
    if RICH_AVAILABLE:
        console.print(f"\n[bold cyan]{'='*50}[/bold cyan]")
        if success:
            console.print(f"[bold green]✓ 训练完成！[/bold green]")
        else:
            console.print(f"[bold yellow]⚠ 训练未完成[/bold yellow]")
        console.print(f"[bold cyan]{'='*50}[/bold cyan]")
        console.print(f"[dim]总耗时: {total_elapsed/60:.1f} 分钟 ({total_elapsed/3600:.1f} 小时)[/dim]")

        if success:
            console.print(f"\n[bold green]最终模型: checkpoints/qwen_final/[/bold green]")
            console.print(f"[bold green]合并模型: checkpoints/qwen_final/merged/[/bold green]")
    else:
        print(f"\n{'='*50}")
        if success:
            print("✓ 训练完成！")
        else:
            print("⚠ 训练未完成")
        print(f"{'='*50}")
        print(f"总耗时: {total_elapsed/60:.1f} 分钟 ({total_elapsed/3600:.1f} 小时)")

        if success:
            print(f"\n最终模型: checkpoints/qwen_final/")
            print(f"合并模型: checkpoints/qwen_final/merged/")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
