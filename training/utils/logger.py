"""
优化的训练日志格式
使用 rich 库实现美观的终端输出
"""

import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.text import Text
from rich import box


class BeautifulLogger:
    """美观的训练日志记录器"""

    def __init__(self):
        self.console = Console()
        self.start_time = None
        self.step = 0
        self.total_steps = 0

    def print_header(self, title: str, subtitle: str = ""):
        """打印标题"""
        self.console.print()
        self.console.print(Panel(
            f"[bold cyan]{title}[/bold cyan]",
            title="LLM Training" if not subtitle else "",
            subtitle=subtitle,
            box=box.DOUBLE_EDGE,
            border_style="bright_blue"
        ))

    def print_config(self, config: dict, model_info: dict):
        """打印配置信息"""
        table = Table(title="[bold yellow]配置信息[/bold yellow]", box=box.ROUNDED)
        table.add_column("[cyan]参数[/cyan]", style="cyan")
        table.add_column("[yellow]值[/yellow]", style="yellow")

        for key, value in config.items():
            if isinstance(value, list) and len(value) > 5:
                value = f"[{len(value)} 项] {', '.join(map(str, value[:3]))}..."
            table.add_row(key, str(value))

        for key, value in model_info.items():
            table.add_row(key, str(value))

        self.console.print(table)

    def print_model_stats(self, model):
        """打印模型统计"""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        stats_table = Table(title="[bold green]模型统计[/bold green]", box=box.SIMPLE_HEAD)
        stats_table.add_column("指标", justify="left")
        stats_table.add_column("值", justify="right")
        stats_table.add_row("总参数", f"[white]{total:,}[/white]")
        stats_table.add_row("可训练参数", f"[green]{trainable:,}[/green]")
        stats_table.add_row("训练占比", f"[yellow]{trainable/total*100:.4f}%[/yellow]")

        self.console.print(stats_table)

    def print_data_info(self, dataset):
        """打印数据信息"""
        self.console.print(f"[dim]📚 训练数据: {len(dataset)} 条[/dim]")

    def start_training(self, total_steps: int):
        """开始训练进度条"""
        self.total_steps = total_steps
        self.start_time = time.time()

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
            transient=True
        )

        self.task = self.progress.add_task(
            "[cyan]训练中...",
            total=total_steps
        )

        self.progress.start()

    def update_step(self, step: int, loss: float, learning_rate: float):
        """更新训练进度"""
        self.progress.update(self.task, completed=step, description=f"[cyan]Step {step}/{self.total_steps}")

    def finish_step(self, step: int, loss: float, duration: float):
        """完成一个步骤"""
        elapsed = time.time() - self.start_time
        remaining = (elapsed / step) * (self.total_steps - step)

        self.console.print(f"  [dim]Step {step:2d}[/dim] | "
                          f"[cyan]Loss: {loss:.4f}[/cyan] | "
                          f"[green]用时: {duration:.1f}s[/green] | "
                          f"[dim]预计剩余: {remaining/60:.1f}min[/dim]")

    def print_training_summary(self, results: dict):
        """打印训练汇总"""
        elapsed = time.time() - self.start_time

        summary_table = Table(title="[bold green]训练完成[/bold green]", box=box.DOUBLE_EDGE)
        summary_table.add_column("[cyan]指标[/cyan]", justify="left")
        summary_table.add_column("[yellow]值[/yellow]", justify="right")

        for key, value in results.items():
            summary_table.add_row(key, str(value))

        summary_table.add_row("总用时", f"{elapsed:.1f}秒")

        self.console.print(summary_table)

    def print_save_info(self, path: str, size_mb: float):
        """打印保存信息"""
        self.console.print(f"[dim]💾 保存到: {path}[/dim]")
        self.console.print(f"[dim]📦 大小: {size_mb:.2f} MB[/dim]")

    def print_separator(self):
        """打印分隔线"""
        self.console.print()

    def print_error(self, message: str):
        """打印错误"""
        self.console.print(f"[red]✗ {message}[/red]")

    def print_success(self, message: str):
        """打印成功"""
        self.console.print(f"[green]✓ {message}[/green]")

    def print_info(self, message: str):
        """打印信息"""
        self.console.print(f"[dim]ℹ {message}[/dim]")


# 简化的彩色输出（用于不想依赖 rich 的情况）
class SimpleLogger:
    """简单彩色日志记录器"""

    @staticmethod
    def print_header(title: str):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}\n")

    @staticmethod
    def print_config(config: dict):
        print("配置信息:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print()

    @staticmethod
    def print_model_stats(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型统计:")
        print(f"  总参数: {total:,}")
        print(f"  可训练参数: {trainable:,} ({trainable/total*100:.4f}%)")
        print()

    @staticmethod
    def print_step(step, total, loss, duration):
        print(f"Step {step}/{total} | Loss: {loss:.4f} | 用时: {duration:.1f}s")

    @staticmethod
    def print_summary(results, elapsed):
        print(f"\n{'='*60}")
        print("  训练完成")
        print(f"{'='*60}")
        for key, value in results.items():
            print(f"  {key}: {value}")
        print(f"  总用时: {elapsed:.1f}秒")
        print(f"{'='*60}\n")


# ANSI 颜色代码
class Colors:
    HEADER = "\033[1;36m"  # Cyan bold
    SUCCESS = "\033[1;32m"  # Green bold
    WARNING = "\033[1;33m"  # Yellow bold
    ERROR = "\033[1;31m"   # Red bold
    INFO = "\033[0;36m"    # Cyan
    RESET = "\033[0m"

    @staticmethod
    def header(text):
        return f"{Colors.HEADER}{text}{Colors.RESET}"

    @staticmethod
    def success(text):
        return f"{Colors.SUCCESS}{text}{Colors.RESET}"

    @staticmethod
    def warning(text):
        return f"{Colors.WARNING}{text}{Colors.RESET}"

    @staticmethod
    def error(text):
        return f"{Colors.ERROR}{text}{Colors.RESET}"

    @staticmethod
    def info(text):
        return f"{Colors.INFO}{text}{Colors.RESET}"


if __name__ == "__main__":
    logger = SimpleLogger()
    logger.print_header("测试日志输出")
    print(Colors.header("彩色标题"))
    print(Colors.success("成功消息"))
    print(Colors.warning("警告消息"))
    print(Colors.error("错误消息"))
    print(Colors.info("信息消息"))
