"""
阶段 0: 环境验证脚本

验证 HiveMind 项目所需的基础设施是否就绪。
"""

import os
import sys
import time
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# 颜色支持
from logger import Colors, SimpleLogger

# 初始化
if RICH_AVAILABLE:
    console = Console()
else:
    console = None
logger = SimpleLogger()


class EnvironmentChecker:
    """环境检查器"""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.project_root = Path(__file__).parent.parent

    def add_result(self, category: str, item: str, status: bool, message: str, details: str = ""):
        """添加检查结果"""
        self.results.append({
            "category": category,
            "item": item,
            "status": status,
            "message": message,
            "details": details,
        })

    def print_header(self, title: str, subtitle: str = ""):
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
            print(f"\n{'='*60}")
            print(f"  {title}")
            if subtitle:
                print(f"  {subtitle}")
            print(f"{'='*60}\n")

    def print_results_table(self):
        """打印结果表格"""
        if RICH_AVAILABLE:
            table = Table(title="[bold yellow]检查结果[/bold yellow]", box=box.ROUNDED)
            table.add_column("[cyan]类别[/cyan]", width=15)
            table.add_column("[yellow]项目[/yellow]", width=20)
            table.add_column("[green]状态[/green]", width=10)
            table.add_column("[white]说明[/white]", width=30)

            for r in self.results:
                status_icon = "[green]✓[/green]" if r["status"] else "[red]✗[/red]"
                status_text = "[green]通过[/green]" if r["status"] else "[red]失败[/red]"
                table.add_row(
                    r["category"],
                    r["item"],
                    f"{status_icon} {status_text}",
                    r["message"]
                )

            console.print(table)
        else:
            print("\n检查结果:")
            for r in self.results:
                status = "✓ 通过" if r["status"] else "✗ 失败"
                print(f"  [{r['category']}] {r['item']}: {status} - {r['message']}")

    def check_mps_device(self) -> bool:
        """检查 MPS 设备"""
        if RICH_AVAILABLE:
            console.print("[dim]检查 MPS 设备...[/dim]")
        else:
            print("\n检查 MPS 设备...")

        try:
            available = torch.backends.mps.is_available()
            built = torch.backends.mps.is_built()

            if not built:
                self.add_result("硬件", "MPS 支持", False, "PyTorch 未编译 MPS 支持")
                return False

            if not available:
                self.add_result("硬件", "MPS 可用性", False, "MPS 设备不可用")
                return False

            # 测试基本操作
            device = torch.device("mps")
            x = torch.randn(100, 100).to(device)
            y = x @ x.T
            y.cpu()

            # 获取内存信息
            try:
                # 尝试获取 MPS 内存信息
                alloc = torch.mps.current_allocated_memory()
                reserved = torch.mps.current_reserved_memory()
                self.add_result(
                    "硬件",
                    "MPS 设备",
                    True,
                    f"已分配 {alloc / 1024**3:.2f}GB"
                )
            except:
                self.add_result(
                    "硬件",
                    "MPS 设备",
                    True,
                    "运行正常"
                )

            return True

        except Exception as e:
            self.add_result("硬件", "MPS 设备", False, f"错误: {str(e)}")
            return False

    def check_dependencies(self) -> bool:
        """检查依赖包"""
        if RICH_AVAILABLE:
            console.print("[dim]检查依赖包...[/dim]")
        else:
            print("\n检查依赖包...")

        required_packages = {
            "torch": "PyTorch",
            "transformers": "Transformers",
            "datasets": "Datasets",
            "peft": "PEFT",
        }

        all_ok = True
        for package, name in required_packages.items():
            try:
                mod = __import__(package)
                version = getattr(mod, "__version__", "unknown")
                self.add_result("依赖", name, True, f"v{version}")
            except ImportError:
                self.add_result("依赖", name, False, "未安装")
                all_ok = False

        return all_ok

    def check_data_directories(self) -> bool:
        """检查数据目录"""
        if RICH_AVAILABLE:
            console.print("[dim]检查数据目录...[/dim]")
        else:
            print("\n检查数据目录...")

        required_dirs = [
            "data/stage0",
            "data/stage1",
            "data/stage2",
            "data/stage3",
            "data/stage4",
            "checkpoints/stage0",
            "checkpoints/stage1",
            "checkpoints/stage2",
            "checkpoints/stage3",
            "checkpoints/stage4",
        ]

        all_ok = True
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            exists = full_path.exists()
            if exists:
                self.add_result("目录", dir_path, True, "存在")
            else:
                # 尝试创建
                try:
                    full_path.mkdir(parents=True, exist_ok=True)
                    self.add_result("目录", dir_path, True, "已创建")
                except Exception as e:
                    self.add_result("目录", dir_path, False, f"创建失败: {e}")
                    all_ok = False

        return all_ok

    def check_mcp_connection(self) -> bool:
        """检查 MCP 连接"""
        if RICH_AVAILABLE:
            console.print("[dim]检查 MCP 连接...[/dim]")
        else:
            print("\n检查 MCP 连接...")

        # 检查 MCP 服务器
        try:
            from mcp import list_mcp_servers

            servers = list_mcp_servers()

            if not servers:
                self.add_result("MCP", "服务器", False, "未配置 MCP 服务器")
                return False

            self.add_result("MCP", "服务器", True, f"发现 {len(servers)} 个服务器")

            # 检查特定服务器
            required_servers = ["web-reader", "web-search-prime"]
            for server in required_servers:
                found = any(s in str(servers) for s in [server])
                if found:
                    self.add_result("MCP", server, True, "可用")
                else:
                    self.add_result("MCP", server, False, "未找到")

            return True

        except ImportError:
            self.add_result("MCP", "连接", False, "MCP 客户端未安装")
            return False
        except Exception as e:
            self.add_result("MCP", "连接", False, f"错误: {str(e)}")
            return False

    def check_swarm_module(self) -> bool:
        """检查蜂群模块"""
        if RICH_AVAILABLE:
            console.print("[dim]检查蜂群模块...[/dim]")
        else:
            print("\n检查蜂群模块...")

        try:
            from swarm import create_swarm_model

            # 创建小型测试模型
            model = create_swarm_model(num_experts=2, expert_size="small")

            # 获取参数量
            num_params = model.get_num_params()

            self.add_result("模块", "SwarmModel", True, f"{num_params:,} 参数")

            return True

        except Exception as e:
            self.add_result("模块", "SwarmModel", False, f"错误: {str(e)}")
            return False

    def run_all_checks(self) -> bool:
        """运行所有检查"""
        self.print_header("HiveMind 环境验证", "阶段 0")

        # 运行检查
        mps_ok = self.check_mps_device()
        deps_ok = self.check_dependencies()
        dirs_ok = self.check_data_directories()
        mcp_ok = self.check_mcp_connection()
        swarm_ok = self.check_swarm_module()

        # 打印结果
        print()
        self.print_results_table()

        # 统计
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"])
        failed = total - passed

        if RICH_AVAILABLE:
            console.print("")
            if failed == 0:
                console.print("[bold green]✓ 所有检查通过！环境已就绪。[/bold green]")
            else:
                console.print(f"[bold yellow]⚠ {passed}/{total} 检查通过，{failed} 项失败[/bold yellow]")
        else:
            print(f"\n{passed}/{total} 检查通过，{failed} 项失败")

        return failed == 0


def main():
    """主函数"""
    start_time = time.time()

    checker = EnvironmentChecker()
    all_ok = checker.run_all_checks()

    elapsed = time.time() - start_time

    if RICH_AVAILABLE:
        console.print(f"[dim]验证耗时: {elapsed:.1f}秒[/dim]")
    else:
        print(f"\n验证耗时: {elapsed:.1f}秒")

    # 返回退出码
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
