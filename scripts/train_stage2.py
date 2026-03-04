"""
阶段 2: 专业分化训练

让专家向不同领域分化，每个专家对特定领域进行微调。
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm import SwarmExpert, SwarmExpertPool, ExpertConfig

# Rich 支持
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# 初始化
if RICH_AVAILABLE:
    console = Console()
else:
    console = None


# 领域配置
DOMAIN_CONFIGS = {
    "code": {
        "experts": [0, 1],
        "data_file": "data/stage2/code_sample.txt",
        "name": "Code Programming",
        "emoji": "💻"
    },
    "writing": {
        "experts": [2, 3],
        "data_file": "data/stage2/writing_sample.txt",
        "name": "Literary Writing",
        "emoji": "📝"
    },
    "math": {
        "experts": [4, 5],
        "data_file": "data/stage2/math_sample.txt",
        "name": "Math Reasoning",
        "emoji": "🔢"
    },
    "knowledge": {
        "experts": [6, 7],
        "data_file": "data/stage2/knowledge_sample.txt",
        "name": "Knowledge Q&A",
        "emoji": "📚"
    },
}


@dataclass
class Stage2Config:
    """阶段2训练配置"""

    # 数据
    max_seq_length: int = 512  # 阶段2使用更长序列
    vocab_size: int = 151936

    # 专家配置 (应与阶段1一致)
    num_experts: int = 8
    expert_hidden_size: int = 128
    expert_num_layers: int = 2
    expert_num_heads: int = 4
    expert_intermediate_size: int = 512

    # 训练配置
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-4  # 较低学习率用于微调
    num_epochs: int = 30  # 更多轮次用于领域适应
    warmup_steps: int = 30

    # 系统
    device: str = "mps"
    stage1_checkpoint_dir: str = "checkpoints/stage1"
    output_dir: str = "checkpoints/stage2"

    def to_expert_config(self) -> ExpertConfig:
        """转换为专家配置"""
        return ExpertConfig(
            hidden_size=self.expert_hidden_size,
            num_layers=self.expert_num_layers,
            num_heads=self.expert_num_heads,
            intermediate_size=self.expert_intermediate_size,
            max_seq_length=self.max_seq_length,
        )


class DomainDataset(Dataset):
    """领域特定数据集"""

    def __init__(self, data_path: str, max_length: int = 512, vocab_size: int = 151936):
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.vocab_size = vocab_size

        # 检查文件是否存在
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        # 加载数据
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.texts = [line.strip() for line in f if line.strip()]

        print(f"  加载了 {len(self.texts)} 条 {self.data_path.stem} 训练样本")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # 简单分词 (使用字符级别作为占位符)
        tokens = [ord(c) % self.vocab_size for c in text[:self.max_length]]

        # Padding
        if len(tokens) < self.max_length:
            tokens.extend([0] * (self.max_length - len(tokens)))

        # 输入和标签 (移位预测)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class DomainSpecializer:
    """领域专门化训练器"""

    def __init__(
        self,
        experts: List[SwarmExpert],
        config: Stage2Config,
        domain_name: str,
        domain_config: Dict,
    ):
        self.experts = experts
        self.config = config
        self.domain_name = domain_name
        self.domain_config = domain_config
        self.device = torch.device(config.device)

        # 为每个专家创建优化器
        self.optimizers = []
        self.schedulers = []

        for expert in self.experts:
            optimizer = torch.optim.AdamW(
                expert.parameters(),
                lr=config.learning_rate,
                weight_decay=0.01,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.warmup_steps,
            )
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)

        # 统计
        self.train_losses = []

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """训练一个 epoch"""
        for expert in self.experts:
            expert.train()

        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # 前向传播 (所有专家)
            batch_size, seq_len = input_ids.shape
            hidden_size = self.config.expert_hidden_size

            # 简单嵌入 (共享)
            embedding = nn.Embedding(self.config.vocab_size, hidden_size).to(self.device)
            hidden_states = embedding(input_ids)

            # 计算每个专家的损失
            losses = []
            for i, expert in enumerate(self.experts):
                expert_output = expert(hidden_states)

                # 输出投影
                output_proj = nn.Linear(hidden_size, self.config.vocab_size).to(self.device)
                logits = output_proj(expert_output)

                # 计算损失
                loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    labels.view(-1),
                    ignore_index=-100,
                )
                losses.append(loss)

            # 平均损失
            avg_loss = sum(losses) / len(losses)

            # 反向传播
            avg_loss = avg_loss / self.config.gradient_accumulation_steps
            avg_loss.backward()

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                for optimizer in self.optimizers:
                    torch.nn.utils.clip_grad_norm_(self.experts[0].parameters(), 1.0)

                for i, optimizer in enumerate(self.optimizers):
                    optimizer.step()
                    self.schedulers[i].step()
                    optimizer.zero_grad()

            total_loss += avg_loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

        return total_loss / num_batches

    def train(self, dataloader: DataLoader, num_epochs: int):
        """完整训练"""
        for epoch in range(num_epochs):
            loss = self.train_epoch(dataloader, epoch)
            self.train_losses.append(loss)

            expert_ids = [e.expert_id for e in self.experts]
            if RICH_AVAILABLE:
                console.print(
                    f"[dim]    Experts {expert_ids} | Epoch {epoch+1}/{num_epochs} | "
                    f"Loss: {loss:.4f}[/dim]"
                )
            else:
                print(f"    Experts {expert_ids} | Epoch {epoch+1}/{num_epochs} | Loss: {loss:.4f}")


def load_stage1_experts(config: Stage2Config) -> List[SwarmExpert]:
    """加载阶段1训练的专家"""
    stage1_dir = Path(config.stage1_checkpoint_dir)

    if not stage1_dir.exists():
        raise FileNotFoundError(
            f"阶段1检查点不存在: {stage1_dir}\n"
            f"请先运行阶段1训练: uv run python scripts/train_stage1.py"
        )

    experts = []

    for i in range(config.num_experts):
        expert_path = stage1_dir / f"expert_{i}.pt"

        if not expert_path.exists():
            raise FileNotFoundError(f"专家检查点不存在: {expert_path}")

        expert = SwarmExpert.load(str(expert_path), device=config.device)
        experts.append(expert)

        if RICH_AVAILABLE:
            console.print(f"[dim]  加载 Expert {i}: {expert_path}[/dim]")
        else:
            print(f"  加载 Expert {i}: {expert_path}")

    return experts


def train_stage2(config: Stage2Config):
    """阶段2训练主函数"""

    # 打印标题
    if RICH_AVAILABLE:
        console.print(Panel(
            f"[bold cyan]专业分化训练[/bold cyan]",
            title="HiveMind",
            subtitle="阶段 2",
            box=box.DOUBLE_EDGE,
            border_style="bright_blue"
        ))

        # 打印配置
        table = Table(title="[bold yellow]训练配置[/bold yellow]", box=box.ROUNDED)
        table.add_column("[cyan]参数[/cyan]", style="cyan")
        table.add_column("[yellow]值[/yellow]", style="yellow")

        for key, value in [
            ("专家数量", config.num_experts),
            ("学习率", config.learning_rate),
            ("训练轮数", config.num_epochs),
            ("批次大小", config.batch_size),
            ("梯度累积", config.gradient_accumulation_steps),
            ("阶段1检查点", config.stage1_checkpoint_dir),
        ]:
            table.add_row(key, str(value))

        console.print(table)
        console.print()
    else:
        print("\n" + "="*50)
        print("  HiveMind - 阶段2: 专业分化训练")
        print("="*50 + "\n")

    # 创建输出目录
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # 加载阶段1的专家
    if RICH_AVAILABLE:
        console.print(f"[bold yellow]加载阶段1专家[/bold yellow]")
    else:
        print("加载阶段1专家")

    experts = load_stage1_experts(config)

    # 开始训练
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console if RICH_AVAILABLE else None,
        disable=not RICH_AVAILABLE,
    ) as progress:
        task = progress.add_task("领域分化...", total=len(DOMAIN_CONFIGS))

        for domain_key, domain_config in DOMAIN_CONFIGS.items():
            emoji = domain_config["emoji"]
            domain_name = domain_config["name"]
            expert_ids = domain_config["experts"]
            data_file = domain_config["data_file"]

            if RICH_AVAILABLE:
                console.print(f"\n[bold yellow]{emoji} 训练领域: {domain_name}[/bold yellow]")
                console.print(f"[dim]  专家: {expert_ids}[/dim]")
                console.print(f"[dim]  数据: {data_file}[/dim]")
            else:
                print(f"\n{emoji} 训练领域: {domain_name}")
                print(f"  专家: {expert_ids}")
                print(f"  数据: {data_file}")

            # 加载数据
            try:
                dataset = DomainDataset(data_file, config.max_seq_length, config.vocab_size)
                dataloader = DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=0,
                )
            except FileNotFoundError as e:
                if RICH_AVAILABLE:
                    console.print(f"[red]  跳过: {e}[/red]")
                else:
                    print(f"  跳过: {e}")
                progress.update(task, advance=1)
                continue

            # 获取对应的专家
            domain_experts = [experts[i] for i in expert_ids]

            # 训练
            specializer = DomainSpecializer(
                domain_experts, config, domain_key, domain_config
            )
            specializer.train(dataloader, config.num_epochs)

            # 保存专家
            for expert in domain_experts:
                expert_path = f"{config.output_dir}/expert_{expert.expert_id}.pt"
                expert.save(expert_path)

                # 更新域标签
                expert.domain = domain_key

            if RICH_AVAILABLE:
                console.print(f"[dim]  已保存专家到 {config.output_dir}/[/dim]")

            progress.update(task, advance=1)

    # 保存专家池配置
    if RICH_AVAILABLE:
        console.print(f"\n[dim]保存专家池配置...[/dim]")

    # 创建新的专家池配置
    pool_config = {
        "num_experts": config.num_experts,
        "config": config.to_expert_config(),
        "domains": {i: experts[i].domain for i in range(config.num_experts)},
        "expert_usage_count": torch.zeros(config.num_experts),
    }

    torch.save(pool_config, f"{config.output_dir}/pool_config.pt")

    elapsed = time.time() - start_time

    # 打印总结
    if RICH_AVAILABLE:
        console.print(f"\n[bold green]✓ 阶段2训练完成！[/bold green]")
        console.print(f"[dim]耗时: {elapsed/60:.1f} 分钟[/dim]")
        console.print(f"[dim]输出: {config.output_dir}/[/dim]")

        # 打印专家领域分配
        table = Table(title="[bold yellow]专家领域分配[/bold yellow]", box=box.ROUNDED)
        table.add_column("[cyan]专家 ID[/cyan]", style="cyan")
        table.add_column("[yellow]领域[/yellow]", style="yellow")
        table.add_column("[green]数据源[/green]", style="green")

        for domain_key, domain_config in DOMAIN_CONFIGS.items():
            for expert_id in domain_config["experts"]:
                table.add_row(
                    str(expert_id),
                    f"{domain_config['emoji']} {domain_key}",
                    domain_config["data_file"]
                )

        console.print(table)
    else:
        print(f"\n✓ 阶段2训练完成！")
        print(f"耗时: {elapsed/60:.1f} 分钟")
        print(f"输出: {config.output_dir}/")
        print("\n专家领域分配:")
        for domain_key, domain_config in DOMAIN_CONFIGS.items():
            print(f"  Experts {domain_config['experts']}: {domain_key}")


def main():
    """主函数"""
    config = Stage2Config()

    # 检查 MPS 可用性
    if not torch.backends.mps.is_available():
        config.device = "cpu"
        print("警告: MPS 不可用，使用 CPU")

    train_stage2(config)


if __name__ == "__main__":
    main()
