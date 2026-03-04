"""
阶段 1: 专家初始化训练

训练 8 个基础专家，获得通用语言能力。
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm import SwarmExpertPool, ExpertConfig
from logger import Colors

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


@dataclass
class Stage1Config:
    """阶段1训练配置"""

    # 数据
    data_file: str = "data/stage1/pretrain_data.txt"
    max_seq_length: int = 256  # 阶段1使用较短序列
    vocab_size: int = 151936  # Qwen tokenizer vocab size

    # 专家配置
    num_experts: int = 8
    expert_hidden_size: int = 128
    expert_num_layers: int = 2
    expert_num_heads: int = 4
    expert_intermediate_size: int = 512

    # 训练配置
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-3
    num_epochs: int = 3
    warmup_steps: int = 50

    # 系统
    device: str = "mps"
    output_dir: str = "checkpoints/stage1"

    def to_expert_config(self) -> ExpertConfig:
        """转换为专家配置"""
        return ExpertConfig(
            hidden_size=self.expert_hidden_size,
            num_layers=self.expert_num_layers,
            num_heads=self.expert_num_heads,
            intermediate_size=self.expert_intermediate_size,
            max_seq_length=self.max_seq_length,
        )


class TextDataset(Dataset):
    """简单文本数据集"""

    def __init__(self, data_path: str, max_length: int = 256, vocab_size: int = 151936):
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.vocab_size = vocab_size

        # 加载数据
        if not self.data_path.exists():
            # 创建示例数据
            self._create_sample_data()

        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.texts = [line.strip() for line in f if line.strip()]

        print(f"  加载了 {len(self.texts)} 条训练样本")

    def _create_sample_data(self):
        """创建示例训练数据"""
        sample_texts = [
            "人工智能是计算机科学的一个分支。",
            "机器学习使计算机能够从数据中学习。",
            "深度学习使用多层神经网络模拟人脑。",
            "自然语言处理是AI的重要应用领域。",
            "计算机视觉让机器能够理解和分析图像。",
            "强化学习通过奖励机制训练智能体。",
            "神经网络是深度学习的基础组件。",
            "大语言模型展现了惊人的语言理解能力。",
            "数据预处理是机器学习流程的重要步骤。",
            "模型评估需要使用独立的测试集。",
        ]

        # 重复生成足够的数据
        extended_texts = []
        for _ in range(5000):  # 生成 50k 条
            import random
            texts = sample_texts.copy()
            random.shuffle(texts)
            extended_texts.extend(texts)

        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_path, 'w', encoding='utf-8') as f:
            for text in extended_texts:
                f.write(text + '\n')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # 简单分词 (使用字符级别作为占位符)
        # 实际应用中应该使用正确的 tokenizer
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


class ExpertTrainer:
    """单个专家训练器"""

    def __init__(
        self,
        expert: nn.Module,
        config: Stage1Config,
        expert_id: int,
    ):
        self.expert = expert
        self.config = config
        self.expert_id = expert_id
        self.device = torch.device(config.device)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.expert.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.warmup_steps,
        )

        # 统计
        self.train_losses = []

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """训练一个 epoch"""
        self.expert.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # 前向传播 (简化版，直接使用专家)
            # 创建简单的嵌入
            batch_size, seq_len = input_ids.shape
            hidden_size = self.config.expert_hidden_size

            # 简单嵌入
            embedding = nn.Embedding(self.config.vocab_size, hidden_size).to(self.device)
            hidden_states = embedding(input_ids)

            # 专家处理
            expert_output = self.expert(hidden_states)

            # 输出投影
            output_proj = nn.Linear(hidden_size, self.config.vocab_size).to(self.device)
            logits = output_proj(expert_output)

            # 计算损失
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
                ignore_index=-100,
            )

            # 反向传播
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.expert.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self, dataloader: DataLoader, num_epochs: int):
        """完整训练"""
        for epoch in range(num_epochs):
            loss = self.train_epoch(dataloader, epoch)
            self.train_losses.append(loss)

            expert_type = "info" if RICH_AVAILABLE else "print"
            if RICH_AVAILABLE:
                console.print(f"[dim]    Expert {self.expert_id} | Epoch {epoch+1}/{num_epochs} | Loss: {loss:.4f}[/dim]")
            else:
                print(f"    Expert {self.expert_id} | Epoch {epoch+1}/{num_epochs} | Loss: {loss:.4f}")


def train_stage1(config: Stage1Config):
    """阶段1训练主函数"""

    # 打印标题
    if RICH_AVAILABLE:
        console.print(Panel(
            f"[bold cyan]专家初始化训练[/bold cyan]",
            title="HiveMind",
            subtitle="阶段 1",
            box=box.DOUBLE_EDGE,
            border_style="bright_blue"
        ))

        # 打印配置
        table = Table(title="[bold yellow]训练配置[/bold yellow]", box=box.ROUNDED)
        table.add_column("[cyan]参数[/cyan]", style="cyan")
        table.add_column("[yellow]值[/yellow]", style="yellow")

        for key, value in [
            ("专家数量", config.num_experts),
            ("隐藏层大小", config.expert_hidden_size),
            ("批次大小", config.batch_size),
            ("梯度累积", config.gradient_accumulation_steps),
            ("学习率", config.learning_rate),
            ("训练轮数", config.num_epochs),
        ]:
            table.add_row(key, str(value))

        console.print(table)
    else:
        print("\n" + "="*50)
        print("  HiveMind - 阶段1: 专家初始化训练")
        print("="*50 + "\n")

    # 创建输出目录
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # 加载数据
    if RICH_AVAILABLE:
        console.print(f"[dim]加载数据: {config.data_file}[/dim]")
    else:
        print(f"加载数据: {config.data_file}")

    dataset = TextDataset(config.data_file, config.max_seq_length, config.vocab_size)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # 创建专家池
    if RICH_AVAILABLE:
        console.print(f"[dim]创建专家池: {config.num_experts} 个专家[/dim]")
    else:
        print(f"创建专家池: {config.num_experts} 个专家")

    expert_pool = SwarmExpertPool(
        num_experts=config.num_experts,
        config=config.to_expert_config(),
    )

    # 训练每个专家
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
        task = progress.add_task("训练专家...", total=config.num_experts)

        for expert_id in range(config.num_experts):
            if RICH_AVAILABLE:
                console.print(f"\n[bold yellow]训练 Expert {expert_id}[/bold yellow]")
            else:
                print(f"\n训练 Expert {expert_id}")

            expert = expert_pool.experts[expert_id].to(torch.device(config.device))

            trainer = ExpertTrainer(expert, config, expert_id)
            trainer.train(dataloader, config.num_epochs)

            # 保存专家
            expert_path = f"{config.output_dir}/expert_{expert_id}.pt"
            expert.save(expert_path)

            if RICH_AVAILABLE:
                console.print(f"[dim]    已保存: {expert_path}[/dim]")

            progress.update(task, advance=1)

    # 保存专家池配置
    expert_pool.save_pool(config.output_dir)

    elapsed = time.time() - start_time

    # 打印总结
    if RICH_AVAILABLE:
        console.print(f"\n[bold green]✓ 阶段1训练完成！[/bold green]")
        console.print(f"[dim]耗时: {elapsed/60:.1f} 分钟[/dim]")
        console.print(f"[dim]输出: {config.output_dir}/[/dim]")
    else:
        print(f"\n✓ 阶段1训练完成！")
        print(f"耗时: {elapsed/60:.1f} 分钟")
        print(f"输出: {config.output_dir}/")


def main():
    """主函数"""
    config = Stage1Config()

    # 检查 MPS 可用性
    if not torch.backends.mps.is_available():
        config.device = "cpu"
        print("警告: MPS 不可用，使用 CPU")

    train_stage1(config)


if __name__ == "__main__":
    main()
