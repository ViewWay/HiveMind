"""
阶段 4: 端到端精调

整体优化，达到顶尖水平。
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

from swarm import SwarmModel, SwarmModelConfig

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
class Stage4Config:
    """阶段4训练配置"""

    # 数据
    max_seq_length: int = 1024  # 更长序列
    vocab_size: int = 151936

    # 模型配置
    num_experts: int = 8
    expert_hidden_size: int = 128
    expert_num_layers: int = 2
    expert_num_heads: int = 4
    expert_intermediate_size: int = 512

    # 路由配置
    router_task_feature_dim: int = 128
    router_temperature: float = 0.5
    router_min_k: int = 2
    router_max_k: int = 6

    # 训练配置 (关键: 非常低的学习率)
    batch_size: int = 2  # 更小批次
    gradient_accumulation_steps: int = 16  # 更多累积
    learning_rate: float = 1e-5  # 极低学习率
    num_epochs: int = 25  # 端到端精调需要足够时间收敛
    warmup_steps: int = 50
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # 系统
    device: str = "mps"
    stage3_checkpoint_dir: str = "checkpoints/stage3"
    output_dir: str = "checkpoints/stage4"

    def to_model_config(self) -> SwarmModelConfig:
        """转换为模型配置"""
        return SwarmModelConfig(
            num_experts=self.num_experts,
            expert_hidden_size=self.expert_hidden_size,
            expert_num_layers=self.expert_num_layers,
            expert_num_heads=self.expert_num_heads,
            expert_intermediate_size=self.expert_intermediate_size,
            router_task_feature_dim=self.router_task_feature_dim,
            router_temperature=self.router_temperature,
            router_min_k=self.router_min_k,
            router_max_k=self.router_max_k,
            vocab_size=self.vocab_size,
            max_seq_length=self.max_seq_length,
        )


class E2EDataset(Dataset):
    """
    端到端训练数据集

    混合所有高质量数据。
    """

    def __init__(self, data_dir: str, max_length: int = 1024, vocab_size: int = 151936):
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.samples = []

        data_path = Path(data_dir)

        # 收集所有数据文件
        for data_file in data_path.glob("stage2/*.txt"):
            domain = data_file.stem.split("_")[0]  # code, writing, math, knowledge

            with open(data_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]

            for text in texts:
                if len(text) >= 50:  # 只保留较长的样本
                    self.samples.append({
                        "text": text,
                        "domain": domain,
                    })

        print(f"  加载了 {len(self.samples)} 条高质量训练样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]

        # 简单分词
        tokens = [ord(c) % self.vocab_size for c in text[:self.max_length]]

        # Padding
        if len(tokens) < self.max_length:
            tokens.extend([0] * (self.max_length - len(tokens)))

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = torch.tensor(tokens, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class E2ETrainer:
    """端到端训练器"""

    def __init__(
        self,
        model: SwarmModel,
        config: Stage4Config,
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

        # 解冻所有参数
        self.model.unfreeze_experts()
        self.model.unfreeze_router()

        # 优化器 (所有参数)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # 余弦退火调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs * 100,  # 估计的总步数
            eta_min=1e-7,
        )

        # 统计
        self.train_losses = []
        self.learning_rates = []

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                labels=labels,
                training=True,
            )

            loss = outputs["loss"]

            # 反向传播
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            # 梯度累积
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # 记录学习率
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.learning_rates.append(current_lr)

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            # 每20个batch打印一次
            if (batch_idx + 1) % 20 == 0 and RICH_AVAILABLE:
                console.print(
                    f"[dim]    Batch {batch_idx+1} | Loss: {loss.item()*self.config.gradient_accumulation_steps:.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}[/dim]"
                )

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self, dataloader: DataLoader, num_epochs: int):
        """完整训练"""
        for epoch in range(num_epochs):
            loss = self.train_epoch(dataloader, epoch)
            self.train_losses.append(loss)

            if RICH_AVAILABLE:
                console.print(
                    f"[dim]    Epoch {epoch+1}/{num_epochs} | "
                    f"Loss: {loss:.4f} | LR: {self.learning_rates[-1]:.2e}[/dim]"
                )
            else:
                print(f"    Epoch {epoch+1}/{num_epochs} | Loss: {loss:.4f}")

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    training=False,
                )

                total_loss += outputs["loss"].item()
                num_batches += 1

        avg_loss = total_loss / num_batches

        # 计算困惑度
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
        }


def load_stage3_model(config: Stage4Config) -> SwarmModel:
    """加载阶段3的模型"""
    stage3_dir = Path(config.stage3_checkpoint_dir)

    if not stage3_dir.exists():
        raise FileNotFoundError(
            f"阶段3检查点不存在: {stage3_dir}\n"
            f"请先运行阶段3训练: uv run python scripts/train_stage3.py"
        )

    # 检查路由器
    router_path = stage3_dir / "router.pt"
    if not router_path.exists():
        raise FileNotFoundError(f"路由器检查点不存在: {router_path}")

    # 检查专家
    stage2_dir = Path(config.stage3_checkpoint_dir).parent / "stage2"
    if not stage2_dir.exists():
        raise FileNotFoundError(f"阶段2检查点不存在: {stage2_dir}")

    # 创建新模型
    model = SwarmModel(config.to_model_config())

    # 加载路由器
    router_checkpoint = torch.load(router_path, map_location=config.device)
    model.router.load_state_dict(router_checkpoint["router_state_dict"])

    if RICH_AVAILABLE:
        console.print(f"[dim]  加载路由器: {router_path}[/dim]")

    # 加载专家
    for i in range(config.num_experts):
        expert_path = stage2_dir / f"expert_{i}.pt"
        if not expert_path.exists():
            raise FileNotFoundError(f"专家检查点不存在: {expert_path}")

        expert_checkpoint = torch.load(expert_path, map_location=config.device)
        model.expert_pool.experts[i].load_state_dict(expert_checkpoint["state_dict"])

        if RICH_AVAILABLE:
            console.print(f"[dim]  加载 Expert {i}: {expert_path}[/dim]")

    return model


def train_stage4(config: Stage4Config):
    """阶段4训练主函数"""

    # 打印标题
    if RICH_AVAILABLE:
        console.print(Panel(
            f"[bold cyan]端到端精调[/bold cyan]",
            title="HiveMind",
            subtitle="阶段 4",
            box=box.DOUBLE_EDGE,
            border_style="bright_blue"
        ))

        # 打印配置
        table = Table(title="[bold yellow]训练配置[/bold yellow]", box=box.ROUNDED)
        table.add_column("[cyan]参数[/cyan]", style="cyan")
        table.add_column("[yellow]值[/yellow]", style="yellow")

        for key, value in [
            ("学习率", config.learning_rate),
            ("训练轮数", config.num_epochs),
            ("批次大小", config.batch_size),
            ("梯度累积", config.gradient_accumulation_steps),
            ("权重衰减", config.weight_decay),
            ("梯度裁剪", config.max_grad_norm),
            ("阶段3检查点", config.stage3_checkpoint_dir),
        ]:
            table.add_row(key, str(value))

        console.print(table)
        console.print()
    else:
        print("\n" + "="*50)
        print("  HiveMind - 阶段4: 端到端精调")
        print("="*50 + "\n")

    # 创建输出目录
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # 加载模型
    if RICH_AVAILABLE:
        console.print(f"[bold yellow]加载阶段3模型[/bold yellow]")
    else:
        print("加载阶段3模型")

    model = load_stage3_model(config)
    model.to(config.device)

    # 打印模型信息
    model_info = model.get_model_info()
    if RICH_AVAILABLE:
        console.print(f"[dim]  总参数量: {model_info['total_params']:,}[/dim]")
        console.print(f"[dim]  专家参数: {model_info['expert_params']:,}[/dim]")
        console.print(f"[dim]  路由参数: {model_info['router_params']:,}[/dim]")

    # 准备数据
    if RICH_AVAILABLE:
        console.print(f"[bold yellow]准备训练数据[/bold yellow]")
    else:
        print("准备训练数据")

    dataset = E2EDataset("data", config.max_seq_length, config.vocab_size)

    # 分割训练集和验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 训练
    if RICH_AVAILABLE:
        console.print(f"[bold yellow]开始端到端训练[/bold yellow]")
    else:
        print("开始端到端训练")

    start_time = time.time()

    trainer = E2ETrainer(model, config)
    trainer.train(train_dataloader, config.num_epochs)

    elapsed = time.time() - start_time

    # 评估
    if RICH_AVAILABLE:
        console.print(f"\n[bold yellow]评估模型[/bold yellow]")

    val_metrics = trainer.evaluate(val_dataloader)

    # 保存模型
    if RICH_AVAILABLE:
        console.print(f"[bold yellow]保存模型[/bold yellow]")

    model.save_model(f"{config.output_dir}/hivemind_final.pt")

    if RICH_AVAILABLE:
        console.print(f"[dim]  已保存: {config.output_dir}/hivemind_final.pt[/dim]")

    # 打印总结
    if RICH_AVAILABLE:
        console.print(f"\n[bold green]✓ 阶段4训练完成！[/bold green]")
        console.print(f"[dim]耗时: {elapsed/60:.1f} 分钟[/dim]")
        console.print(f"[dim]输出: {config.output_dir}/[/dim]")

        table = Table(title="[bold yellow]最终指标[/bold yellow]", box=box.ROUNDED)
        table.add_column("[cyan]指标[/cyan]", style="cyan")
        table.add_column("[yellow]值[/yellow]", style="yellow")

        table.add_row("验证损失", f"{val_metrics['loss']:.4f}")
        table.add_row("困惑度", f"{val_metrics['perplexity']:.2f}")
        table.add_row("训练损失", f"{trainer.train_losses[-1]:.4f}")

        console.print(table)
    else:
        print(f"\n✓ 阶段4训练完成！")
        print(f"耗时: {elapsed/60:.1f} 分钟")
        print(f"输出: {config.output_dir}/")
        print(f"\n最终指标:")
        print(f"  验证损失: {val_metrics['loss']:.4f}")
        print(f"  困惑度: {val_metrics['perplexity']:.2f}")


def main():
    """主函数"""
    config = Stage4Config()

    # 检查 MPS 可用性
    if not torch.backends.mps.is_available():
        config.device = "cpu"
        print("警告: MPS 不可用，使用 CPU")

    train_stage4(config)


if __name__ == "__main__":
    main()
