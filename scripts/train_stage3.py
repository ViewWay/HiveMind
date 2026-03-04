"""
阶段 3: 路由层训练

训练智能路由层，学会选择正确的专家。
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

from swarm import SwarmExpert, SwarmExpertPool, ExpertConfig, HybridRouter, RouterConfig

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


# 领域到专家的映射 (与阶段2一致)
DOMAIN_TO_EXPERTS = {
    "code": [0, 1],
    "writing": [2, 3],
    "math": [4, 5],
    "knowledge": [6, 7],
}

# 专家到领域标签的映射
EXPERT_TO_DOMAIN = {}
for domain, experts in DOMAIN_TO_EXPERTS.items():
    for e in experts:
        EXPERT_TO_DOMAIN[e] = domain


@dataclass
class Stage3Config:
    """阶段3训练配置"""

    # 数据
    max_seq_length: int = 512
    vocab_size: int = 151936

    # 专家配置 (应与阶段1、2一致)
    num_experts: int = 8
    expert_hidden_size: int = 128

    # 路由器配置
    task_feature_dim: int = 128
    router_temperature: float = 0.5
    router_min_k: int = 2
    router_max_k: int = 6
    load_balance_weight: float = 0.1

    # 训练配置
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-3
    num_epochs: int = 50  # 路由需要更多训练学会正确选择
    warmup_steps: int = 100

    # 系统
    device: str = "mps"
    stage2_checkpoint_dir: str = "checkpoints/stage2"
    output_dir: str = "checkpoints/stage3"


class RouterDataset(Dataset):
    """
    路由器训练数据集

    混合所有领域的数据，并标记目标专家。
    """

    def __init__(self, data_files: Dict[str, str], max_length: int = 512, vocab_size: int = 151936):
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.samples = []

        # 加载所有领域的数据
        for domain, data_file in data_files.items():
            data_path = Path(data_file)
            if not data_path.exists():
                if RICH_AVAILABLE:
                    console.print(f"[yellow]警告: 数据文件不存在: {data_file}[/yellow]")
                continue

            with open(data_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]

            # 目标专家
            target_experts = DOMAIN_TO_EXPERTS[domain]

            for text in texts:
                self.samples.append({
                    "text": text,
                    "domain": domain,
                    "target_experts": target_experts,
                })

        print(f"  加载了 {len(self.samples)} 条混合训练样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        target_experts = sample["target_experts"]

        # 简单分词
        tokens = [ord(c) % self.vocab_size for c in text[:self.max_length]]

        # Padding
        if len(tokens) < self.max_length:
            tokens.extend([0] * (self.max_length - len(tokens)))

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        # 目标专家分布 (one-hot)
        target_distribution = torch.zeros(self.vocab_size, dtype=torch.float)  # 占位
        # 实际使用 num_experts 大小
        # target_distribution = torch.zeros(8, dtype=torch.float)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "domain": sample["domain"],
            "target_experts": target_experts,
        }


class RouterTrainer:
    """路由器训练器"""

    def __init__(
        self,
        router: HybridRouter,
        experts: List[SwarmExpert],
        config: Stage3Config,
    ):
        self.router = router
        self.experts = experts
        self.config = config
        self.device = torch.device(config.device)

        # 冻结专家参数
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False

        # 只训练路由器
        self.optimizer = torch.optim.AdamW(
            self.router.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.warmup_steps,
        )

        # 统计
        self.train_losses = []
        self.routing_accuracies = []

    def compute_router_loss(
        self,
        routing_result: Dict[str, torch.Tensor],
        target_domain: str,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算路由损失

        Args:
            routing_result: 路由器输出
            target_domain: 目标领域

        Returns:
            (损失, 指标字典)
        """
        expert_weights = routing_result["expert_weights"]
        batch_size = expert_weights.shape[0]

        # 1. 负载均衡损失
        load_balance_loss = self.router.load_balance_loss(expert_weights)

        # 2. 领域对齐损失 (鼓励路由器将正确的领域专家权重设高)
        target_experts = DOMAIN_TO_EXPERTS[target_domain]
        target_mask = torch.zeros(expert_weights.shape[1], device=self.device)
        target_mask[target_experts] = 1.0

        # 目标分布: 给目标专家更高的权重
        target_distribution = target_mask.unsqueeze(0).expand_as(expert_weights)
        target_distribution = target_distribution / target_distribution.sum(dim=-1, keepdim=True)

        # KL 散度损失
        alignment_loss = F.kl_div(
            expert_weights.log() + 1e-8,
            target_distribution,
            reduction="batchmean",
        )

        # 3. 总损失
        total_loss = (
            alignment_loss +
            self.config.load_balance_weight * load_balance_loss
        )

        # 4. 计算路由准确率
        selected_experts = routing_result["selected_experts"]  # [batch, k]
        # 检查选中的专家中是否有目标专家
        hits = 0
        total = 0
        for i in range(batch_size):
            selected = selected_experts[i].cpu().tolist()
            if any(e in target_experts for e in selected):
                hits += 1
            total += 1

        accuracy = hits / total if total > 0 else 0.0

        metrics = {
            "total_loss": total_loss.item(),
            "alignment_loss": alignment_loss.item(),
            "load_balance_loss": load_balance_loss.item(),
            "accuracy": accuracy,
            "avg_k": routing_result["k"],
        }

        return total_loss, metrics

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """训练一个 epoch"""
        self.router.train()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            domains = batch["domain"]

            # 嵌入
            batch_size, seq_len = input_ids.shape
            hidden_size = self.config.expert_hidden_size

            embedding = nn.Embedding(self.config.vocab_size, hidden_size).to(self.device)
            hidden_states = embedding(input_ids)

            # 路由
            routing_result = self.router(hidden_states, training=True)

            # 计算损失 (使用批次中第一个样本的领域作为简化)
            # 实际应用中应该逐样本处理
            target_domain = domains[0] if isinstance(domains, list) else domains
            loss, metrics = self.compute_router_loss(routing_result, target_domain)

            # 反向传播
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.router.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            total_accuracy += metrics["accuracy"]
            num_batches += 1

            # 每10个batch打印一次
            if (batch_idx + 1) % 10 == 0 and RICH_AVAILABLE:
                console.print(
                    f"[dim]    Batch {batch_idx+1} | Loss: {metrics['total_loss']:.4f} | "
                    f"Acc: {metrics['accuracy']:.2f} | K: {metrics['avg_k']}[/dim]"
                )

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        return avg_loss, avg_accuracy

    def train(self, dataloader: DataLoader, num_epochs: int):
        """完整训练"""
        for epoch in range(num_epochs):
            loss, accuracy = self.train_epoch(dataloader, epoch)
            self.train_losses.append(loss)
            self.routing_accuracies.append(accuracy)

            if RICH_AVAILABLE:
                console.print(
                    f"[dim]    Epoch {epoch+1}/{num_epochs} | "
                    f"Loss: {loss:.4f} | Accuracy: {accuracy:.2f}[/dim]"
                )
            else:
                print(f"    Epoch {epoch+1}/{num_epochs} | Loss: {loss:.4f} | Accuracy: {accuracy:.2f}")


def load_stage2_experts(config: Stage3Config) -> List[SwarmExpert]:
    """加载阶段2训练的专家"""
    stage2_dir = Path(config.stage2_checkpoint_dir)

    if not stage2_dir.exists():
        raise FileNotFoundError(
            f"阶段2检查点不存在: {stage2_dir}\n"
            f"请先运行阶段2训练: uv run python scripts/train_stage2.py"
        )

    experts = []

    for i in range(config.num_experts):
        expert_path = stage2_dir / f"expert_{i}.pt"

        if not expert_path.exists():
            raise FileNotFoundError(f"专家检查点不存在: {expert_path}")

        expert = SwarmExpert.load(str(expert_path), device=config.device)
        experts.append(expert)

        if RICH_AVAILABLE:
            console.print(f"[dim]  加载 Expert {i}: {expert_path}[/dim]")
        else:
            print(f"  加载 Expert {i}: {expert_path}")

    return experts


def train_stage3(config: Stage3Config):
    """阶段3训练主函数"""

    # 打印标题
    if RICH_AVAILABLE:
        console.print(Panel(
            f"[bold cyan]路由层训练[/bold cyan]",
            title="HiveMind",
            subtitle="阶段 3",
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
            ("负载均衡权重", config.load_balance_weight),
            ("阶段2检查点", config.stage2_checkpoint_dir),
        ]:
            table.add_row(key, str(value))

        console.print(table)
        console.print()
    else:
        print("\n" + "="*50)
        print("  HiveMind - 阶段3: 路由层训练")
        print("="*50 + "\n")

    # 创建输出目录
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # 加载阶段2的专家
    if RICH_AVAILABLE:
        console.print(f"[bold yellow]加载阶段2专家[/bold yellow]")
    else:
        print("加载阶段2专家")

    experts = load_stage2_experts(config)

    # 创建路由器
    if RICH_AVAILABLE:
        console.print(f"[bold yellow]创建混合路由器[/bold yellow]")
    else:
        print("创建混合路由器")

    router_config = RouterConfig(
        num_experts=config.num_experts,
        hidden_size=config.exppert_hidden_size if hasattr(config, 'exppert_hidden_size') else config.expert_hidden_size,
        task_feature_dim=config.task_feature_dim,
        temperature=config.router_temperature,
        min_k=config.router_min_k,
        max_k=config.router_max_k,
        load_balance_weight=config.load_balance_weight,
    )

    router = HybridRouter(router_config).to(config.device)

    # 准备数据
    if RICH_AVAILABLE:
        console.print(f"[bold yellow]准备训练数据[/bold yellow]")
    else:
        print("准备训练数据")

    data_files = {
        "code": "data/stage2/code_sample.txt",
        "writing": "data/stage2/writing_sample.txt",
        "math": "data/stage2/math_sample.txt",
        "knowledge": "data/stage2/knowledge_sample.txt",
    }

    dataset = RouterDataset(data_files, config.max_seq_length, config.vocab_size)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # 训练
    if RICH_AVAILABLE:
        console.print(f"[bold yellow]开始训练路由器[/bold yellow]")
    else:
        print("开始训练路由器")

    start_time = time.time()

    trainer = RouterTrainer(router, experts, config)
    trainer.train(dataloader, config.num_epochs)

    elapsed = time.time() - start_time

    # 保存路由器
    if RICH_AVAILABLE:
        console.print(f"\n[bold yellow]保存路由器[/bold yellow]")

    torch.save({
        "router_state_dict": router.state_dict(),
        "config": router_config,
        "expert_domains": EXPERT_TO_DOMAIN,
        "training_losses": trainer.train_losses,
        "routing_accuracies": trainer.routing_accuracies,
    }, f"{config.output_dir}/router.pt")

    if RICH_AVAILABLE:
        console.print(f"[dim]  已保存: {config.output_dir}/router.pt[/dim]")

    # 打印统计
    stats = router.get_routing_stats()

    if RICH_AVAILABLE:
        console.print(f"\n[bold green]✓ 阶段3训练完成！[/bold green]")
        console.print(f"[dim]耗时: {elapsed/60:.1f} 分钟[/dim]")
        console.print(f"[dim]输出: {config.output_dir}/[/dim]")

        # 打印专家使用统计
        table = Table(title="[bold yellow]专家使用统计[/bold yellow]", box=box.ROUNDED)
        table.add_column("[cyan]专家 ID[/cyan]", style="cyan")
        table.add_column("[yellow]领域[/yellow]", style="yellow")
        table.add_column("[green]使用率[/green]", style="green")

        for i, usage in enumerate(stats["expert_usage"]):
            domain = EXPERT_TO_DOMAIN.get(i, "unknown")
            table.add_row(str(i), domain, f"{usage:.2%}")

        console.print(table)
    else:
        print(f"\n✓ 阶段3训练完成！")
        print(f"耗时: {elapsed/60:.1f} 分钟")
        print(f"输出: {config.output_dir}/")
        print("\n专家使用统计:")
        for i, usage in enumerate(stats["expert_usage"]):
            domain = EXPERT_TO_DOMAIN.get(i, "unknown")
            print(f"  Expert {i} ({domain}): {usage:.2%}")


def main():
    """主函数"""
    config = Stage3Config()

    # 检查 MPS 可用性
    if not torch.backends.mps.is_available():
        config.device = "cpu"
        print("警告: MPS 不可用，使用 CPU")

    train_stage3(config)


if __name__ == "__main__":
    main()
