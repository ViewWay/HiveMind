"""
基于 Qwen3.5-4B 的 MoE 训练 - Stage 4: 端到端精调

最终的整体优化，达到顶尖性能。
"""

import os
import sys
import time
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import PeftModel, LoraConfig, TaskType

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

if RICH_AVAILABLE:
    console = Console()

QWEN_MODEL_PATH = "/Users/yimiliya/github/Qwen3.5-4B"


@dataclass
class QwenMoEStage4Config:
    """Stage 4 配置"""

    model_path: str = QWEN_MODEL_PATH
    stage3_checkpoint: str = "checkpoints/qwen_stage3"

    # 最终精调配置
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = None

    # 训练配置 - 极低学习率精调
    max_seq_length: int = 1024  # 更长序列
    batch_size: int = 1
    gradient_accumulation_steps: int = 32  # 更多累积
    learning_rate: float = 5e-5  # 极低学习率
    num_epochs: int = 3
    warmup_steps: int = 50

    load_in_8bit: bool = True
    device: str = "mps"
    output_dir: str = "checkpoints/qwen_final"

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]


class FinalDataset(Dataset):
    """最终训练数据集 - 混合所有高质量数据"""

    def __init__(self, data_dir: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        data_path = Path(data_dir)

        # 收集所有数据
        for data_file in data_path.glob("stage2/*.txt"):
            with open(data_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip() and len(line.strip()) >= 50]

            self.samples.extend(texts)

        print(f"  加载了 {len(self.samples)} 条高质量训练样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]

        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_stage3_model(config):
    """加载 Stage 3 模型"""

    if RICH_AVAILABLE:
        console.print(f"[bold yellow]加载 Stage 3 检查点[/bold yellow]")

    tokenizer = AutoTokenizer.from_pretrained(
        config.stage3_checkpoint,
        trust_remote_code=True,
    )

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=config.load_in_8bit,
    ) if config.load_in_8bit else None

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(
        base_model,
        config.stage3_checkpoint,
    )

    # 解冻所有 LoRA 参数进行最终精调
    for param in model.parameters():
        param.requires_grad = True

    if RICH_AVAILABLE:
        model.print_trainable_parameters()

    return model, tokenizer


def train_qwen_moe_stage4(config: QwenMoEStage4Config):
    """Stage 4 训练主函数"""

    if RICH_AVAILABLE:
        console.print(Panel(
            f"[bold cyan]Qwen3.5-4B MoE - 端到端精调[/bold cyan]",
            title="HiveMind",
            subtitle="Stage 4: 最终优化",
            box=box.DOUBLE_EDGE,
            border_style="bright_blue"
        ))

        table = Table(title="[bold yellow]训练配置[/bold yellow]", box=box.ROUNDED)
        table.add_column("[cyan]参数[/cyan]", style="cyan")
        table.add_column("[yellow]值[/yellow]", style="yellow")

        for key, value in [
            ("学习率", config.learning_rate),
            ("训练轮数", config.num_epochs),
            ("序列长度", config.max_seq_length),
            ("梯度累积", config.gradient_accumulation_steps),
        ]:
            table.add_row(key, str(value))

        console.print(table)
        console.print()

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # 加载模型
    model, tokenizer = load_stage3_model(config)

    # 准备数据
    if RICH_AVAILABLE:
        console.print(f"[bold yellow]准备最终训练数据[/bold yellow]")

    dataset = FinalDataset("data", tokenizer, config.max_seq_length)

    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    if RICH_AVAILABLE:
        console.print(f"[dim]  训练集: {train_size} 条[/dim]")
        console.print(f"[dim]  验证集: {val_size} 条[/dim]")

    # 训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
        save_total_limit=3,
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # 开始训练
    if RICH_AVAILABLE:
        console.print(f"[bold yellow]开始端到端精调[/bold yellow]")

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    # 保存最终模型
    if RICH_AVAILABLE:
        console.print(f"[bold yellow]保存最终模型[/bold yellow]")

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # 合并 LoRA 权重 (可选)
    if RICH_AVAILABLE:
        console.print(f"[dim]合并 LoRA 权重...[/dim]")

    merged_model = PeftModel.merge_and_unload(
        model.base_model.model if hasattr(model, 'base_model') else model.model,
        model,
    )

    merged_path = f"{config.output_dir}/merged"
    Path(merged_path).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

    if RICH_AVAILABLE:
        console.print(f"\n[bold green]✓ Stage 4 训练完成！[/bold green]")
        console.print(f"[dim]耗时: {elapsed/60:.1f} 分钟[/dim]")
        console.print(f"[dim]LoRA 模型: {config.output_dir}/[/dim]")
        console.print(f"[dim]合并模型: {merged_path}/[/dim]")

        # 打印最终指标
        if trainer.state.log_history:
            table = Table(title="[bold yellow]最终训练指标[/bold yellow]", box=box.ROUNDED)
            table.add_column("[cyan]指标[/cyan]", style="cyan")
            table.add_column("[yellow]值[/yellow]", style="yellow")

            final_loss = trainer.state.log_history[-1].get("train_loss", "N/A")
            if final_loss != "N/A":
                table.add_row("训练损失", f"{final_loss:.4f}")

            eval_losses = [h.get("eval_loss") for h in trainer.state.log_history if "eval_loss" in h]
            if eval_losses:
                table.add_row("验证损失", f"{eval_losses[-1]:.4f}")

            console.print(table)


def main():
    config = QwenMoEStage4Config()

    if torch.cuda.is_available():
        config.device = "cuda"
    elif torch.backends.mps.is_available():
        config.device = "mps"
    else:
        config.device = "cpu"
        config.load_in_8bit = False

    train_qwen_moe_stage4(config)


if __name__ == "__main__":
    main()
