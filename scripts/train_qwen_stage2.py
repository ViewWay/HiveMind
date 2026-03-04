"""
基于 Qwen3.5-4B 的 MoE 训练 - Stage 2: 领域分化

每个专家专注于特定领域进行微调。
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

# 领域配置
DOMAIN_CONFIGS = {
    "code": {
        "data_file": "data/stage2/code_sample.txt",
        "name": "Code Programming",
        "emoji": "💻"
    },
    "writing": {
        "data_file": "data/stage2/writing_sample.txt",
        "name": "Literary Writing",
        "emoji": "📝"
    },
    "math": {
        "data_file": "data/stage2/math_sample.txt",
        "name": "Math Reasoning",
        "emoji": "🔢"
    },
    "knowledge": {
        "data_file": "data/stage2/knowledge_sample.txt",
        "name": "Knowledge Q&A",
        "emoji": "📚"
    },
}


@dataclass
class QwenMoEStage2Config:
    """Stage 2 配置"""

    model_path: str = QWEN_MODEL_PATH
    stage1_checkpoint: str = "checkpoints/qwen_stage1"

    # LoRA 配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # 训练配置
    max_seq_length: int = 512
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4  # 更低的学习率
    num_epochs: int = 3  # 每个领域训练轮数
    warmup_steps: int = 50

    # 量化
    load_in_8bit: bool = True

    device: str = "mps"
    output_dir: str = "checkpoints/qwen_stage2"


class DomainDataset(Dataset):
    """领域数据集"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, 'r', encoding='utf-8') as f:
            self.texts = [line.strip() for line in f if line.strip()]

        print(f"  加载了 {len(self.texts)} 条 {Path(data_path).stem} 样本")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

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


def load_stage1_model(config):
    """加载 Stage 1 模型"""

    if RICH_AVAILABLE:
        console.print(f"[bold yellow]加载 Stage 1 检查点[/bold yellow]")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.stage1_checkpoint,
        trust_remote_code=True,
    )

    # 加载基础模型
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

    # 加载 LoRA 权重
    model = PeftModel.from_pretrained(
        base_model,
        config.stage1_checkpoint,
    )

    if RICH_AVAILABLE:
        console.print(f"[dim]  模型加载完成[/dim]")

    return model, tokenizer


def train_domain(model, tokenizer, config, domain_name, domain_config):
    """训练单个领域"""

    data_file = domain_config["data_file"]
    output_dir = f"{config.output_dir}/{domain_name}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 准备数据
    if RICH_AVAILABLE:
        console.print(f"[dim]  准备 {domain_name} 数据[/dim]")

    dataset = DomainDataset(data_file, tokenizer, config.max_seq_length)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=10,
        save_steps=500,
        save_total_limit=1,
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # 训练
    if RICH_AVAILABLE:
        console.print(f"[bold yellow]  训练 {domain_name}...[/bold yellow]")

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    # 保存
    trainer.save_model(output_dir)

    if RICH_AVAILABLE:
        console.print(f"[dim]  完成: {domain_name} ({elapsed/60:.1f}分钟)[/dim]")

    return model


def train_qwen_moe_stage2(config: QwenMoEStage2Config):
    """Stage 2 训练主函数"""

    if RICH_AVAILABLE:
        console.print(Panel(
            f"[bold cyan]Qwen3.5-4B MoE - 领域分化[/bold cyan]",
            title="HiveMind",
            subtitle="Stage 2",
            box=box.DOUBLE_EDGE,
            border_style="bright_blue"
        ))

        table = Table(title="[bold yellow]训练配置[/bold yellow]", box=box.ROUNDED)
        table.add_column("[cyan]参数[/cyan]", style="cyan")
        table.add_column("[yellow]值[/yellow]", style="yellow")

        for key, value in [
            ("Stage 1 检查点", config.stage1_checkpoint),
            ("学习率", config.learning_rate),
            ("训练轮数", config.num_epochs),
        ]:
            table.add_row(key, str(value))

        console.print(table)
        console.print()

    # 加载模型
    model, tokenizer = load_stage1_model(config)

    # 训练各个领域
    start_time = time.time()

    for domain_name, domain_config in DOMAIN_CONFIGS.items():
        emoji = domain_config["emoji"]
        domain_display_name = domain_config["name"]

        if RICH_AVAILABLE:
            console.print(f"\n[bold yellow]{emoji} 训练领域: {domain_display_name}[/bold yellow]")

        model = train_domain(model, tokenizer, config, domain_name, domain_config)

    elapsed = time.time() - start_time

    if RICH_AVAILABLE:
        console.print(f"\n[bold green]✓ Stage 2 训练完成！[/bold green]")
        console.print(f"[dim]总耗时: {elapsed/60:.1f} 分钟[/dim]")
        console.print(f"[dim]输出: {config.output_dir}/[/dim]")


def main():
    config = QwenMoEStage2Config()

    if torch.cuda.is_available():
        config.device = "cuda"
    elif torch.backends.mps.is_available():
        config.device = "mps"
    else:
        config.device = "cpu"
        config.load_in_8bit = False

    train_qwen_moe_stage2(config)


if __name__ == "__main__":
    main()
