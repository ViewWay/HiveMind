"""
基于 Qwen3.5-4B 的 MoE 训练 - Stage 3: 路由层训练

训练智能路由层，学会选择正确的专家。
"""

import os
import sys
import time
import torch
from torch.utils.data import Dataset, DataLoader
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
class QwenMoEStage3Config:
    """Stage 3 配置"""

    model_path: str = QWEN_MODEL_PATH
    stage2_checkpoint_dir: str = "checkpoints/qwen_stage2"

    # 路由器训练配置
    lora_r: int = 32  # 更大的 r 用于路由器
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = None

    # 训练配置
    max_seq_length: int = 512
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-4
    num_epochs: int = 5
    warmup_steps: int = 50

    load_in_8bit: bool = True
    device: str = "mps"
    output_dir: str = "checkpoints/qwen_stage3"

    def __post_init__(self):
        if self.target_modules is None:
            # 专注于注意力层作为路由
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate",  # MLP 门控
            ]


class MixedDomainDataset(Dataset):
    """混合领域数据集，用于路由器训练"""

    def __init__(self, data_dir: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        data_path = Path(data_dir)

        # 收集所有领域数据
        for data_file in data_path.glob("stage2/*.txt"):
            domain = data_file.stem.split("_")[0]

            with open(data_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]

            for text in texts:
                self.samples.append({
                    "text": text,
                    "domain": domain,
                })

        print(f"  加载了 {len(self.samples)} 条混合训练样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]

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
            "domain": sample["domain"],
        }


def load_base_model(config):
    """加载基础模型"""

    if RICH_AVAILABLE:
        console.print(f"[bold yellow]加载 Qwen 基础模型[/bold yellow]")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,
    )

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=config.load_in_8bit,
    ) if config.load_in_8bit else None

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    return model, tokenizer


def merge_stage2_checkpoints(base_model, tokenizer, config):
    """合并 Stage 2 的领域检查点"""

    if RICH_AVAILABLE:
        console.print(f"[bold yellow]合并 Stage 2 检查点[/bold yellow]")

    stage2_dir = Path(config.stage2_checkpoint_dir)

    # 加载各个领域的 LoRA 权重并平均
    merged_weights = None
    num_loaded = 0

    for domain_dir in stage2_dir.iterdir():
        if not domain_dir.is_dir():
            continue

        adapter_path = domain_dir / "adapter_model.bin"
        if not adapter_path.exists():
            continue

        if RICH_AVAILABLE:
            console.print(f"[dim]  加载: {domain_dir.name}[/dim]")

        # 加载 LoRA 权重
        lora_weights = torch.load(adapter_path, map_location="cpu")

        if merged_weights is None:
            merged_weights = {k: v.clone() for k, v in lora_weights.items()}
        else:
            for k, v in lora_weights.items():
                if k in merged_weights:
                    merged_weights[k] += v
                else:
                    merged_weights[k] = v.clone()

        num_loaded += 1

    # 平均权重
    if num_loaded > 0 and merged_weights:
        for k in merged_weights:
            merged_weights[k] /= num_loaded

        # 创建新的 LoRA 配置并加载合并的权重
        from peft import get_peft_model, LoraConfig, TaskType

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(base_model, lora_config)

        # 加载合并的权重
        model.load_state_dict(merged_weights, strict=False)

        if RICH_AVAILABLE:
            console.print(f"[dim]  合并了 {num_loaded} 个领域权重[/dim]")

        return model, tokenizer

    # 如果没有找到检查点，直接返回基础模型
    return base_model, tokenizer


def train_qwen_moe_stage3(config: QwenMoEStage3Config):
    """Stage 3 训练主函数"""

    if RICH_AVAILABLE:
        console.print(Panel(
            f"[bold cyan]Qwen3.5-4B MoE - 路由层训练[/bold cyan]",
            title="HiveMind",
            subtitle="Stage 3",
            box=box.DOUBLE_EDGE,
            border_style="bright_blue"
        ))

        table = Table(title="[bold yellow]训练配置[/bold yellow]", box=box.ROUNDED)
        table.add_column("[cyan]参数[/cyan]", style="cyan")
        table.add_column("[yellow]值[/yellow]", style="yellow")

        for key, value in [
            ("LoRA r", config.lora_r),
            ("学习率", config.learning_rate),
            ("训练轮数", config.num_epochs),
        ]:
            table.add_row(key, str(value))

        console.print(table)
        console.print()

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # 加载模型
    base_model, tokenizer = load_base_model(config)
    model, tokenizer = merge_stage2_checkpoints(base_model, tokenizer, config)

    # 准备数据
    if RICH_AVAILABLE:
        console.print(f"[bold yellow]准备混合训练数据[/bold yellow]")

    dataset = MixedDomainDataset("data", tokenizer, config.max_seq_length)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # 训练
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        save_total_limit=2,
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    if RICH_AVAILABLE:
        console.print(f"[bold yellow]开始路由层训练[/bold yellow]")

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    if RICH_AVAILABLE:
        console.print(f"\n[bold green]✓ Stage 3 训练完成！[/bold green]")
        console.print(f"[dim]耗时: {elapsed/60:.1f} 分钟[/dim]")
        console.print(f"[dim]输出: {config.output_dir}/[/dim]")


def main():
    config = QwenMoEStage3Config()

    if torch.cuda.is_available():
        config.device = "cuda"
    elif torch.backends.mps.is_available():
        config.device = "mps"
    else:
        config.device = "cpu"
        config.load_in_8bit = False

    train_qwen_moe_stage3(config)


if __name__ == "__main__":
    main()
