"""
基于 Qwen3.5-4B 的 MoE 训练 - Stage 1: 加载基础模型并添加 MoE 层

使用 Qwen3.5-4B 作为基础模型，在特定层添加 MoE 专家。
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

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

if RICH_AVAILABLE:
    console = Console()
else:
    console = None


# Qwen3.5-4B 模型路径
QWEN_MODEL_PATH = "/Users/yimiliya/github/Qwen3.5-4B"


@dataclass
class QwenMoEConfig:
    """Qwen MoE 训练配置"""

    # 模型路径
    model_path: str = QWEN_MODEL_PATH

    # MoE 配置
    num_experts: int = 8
    num_experts_per_token: int = 2  # Top-K
    moe_layers: List[int] = None  # 在哪些层添加 MoE

    # LoRA 配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None

    # 训练配置
    max_seq_length: int = 512
    batch_size: int = 1  # Qwen 较大，小批次
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    num_epochs: int = 5  # LoRA 适应基础模型
    warmup_steps: int = 100

    # 量化配置
    load_in_8bit: bool = True
    load_in_4bit: bool = False

    # 系统
    device: str = "mps"  # 或 "cuda"
    output_dir: str = "checkpoints/qwen_stage1"

    def __post_init__(self):
        if self.moe_layers is None:
            # 在中间层添加 MoE (Qwen3.5-4B 有 32 层)
            self.moe_layers = [8, 16, 24]

        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


class MoELayer(nn.Module):
    """
    MoE 层，插入到 Qwen 的 MLP 层位置
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_token: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token

        # 门控网络 - 决定使用哪些专家
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_size * 2),
                nn.SiLU(),
                nn.Linear(intermediate_size * 2, hidden_size),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_size]

        Returns:
            [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.shape

        # 计算门控权重
        gate_logits = self.gate(x)  # [batch, seq_len, num_experts]
        gate_probs = torch.softmax(gate_logits, dim=-1)

        # Top-K 选择
        top_k_probs, top_k_indices = torch.topk(
            gate_probs,
            k=self.num_experts_per_token,
            dim=-1,
        )

        # 初始化输出
        output = torch.zeros_like(x)

        # 对每个专家进行处理
        for expert_idx in range(self.num_experts):
            # 找到使用这个专家的位置
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # [batch, seq_len]

            if not expert_mask.any():
                continue

            # 获取需要处理的输入
            expert_input = x[expert_mask]  # [N, hidden_size]
            expert_weight = top_k_probs[expert_mask]  # [N, k]

            # 专家计算
            expert_output = self.experts[expert_idx](expert_input)

            # 加权累加
            for i in range(self.num_experts_per_token):
                w = expert_weight[:, i:i+1]  # [N, 1]
                mask = (top_k_indices[expert_mask, i] == expert_idx)
                output[expert_mask] += (expert_output * w * mask.unsqueeze(-1))

        return output


class QwenMoEDataset(Dataset):
    """
    Qwen MoE 训练数据集
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        data_path = Path(data_path)

        # 收集所有数据文件
        for data_file in data_path.glob("stage2/*.txt"):
            with open(data_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]

            self.samples.extend(texts)

        print(f"  加载了 {len(self.samples)} 条训练样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]

        # 使用 Qwen tokenizer
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        # 标签是输入的移位版本 (用于语言建模)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_quantization_config(config: QwenMoEConfig):
    """创建量化配置"""
    if config.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif config.load_in_8bit:
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    return None


def load_qwen_model(config: QwenMoEConfig):
    """加载 Qwen3.5-4B 模型"""

    if RICH_AVAILABLE:
        console.print(f"[bold yellow]加载 Qwen3.5-4B 模型[/bold yellow]")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,
    )

    # 设置特殊 token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if RICH_AVAILABLE:
        console.print(f"[dim]  Vocab size: {len(tokenizer)}[/dim]")

    # 量化配置
    quantization_config = create_quantization_config(config)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # 准备 kbit 训练
    if config.load_in_8bit or config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if RICH_AVAILABLE:
        console.print(f"[dim]  模型加载完成[/dim]")
        console.print(f"[dim]  参数量: {model.num_parameters() / 1e9:.2f}B[/dim]")

    return model, tokenizer


def setup_lora(model, config: QwenMoEConfig):
    """设置 LoRA"""

    if RICH_AVAILABLE:
        console.print(f"[bold yellow]配置 LoRA[/bold yellow]")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    if RICH_AVAILABLE:
        model.print_trainable_parameters()

    return model


def train_qwen_moe_stage1(config: QwenMoEConfig):
    """Stage 1 训练主函数"""

    # 打印标题
    if RICH_AVAILABLE:
        console.print(Panel(
            f"[bold cyan]Qwen3.5-4B MoE 训练[/bold cyan]",
            title="HiveMind",
            subtitle="Stage 1: 基础模型",
            box=box.DOUBLE_EDGE,
            border_style="bright_blue"
        ))

        table = Table(title="[bold yellow]训练配置[/bold yellow]", box=box.ROUNDED)
        table.add_column("[cyan]参数[/cyan]", style="cyan")
        table.add_column("[yellow]值[/yellow]", style="yellow")

        for key, value in [
            ("模型", config.model_path),
            ("专家数量", config.num_experts),
            ("Top-K", config.num_experts_per_token),
            ("LoRA r", config.lora_r),
            ("学习率", config.learning_rate),
            ("训练轮数", config.num_epochs),
            ("8bit量化", config.load_in_8bit),
        ]:
            table.add_row(key, str(value))

        console.print(table)
        console.print()
    else:
        print("\n" + "="*50)
        print("  HiveMind - Qwen3.5-4B MoE Stage 1")
        print("="*50 + "\n")

    # 创建输出目录
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # 加载模型和 tokenizer
    model, tokenizer = load_qwen_model(config)

    # 设置 LoRA
    model = setup_lora(model, config)

    # 准备数据
    if RICH_AVAILABLE:
        console.print(f"[bold yellow]准备训练数据[/bold yellow]")

    dataset = QwenMoEDataset("data", tokenizer, config.max_seq_length)

    # 分割训练集和验证集
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
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        save_total_limit=2,
        fp16=False,  # MPS 不完全支持 fp16
        bf16=False,  # MPS 不支持 bf16
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # 开始训练
    if RICH_AVAILABLE:
        console.print(f"[bold yellow]开始训练[/bold yellow]")

    start_time = time.time()

    trainer.train()

    elapsed = time.time() - start_time

    # 保存模型
    if RICH_AVAILABLE:
        console.print(f"[bold yellow]保存模型[/bold yellow]")

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    if RICH_AVAILABLE:
        console.print(f"\n[bold green]✓ Stage 1 训练完成！[/bold green]")
        console.print(f"[dim]耗时: {elapsed/60:.1f} 分钟[/dim]")
        console.print(f"[dim]输出: {config.output_dir}/[/dim]")
    else:
        print(f"\n✓ Stage 1 训练完成！")
        print(f"耗时: {elapsed/60:.1f} 分钟")


def main():
    """主函数"""
    config = QwenMoEConfig()

    # 检查设备
    if torch.cuda.is_available():
        config.device = "cuda"
    elif torch.backends.mps.is_available():
        config.device = "mps"
    else:
        config.device = "cpu"
        config.load_in_8bit = False
        config.load_in_4bit = False

    train_qwen_moe_stage1(config)


if __name__ == "__main__":
    main()
