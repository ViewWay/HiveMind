"""
Qwen LoRA Training Script for M4 Pro (MPS) - 优化日志版本
使用纯文本语料进行 LoRA 微调
"""

import os
import sys
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from pathlib import Path
from logger import SimpleLogger, Colors


# ANSI 颜色支持
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# ========== 配置区域 ==========
CONFIG = {
    # 模型路径
    "model_path": os.getenv("MODEL_PATH", "../Qwen3.5-4B"),

    # 数据路径
    "data_dir": "./data",
    "data_file": "train_large.txt",  # 使用扩展数据

    # 输出路径
    "output_dir": "./output/lora-adapter-v2",

    # LoRA 配置
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],

    # 训练参数
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "logging_steps": 10,
    "save_steps": 500,

    # 序列长度
    "max_seq_length": 2048,

    # MPS 优化
    "use_mps": True,
}
# =================================


def load_text_data(data_path: str) -> Dataset:
    """从文本文件加载数据"""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    with open(path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    print(f"  📚 数据加载: {len(texts)} 条")
    return Dataset.from_dict({"text": texts})


def tokenize_function(examples, tokenizer, max_length):
    """分词函数"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )


class PrettyTrainer(Trainer):
    """带美化日志的 Trainer"""

    def __init__(self, *args, logger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger
        self.step_start_time = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        """重写训练步骤以添加日志"""
        self.step_start_time = time.time()
        return super().training_step(model, inputs, num_items_in_batch)

    def log(self, logs):
        """重写日志方法"""
        super().log(logs)

        # 自定义日志输出
        if self.state.global_step % self.args.logging_steps == 0:
            step = self.state.global_step
            loss = logs.get("loss", 0)
            duration = time.time() - self.step_start_time
            
            total_steps = self.state.max_steps
            elapsed = time.time() - self.train_start_time
            remaining = (elapsed / step) * (total_steps - step) if step > 0 else 0
            
            # 美化日志输出
            print(f"\r  {Colors.info(f'Step {step:3d}/{total_steps}')}"
                  f" │ {Colors.warning(f'Loss: {loss:.4f}')}"
                  f" │ {Colors.success(f'{duration:.1f}s')}"
                  f" │ {Colors.info(f'剩余: {remaining/60:.1f}min')}"
                  f"{' ' * 20}", end="", flush=True)


def print_header(console, title, subtitle=""):
    """打印标题"""
    if RICH_AVAILABLE:
        console.print(Panel(
            f"[bold cyan]{title}[/bold cyan]",
            title="LLM Training",
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


def print_config_table(console, config):
    """打印配置表格"""
    if RICH_AVAILABLE:
        table = Table(title="[bold yellow]配置信息[/bold yellow]", box=box.ROUNDED)
        table.add_column("[cyan]参数[/cyan]", style="cyan")
        table.add_column("[yellow]值[/yellow]", style="yellow")
        
        for key, value in config.items():
            if isinstance(value, list) and len(value) > 5:
                value = f"[{len(value)} 项] {', '.join(map(str, value[:3]))}..."
            table.add_row(key, str(value))
        
        console.print(table)
    else:
        print("配置信息:")
        for key, value in config.items():
            if isinstance(value, list) and len(value) > 5:
                value = f"[{len(value)} 项] ..."
            print(f"  {key}: {value}")


def print_model_stats(console, model):
    """打印模型统计"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if RICH_AVAILABLE:
        table = Table(title="[bold green]模型统计[/bold green]", box=box.SIMPLE_HEAD)
        table.add_column("指标", justify="left")
        table.add_column("值", justify="right")
        table.add_row("总参数", f"{total:,}")
        table.add_row("可训练参数", f"[green]{trainable:,}[/green]")
        table.add_row("训练占比", f"[yellow]{trainable/total*100:.4f}%[/yellow]")
        console.print(table)
    else:
        print(f"模型统计:")
        print(f"  总参数: {total:,}")
        print(f"  可训练参数: {trainable:,} ({trainable/total*100:.4f}%)")


def print_training_results(console, results, elapsed_time):
    """打印训练结果"""
    if RICH_AVAILABLE:
        table = Table(title="[bold green]训练完成[/bold green]", box=box.DOUBLE_EDGE)
        table.add_column("[cyan]指标[/cyan]", justify="left")
        table.add_column("[yellow]值[/yellow]", justify="right")
        
        for key, value in results.items():
            table.add_row(key, str(value))
        table.add_row("总用时", f"{elapsed_time:.1f}秒")
        
        console.print(table)
    else:
        print(f"\n{'='*60}")
        print("  训练完成")
        print(f"{'='*60}")
        for key, value in results.items():
            print(f"  {key}: {value}")
        print(f"  总用时: {elapsed_time:.1f}秒")


def main():
    # 初始化
    if RICH_AVAILABLE:
        console = Console()
    else:
        console = None
    
    logger = SimpleLogger()
    
    # 打印标题
    print_header(console, "Qwen3.5-4B LoRA 训练", "M4 Pro | MPS 加速")
    
    # 打印配置
    print_config_table(console, CONFIG)
    
    # 1. 设置设备
    device = torch.device("mps" if CONFIG["use_mps"] and torch.backends.mps.is_available() else "cpu")
    console.print(f"[dim]使用设备: {device}[/dim]" if RICH_AVAILABLE else f"使用设备: {device}")
    
    # 2. 加载 Tokenizer
    console.print(f"[dim]加载 Tokenizer...[/dim]" if RICH_AVAILABLE else "加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 3. 加载模型
    console.print(f"[dim]加载模型...[/dim]" if RICH_AVAILABLE else "加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_path"],
        trust_remote_code=True,
        dtype=torch.float16,
    ).to(device)
    
    print_model_stats(console, model)
    
    # 4. 启用梯度检查点
    model.gradient_checkpointing_enable()
    console.print("[dim]✓ 已启用梯度检查点[/dim]" if RICH_AVAILABLE else "✓ 已启用梯度检查点")
    
    # 5. 配置 LoRA
    console.print(f"[dim]配置 LoRA (r={CONFIG['lora_r']})...[/dim]" if RICH_AVAILABLE else "配置 LoRA...")
    
    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 6. 加载数据
    data_path = os.path.join(CONFIG["data_dir"], CONFIG["data_file"])
    console.print(f"[dim]加载数据: {data_path}[/dim]" if RICH_AVAILABLE else f"加载数据: {data_path}")
    dataset = load_text_data(data_path)
    
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, CONFIG["max_seq_length"]),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    
    # 7. 训练参数
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["num_train_epochs"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        warmup_steps=CONFIG["warmup_steps"],
        logging_steps=CONFIG["logging_steps"],
        save_steps=CONFIG["save_steps"],
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        report_to=["none"],
        save_strategy="steps",
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # 8. 创建 Trainer
    trainer = PrettyTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        logger=logger,
    )
    trainer.train_start = time.time  # 记录开始时间
    
    # 9. 开始训练
    console.print("")  # 空行
    if RICH_AVAILABLE:
        console.print("[bold yellow]开始训练...[/bold yellow]")
    else:
        print("\n开始训练...")
    
    trainer.train()
    
    # 10. 保存结果
    elapsed = time.time() - trainer.train_start
    
    results = {
        "训练轮数": CONFIG["num_train_epochs"],
        "训练步数": trainer.state.global_step,
        "最终损失": trainer.state.log_history[-1].get("loss", "N/A"),
        "学习率": CONFIG["learning_rate"],
        "批大小": CONFIG["per_device_train_batch_size"],
    }
    
    print_training_results(console, results, elapsed)
    
    trainer.save_model(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])
    
    # 打印保存信息
    import shutil
    adapter_size = sum(f.stat(os.path.join(CONFIG["output_dir"], f)).st_size
                      for f in os.listdir(CONFIG["output_dir"]) if f.endswith('.bin') or f.endswith('.safetensors')) / (1024*1024)

    if RICH_AVAILABLE:
        console.print(f"[dim]💾 保存到: {CONFIG['output_dir']}[/dim]")
        console.print(f"[dim]📦 适配器大小: {adapter_size:.2f} MB[/dim]")
    else:
        print(f"💾 保存到: {CONFIG['output_dir']}")
        print(f"📦 适配器大小: {adapter_size:.2f} MB")
    console.print("")  # 空行


if __name__ == "__main__":
    main()
