"""
LoRA 配置对比实验
对比不同 target_modules 的训练效果
"""

import os
import torch
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from pathlib import Path


# ========== LoRA 配置对比 ==========
LORA_CONFIGS = {
    "attention_only": {
        "name": "仅注意力层",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "description": "只训练注意力层，参数最少，关注语义理解"
    },
    "mlp_only": {
        "name": "仅MLP层",
        "target_modules": ["gate_proj", "up_proj", "down_proj"],
        "description": "只训练前馈神经网络层，关注知识存储"
    },
    "full": {
        "name": "全部层",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "description": "训练注意力和MLP层，参数最多，效果通常最好"
    }
}


def load_data(data_path: str) -> Dataset:
    """加载训练数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
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


def run_experiment(
    config_name: str,
    lora_config: dict,
    model_path: str,
    dataset: Dataset,
    tokenizer,
    output_dir: str,
    max_seq_length: int = 512,
    num_train_epochs: int = 2,
):
    """
    运行单个训练实验
    """
    print(f"\n{'='*60}")
    print(f"实验: {lora_config['name']} ({config_name})")
    print(f"描述: {lora_config['description']}")
    print(f"{'='*60}\n")

    # 加载模型
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
    ).to("mps")

    model.gradient_checkpointing_enable()

    # 配置 LoRA
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=lora_config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)

    # 打印可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.4f}%)")

    # 分词
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_seq_length),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    # 训练参数
    experiment_output_dir = os.path.join(output_dir, config_name)
    training_args = TrainingArguments(
        output_dir=experiment_output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=5,
        save_steps=100,
        save_total_limit=1,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        report_to=["none"],
        save_strategy="steps",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 训练
    print("\n开始训练...")
    result = trainer.train()

    # 保存结果
    final_loss = result.training_loss
    trainer.save_model()

    metrics = {
        "config_name": config_name,
        "name": lora_config["name"],
        "trainable_params": trainable_params,
        "trainable_percent": trainable_params / total_params * 100,
        "final_loss": final_loss,
        "target_modules": lora_config["target_modules"],
    }

    # 保存指标
    with open(os.path.join(experiment_output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ 实验完成! 最终损失: {final_loss:.4f}")
    print(f"   保存位置: {experiment_output_dir}")

    return metrics


def main():
    print("="*60)
    print("LoRA 配置对比实验")
    print("="*60)

    # 配置
    MODEL_PATH = os.getenv("MODEL_PATH", "../Qwen3.5-4B")
    DATA_PATH = "./data/train_large.txt"
    OUTPUT_DIR = "./output/comparison"
    NUM_EPOCHS = 2

    # 加载 tokenizer
    print(f"\n加载 tokenizer: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据
    print(f"加载数据: {DATA_PATH}")
    dataset = load_data(DATA_PATH)
    print(f"数据量: {len(dataset)} 条\n")

    # 运行所有实验
    all_results = {}

    for config_name, lora_config in LORA_CONFIGS.items():
        try:
            metrics = run_experiment(
                config_name=config_name,
                lora_config=lora_config,
                model_path=MODEL_PATH,
                dataset=dataset,
                tokenizer=tokenizer,
                output_dir=OUTPUT_DIR,
                num_train_epochs=NUM_EPOCHS,
            )
            all_results[config_name] = metrics
        except Exception as e:
            print(f"❌ 实验失败: {config_name} - {e}")
            all_results[config_name] = {"error": str(e)}

    # 保存汇总结果
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # 打印对比结果
    print("\n" + "="*60)
    print("实验结果对比")
    print("="*60)
    print(f"{'配置':<15} {'可训练参数':<12} {'占比':<10} {'最终损失':<10}")
    print("-"*60)

    for config_name, metrics in all_results.items():
        if "error" not in metrics:
            print(f"{metrics['name']:<15} {metrics['trainable_params']:<12,} "
                  f"{metrics['trainable_percent']:<10.4f}% {metrics['final_loss']:<10.4f}")
        else:
            print(f"{config_name:<15} 失败")

    print("\n✅ 所有实验完成!")
    print(f"结果保存于: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
