"""
优化的推理脚本
支持交互模式、批量测试、多种采样策略
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse


class ChatModel:
    """聊天模型封装"""

    def __init__(self, model_path: str, adapter_path: str = None, device: str = "mps"):
        self.device = device
        self.model_path = model_path
        self.adapter_path = adapter_path

        print("加载 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("加载模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto",
        )

        if adapter_path:
            print(f"加载 LoRA adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
    ) -> str:
        """生成回复"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除输入部分
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        return response

    def chat(self, system_prompt: str = ""):
        """交互式聊天模式"""
        print("\n" + "="*50)
        print("交互式聊天模式 (输入 'quit' 退出)")
        print("="*50)

        if system_prompt:
            print(f"系统提示: {system_prompt}\n")

        history = []

        while True:
            try:
                user_input = input("\n你: ").strip()
                if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                    print("再见!")
                    break

                if not user_input:
                    continue

                # 构建完整提示
                full_prompt = user_input
                if history:
                    context = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in history[-3:]])
                    full_prompt = f"{context}\n\nQ: {user_input}\nA:"

                print("AI: ", end="", flush=True)
                response = self.generate(full_prompt)
                print(response)

                history.append({"q": user_input, "a": response})

            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except Exception as e:
                print(f"\n错误: {e}")

    def batch_test(self, prompts: list, output_file: str = None):
        """批量测试"""
        print(f"\n批量测试 {len(prompts)} 个提示...")

        results = []
        for i, prompt in enumerate(prompts, 1):
            print(f"[{i}/{len(prompts)}] {prompt[:50]}...")

            try:
                response = self.generate(prompt)
                results.append({
                    "prompt": prompt,
                    "response": response,
                    "success": True
                })
            except Exception as e:
                print(f"  错误: {e}")
                results.append({
                    "prompt": prompt,
                    "response": str(e),
                    "success": False
                })

        if output_file:
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {output_file}")

        return results

    def compare_samples(
        self,
        prompt: str,
        temperatures: list = [0.3, 0.7, 1.0],
        top_ps: list = [0.8, 0.9, 1.0],
    ):
        """对比不同采样参数的生成效果"""
        print(f"\n提示: {prompt}")
        print("="*50)

        results = []

        # 对比不同温度
        print("\n温度对比:")
        for temp in temperatures:
            response = self.generate(prompt, temperature=temp, do_sample=True)
            print(f"\nTemperature={temp}:")
            print(f"  {response[:200]}...")

        # 对比不同 top_p
        print("\nTop-p 对比:")
        for tp in top_ps:
            response = self.generate(prompt, top_p=tp, temperature=0.7, do_sample=True)
            print(f"\nTop-p={tp}:")
            print(f"  {response[:200]}...")

        # 贪婪解码
        print("\n贪婪解码 (do_sample=False):")
        response = self.generate(prompt, do_sample=False)
        print(f"  {response[:200]}...")


def main():
    parser = argparse.ArgumentParser(description="LLM 推理脚本")
    parser.add_argument("--model", type=str, default="../Qwen3.5-4B", help="模型路径")
    parser.add_argument("--adapter", type=str, default="./output/lora-adapter", help="LoRA adapter 路径")
    parser.add_argument("--mode", type=str, choices=["single", "chat", "batch", "compare"], default="single")
    parser.add_argument("--prompt", type=str, help="单次推理的提示词")
    parser.add_argument("--data", type=str, help="批量测试的数据文件")
    parser.add_argument("--output", type=str, help="批量测试输出文件")
    parser.add_argument("--max-tokens", type=int, default=256, help="最大生成 token 数")
    parser.add_argument("--temp", type=float, default=0.7, help="温度参数")

    args = parser.parse_args()

    # 加载模型
    model = ChatModel(args.model, args.adapter)

    if args.mode == "single":
        if not args.prompt:
            args.prompt = "什么是人工智能？"
        print(f"提示: {args.prompt}\n")
        response = model.generate(args.prompt, max_new_tokens=args.max_tokens, temperature=args.temp)
        print(f"回复:\n{response}")

    elif args.mode == "chat":
        model.chat()

    elif args.mode == "batch":
        if not args.data:
            print("批量模式需要指定 --data 参数")
            return

        with open(args.data, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]

        output_file = args.output or "./output/batch_results.json"
        model.batch_test(prompts, output_file)

    elif args.mode == "compare":
        prompt = args.prompt or "请解释一下机器学习的基本概念。"
        model.compare_samples(prompt)


if __name__ == "__main__":
    main()
