import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_PATH = "../Qwen3.5-4B"
ADAPTER_PATH = "./output/lora-adapter"

print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("加载模型...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

prompt = "什么是人工智能？"
print(f"\n问题: {prompt}\n")

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"回复:\n{response}")
