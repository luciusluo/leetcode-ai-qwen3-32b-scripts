from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch

model_path = "/hy-tmp/model_hub/qwen/Qwen3-32B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "你今天过的怎么样？"
inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
outputs = model.generate(**inputs, max_new_tokens=256)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))