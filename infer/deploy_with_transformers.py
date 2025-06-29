from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
import deepspeed
import uvicorn
import asyncio


model_path = "/hy-tmp/model_hub/qwen/Qwen3-32B"
app = FastAPI(title="Qwen3-32B API Service")


# 请求和响应模型
class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 4096
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: float = 1

class GenerationResponse(BaseModel):
    response: str


# 模型加载函数
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    )
    model.eval()

    return tokenizer, model


# 全局加载模型
tokenizer, model = load_model()

# 定义API端点
@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    try:
        # 编码输入
        print(f"llm输入: {request.prompt}")
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        
        # 生成响应
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        )
        
        # 解码输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# 启动服务
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8056, reload=False)

