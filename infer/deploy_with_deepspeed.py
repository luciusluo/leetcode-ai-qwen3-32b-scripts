from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
import deepspeed
import uvicorn
import asyncio

# 获取当前进程的rank（分布式环境中唯一标识）
local_rank = int(os.environ.get("LOCAL_RANK", 0))
rank = int(os.environ.get("RANK", 0))

model_path = "/hy-tmp/model_hub/qwen/Qwen3-32B"
device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Qwen3-32B API Service") if rank == 0 else None


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
        # low_cpu_mem_usage=True,
        trust_remote_code=True,
        # device_map="auto"
    )

    # DeepSpeed推理优化配置
    ds_model = deepspeed.init_inference(
        model=model,                  # 基础模型
        tensor_parallel={"tp_size": 4},                    # 模型并行度，设置为GPU数量
        dtype=torch.bfloat16,          # 推理精度
        replace_method="auto",        # 自动替换方法
        # replace_with_kernel_inject=True,  # 使用内核注入以获得更好性能
        # 可选：启用量化以减少显存使用
        # quantization_config=deepspeed.inference.config.InferenceQuantizationConfig(
        #     quantization_type=deepspeed.inference.config.QuantizationType.INT8
        # )
    )

    return tokenizer, ds_model.module


# 全局加载模型
tokenizer, model = load_model()

# 定义API端点
if rank == 0:
    @app.post("/generate", response_model=GenerationResponse)
    async def generate(request: GenerationRequest):
        try:
            # 编码输入
            print(f"llm输入: {request.prompt}")
            inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
            
            # 生成响应
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=request.max_length,
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
    if rank == 0:
        uvicorn.run(app, host="0.0.0.0", port=8056)
    else:
        # 其他进程进入等待状态（保持模型加载，支持并行计算）
        while True:
            asyncio.sleep(3600)  # 无限等待，直到主进程结束

