import requests

# API端点
API_URL = "http://localhost:8056/generate"

# 请求数据
data = {
    "prompt": "请介绍一下量子计算的基本原理。",
    "max_length": 2048,
    "temperature": 0.7
}

# 发送请求
response = requests.post(API_URL, json=data)

# 处理响应
if response.status_code == 200:
    result = response.json()
    print("模型回复:", result["response"])
else:
    print(f"请求失败，状态码: {response.status_code}，错误信息: {response.text}")
