"""第06章·6.1 为什么需要结构化输出：靠"提示词 + json.loads"解析有多脆。"""
import json
import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

model = init_chat_model(
    "deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)

# 用提示词"要求"模型输出 JSON
resp = model.invoke([
    SystemMessage("你是商品信息提取器。输出 JSON，包含：商品名称、价格、是否现货。"),
    HumanMessage("请提取以下商品信息并输出 JSON：小米14 Ultra 512G 黑色，售价 4599 元，今日现货。"),
])

raw = resp.content
print("—— 模型原始输出 ——")
print(raw)
print()

# 直接拿给 json.loads 吃
try:
    data = json.loads(raw)
    print("json.loads 解析成功 ✅")
    print("name  =", data.get("名称") or data.get("name"))
    print("price =", data.get("价格") or data.get("price"))
except Exception as e:
    print("json.loads 解析失败 ❌")
    print("报错:", e)
