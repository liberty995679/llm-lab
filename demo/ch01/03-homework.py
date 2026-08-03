import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(override= True)

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

model = init_chat_model(
    model="deepseek-v4-flash",
    base_url=DEEPSEEK_BASE_URL,   # 来自 .env
    api_key=DEEPSEEK_API_KEY,     # 来自系统环境变量
    temperature=0,
)

for i in range(3):
    resp = model.invoke("给我一个1-100的随机整数")
    print("模型回复：", resp.content)

print("="*10)

model = init_chat_model(
    model="deepseek-v4-flash",
    base_url=DEEPSEEK_BASE_URL,   # 来自 .env
    api_key=DEEPSEEK_API_KEY,     # 来自系统环境变量
    temperature=1.5,
)

for i in range(3):
    resp = model.invoke("给我一个1-100的随机整数")
    print("模型回复：", resp.content)

