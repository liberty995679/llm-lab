"""第01章·环境连通性验证：用 LangChain 的 ChatOpenAI 兼容模式调用 DeepSeek。"""
import os

from dotenv import load_dotenv

# 加载 .env（默认不覆盖已存在的系统环境变量）
load_dotenv()

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url=os.environ["DEEPSEEK_BASE_URL"],   # 来自 .env
    api_key=os.environ["DEEPSEEK_API_KEY"],     # 来自系统环境变量
    temperature=0.3,
)

resp = model.invoke("请用一句话回答：LangChain 是什么？")
print("模型回复：", resp.content)
