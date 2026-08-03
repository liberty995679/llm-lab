"""第01章·把 ChatOpenAI 的返回对象拆开看看。"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

model = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url=os.environ["DEEPSEEK_BASE_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
)

resp = model.invoke("用三个词描述一下你自己")

print("返回值的类型:", type(resp).__name__)
print("是 AIMessage 吗:", isinstance(resp, AIMessage))
print()
print("[content] 正文:", resp.content)
print()
print("[additional_kwargs] 额外字段:", resp.additional_kwargs)
print()
print("[response_metadata] 元数据:", resp.response_metadata)
print()
print("[usage_metadata] token 用量:", resp.usage_metadata)
