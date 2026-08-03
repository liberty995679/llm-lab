"""第04章·SystemMessage：一句话给模型立规矩。"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

model = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url=os.environ["DEEPSEEK_BASE_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
)

print("① 角色定义：SystemMessage 是给模型的'人设与行为守则'")
resp = model.invoke(
    [
        SystemMessage("你是一个中英翻译官。用户说中文，你只回复对应的英文，不解释，不加多余的话。"),
        HumanMessage("今天天气真好"),
    ]
)
print("  模型回复:", resp.content)

print()
print("② 行为边界：知识库 Agent 的雏形——'不知道就说不知道'")
resp2 = model.invoke(
    [
        SystemMessage(
            "你是一个知识库助手。你只能依据'提供的资料'回答；"
            "资料里没有的内容，必须明确回答'资料库中没有相关信息'，禁止编造。"
        ),
        HumanMessage("请告诉我：LangChain 的作者是谁？（注意：资料中未提供此信息）"),
    ]
)
print("  模型回复:", resp2.content)
