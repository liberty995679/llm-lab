"""第04章·模板高级特性：partial 部分填充 + few-shot 示例注入。"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

model = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url=os.environ["DEEPSEEK_BASE_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
)

# ① partial：把"固定"的变量提前绑死，调用时不用再传（零成本，不调模型）
base = ChatPromptTemplate.from_messages([
    ("system", "你是{role}。"),
    ("human", "{question}"),
]).partial(role="知识库助手")

msgs = base.invoke({"question": "只传 question 就能跑，role 已被 partial 固定"})
print("【partial 产出】")
for m in msgs.to_messages():
    print(f"  {type(m).__name__}: {m.content}")

# ② few-shot：塞两条示例，让模型"照格式答"（只花 1 次调用）
few_shot = ChatPromptTemplate.from_messages([
    ("system",
     "你是知识库助手。回答必须严格模仿下面示例的格式："
     "资料里有的写'根据资料：...'，没有的写'资料库中没有相关信息。'"),
    ("human", "知识库问题：什么是 LangChain？"),
    ("ai", "根据资料：LangChain 是一个大语言模型应用开发框架。"),
    ("human", "知识库问题：北京今天下雨吗？"),
    ("ai", "资料库中没有相关信息。"),
    ("human", "知识库问题：{question}"),
])

chain = few_shot | model
resp = chain.invoke({"question": "LangChain 的作者是谁？"})
print()
print("【few-shot 结果】", resp.content)
