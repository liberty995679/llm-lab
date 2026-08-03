"""第04章·ChatPromptTemplate 基础：固定结构 + 变化变量。"""
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

# 1) 定义模板：角色用元组简写 (system / human / ai, 模板文本)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是{role}。你只能根据'{source}'回答，资料里没有的必须明确说'资料库中没有相关信息'。"),
    ("human", "{question}"),
])

# 2) 只注入变量，看模板产出什么（这一步不调用模型，零成本）
messages = prompt.invoke({
    "role": "知识库助手",
    "source": "公司内部产品文档",
    "question": "我们的产品支持哪些语言？",
})
print("【模板产出的消息】")
for m in messages.to_messages():
    print(f"  {type(m).__name__}: {m.content}")

# 3) 拼上模型跑一遍 → 这正是第5节"三段式组合"的雏形
chain = prompt | model
resp = chain.invoke({
    "role": "知识库助手",
    "source": "公司内部产品文档",
    "question": "请用一句话介绍我们的产品叫什么名字。",
})
print()
print("【完整链条输出】", resp.content)
