import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

model = init_chat_model(
    model= "deepseek-v4-flash",
    api_key = DEEPSEEK_API_KEY,
    base_url = DEEPSEEK_BASE_URL
)

prompt = ChatPromptTemplate.from_messages([
    ("system","你是一个知识库助手，你需要根据{faq}的描述回答问题,资料里没有的就说'资料库中没有相关信息。'"),
    ("human","{question}")
])

chain = prompt | model | StrOutputParser()

questions = ["你们的客服电话是多少？", "你们支持Linux吗", "什么是langchain"]

for q in questions:
    res = chain.invoke({
        'faq':"""
        FAQ:
        - 产品名：灵犀笔记
        - 客服电话：400-123-4567
        - 支持平台：Windows、macOS、iOS、Android
        - 免费版：100 篇笔记
        """,
        'question': q
    })
    print(res)