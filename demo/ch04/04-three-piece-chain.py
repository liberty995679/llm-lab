"""第04章·三段式组合：prompt | model | parser。"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url=os.environ["DEEPSEEK_BASE_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是知识库助手，只根据'{source}'回答，资料里没有的就说'资料库中没有相关信息。'"),
    ("human", "{question}"),
])

# 三段拼成一条链：dict -> 消息列表 -> AIMessage -> 纯字符串
chain = prompt | model | StrOutputParser()

# ① 不花钱的"解剖"：看这条链由哪三块组成
print("链的内部结构:", [type(s).__name__ for s in chain.steps])

# ② 一次调用，感受"出口类型"变了
result = chain.invoke({
    "source": "公司 FAQ 文档",
    "question": "退换货需要几天？",
})
print("返回类型:", type(result).__name__, "| 是字符串吗:", isinstance(result, str))
print("结果:", result)

# ③ 拼好的链条本身也是一个 Runnable：全家桶方法都能用
methods = [m for m in ("invoke", "batch", "stream", "ainvoke", "astream", "abatch")
           if hasattr(chain, m)]
print("链支持的方法:", methods)
