"""第10章 格6：完整 RAG——检索 + 拼上下文 + DeepSeek 生成（1 次 embedding + 1 次对话）"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv(r"D:/development/langchain01/.env", override=True)

DB_DIR = r"D:/development/langchain01/demo/ch10/data/chroma_db"

# 1. 检索
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url=os.getenv("OPENROUTER_API_BASE"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)

question = "基础版 API 调用超额了会怎样？"
hits = vectorstore.similarity_search(question, k=3)
context = "\n\n".join(f"[片段{i}] {d.page_content}" for i, d in enumerate(hits, 1))
print("=== 检索到的上下文 ===\n", context, "\n")

# 2. 生成
model = init_chat_model("deepseek-v4-flash", temperature=0)
resp = model.invoke([
    ("system", "你是一个问答助手。请仅根据检索到的上下文回答问题，不要使用内部知识补充。"
               "如果上下文不足以回答，请直接回答：资料库中没有相关信息。"),
    ("human", f"问题：{question}\n\n上下文：\n{context}"),
])
print("=== DeepSeek 最终回答 ===\n", resp.content)
