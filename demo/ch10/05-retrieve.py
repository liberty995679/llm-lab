"""第10章 格5：检索——打开已落盘的 Chroma，相似度搜索 Top-K（0 次 embedding 调用）"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv(r"D:/development/langchain01/.env", override=True)

DB_DIR = r"D:/development/langchain01/demo/ch10/data/chroma_db"

# 打开已有库（不重新向量化）
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url=os.getenv("OPENROUTER_API_BASE"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
vectorstore = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embedding_model,
)

question = "基础版 API 调用超额了会怎样？"
hits = vectorstore.similarity_search(question, k=3)

print(f"问题: {question}")
print(f"召回了 {len(hits)} 条，按相似度排序:\n")
for i, d in enumerate(hits, 1):
    print(f"--- 第{i}条 (source: {d.metadata.get('source')}) ---")
    print(d.page_content[:120])
    print()
