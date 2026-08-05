"""诊断：查看作业库的实际召回内容（尤其第2问的上下文是否真的包含退款信息）"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv(r"D:/development/langchain01/.env", override=True)

DB_DIR = r"D:/development/langchain01/demo/ch10/homework.db"
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url=os.getenv("OPENROUTER_API_BASE"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)
print("库内条数:", vectorstore._collection.count(), "(预期 31，若更大说明重复写入)")

q = "为什么我申请退款被拒了？"
hits = vectorstore.similarity_search(q, k=3)
print(f"\n===== 第2问召回的 3 条片段 =====")
for i, d in enumerate(hits, 1):
    print(f"--- 片段{i} ---")
    print(d.page_content)
    print()
