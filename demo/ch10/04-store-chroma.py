"""第10章 格4：向量存进 Chroma 本地库（1 次 embedding 调用，之后落盘）"""
import sys
import os
import shutil
sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv(r"D:/development/langchain01/.env", override=True)

DB_DIR = r"D:/development/langchain01/demo/ch10/data/chroma_db"

# 加载 + 切分（0 API）
loader = TextLoader(r"D:/development/langchain01/demo/ch10/data/knowledge.txt", encoding="utf-8")
chunks = RecursiveCharacterTextSplitter(
    chunk_size=120, chunk_overlap=30,
    separators=["\n==============================\n", "\n\n", "\n", "。", "，", " ", ""],
).split_documents(loader.load())

# 建库前先清掉旧库，避免重复写入堆积
if os.path.exists(DB_DIR):
    shutil.rmtree(DB_DIR)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url=os.getenv("OPENROUTER_API_BASE"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# 一步到位：切分好的文档 → 向量化 → 存盘
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=DB_DIR,
)
print("已写入 Chroma，库内条数:", vectorstore._collection.count())
print("落盘目录:", DB_DIR)
print("目录内容:", os.listdir(DB_DIR))
