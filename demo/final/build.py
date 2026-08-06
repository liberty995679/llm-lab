import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownTextSplitter
from langchain_chroma import Chroma
load_dotenv(override=True)

text = Path(r"d:/development/langchain01/demo/final/中国地区经济报告2026.md").read_text(encoding='utf-8')

docs = MarkdownTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
).create_documents([text])

embed = OpenAIEmbeddings(
      model="text-embedding-3-small",
      base_url=os.getenv("OPENROUTER_API_BASE"),
      api_key=os.getenv("OPENROUTER_API_KEY"),
)

Chroma.from_documents(
    docs, embed,
    persist_directory=r"D:/development/langchain01/demo/final/report.db"
)

print(f"切分成了 {len(docs)}, 已经写入")