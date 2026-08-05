"""第10章 格2：RecursiveCharacterTextSplitter 切分 Document -> chunks（0 API 调用）"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader(r"D:/development/langchain01/demo/ch10/data/knowledge.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,      # 每个块最大字符数
    chunk_overlap=30,    # 相邻块重叠字符数
    separators=["\n==============================\n", "\n\n", "\n", "。", "，", " ", ""],
)
chunks = splitter.split_documents(docs)

print("切出块数:", len(chunks))
print("原始文档长度:", len(docs[0].page_content))
for i, c in enumerate(chunks[:4]):
    print(f"\n--- 块{i} len={len(c.page_content)} ---")
    print(c.page_content)
