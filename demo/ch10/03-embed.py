"""第10章 格3：embed_documents 批量向量化 + 语义相似度演示（仅 1 次 API 调用）"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv(r"D:/development/langchain01/.env", override=True)

# 加载 + 切分（复用前两格，0 API）
loader = TextLoader(r"D:/development/langchain01/demo/ch10/data/knowledge.txt", encoding="utf-8")
chunks = RecursiveCharacterTextSplitter(
    chunk_size=120, chunk_overlap=30,
    separators=["\n==============================\n", "\n\n", "\n", "。", "，", " ", ""],
).split_documents(loader.load())

# 向量化：31 个 chunk 一批发，1 次调用
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url=os.getenv("OPENROUTER_API_BASE"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
texts = [c.page_content for c in chunks]
vecs = embedding_model.embed_documents(texts)
print("向量数量:", len(vecs), "| 每个维度:", len(vecs[0]))

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# 用已算出的向量做语义对比（0 额外调用）
refund = next(i for i, t in enumerate(texts) if "退款规则" in t and "四、" in t)
invoice = next(i for i, t in enumerate(texts) if "发票规则" in t and "五、" in t)
intro = next(i for i, t in enumerate(texts) if "产品简介" in t and "一、" in t)
print(f"块{refund}(退款) vs 块{invoice}(发票): 余弦={cos(vecs[refund], vecs[invoice]):.3f}")
print(f"块{refund}(退款) vs 块{intro}(产品):  余弦={cos(vecs[refund], vecs[intro]):.3f}")
