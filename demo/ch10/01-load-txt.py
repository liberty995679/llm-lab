"""第10章 格1：TextLoader 加载 txt -> Document 对象（0 API 调用）"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain_community.document_loaders import TextLoader

loader = TextLoader(r"D:/development/langchain01/demo/ch10/data/knowledge.txt", encoding="utf-8")
docs = loader.load()  # list[Document]

print("docs 类型:", type(docs), "| 长度:", len(docs))
doc = docs[0]
print("元素类型:", type(doc))
print("metadata:", doc.metadata)
print("page_content 前 150 字:")
print(doc.page_content[:150])
