"""第10章 加餐：把检索包成 @tool，create_agent 自主决定何时检索（0 次 embedding 建库）"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv(r"D:/development/langchain01/.env", override=True)
DB_DIR = r"D:/development/langchain01/demo/ch10/homework.db"

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url=os.getenv("OPENROUTER_API_BASE"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
vector = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)


@tool
def search_knowledge(query: str) -> str:
    """在知识库中检索与问题最相关的 3 个文本片段，返回原文。"""
    hits = vector.similarity_search(query, k=3)
    return "\n\n".join(f"[片段{i}] {d.page_content}" for i, d in enumerate(hits, 1))


model = init_chat_model("deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[search_knowledge],
    system_prompt=(
        "你是知识库问答助手。回答前必须先调用 search_knowledge 检索相关片段，"
        "并仅根据检索结果回答，不要用内部知识补充。"
        "若检索结果不足以回答，直接回答：资料库中没有相关信息。"
    ),
)

result = agent.invoke({"messages": [("user", "试用版到期后数据会立刻删除吗？")]})
for m in result["messages"]:
    print(f"\n=== {type(m).__name__} ===")
    if getattr(m, "tool_calls", None):
        print("tool_calls:", [(tc["name"], tc["args"]) for tc in m.tool_calls])
    print(m.content[:500] if m.content else "(无文本内容)")
