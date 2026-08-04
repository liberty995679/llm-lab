"""第09章·第八格B：新进程，用同一个 sqlite 文件问同一个会话。"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver

model = init_chat_model("deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"), temperature=0)

DB = r"D:/development/langchain01/demo/ch09/ch09_mem.db"

with SqliteSaver.from_conn_string(DB) as checkpointer:
    agent = create_agent(model=model, tools=[], checkpointer=checkpointer)
    cfg = {"configurable": {"thread_id": "会话A"}}
    r2 = agent.invoke({"messages": [{"role": "user", "content": "我叫什么名字？"}]}, config=cfg)
    print("回答 →", r2["messages"][-1].content)
