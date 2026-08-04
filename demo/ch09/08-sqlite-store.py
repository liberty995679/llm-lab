"""第09章·第八格A：SqliteSaver —— 把记忆写进本地 sqlite 文件（跨进程持久）。"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver

model = init_chat_model("deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"), temperature=0)

# 唯一区别：把 InMemorySaver 换成 SqliteSaver，指向本地文件
# 注意：from_conn_string 要的是普通文件路径，不是 sqlite:/// URI（源码 docstring 有示例）
DB = r"D:/development/langchain01/demo/ch09/ch09_mem.db"

with SqliteSaver.from_conn_string(DB) as checkpointer:
    agent = create_agent(model=model, tools=[], checkpointer=checkpointer)
    cfg = {"configurable": {"thread_id": "会话A"}}
    agent.invoke({"messages": [{"role": "user", "content": "记住：我叫小明。"}]}, config=cfg)
    print("已写进本地文件。进程内仓库消息数 =", len(agent.get_state(cfg).values["messages"]))

print("进程结束 → 数据已落盘到 ch09_mem.db，不随进程消失")
