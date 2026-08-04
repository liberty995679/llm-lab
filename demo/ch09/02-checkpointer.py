"""第09章·第二格：加一个 checkpointer（记忆存储器），Agent 就记住了。"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

model = init_chat_model("deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"), temperature=0)

# 关键①：创建 Agent 时传入 checkpointer = 给它装上一个"记忆存储器"
agent = create_agent(model=model, tools=[], checkpointer=InMemorySaver())

# 关键②：invoke 时带上 config，里面 thread_id = "会话编号"
cfg = {"configurable": {"thread_id": "会话A"}}

cfg02 = {"configurable": {"thread_id": "会话B"}}

r1 = agent.invoke({"messages": [{"role": "user", "content": "记住：我叫小明，喜欢喝咖啡。"}]}, config=cfg)
print("第一轮 →", r1["messages"][-1].content)

r2 = agent.invoke({"messages": [{"role": "user", "content": "我叫什么名字？"}]}, config=cfg02)
print("第二轮 →", r2["messages"][-1].content)
