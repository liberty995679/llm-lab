"""第09章·第七格B：新进程里问同一个会话（同一个 thread_id，换了个进程）。"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

model = init_chat_model("deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"), temperature=0)

agent = create_agent(model=model, tools=[], checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "会话A"}}

r2 = agent.invoke({"messages": [{"role": "user", "content": "我叫什么名字？"}]}, config=cfg)
print("回答 →", r2["messages"][-1].content)
print("仓库消息数 =", len(agent.get_state(cfg).values["messages"]))
