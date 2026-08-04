"""第09章·第四格：记忆的代价 —— 历史消息越积越多。"""
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

# 连续 3 轮对话，每轮后偷看仓库消息数
for i, q in enumerate(["我叫小明。", "我喜欢喝咖啡。", "我住在上海。"], start=1):
    agent.invoke({"messages": [{"role": "user", "content": q}]}, config=cfg)
    snap = agent.get_state(cfg)
    print(f"第{i}轮后 仓库消息数 = {len(snap.values['messages'])}")
