"""第09章·第三格：偷看 checkpointer 仓库里存了什么（不调模型）。"""
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

r1 = agent.invoke({"messages": [{"role": "user", "content": "我叫小明，我喜欢喝咖啡。"}]}, config=cfg)
print("回答 →", r1["messages"][-1].content)

# 零成本内省：get_state 从本地仓库读当前快照，不再花钱调模型
snap = agent.get_state(cfg)
print("state 类型:", type(snap).__name__)
print("仓库里已存消息数:", len(snap.values["messages"]))
for m in snap.values["messages"]:
    print(f"  [{m.type}] {m.content}")
