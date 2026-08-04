"""第09章·第五格：记忆管理(1) —— 裁剪，把太老的消息丢掉。"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import trim_messages

model = init_chat_model("deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"), temperature=0)

agent = create_agent(model=model, tools=[], checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "会话A"}}

for q in ["我叫小明，我喜欢喝咖啡。", "我住在上海。", "我明天想去公园。"]:
    agent.invoke({"messages": [{"role": "user", "content": q}]}, config=cfg)

history = agent.get_state(cfg).values["messages"]
print("裁剪前", len(history), "条：")
for m in history:
    print(f"  [{m.type}] {m.content}")

# 裁剪：只保留最后 4 条（这里用消息条数当"token"来计数）
trimmed = trim_messages(history, max_tokens=4, token_counter=len, strategy="last")
print("裁剪后", len(trimmed), "条：")
for m in trimmed:
    print(f"  [{m.type}] {m.content}")
