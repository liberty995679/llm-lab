"""第09章·第六格：记忆管理(2) —— 压缩，把旧消息总结成摘要（不丢信息，只丢精度）。"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage

model = init_chat_model("deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"), temperature=0)

agent = create_agent(model=model, tools=[], checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "会话A"}}

for q in ["我叫小明，我喜欢喝咖啡。", "我住在上海，养了只猫。", "我明天想去公园。"]:
    agent.invoke({"messages": [{"role": "user", "content": q}]}, config=cfg)

history = agent.get_state(cfg).values["messages"]
print("压缩前", len(history), "条对话：")
for m in history:
    print(f"  [{m.type}] {m.content}")

# 把前 4 条"太老的对话"交给模型，压缩成一句摘要（这一步花 1 次调用）
old_msgs = history[:-2]
summ = model.invoke([
    SystemMessage(content="把以下对话里用户提到的所有个人信息总结成一句中文，只保留事实。"),
    *old_msgs,
])
print()
print("压缩出的摘要 →", summ.content)

# 两种方案对比
print()
print("裁剪方案：前 4 条直接丢掉 → 信息没了")
print(f"压缩方案：前 4 条换成这句摘要 → 信息基本保住，token 却变少了")
