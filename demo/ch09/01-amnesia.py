"""第09章·第一格：Agent 是"失忆"的（还没有加记忆）。"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

model = init_chat_model("deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"), temperature=0)

agent = create_agent(model=model, tools=[])

# 第一次对话：告诉它一个事实
r1 = agent.invoke({"messages": [{"role": "user", "content": "记住：我叫小明，喜欢喝咖啡。"}]})
print("第一轮回答 →", r1["messages"][-1].content)

# 第二次对话：问它这个事实
r2 = agent.invoke({"messages": [{"role": "user", "content": "我叫什么名字？"}]})
print("第二轮回答 →", r2["messages"][-1].content)
