"""第07章·反转课堂第三格：看思考痕迹（调工具前它在想什么）。"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.agents import create_agent


@tool
def get_time(city: str) -> str:
    """返回城市当前时间。"""
    return {"北京": "08:30", "上海": "08:30"}[city]


model = init_chat_model("deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"), temperature=0)

agent = create_agent(model=model, tools=[get_time])
resp = agent.invoke({"messages": [{"role": "user", "content": "北京现在几点？"}]})

for m in resp["messages"]:
    if m.type == "ai":
        thinking = (m.additional_kwargs.get("reasoning_content") or "")
        print(f"[ai] content={m.content!r}")
        print(f"     思考: {thinking[:200]}")
        if m.tool_calls:
            print(f"     决定调: {[c['name'] for c in m.tool_calls]}")
    else:
        print(f"[{m.type}] {m.content}")
