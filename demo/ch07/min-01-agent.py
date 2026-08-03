"""第07章·反转课堂第一格：最小的能调工具的 Agent。"""
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
resp = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})

for m in resp["messages"]:
    print(m.type, "→", m.content)
    if m.type == 'ai' and m.tool_calls:
        print("  tool_call:", m.tool_calls)
