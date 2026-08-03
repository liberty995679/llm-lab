import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()

model = init_chat_model(
    "deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

@tool
def get_time(city: str) -> str:
    """返回城市当前时间。"""
    return {"北京": "08:30", "上海": "10:30"}[city]

@tool
def get_weather(city: str) -> str:
    """返回城市天气。"""
    return {"北京": "晴天", "上海": "暴雨"}[ city]

agent = create_agent(model=model, tools=[get_time, get_weather])

question = ["北京今天天气如何？", "上海今天几点？", "北京天气和上海几点?"]

for q in question:
    res = agent.invoke({"messages": [{"role": "user", "content": q}]})
    for m in res["messages"]:
        print(m.type, "→", m.content)
        if m.type == 'ai' and m.tool_calls:
            print(" 使用tool_call:", m.tool_calls)