"""第07章·7.2 create_agent：模型 + 工具 + 循环，一行创建智能体。"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.agents import create_agent


# 复用 05 章的工具：工具集 = Agent 的能力范围
@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。仅支持北京、上海、深圳。"""
    data = {"北京": "晴，25°C", "上海": "小雨，28°C", "深圳": "多云，30°C"}
    return data.get(city, "没有该城市的数据")


model = init_chat_model(
    "deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)  # 故意不关 thinking，验证 Agent 工具循环会不会踩 tool_choice 的坑

agent = create_agent(model=model, tools=[get_weather])
print("agent 类型:", type(agent).__name__)

resp = agent.invoke({"messages": [{"role": "user", "content": "北京和深圳今天天气怎么样？"}]})

# 打印完整消息历史 = 展示框架替我们跑完的"整个循环"
print()
print("—— 框架跑出的完整消息历史 ——")
for m in resp["messages"]:
    if m.type == "ai" and m.tool_calls:
        calls = ", ".join(f"{c['name']}({c['args']})" for c in m.tool_calls)
        print(f"[AI 意图] 调工具 → {calls}")
    elif m.type == "tool":
        print(f"[工具结果] {m.content}")
    else:
        print(f"[{m.type}] {m.content}")
