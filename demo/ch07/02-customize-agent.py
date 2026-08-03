"""第07章·7.4 定制化：system_prompt 边界 + 结构化输出（final_answer 工具契约）。

踩坑记录：create_agent(response_format=...) 在 DeepSeek 上会 400
（"This response_format type is unavailable now"）——它内部是先跑循环再单独
调 with_structured_output，DeepSeek 不支持。跨厂商稳健做法 = 结构化输出工具。
"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.agents import create_agent


@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。仅支持北京、上海、深圳。"""
    data = {"北京": "晴，25°C", "上海": "小雨，28°C", "深圳": "多云，30°C"}
    return data.get(city, "没有该城市的数据")


# 结构化契约就藏在这个工具的 schema 里（06 章三字段思想的 Agent 版）
@tool
def final_answer(cities: list[str], summary: str, out_of_scope: bool) -> str:
    """给出最终的结构化回答。调用本工具即代表本次回答结束。参数：
    - cities: 本次回答涉及的城市列表
    - summary: 天气总结，一句话
    - out_of_scope: 问题是否超出天气范围，无关为 True"""
    return "已记录最终答案。"


model = init_chat_model(
    "deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)  # thinking 保留：Agent 自由决策调工具，不踩 tool_choice 的墙

agent = create_agent(
    model=model,
    tools=[get_weather, final_answer],
    system_prompt="你是天气预报助手，只负责天气问题。最终必须调用 final_answer 工具给出结构化回答。",
)


def extract_structured(messages):
    """从消息历史里取出最后一次 final_answer 调用的参数。"""
    for m in reversed(messages):
        if m.type == "ai" and m.tool_calls:
            for tc in m.tool_calls:
                if tc["name"] == "final_answer":
                    return tc["args"]
    return None


resp = agent.batch([
    {"messages": [{"role": "user", "content": "上海今天天气怎么样？"}]},
    {"messages": [{"role": "user", "content": "帮我写一首关于大海的诗"}]},
])

for i, r in enumerate(resp):
    print(f"—— 问题{i+1} ——")
    print("  结构化结果:", extract_structured(r["messages"]))
    print()
