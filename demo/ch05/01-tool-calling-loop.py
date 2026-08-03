"""第05章·工具调用核心机制：意图与执行分离。"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage


# ① @tool 把普通函数变成工具：函数名 / docstring / 类型注解 → 自动生成 schema
@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。仅支持北京、上海、深圳。"""
    data = {"北京": "晴，25°C", "上海": "小雨，28°C", "深圳": "多云，30°C"}
    return data.get(city, "没有该城市的数据")


# 看看工具自动生成的 schema 长什么样（零成本，不调模型）
print("工具自动生成的 schema:")
print(get_weather.args_schema.model_json_schema())
print()

# ② 把工具"绑"到模型上：模型从此"知道"有这个工具可以用
model = init_chat_model(
    "deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
).bind_tools([get_weather])

messages = [
    SystemMessage("你是一个天气助手。你必须调用工具获取天气后再回答，禁止直接编造天气。"),
    HumanMessage("北京今天天气怎么样？"),
]

# ③ 第一轮：模型返回的不是文字，而是"工具调用意图"
resp = model.invoke(messages)
print("模型回复类型:", type(resp).__name__)
print("tool_calls:", resp.tool_calls)

# ④ 你（代码）执行工具
call = resp.tool_calls[0]
result = get_weather.invoke(call["args"])
print("工具执行结果:", result)

# ⑤ 把结果作为 ToolMessage 喂回模型，模型基于结果给最终回答
# 注意：langchain 1.5 里 content 不能为 None，用空字符串承载 tool_calls
messages.append(AIMessage(content="", tool_calls=[call]))
messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

final = model.invoke(messages)
print("最终回答:", final.content)
