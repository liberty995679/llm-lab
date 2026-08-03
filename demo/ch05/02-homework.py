import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

model = init_chat_model(
    model= "deepseek-v4-flash",
    api_key = DEEPSEEK_API_KEY,
    base_url = DEEPSEEK_BASE_URL
)

@tool
def get_user_balance(user_id: str) -> str:
    """查询指定用户的余额。"""
    data = {
        "1001": "1000.00元",
        "1002": "500.00元",
    }
    return data.get(user_id, "没有该用户的数据")

@tool
def get_user_level(user_id: str) -> str:
    """查询指定用户的等级。"""
    data = {
        "1001": "VIP",
        "1002": "普通用户",
    }
    return data.get(user_id, "没有该用户数据")

message = [('system','你是客服助手，必须用工具查询后再回答，禁止编造。'),(
    'human','1001用户的余额和会员等级分别是多少？')
]

bound_tool = model.bind_tools([get_user_balance, get_user_level])
res = bound_tool.invoke( message)

tool_map = {
    "get_user_balance": get_user_balance,
    "get_user_level": get_user_level,
}

while res.tool_calls:
    message.append(AIMessage(content='',tool_calls=res.tool_calls))
    for call in res.tool_calls:
        tool = tool_map[call['name']]
        result = tool.invoke(call['args'])
        message.append(ToolMessage(tool_call_id=call['id'], content=result))
    res = bound_tool.invoke(message)

print(res.content)
