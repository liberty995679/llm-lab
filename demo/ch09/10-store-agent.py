"""第09章·第十格：长期记忆(3) —— 在工具中访问 store（课件3.3节 DeepSeek 版）。"""
import os

from dotenv import load_dotenv

load_dotenv(override=True)

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from typing import NotRequired
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore

model = init_chat_model("deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"), temperature=0)

store = InMemoryStore()

# 自定义状态：默认 messages 之外，多加一个 user_id 字段
class CustomState(AgentState):
    user_id: NotRequired[str]

# 写工具：把 user_id 的用户名写进长期记忆
@tool(parse_docstring=True)
def save_user_info(name: str, runtime: ToolRuntime) -> str:
    """将用户信息保存在长期记忆中

    Args:
        name: 用户名

    Returns:
        str: 保存状态
    """
    runtime.store.put(("users",), runtime.state["user_id"], {"name": name})
    return "saved"

# 读工具：从长期记忆里把用户信息捞回来
@tool(parse_docstring=True)
def get_user_info(runtime: ToolRuntime) -> str:
    """从长期记忆中读取用户信息

    Returns:
        str: 用户信息
    """
    item = runtime.store.get(("users",), runtime.state["user_id"])
    return str(item.value) if item else "unknown"

agent = create_agent(
    model=model,
    tools=[save_user_info, get_user_info],
    store=store,
    system_prompt="用户提及个人信息时及时记录，用户询问个人信息时尝试用工具检索",
    state_schema=CustomState,
)

# 两个独立会话（无 config 串联），但共享同一个 store
r1 = agent.invoke({"messages": [HumanMessage("你好，我是小花")], "user_id": "user-1"})
r2 = agent.invoke({"messages": [HumanMessage("我是谁？")], "user_id": "user-1"})
print("第一个会话回答 →", r1["messages"][-1].content)
print("第二个会话回答 →", r2["messages"][-1].content)
