import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent,AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langgraph.store.memory import InMemoryStore

from typing import NotRequired


load_dotenv(override=True)

store = InMemoryStore()

class Customuser(AgentState):
    user_id: NotRequired[str]

model = init_chat_model("deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"), temperature=0
)

@tool(parse_docstring = True)
def sav_user_info(name:str,runtime:ToolRuntime):
    """保存用户的会话记录

    Args:
        name:用户名

    Returns:
        保存状态
    """
    runtime.store.put(("users",),runtime.state['user_id'],{'name':name})
    return "saved"

@tool(parse_docstring=True)
def get_user_info(runtime:ToolRuntime):
    """从长期记忆中读取用户信息

    Returns:
        str:用户信息
    """
    item = runtime.store.get(("users",),runtime.state['user_id'])
    return (item.value) if item else "unkown"

cfg = {"configurable":{"thread_id":"t1"}}

agent = create_agent(
    model = model,
    tools = [sav_user_info, get_user_info],
    store = store,
    state_schema=Customuser,
    checkpointer=InMemorySaver(),
    system_prompt="按照用户说的信息来，身份以 store 为准，不要从对话里猜，优先调用已有的工具，20字以内答复"
)

res01 = agent.invoke({"messages":[HumanMessage("我叫张三")],'user_id':'user-A'},config=cfg)
print("01：我叫张三：")
print(res01['messages'][-1].content)

res02 = agent.invoke({"messages":[HumanMessage("我叫什么？")],'user_id':'user-B'},config=cfg)
print("02: 我叫什么？（换成了B用户，结果应该是不知道）")
print(res02['messages'][-1].content)

res03 = agent.invoke({"messages":[HumanMessage("我叫李四，请记住")],'user_id':'user-B'},config=cfg)
print("03: 我叫李四请记住")
print(res03['messages'][-1].content)

res04 = agent.invoke({"messages":[HumanMessage("我叫什么？")],'user_id':'user-A'},config=cfg)
print("04:我叫什么？因为是A用户，应该为张三")
print(res04['messages'][-1].content)




