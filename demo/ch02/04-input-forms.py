"""第02章·输入形式的统一：字符串 vs 消息列表。"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler

model = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url=os.environ["DEEPSEEK_BASE_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
)


class Spy(BaseCallbackHandler):
    """偷看模型实际收到的消息列表长什么样。"""

    def on_chat_model_start(self, serialized, messages, **kwargs):
        seen = [(type(m).__name__, m.content[:30]) for m in messages[0]]
        print("  模型实际收到的消息:", seen)


print("① 传字符串：")
model.invoke("你是谁？", config={"callbacks": [Spy()]})

print("② 传单条 HumanMessage：")
model.invoke([HumanMessage("你是谁？")], config={"callbacks": [Spy()]})

print("③ 传多轮对话（消息列表的真正意义）：")
conversation = [
    HumanMessage("你好，我叫小明。"),
    AIMessage("你好小明，很高兴认识你！"),
    HumanMessage("我刚才自我介绍了吗？我叫什么？"),
]
resp = model.invoke(conversation, config={"callbacks": [Spy()]})
print("  模型的回答:", resp.content)
