"""第02章·Runnable 统一协议与 config。"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler

model = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url=os.environ["DEEPSEEK_BASE_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
)

# ① 统一协议：model 本身就是一个 Runnable，你上节刚用过它的全部异步方法
methods = [m for m in ("invoke", "batch", "stream", "ainvoke", "astream", "abatch")
           if hasattr(model, m)]
print("model 具备的调用方法:", methods)

# ② 组合：用 | 把 model 接在 StrOutputParser 前面，组成一条"链"
#    StrOutputParser 作用：把 AIMessage 剥成纯字符串（第06章主角，先借用）
chain = model | StrOutputParser()
print("chain 的类型:", type(chain).__name__)

# ③ 链和模型用同一套接口——这就是"统一协议"的威力
r = chain.invoke("用三个词描述 LangChain")
print("chain.invoke 结果:", repr(r))

# ④ bind：把参数"绑死"到模型上，返回一条新链
short = model.bind(max_tokens=8)
print("bind 后截断:", repr(short.invoke("请写一段100字的自我介绍")))

# ⑤ config：每次调用时注入的"运行配置"
class Spy(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        # messages 是"批量序列"：list[list[Message]]，取第一条序列的最后一条消息
        print("  [窃听] 发给模型的最后一条消息:", messages[0][-1].content)

    def on_llm_end(self, response, **kwargs):
        print("  [窃听] 模型返回:", response.generations[0][0].text)

model.invoke(
    "2+2=?",
    config={"metadata": {"章节": "ch02"}, "tags": ["demo"], "callbacks": [Spy()]},
)
