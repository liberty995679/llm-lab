"""第06章·6.5 关联最终项目：知识库 Agent 的三字段答案契约。"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field


# 最终 Agent 的"答案契约"：answer + citations + out_of_scope
class KBAnswer(BaseModel):
    """知识库 Agent 对用户问题的结构化回答。"""

    answer: str = Field(description="基于资料给出的答案，不得编造资料外的信息")
    citations: list[str] = Field(description="支撑答案的资料条目编号，如 ['2']；无支撑则为空列表 []")
    out_of_scope: bool = Field(description="问题是否超出资料范围。资料中找不到答案时为 True")


# 模拟知识库资料（ch04 作业同款；第 10 章 RAG 会把这个换成"检索出来的段落"）
KB_CONTEXT = """【1】客服电话：400-888-0000，工作时间 9:00-21:00。
【2】退款政策：收货后 7 天内无理由退款，运费自理。
【3】发货时间：下单后 48 小时内发货。
【4】保修：整机保修 1 年，屏幕保修 6 个月。"""

SYSTEM = (
    "你是知识库问答助手。只能根据下面资料回答，资料里没有的信息，"
    "回答「资料库中没有相关信息」。\n资料：\n" + KB_CONTEXT
)

model = init_chat_model(
    "deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
    model_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
).with_structured_output(KBAnswer)

questions = [
    "怎么申请退款？",
    "你们的创始人的星座是什么？",
]

# batch：一次请求并发生成（复用 02 章知识，省时间）
results = model.batch([[SystemMessage(SYSTEM), HumanMessage(q)] for q in questions])

for q, r in zip(questions, results):
    print("问题:", q)
    print("  answer      :", r.answer)
    print("  citations   :", r.citations)
    print("  out_of_scope:", r.out_of_scope)
    print()
