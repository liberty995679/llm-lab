import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel,Field

load_dotenv()

class WorkOrder(BaseModel):
    """客户工单信息。"""
    customer_id: str = Field(default="",description="客户编号，形如 CUS-xxxx；没提到则填空字符串")
    issue_type: str = Field(description="问题类型：投诉/咨询/售后 三选一")
    product: str = Field(description="涉及的产品名，没提到则填空字符串")
    urgency: str = Field(description="紧急程度：紧急/一般。含'很急/尽快/今天'等词为紧急")
    resolved: bool = Field(description="对话结束时问题是否已解决")
    summary: str = Field(description="一句话摘要，30字以内")


model = init_chat_model(
    "deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,

).with_structured_output(WorkOrder, method="json_mode", include_raw=True)


CASES = [
  "你好，我上个月买的蓝牙耳机左边没声音了，很急，帮我看看怎么办",
  "我想问下你们店什么时候发货？订单号 CUS-1024",
  "垃圾产品！！用了三天就坏，再也不来了（没有订单号，情绪激动）",
]

system = """你是一个客户服务小助手，必须输出严格 JSON 对象，不得包含多余文字。
字段 strictly must strictly follow the following definition：
- customer_id: str，客户编号，形如 CUS-xxxx；没提到则填空字符串
- issue_type: str，问题类型：投诉/咨询/售后 三选一
- product: str，涉及产品名，没提到则填空字符串
- urgency: str，紧急程度：紧急/一般。含'很急/尽快/今天'等词为紧急
- resolved: bool，对话结束时问题是否已解决
- summary: str，一句话摘要，30字以内
"""

res = model.batch([[SystemMessage(system), HumanMessage(q)]  for q in CASES])

jsch = WorkOrder.model_json_schema()

print("JSON Schema:", jsch)

for q, r in zip(CASES, res):
    print("问题:", q)

    #  检查解析是否成功
    if r["parsing_error"]:
        print(f"  解析失败：{r['parsing_error']}")
        print(f"  原始输出：{r['raw'].content}")
        print()
        continue

    parsed = r["parsed"]  # ← 取出 WorkOrder 对象
    print("  customer_id:", parsed.customer_id)
    print("  issue_type:", parsed.issue_type)
    print("  product:", parsed.product)
    print("  urgency:", parsed.urgency)
    print("  resolved:", parsed.resolved)
    print("  summary:", parsed.summary)
    print()

