"""第06章·6.3.2 method="json_mode"：保留思考模式的结构化输出。"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field


class Product(BaseModel):
    """从商品描述中提取的信息。"""

    product_name: str = Field(description="商品名称，例如：小米14 Ultra")
    price: float = Field(description="价格，单位元，纯数字。把'4599元'拆出数字 4599")
    in_stock: bool = Field(description="是否现货，是/有货=True，否/缺货=False")
    source_text: str = Field(description="原文中与商品相关的完整描述")


# 关键区别：不关 thinking（不传 extra_body），改用 method="json_mode"
# include_raw=True：返回 dict，同时给你"原始消息"和"解析后的对象"
model = init_chat_model(
    "deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
).with_structured_output(Product, method="json_mode", include_raw=True)

# 踩坑教训1：DeepSeek 的 json_mode 要求 prompt 里必须出现 "json" 字样，否则 400
# 踩坑教训2：json_mode 不会自动把 schema 发给模型（function_calling 会），
#           字段契约必须自己写进 prompt，否则模型自由发挥键名 → 解析失败
out = model.invoke([
    SystemMessage("""你是商品信息提取器，必须输出严格 JSON 对象，不得包含多余文字。
字段严格按以下定义：
- product_name: str，商品名称
- price: float，价格，纯数字，单位元
- in_stock: bool，是否现货
- source_text: str，原文中与商品相关的完整描述"""),
    HumanMessage("提取：小米14 Ultra 512G 黑色，售价 4599 元，今日现货。"),
])

print("返回结构:", list(out.keys()))
print()
print("解析后的对象:", out["parsed"])
print("解析错误:", out["parsing_error"])
print()

# thinking 开启时，推理过程会留在 raw 的 additional_kwargs 里
reasoning = out["raw"].additional_kwargs.get("reasoning_content")
print("思考痕迹（前 100 字）:", reasoning[:100] if reasoning else "（无，说明本次没走思考）")
