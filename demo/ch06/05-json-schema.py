"""第06章·6.3.3 method="json_schema"：把 JSON Schema 作为 response_format 传给模型。"""
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


model = init_chat_model(
    "deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
    model_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
).with_structured_output(Product, method="json_schema", include_raw=True)

out = model.invoke([
    SystemMessage("你是商品信息提取器。"),
    HumanMessage("提取：小米14 Ultra 512G 黑色，售价 4599 元，今日现货。"),
])

print("解析后的对象:", out["parsed"])
print("解析错误:", out["parsing_error"])
