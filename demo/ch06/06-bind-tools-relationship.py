"""第06章·6.4 with_structured_output 与 bind_tools 的关系：零成本内省。"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field


class Product(BaseModel):
    """从商品描述中提取的信息。"""

    product_name: str = Field(description="商品名称")
    price: float = Field(description="价格，单位元，纯数字")
    in_stock: bool = Field(description="是否现货")
    source_text: str = Field(description="原文相关描述")


model = init_chat_model(
    "deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
    model_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
)

structured = model.with_structured_output(Product)

print("返回类型:", type(structured).__name__)
print()
print("—— 内部结构 repr ——")
print(structured)
print()

# 对比：05章手动 bind_tools 时，传给模型的"工具列表"长什么样
print("—— 契约被包装成工具的 schema（和 bind_tools 看到的同款）——")
tool_schema = Product.model_json_schema()
print({"type": "function", "function": {"name": "Product", "parameters": tool_schema}})
