"""第06章·6.3.1 with_structured_output：method="function_calling"（默认姿势）。"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field


# 复用 6.2 的契约
class Product(BaseModel):
    """从商品描述中提取的信息。"""

    product_name: str = Field(description="商品名称，例如：小米14 Ultra")
    price: float = Field(description="价格，单位元，纯数字。把'4599元'拆出数字 4599")
    in_stock: bool = Field(description="是否现货，是/有货=True，否/缺货=False")
    source_text: str = Field(description="原文中与商品相关的完整描述")


# with_structured_output 返回一个"新的 Runnable"：输入文本，输出 Product 对象
# 踩坑教训：deepseek-v4-flash 默认思考模式禁止 tool_choice（函数调用通道），
# 必须用 extra_body 关掉 thinking 才能走 method="function_calling"
model = init_chat_model(
    "deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
    model_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
).with_structured_output(Product)

# 输入可以就是一句人话（语法糖：内部自动包成 HumanMessage）
result = model.invoke("提取：小米14 Ultra 512G 黑色，售价 4599 元，今日现货。")

print("返回类型:", type(result).__name__)
print("—— 提取结果 ——")
print(result)
print()
print("result.price    类型:", type(result.price).__name__, "| 值:", result.price)
print("result.in_stock 类型:", type(result.in_stock).__name__, "| 值:", result.in_stock)
