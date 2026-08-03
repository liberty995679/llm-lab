"""第06章·6.2 定义 schema：Pydantic BaseModel vs TypedDict，以及契约长什么样。"""
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# ① 契约主力：Pydantic BaseModel（带校验 + 字段说明）
class Product(BaseModel):
    """从商品描述中提取的信息。"""

    product_name: str = Field(description="商品名称，例如：小米14 Ultra")
    price: float = Field(description="价格，单位元，纯数字。把'4599元'拆出数字 4599")
    in_stock: bool = Field(description="是否现货，是/有货=True，否/缺货=False")
    source_text: str = Field(description="原文中与商品相关的完整描述")


# 关键：看看这份契约会被翻译成什么——模型就是照着它填的
print("—— Product 的 JSON Schema ——")
print(Product.model_json_schema())
print()

# ② 轻量替代：TypedDict（只给类型检查工具看，运行时不校验、无字段说明）
class ProductTD(TypedDict):
    product_name: str
    price: float
    in_stock: bool

print("—— 对比：TypedDict 完全没有 model_json_schema ——")
print("ProductTD 只是类型标注，运行时没有任何约束")
print()

# ③ 校验能力：Pydantic 在"数据不对"时会当场报错（这就是契约的执行力）
try:
    bad = Product(product_name="小米", price="4599元", in_stock="是", source_text="...")
    print("校验通过：", bad)
except Exception as e:
    print("校验失败 ❌：", e)
