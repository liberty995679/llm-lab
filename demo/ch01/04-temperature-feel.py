"""第01章·正确地感受 temperature：用开放式文本，而不是随机数。"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI


def make(t):
    return ChatOpenAI(
        model="deepseek-v4-flash",
        base_url=os.environ["DEEPSEEK_BASE_URL"],
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=t,
    )


prompt = "请写一句20字以内的广告语，给一家专做珍珠奶茶的店。"

print("===== temperature = 0.0 （每次几乎都一样：永远挑概率最高的词） =====")
m0 = make(0.0)
for i in range(3):
    print(f"  [{i+1}] {m0.invoke(prompt).content.strip()}")

print()
print("===== temperature = 1.5 （每次都不一样：给低概率的词更多出场机会） =====")
m15 = make(1.5)
for i in range(3):
    print(f"  [{i+1}] {m15.invoke(prompt).content.strip()}")
