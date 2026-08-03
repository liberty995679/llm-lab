"""第02章·异步三兄弟：ainvoke / abatch / astream"""
import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url=os.environ["DEEPSEEK_BASE_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
)


async def main():
    # 1) ainvoke：异步单次调用
    resp = await model.ainvoke("用一句话解释 async/await")
    print("【ainvoke】", resp.content[:50])

    # 2) abatch：异步批量，一次性发多条
    results = await model.abatch(["1+1=?", "2+2=?", "3+3=?"])
    for i, r in enumerate(results, 1):
        print(f"【abatch 第{i}条】{r.content}")

    # 3) astream：异步流式，逐 token 吐
    print("【astream】")
    async for chunk in model.astream("用一句话描述北京的秋天"):
        print(chunk.content, end="", flush=True)
    print()


asyncio.run(main())
