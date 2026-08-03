"""第02章·用计时证明异步的价值：同步6连 vs 并发6连。"""
import asyncio
import os
import time

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url=os.environ["DEEPSEEK_BASE_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
)

questions = ["1+1=?", "2+2=?", "3+3=?", "9+9=?", "8+8=?", "7+7=?"]

# ---- 同步版：一个接一个，总耗时 ≈ 单条 × 6 ----
t0 = time.perf_counter()
sync_answers = [model.invoke(q).content for q in questions]
t_sync = time.perf_counter() - t0
print(f"同步 6 连：耗时 {t_sync:.2f}s")

# ---- 异步并发版：一次性全部发出去 ----
async def run_all():
    # asyncio.gather：让这 6 个协程同时跑，等全部完成
    return await asyncio.gather(*[model.ainvoke(q) for q in questions])

t0 = time.perf_counter()
async_answers = asyncio.run(run_all())
t_async = time.perf_counter() - t0
print(f"并发 6 连：耗时 {t_async:.2f}s")

print(f"\n加速比：{t_sync / t_async:.1f}x")
print("同步答案:", sync_answers)
print("并发答案:", [a.content for a in async_answers])
print("两边答案一致吗:", sync_answers == [a.content for a in async_answers])
