import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(override=True)
DIR = r"D:/development/langchain01/demo/final/report.db"

model = init_chat_model(
    model='deepseek-v4-flash',
    base_url = os.getenv('DEEPSEEK_BASE_URL'),
    api_key = os.getenv('DEEPSEEK_API_KEY'),
)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url=os.getenv("OPENROUTER_API_BASE"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

vector = Chroma(persist_directory=DIR, embedding_function=embedding_model)

@tool
def search_info(query:str):
    """查询知识库的信息"""
    text = vector.similarity_search(query, k=3)
    return ("\n\n".join(f"片段 [{i}] {d.page_content}" for i,d in enumerate(text, 1)))

agent = create_agent(
    model= model,
    tools=[search_info],
    system_prompt=(
            "你是知识库问答助手。回答前可以调用 search_info 检索相关片段，"
            "并仅根据检索结果回答，不要用内部知识补充。"
            "若检索结果不足以回答，直接回答：资料库中没有相关信息。"
    ),
    checkpointer=InMemorySaver(),
)

config = {"configurable":{"thread_id":"report-1"}}
res = agent.invoke({"messages":[HumanMessage("东部地区进出口规模达到了多少")]},config=config)
res = agent.invoke({"messages": [HumanMessage("我刚才问的问题你还记得吗？")]},config=config)
res = agent.invoke({"messages": [HumanMessage("介绍一下法国的葡萄酒产区分布")]},config=config)
for m in res['messages']:
    print(m.type,"->", m.content,'\n')