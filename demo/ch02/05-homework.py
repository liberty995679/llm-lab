import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage,AIMessage

load_dotenv(override= True)

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

model = init_chat_model(
    model="deepseek-v4-flash",
    base_url=DEEPSEEK_BASE_URL,   # 来自 .env
    api_key=DEEPSEEK_API_KEY,     # 来自系统环境变量
    temperature=0,
)

def chat(history: list, user_input):
    history.append(HumanMessage(content=user_input))
    return model.invoke(history)

history = []
for i in range(3):
    abc = input("请输入：")
    resp = chat(history, abc)
    print("模型回复：", resp.content)
    history.append(AIMessage(content=resp.content))
    print(len(history))





