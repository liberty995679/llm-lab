"""第09章·第九格：长期记忆(1) —— store 的四层架构 put()/get()（零API调用）。"""
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# 四层架构：store 仓库 → namespace 分区 → key 键 → value 值
ns = ("users", "alice", "memories")          # namespace = "谁的哪类记忆"

store.put(ns, "pref_food", {"food": "咖啡"})  # 写入一条
item = store.get(ns, "pref_food")             # 读回
print("get() 返回类型:", type(item).__name__)
print("item.value:", item.value)

store.put(ns, "pref_food", {"food": "茶"})    # 同 key 再写 → 覆盖
print("覆盖后:", store.get(ns, "pref_food").value)

print("get() 不存在的 key →", store.get(ns, "not_exist"))
