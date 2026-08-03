"""EngramRouter quickstart example.

Run:
    python examples/quickstart.py
"""

from engram_router import MemoryStore

# ---- 1. Create a memory store --------------------------------------------
store = MemoryStore(enable_vector=False)  # Skip vector for quick demo
print("[1] MemoryStore created")

# ---- 2. Save some memories -----------------------------------------------
mem1 = store.save("张三今年30岁，是程序员，在字节跳动工作", namespace="demo")
mem2 = store.save("李四住北京朝阳区，养了只橘猫叫咪咪", namespace="demo")
mem3 = store.save("张三喜欢钓鱼和爬山，周末经常去户外", namespace="demo")
mem4 = store.save("上次团建去的古北水镇，大家玩得很开心", namespace="demo")
print(f"[2] Saved {mem1}, {mem2}, {mem3}, {mem4}")

# ---- 3. Recall — 三层召回 + 评分 -----------------------------------------
results = store.recall("张三有什么爱好", top_k=3, namespace="demo")
print("[3] Recall: '张三有什么爱好'")
for i, r in enumerate(results):
    print(f"    {i+1}. [{r.score:.2f}] {r.raw_text}  ({r.match_reason})")

# ---- 4. Gap check — 检测记忆空白 -----------------------------------------
gaps = store.gap_check("团队最近去了哪里", namespace="demo")
print(f"[4] Gap check: {gaps}")

# ---- 5. Timeline — 时间线查询 --------------------------------------------
tl = store.timeline.get_timeline(limit=5)
print(f"[5] Timeline: {len(tl)} events")

# ---- 6. Delete — 带 namespace 保护 ---------------------------------------
deleted = store.delete(mem4, namespace="demo")
print(f"[6] Deleted {mem4}: {deleted}")

# ---- 7. Cleanup -----------------------------------------------------------
store.close()
print("[7] Done")
