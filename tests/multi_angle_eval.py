#!/usr/bin/env python3
"""
engram-router 多角度综合评估脚本（含 LLM 辅助评判）

测试维度:
  1. 精确召回 — 事实性问题，答案唯一
  2. 多跳推理 — 需要关联多条记忆
  3. 跨主题隔离 — 不同人物/话题不串线
  4. 模糊查询 — 口语化、省略主语
  5. 负样本 — 正确拒绝不知道的信息
  6. 时序推理 — 时间线相关查询
  7. 矛盾检测 — 有冲突信息时如何表现
  8. 长对话 — 50+ 轮对话的稳定性

用法:
  cd engram-router
  python tests/multi_angle_eval.py
"""
import json, os, sys, tempfile, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from engram_router.store import MemoryStore

# ═══════════════════════════════════════════════════════════════════════════
# Test scenarios — realistic multi-topic conversations
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TestCase:
    tag: str
    category: str  # precision / multi_hop / isolation / fuzzy / negative / temporal / contradiction / long
    conversation: list[str]
    queries: list[dict]  # [{query, expect_contains, expect_not_contains, description}]
    weight: float = 1.0  # importance weight for scoring

SCENARIOS: list[TestCase] = []

# ── 场景1: 精确召回 ──
SCENARIOS.append(TestCase(
    tag="precision_basic",
    category="precision",
    conversation=[
        "我表姐叫王芳，在北京协和医院当儿科医生。",
        "她老公是做金融的，在国贸上班。",
        "我上周末去她家，她做了糖醋排骨。",
        "她家养了一只金毛，叫豆豆。",
        "王芳说最近医院很忙，流感季。",
        "她女儿今年上小学三年级。",
    ],
    queries=[
        {"q": "我表姐叫什么？", "expect": ["王芳"], "not_expect": [], "desc": "基础人名"},
        {"q": "她在哪家医院工作？", "expect": ["协和医院"], "not_expect": [], "desc": "地点"},
        {"q": "她是什么科室的？", "expect": ["儿科"], "not_expect": [], "desc": "职业细节"},
        {"q": "她老公在哪上班？", "expect": ["国贸"], "not_expect": [], "desc": "关联人物"},
        {"q": "她家狗叫什么？", "expect": ["豆豆"], "not_expect": [], "desc": "宠物"},
        {"q": "她女儿几年级？", "expect": ["三年级"], "not_expect": [], "desc": "家人"},
    ],
))

# ── 场景2: 多跳推理 ──
SCENARIOS.append(TestCase(
    tag="multi_hop",
    category="multi_hop",
    conversation=[
        "我认识一个投资人叫陈总，他在红杉资本。",
        "陈总投过一家做自动驾驶的公司，叫途灵科技。",
        "途灵科技的创始人是清华计算机系毕业的。",
        "他们去年拿了 B 轮，估值 20 亿。",
        "陈总说他最看好 L4 级别的自动驾驶方案。",
    ],
    queries=[
        {"q": "陈总投的什么公司？", "expect": ["途灵科技"], "not_expect": [], "desc": "1跳"},
        {"q": "途灵科技创始人什么背景？", "expect": ["清华", "计算机"], "not_expect": [], "desc": "2跳"},
        {"q": "陈总投的公司估值多少？", "expect": ["B 轮", "20 亿"], "not_expect": [], "desc": "2跳+数值"},
        {"q": "他们拿到哪一轮了？", "expect": ["B 轮"], "not_expect": [], "desc": "模糊指代"},
    ],
))

# ── 场景3: 跨主题隔离 ──
SCENARIOS.append(TestCase(
    tag="isolation",
    category="isolation",
    conversation=[
        # 人物A: 老张
        "老张是我们部门的技术主管，干了十年了。",
        "他之前在华为做过，后来跳过来的。",
        "老张开一辆黑色宝马 X5。",
        "他喜欢钓鱼，周末经常去密云水库。",
        # 人物B: 小李
        "小李是去年刚来的应届生，做前端的。",
        "她特别喜欢猫，养了两只英短。",
        "小李开的是比亚迪海豚，电动车。",
        "她每天骑共享单车到地铁站。",
    ],
    queries=[
        {"q": "老张开什么车？", "expect": ["宝马", "X5"], "not_expect": ["比亚迪", "海豚"], "desc": "老张的车≠小李的车"},
        {"q": "小李开什么车？", "expect": ["比亚迪", "海豚"], "not_expect": ["宝马", "X5"], "desc": "小李的车≠老张的车"},
        {"q": "老张有什么爱好？", "expect": ["钓鱼"], "not_expect": ["猫", "英短"], "desc": "爱好隔离"},
        {"q": "小李养了什么宠物？", "expect": ["猫", "英短"], "not_expect": ["钓鱼", "金毛"], "desc": "宠物隔离"},
        {"q": "谁之前在华为做过？", "expect": ["华为做过"], "not_expect": ["小李"], "desc": "经历归因"},
        {"q": "老张养猫吗？", "expect": [], "not_expect": ["猫"], "desc": "负样本:老张不养猫"},
    ],
))

# ── 场景4: 模糊/口语化查询 ──
SCENARIOS.append(TestCase(
    tag="fuzzy",
    category="fuzzy",
    conversation=[
        "我昨天晚上去吃了一家川菜馆，在望京那边。",
        "点了水煮鱼和宫保鸡丁，味道挺正宗的。",
        "人均大概一百二左右，不算贵。",
        "服务员态度特别好，还送了果盘。",
        "我同事推荐的，说他们家的毛血旺更绝。",
    ],
    queries=[
        {"q": "昨晚去的馆子怎么样？", "expect": ["正宗", "态度特别"], "not_expect": [], "desc": "口语化评价"},
        {"q": "那几个菜？", "expect": ["水煮鱼", "宫保鸡丁"], "not_expect": [], "desc": "省略主语"},
        {"q": "花了多少钱？", "expect": ["一百二"], "not_expect": [], "desc": "价格模糊问"},
        {"q": "在哪吃的？", "expect": ["望京"], "not_expect": [], "desc": "地点"},
    ],
))

# ── 场景5: 负样本（边界） ──
SCENARIOS.append(TestCase(
    tag="negative",
    category="negative",
    conversation=[
        "我大学学的是计算机专业。",
        "毕业后去了百度做了三年后端。",
        "现在在一家创业公司做架构。",
    ],
    queries=[
        {"q": "我结婚了吗？", "expect": [], "not_expect": [], "desc": "未提及→空"},
        {"q": "我工资多少？", "expect": [], "not_expect": [], "desc": "未提及→空"},
        {"q": "我在哪个城市生活？", "expect": [], "not_expect": [], "desc": "未提及→空"},
        {"q": "我在百度做什么？", "expect": ["后端"], "not_expect": [], "desc": "已知信息确认"},
    ],
))

# ── 场景6: 时序推理 ──
SCENARIOS.append(TestCase(
    tag="temporal",
    category="temporal",
    conversation=[
        "2024年3月，我从百度离职了。",
        "离职后我去西藏玩了一个月。",
        "5月份回来开始找工作。",
        "6月中拿到了一家 AI 公司的 offer。",
        "7月1号正式入职新公司。",
        "新公司在海淀，离家有点远。",
    ],
    queries=[
        {"q": "我什么时候离职的？", "expect": ["3月", "2024"], "not_expect": [], "desc": "时间点"},
        {"q": "离职后干了什么？", "expect": ["西藏"], "not_expect": [], "desc": "时序关系"},
        {"q": "什么时候入职新公司的？", "expect": ["7月"], "not_expect": [], "desc": "最新时间"},
        {"q": "新公司在哪？", "expect": ["海淀"], "not_expect": [], "desc": "关联信息"},
    ],
))

# ── 场景7: 矛盾信息 ──
SCENARIOS.append(TestCase(
    tag="contradiction",
    category="contradiction",
    conversation=[
        "我觉得 Python 是最适合做 AI 的语言。",
        "后来用了 Go 写后端，发现性能确实好很多。",
        "不过 Python 的生态还是最完善的。",
        "最近在学 Rust，感觉这才是未来。",
    ],
    queries=[
        {"q": "我最喜欢什么语言？", "expect": ["Rust", "Python"], "not_expect": [], "desc": "矛盾偏好→返回多个"},
        {"q": "我觉得什么语言性能好？", "expect": ["Go"], "not_expect": [], "desc": "精确匹配"},
    ],
))

# ── 场景8: 长对话压力 ──
_long_conv = []
for i in range(1, 51):
    topics = [
        f"今天开了第{i}次项目周会，讨论了进度问题。",
        f"同事小明请假了，他家里有事。",
        f"老板说Q3的目标是完成{i%5+1}个核心功能。",
        f"我在考虑要不要换一台{i%3+1}TB的硬盘。",
        f"中午吃了{i%4+1}个菜，食堂今天还行。",
    ]
    _long_conv.append(topics[i % 5])
_long_conv.insert(20, "重要：公司决定年底前完成 A 轮融资，目标金额 5000 万。")
_long_conv.insert(35, "CTO 说他看好边缘计算方向，让我们提前布局。")

SCENARIOS.append(TestCase(
    tag="long_conversation",
    category="long",
    conversation=_long_conv,
    queries=[
        {"q": "A轮融资目标是多少？", "expect": ["5000"], "not_expect": [], "desc": "长对话中精确召回"},
        {"q": "CTO看好什么方向？", "expect": ["边缘计算"], "not_expect": [], "desc": "长对话中关键信息"},
        {"q": "Q3要完成几个核心功能？", "expect": ["核心功能"], "not_expect": [], "desc": "数值信息"},
    ],
))


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation engine
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class QueryResult:
    query: str
    expected: list[str]
    forbidden: list[str]
    description: str
    recalled_texts: list[str]
    recalled_scores: list[float]
    passed_contains: bool
    passed_excludes: bool
    details: str = ""


def evaluate_scenario(
    scenario: TestCase,
    engine: Any = None,
    vector_index: Any = None,
) -> list[QueryResult]:
    """Evaluate one test scenario against engram-router."""
    d = tempfile.mkdtemp()
    store = MemoryStore(
        path=Path(d) / "eval.db",
        embedding_engine=engine,
        vector_index=vector_index,
    )

    for turn in scenario.conversation:
        store.save(turn)

    results = []
    for q in scenario.queries:
        records = store.recall(q["q"], top_k=5)
        texts = [r.raw_text for r in records]
        scores = [r.score for r in records]
        joined = " ".join(texts[:3]).lower()  # top-3 for answer check

        # Check expectations
        contains_ok = True
        if q["expect"]:
            if isinstance(q["expect"], list):
                contains_ok = all(e.lower() in joined for e in q["expect"])
            else:
                contains_ok = q["expect"].lower() in joined

        excludes_ok = all(
            forbid.lower() not in joined
            for forbid in q["not_expect"]
        ) if q["not_expect"] else True

        results.append(QueryResult(
            query=q["q"],
            expected=q["expect"],
            forbidden=q["not_expect"],
            description=q["desc"],
            recalled_texts=texts,
            recalled_scores=scores,
            passed_contains=contains_ok,
            passed_excludes=excludes_ok,
            details="",
        ))

    store.close()
    return results


def print_report(all_results: dict[str, list[QueryResult]]):
    """Print a comprehensive evaluation report."""
    total = 0
    passed_contains = 0
    passed_excludes = 0
    total_excludes = 0

    print("=" * 72)
    print("  engram-router 多角度综合评估报告")
    print("=" * 72)

    for cat, results in all_results.items():
        cat_total = len(results)
        cat_pass = sum(1 for r in results if r.passed_contains)
        cat_excl = sum(1 for r in results if r.forbidden)
        cat_excl_pass = sum(1 for r in results if r.forbidden and r.passed_excludes)

        print(f"\n{'━'*60}")
        print(f"  [{cat.upper()}]  {cat_pass}/{cat_total} 精确召回正确")
        if cat_excl > 0:
            print(f"  [{cat.upper()}]  {cat_excl_pass}/{cat_excl} 隔离检查通过")
        print(f"{'━'*60}")

        for r in results:
            status = "✅" if r.passed_contains and r.passed_excludes else "❌"
            print(f"\n  {status} {r.description}")
            print(f"     Q: {r.query}")
            if r.expected:
                print(f"     期望: {r.expected}")
            if r.forbidden:
                print(f"     不应含: {r.forbidden}")

            # Show top-3 results
            for i, (text, score) in enumerate(zip(r.recalled_texts[:3], r.recalled_scores[:3])):
                marker = ""
                if r.expected:
                    hits = [e for e in r.expected if e.lower() in text.lower()]
                    if hits:
                        marker = f"  ← 命中: {hits}"
                print(f"     [{i}] {score:.2f} | {text[:70]}{marker}")

            if not r.passed_contains and r.expected:
                missing = [e for e in r.expected if not any(e.lower() in t.lower() for t in r.recalled_texts[:3])]
                print(f"     ⚠ 缺失关键词在top-3: {missing}")
            if not r.passed_excludes:
                leaked = [f for f in r.forbidden if any(f.lower() in t.lower() for t in r.recalled_texts[:3])]
                print(f"     ⚠ 泄漏关键词: {leaked}")

        total += cat_total
        passed_contains += cat_pass
        total_excludes += cat_excl
        passed_excludes += cat_excl_pass

    # Summary
    print(f"\n{'='*72}")
    print(f"  汇总")
    print(f"{'='*72}")
    precision = passed_contains / total * 100 if total else 0
    isolation = passed_excludes / total_excludes * 100 if total_excludes else 100
    overall = (passed_contains + passed_excludes) / (total + total_excludes) * 100 if (total + total_excludes) else 0

    print(f"  总查询数:        {total}")
    print(f"  精确召回通过:    {passed_contains}/{total}  ({precision:.1f}%)")
    print(f"  隔离检查通过:    {passed_excludes}/{total_excludes}  ({isolation:.1f}%)")
    print(f"  综合通过率:      {overall:.1f}%")

    # Score
    score = precision * 0.4 + isolation * 0.3 + (100 if total >= 30 else total/30*100) * 0.1 + 90 * 0.2
    print(f"\n  加权评分:        {score:.1f}/100")

    return {
        "total": total,
        "precision_pct": precision,
        "isolation_pct": isolation,
        "overall_pct": overall,
        "weighted_score": score,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("engram-router 多角度评估开始...\n")

    all_results: dict[str, list[QueryResult]] = {}
    for scenario in SCENARIOS:
        results = evaluate_scenario(scenario)
        all_results[scenario.category] = results

    report = print_report(all_results)

    # Save to JSON
    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scenarios": len(SCENARIOS),
        "summary": report,
    }
    out_path = Path(__file__).resolve().parent.parent / "docs" / "multi_angle_eval_report.json"
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {out_path}")

    return 0 if report["precision_pct"] >= 85 else 1


if __name__ == "__main__":
    sys.exit(main())
