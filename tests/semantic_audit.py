#!/usr/bin/env python3
"""
engram-router 语义联想能力审计 — 10个关键场景实测

测试维度（不在现有测试中）:
  S1  同义词: "妈妈多大了" vs "妈妈今年55岁"
  S2  近义词: "HHKB好用吗" vs "我最喜欢机械键盘"
  S3  跨实体: "张三同事的老婆" vs "李丽，张三是前同事" "李丽是家庭主妇"
  S4  否定查询: "李四买特斯拉了吗" vs "张三买了特斯拉"
  S5  时间推理: "去年发生了什么" vs "2025年妈妈退休"
  S6  隐含推理: "谁送我键盘的原因" vs "张三送HHKB，因为是生日"
  S7  语义相似: "我最近胖了" vs "我体重增加了5公斤"
  S8  情感推理: "我为什么心情不好" vs "今天被老板批评了"
  S9  跨话题关联: "我之前说的那个计划" vs "我想明年去日本旅游"
  S10 人称指代: "他送我的东西好用吗" vs "张三送了我一把HHKB键盘"
"""

import os, sys, tempfile, json, time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from engram_router.store import MemoryStore


# ═══════════════════════════════════════════════════════════════════════════
# 10 KEY SEMANTIC SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    id: str
    name: str
    description: str
    store_memories: list[str]       # what we save into the store
    query: str                       # what the user asks
    expected_memory_idx: list[int]   # which store_memories (0-indexed) should be in top-3
    forbidden_memory_idx: list[int]  # which store_memories must NOT be in top-3
    note: str = ""


SCENARIOS = []

# ── S1: 同义词 - "妈妈多大了" vs "妈妈今年55岁" ──
SCENARIOS.append(Scenario(
    id="S1",
    name="同义词联想",
    description="查询用'多大了'，存储用'55岁'，需语义联想年龄",
    store_memories=[
        "妈妈今年55岁，已经退休了。",              # idx 0 ★target
        "爸爸今年58岁，还在上班。",                # idx 1 distractor
        "我今年30岁，在一家互联网公司工作。",       # idx 2 distractor
    ],
    query="妈妈多大了",
    expected_memory_idx=[0],
    forbidden_memory_idx=[],
    note="查询用口语化'多大了'，期望召回含'55岁'的记忆"
))

# ── S2: 近义词 - "HHKB好用吗" vs "我最喜欢机械键盘" ──
SCENARIOS.append(Scenario(
    id="S2",
    name="近义词联想",
    description="查询用'HHKB好用吗'，存储用'机械键盘'，HHKB 应通过别名映射到键盘",
    store_memories=[
        "我最喜欢机械键盘，打字手感特别好。",        # idx 0 ★target
        "鼠标我用的是罗技的无线鼠标。",              # idx 1 distractor
        "显示器是戴尔的27寸4K屏。",                  # idx 2 distractor
    ],
    query="HHKB好用吗",
    expected_memory_idx=[0],
    forbidden_memory_idx=[],
    note="HHKB 在 object_topic_aliases 中映射到'键盘'，'键盘'映射到机械键盘"
))

# ── S3: 跨实体 - "张三同事的老婆" vs "李丽，张三是前同事" "李丽是家庭主妇" ──
SCENARIOS.append(Scenario(
    id="S3",
    name="跨实体推理",
    description="需要跨两条记忆推理：张三是前同事 → 张三的老婆是李丽 → 李丽是家庭主妇",
    store_memories=[
        "李丽是我的大学同学，张三是我的前同事。",     # idx 0 link
        "李丽是家庭主妇，没有上班。",                # idx 1 ★target
        "王芳在北京协和医院工作。",                  # idx 2 distractor
    ],
    query="张三同事的老婆",
    expected_memory_idx=[0, 1],  # at minimum, idx 1 should be visible
    forbidden_memory_idx=[],
    note="跨实体多跳：张三→同事→老婆=李丽→家庭主妇"
))

# ── S4: 否定查询 - "李四买特斯拉了吗" vs "张三买了特斯拉" ──
SCENARIOS.append(Scenario(
    id="S4",
    name="否定查询/跨人物",
    description="问李四买没买特斯拉，存的是张三买的，应正确区分人物",
    store_memories=[
        "张三上个月买了一辆特斯拉。",                 # idx 0 distractor
        "李四最近在考虑买电动车。",                  # idx 1 relevant
        "特斯拉最近降价了。",                         # idx 2 distractor
    ],
    query="李四买特斯拉了吗",
    expected_memory_idx=[1],     # the relevant one about 李四
    forbidden_memory_idx=[],     # 0 should NOT be the top hit -- about 张三
    note="人物区分：李四≠张三，应优先召回李四相关"
))

# ── S5: 时间推理 - "去年发生了什么" vs "2025年妈妈退休" ──
SCENARIOS.append(Scenario(
    id="S5",
    name="时间推理",
    description="查询用相对时间'去年'，存储用绝对时间'2025年'",
    store_memories=[
        "2025年妈妈退休了，回家种花。",              # idx 0 ★target
        "2024年爸爸换了个新工作。",                 # idx 1 distractor
        "我2023年大学毕业。",                        # idx 2 distractor
    ],
    query="去年发生了什么",
    expected_memory_idx=[0],
    forbidden_memory_idx=[],
    note="相对时间→绝对时间映射：'去年'无法直接匹配'2025年'，但'退休'可能通过其他词匹配"
))

# ── S6: 隐含推理 - "谁送我键盘的原因" vs "张三送HHKB，因为是生日" ──
SCENARIOS.append(Scenario(
    id="S6",
    name="隐含推理（因果）",
    description="查询问'原因'，存储含因果标记'因为是生日'",
    store_memories=[
        "张三送我一把HHKB键盘，因为是生日礼物。",     # idx 0 ★target (含因果)
        "李四也送了我一本书，挺好看的。",              # idx 1 distractor
        "我每年生日都会给自己买个礼物。",             # idx 2 distractor
    ],
    query="谁送我键盘的原因",
    expected_memory_idx=[0],
    forbidden_memory_idx=[],
    note="因果推理：'因为是生日'是原因标记，应被正确召回"
))

# ── S7: 语义相似 - "我最近胖了" vs "我体重增加了5公斤" ──
SCENARIOS.append(Scenario(
    id="S7",
    name="语义相似（体重）",
    description="'胖了' vs '体重增加了5公斤'，语义相关但无共享词",
    store_memories=[
        "我这个月体重增加了5公斤，得减肥了。",         # idx 0 ★target
        "我最近开始跑步，每天跑5公里。",              # idx 1 distractor
        "我换了个新手机，拍照效果不错。",              # idx 2 distractor
    ],
    query="我最近胖了",
    expected_memory_idx=[0],
    forbidden_memory_idx=[],
    note="'胖'和'体重增加'是语义相关，但用词完全不同"
))

# ── S8: 情感推理 - "我为什么心情不好" vs "今天被老板批评了" ──
SCENARIOS.append(Scenario(
    id="S8",
    name="情感推理（情绪原因）",
    description="查询问心情不好的原因，存储是引发负面情绪的事件",
    store_memories=[
        "今天在会议室被老板批评了。",                 # idx 0 ★target
        "中午吃了红烧肉，味道还行。",                 # idx 1 distractor
        "昨天晚上睡得很好。",                         # idx 2 distractor
    ],
    query="我为什么心情不好",
    expected_memory_idx=[0],
    forbidden_memory_idx=[],
    note="'心情不好'和'被批评'是因果关系，但无表面词匹配"
))

# ── S9: 跨话题关联 - "我之前说的那个计划" vs "我想明年去日本旅游" ──
SCENARIOS.append(Scenario(
    id="S9",
    name="跨话题模糊指代",
    description="'计划' vs '想明年去日本旅游'，无重叠词",
    store_memories=[
        "我想明年去日本旅游，看看樱花。",              # idx 0 ★target
        "今天的晚饭还不错。",                          # idx 1 distractor
        "Python 3.12 的新特性挺多的。",                # idx 2 distractor
    ],
    query="我之前说的那个计划",
    expected_memory_idx=[0],
    forbidden_memory_idx=[],
    note="'计划'和'想去...旅游'是语义关联，但无直接词重叠"
))

# ── S10: 人称指代 - "他送我的东西好用吗" vs "张三送了我一把HHKB键盘" ──
SCENARIOS.append(Scenario(
    id="S10",
    name="人称指代",
    description="查询用'他'指代，存储用'张三'，需理解指代关系",
    store_memories=[
        "张三送了我一把HHKB键盘。",                  # idx 0 ★target
        "李四说这个键盘不好用。",                     # idx 1 distractor
        "我自己买了一个鼠标。",                        # idx 2 distractor
    ],
    query="他送我的东西好用吗",
    expected_memory_idx=[0],
    forbidden_memory_idx=[],
    note="人称指代：'他'需关联到送礼人，'东西'需关联到键盘"
))


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScenarioResult:
    scenario: Scenario
    recall_results: list       # list of MemoryRecord
    top3_texts: list[str]
    top3_scores: list[float]
    top3_ids: list[str]
    expected_hit_in_top1: bool  # the best expected memory is rank 0
    expected_hit_in_top3: bool  # all expected memories are in top 3
    forbidden_in_top3: list[str]  # forbidden texts that leaked
    passed: bool
    analysis: str


def evaluate_one(store: MemoryStore, scenario: Scenario) -> ScenarioResult:
    """Run one scenario and return detailed results."""
    records = store.recall(scenario.query, top_k=5)
    
    top3_texts = [r.raw_text for r in records[:3]]
    top3_scores = [r.score for r in records[:3]]
    top3_ids = [r.id for r in records[:3]]
    all_texts = [r.raw_text for r in records]
    
    # Check expected
    expected_texts = [scenario.store_memories[i] for i in scenario.expected_memory_idx]
    forbidden_texts = [scenario.store_memories[i] for i in scenario.forbidden_memory_idx]
    
    # Is the FIRST expected memory in position 0?
    expected_hit_top1 = False
    if expected_texts:
        expected_hit_top1 = (expected_texts[0] in top3_texts[:1])
    
    # Are ALL expected memories in top 3?
    expected_hit_top3 = all(
        any(et in t for t in top3_texts) or any(t in et for t in top3_texts)
        for et in expected_texts
    )
    
    # Are any forbidden memories in top 3?
    forbidden_leaked = [
        ft for ft in forbidden_texts
        if any(ft in t for t in top3_texts)
    ]
    
    # Overall pass: expected in top 3 AND no forbidden in top 3
    passed = expected_hit_top3 and len(forbidden_leaked) == 0
    
    # Analysis
    analysis_parts = []
    if expected_hit_top1:
        analysis_parts.append("✅ 最佳匹配在第1位")
    elif expected_hit_top3:
        ranking = next((i for i, t in enumerate(top3_texts) 
                       if expected_texts[0] in t or t in expected_texts[0]), -1)
        analysis_parts.append(f"⚠ 最佳匹配在第{ranking+1}位（非首位）")
    else:
        analysis_parts.append("❌ 期望记忆未进入top-3")
    
    if forbidden_leaked:
        analysis_parts.append(f"❌ 泄漏: {forbidden_leaked}")
    
    return ScenarioResult(
        scenario=scenario,
        recall_results=records,
        top3_texts=top3_texts,
        top3_scores=top3_scores,
        top3_ids=top3_ids,
        expected_hit_in_top1=expected_hit_top1,
        expected_hit_in_top3=expected_hit_top3,
        forbidden_in_top3=forbidden_leaked,
        passed=passed,
        analysis=" | ".join(analysis_parts),
    )


def print_detailed_report(results: list[ScenarioResult]):
    """Print detailed per-scenario results."""
    print("=" * 80)
    print("  engram-router 语义联想能力审计报告 — 10个关键场景")
    print("=" * 80)
    
    passed_count = 0
    top1_count = 0
    
    for r in results:
        s = r.scenario
        status = "✅ 通过" if r.passed else "❌ 失败"
        print(f"\n{'─' * 70}")
        print(f"  [{s.id}] {s.name}  {status}")
        print(f"  描述: {s.description}")
        print(f"  备注: {s.note}")
        
        print(f"\n  📝 存储的记忆:")
        for i, mem in enumerate(s.store_memories):
            marker = " ← ★目标" if i in s.expected_memory_idx else ""
            print(f"    [{i}] {mem}{marker}")
        
        print(f"\n  🔍 查询: '{s.query}'")
        
        print(f"\n  📊 召回结果 (top-3):")
        for i, (text, score, mid) in enumerate(zip(r.top3_texts, r.top3_scores, r.top3_ids)):
            is_expected = any(
                s.store_memories[ei] in text or text in s.store_memories[ei]
                for ei in s.expected_memory_idx
            )
            icon = "⭐" if is_expected else "  "
            print(f"    [{i}] {icon} {score:.2f} | {mid} | {text[:80]}")
        
        # Show full top-5 if there are interesting results
        if len(r.recall_results) > 3:
            print(f"\n  📊 完整 top-5:")
            for i, rec in enumerate(r.recall_results):
                is_expected = any(
                    s.store_memories[ei] in rec.raw_text or rec.raw_text in s.store_memories[ei]
                    for ei in s.expected_memory_idx
                )
                icon = "⭐" if is_expected else "  "
                print(f"    [{i}] {icon} {rec.score:.4f} | {rec.id} | {rec.raw_text[:80]}")
        
        print(f"\n  📋 分析: {r.analysis}")
        
        if r.passed:
            passed_count += 1
        if r.expected_hit_in_top1:
            top1_count += 1
    
    # Summary
    print(f"\n{'=' * 80}")
    print(f"  汇总")
    print(f"{'=' * 80}")
    print(f"  总场景数:          {len(results)}")
    print(f"  通过数:            {passed_count}/{len(results)}  ({passed_count/len(results)*100:.0f}%)")
    print(f"  Top-1 命中:        {top1_count}/{len(results)}  ({top1_count/len(results)*100:.0f}%)")
    
    # Per-category analysis
    print(f"\n  详细分类:")
    for r in results:
        status = "✅" if r.passed else "❌"
        top1 = "🏆" if r.expected_hit_in_top1 else "  "
        print(f"    {status} {top1} [{r.scenario.id}] {r.scenario.name}: {r.analysis}")
    
    return passed_count


def main():
    print("engram-router 语义联想审计开始...")
    
    d = tempfile.mkdtemp()
    db_path = Path(d) / "semantic_audit.db"
    
    results: list[ScenarioResult] = []
    
    for scenario in SCENARIOS:
        # Fresh store per scenario to avoid cross-contamination
        store = MemoryStore(path=db_path)
        
        # Clear any previous data by using a fresh connection
        # Actually we need a separate DB per scenario
        store.close()
        
        scenario_db = Path(d) / f"{scenario.id}.db"
        store = MemoryStore(path=scenario_db)
        
        # Save memories
        for mem in scenario.store_memories:
            store.save(mem)
        
        # Run recall
        result = evaluate_one(store, scenario)
        results.append(result)
        
        store.close()
    
    passed = print_detailed_report(results)
    
    # Save JSON report
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_scenarios": len(results),
        "passed": passed,
        "pass_rate": f"{passed/len(results)*100:.0f}%",
        "scenarios": [
            {
                "id": r.scenario.id,
                "name": r.scenario.name,
                "query": r.scenario.query,
                "passed": r.passed,
                "top1_hit": r.expected_hit_in_top1,
                "top3_hit": r.expected_hit_in_top3,
                "top3_scores": r.top3_scores,
                "top3_texts": r.top3_texts,
                "analysis": r.analysis,
            }
            for r in results
        ],
    }
    
    out_path = Path(__file__).resolve().parent.parent / "docs" / "semantic_audit_report.json"
    with open(out_path, "w") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {out_path}")
    
    return 0 if passed >= 4 else 1  # expect at least 40% pass rate


if __name__ == "__main__":
    sys.exit(main())
