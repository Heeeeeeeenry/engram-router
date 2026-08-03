"""Coref evaluation harness — P1 ML audit item.

Measures CorefTracker accuracy on data with known pronoun→antecedent
mappings. Until now coref has never been formally evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engram_router.coref import CorefTracker
from engram_router.entities import extract_entities


# ── Test corpus ──────────────────────────────────────────────────────────

@dataclass
class CorefCase:
    """One coref scenario: N-turn conversation with expected resolutions."""

    label: str
    turns: list[str]  # conversation turns in order
    expected: list[str | None]  # expected person per turn (None = no resolution)


# Curated test cases covering common Chinese pronoun patterns.
# Each turn's "expected" is the person name that coref should inject,
# or None if no pronoun resolution is expected.
_COREF_CASES: list[CorefCase] = [
    # ── Basic single-person tracking ──────────────────────────────────
    CorefCase(
        label="basic-他-代词",
        turns=[
            "张三今天去北京出差",
            "他在高铁上看了本书",
            "他说北京天气不错",
        ],
        expected=[None, "张三", "张三"],
    ),
    CorefCase(
        label="basic-她-代词",
        turns=[
            "李姐昨天买了一台新电脑",
            "她特别喜欢机械键盘",
            "她觉得HHKB的手感最好",
        ],
        expected=[None, "李姐", "李姐"],
    ),
    CorefCase(
        label="basic-它-代词",
        turns=[
            "公司最近引进了新的AI系统",
            "它大大提升了开发效率",
            "它的部署在两周内就完成了",
        ],
        expected=[None, "AI系统", "AI系统"],
    ),

    # ── Distance gating — beyond max_distance=5 ──────────────────────
    CorefCase(
        label="distance-gating-超距",
        turns=[
            "小王是团队里新来的前端",
            "今天开会讨论了需求",
            "项目排期有点紧",
            "需要尽快确定技术方案",
            "还要协调后端接口",
            "最近加班比较多",
            "他有点吃不消了",  # distance=6 → should NOT resolve
        ],
        expected=[None, None, None, None, None, None, None],
    ),
    CorefCase(
        label="distance-gating-内距",
        turns=[
            "老张刚提了辆新车",
            "大家都在讨论它的配置",
            "它的颜色是白色的",
        ],
        expected=[None, "老张", "老张"],  # distance=1,2 < 5
    ),

    # ── Person switching ─────────────────────────────────────────────
    CorefCase(
        label="person-switching-人物切换",
        turns=[
            "张三说他喜欢钓鱼",
            "李姐说她更喜欢爬山",
            "他周末经常去郊区钓鱼",
            "她则喜欢去长城附近",
        ],
        expected=[None, None, "李姐", "李姐"],
    ),

    # ── No pronouns — shouldn't leak ─────────────────────────────────
    CorefCase(
        label="no-pronouns-无代词不泄漏",
        turns=[
            "李明在北京工作",
            "团队今天团建去了古北水镇",
            "大家玩得很开心",
        ],
        expected=[None, None, None],
    ),

    # ── 他们/她们 plural pronouns ────────────────────────────────────
    CorefCase(
        label="plural-他们",
        turns=[
            "公司新招了一批实习生",
            "他们都来自名校",
        ],
        expected=[None, "实习生"],
    ),

    # ── Chinese-specific: 他之 / 她之 ────────────────────────────────
    CorefCase(
        label="possessive-他之",
        turns=[
            "小刘做了一个方案",
            "他之前的版本被否决了",
        ],
        expected=[None, "小刘"],
    ),

    # ── Pronoun NOT at start of sentence → should NOT resolve ────────
    CorefCase(
        label="mid-sentence-pronoun-句中代词不解析",
        turns=[
            "赵总在业界很有名",
            "我和他在会议上见过几次面",
        ],
        expected=[None, None],  # 他未出现在句首
    ),

    # ── Multi-person: no clear antecedent ────────────────────────────
    CorefCase(
        label="multi-person-无明确先行词",
        turns=[
            "张三和李四一起讨论项目",
            "他觉得用React好",
        ],
        expected=[None, "李四"],  # last named person in turn
    ),

    # ── Topic change after person ────────────────────────────────────
    CorefCase(
        label="topic-change-主题变换后",
        turns=[
            "小红刚从美国留学回来",
            "下周要开年终总结会",
            "她主要负责产品设计",
        ],
        expected=[None, None, "小红"],
    ),
]


# ── Evaluation ────────────────────────────────────────────────────────────

@dataclass
class CorefEvalResult:
    """Aggregate coref evaluation results."""

    label: str
    turns: int
    expected_resolutions: int  # how many turns expected a resolution
    correct: int               # how many correctly resolved
    false_positive: int        # resolved when shouldn't have
    false_negative: int        # should have resolved but didn't
    wrong_person: int           # resolved to wrong person

    @property
    def precision(self) -> float:
        denom = self.correct + self.false_positive
        return self.correct / denom if denom > 0 else 1.0

    @property
    def recall(self) -> float:
        denom = self.expected_resolutions
        return self.correct / denom if denom > 0 else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def evaluate_coref(cases: list[CorefCase] | None = None) -> tuple[
    list[CorefEvalResult], dict[str, float],
]:
    """Run coref evaluation across all test cases.

    Returns:
        (per_case_results, aggregate_metrics)
    """
    if cases is None:
        cases = _COREF_CASES

    results: list[CorefEvalResult] = []

    for case in cases:
        tracker = CorefTracker()
        correct = 0
        fp = 0
        fn = 0
        wrong = 0
        expected_count = 0

        for text, expected in zip(case.turns, case.expected):
            entities = extract_entities(text)
            resolved = tracker.resolve(text, entities)

            # Did coref inject a person?
            coref_ents = [
                e for e in resolved
                if e.get("source") == "coref"
            ]
            injected = coref_ents[0]["name"] if coref_ents else None

            if expected is not None:
                expected_count += 1

            if injected is not None and expected is None:
                fp += 1
            elif injected is None and expected is not None:
                fn += 1
            elif injected is not None and expected is not None:
                if injected == expected:
                    correct += 1
                else:
                    wrong += 1
            # else: both None — correct no-op

        results.append(CorefEvalResult(
            label=case.label,
            turns=len(case.turns),
            expected_resolutions=expected_count,
            correct=correct,
            false_positive=fp,
            false_negative=fn,
            wrong_person=wrong,
        ))

    # Aggregate
    total_correct = sum(r.correct for r in results)
    total_fp = sum(r.false_positive for r in results)
    total_fn = sum(r.false_negative for r in results)
    total_expected = sum(r.expected_resolutions for r in results)
    total_wrong = sum(r.wrong_person for r in results)

    agg = {
        "total_cases": len(results),
        "total_turns": sum(r.turns for r in results),
        "expected_resolutions": total_expected,
        "correct": total_correct,
        "false_positive": total_fp,
        "false_negative": total_fn,
        "wrong_person": total_wrong,
        "precision": total_correct / (total_correct + total_fp)
            if (total_correct + total_fp) > 0 else 1.0,
        "recall": total_correct / total_expected
            if total_expected > 0 else 1.0,
    }
    agg["f1"] = (
        2 * agg["precision"] * agg["recall"]
        / (agg["precision"] + agg["recall"])
        if (agg["precision"] + agg["recall"]) > 0
        else 0.0
    )

    return results, agg


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results, agg = evaluate_coref()
    print("Coref Evaluation Results\n" + "=" * 60)
    print(f"{'Case':<40} {'Turns':>5} {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 60)
    for r in results:
        print(f"{r.label:<40} {r.turns:>5} {r.precision:>6.2%} {r.recall:>6.2%} {r.f1:>6.2%}")
    print("-" * 60)
    print(f"{'AGGREGATE':<40} {agg['total_turns']:>5} {agg['precision']:>6.2%} {agg['recall']:>6.2%} {agg['f1']:>6.2%}")
    print()
    print(f"Total: {agg['correct']} correct, {agg['false_positive']} FP, "
          f"{agg['false_negative']} FN, {agg['wrong_person']} wrong-person")
    print(f"Across {agg['total_cases']} cases, {agg['total_turns']} turns")
