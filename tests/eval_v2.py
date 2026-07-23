#!/usr/bin/env python3
"""Strict evaluator for engram-router — v2.

Replaces the "contains-anywhere-in-top-3" ruler used by
``tests/semantic_audit.py`` and ``tests/multi_angle_eval.py`` with rank-aware
metrics that expose whether the target memory is actually the top hit rather
than just present somewhere in the shortlist.

Metrics (each a standalone function ``metric(records, case) -> float``):
    1. Precision@1
    2. Precision@k
    3. Reciprocal Rank / Mean Reciprocal Rank
    4. nDCG@k (relevance grades 0/1/2)
    5. Recall@k
    6. Contamination@k (fraction of top-k that are forbidden)
    7. Rejection Accuracy (for is_negative cases)

Run:
    ENGRAM_SKIP_VECTOR=1 python tests/eval_v2.py

Exit code:
    0  if aggregate P@1 >= 0.8
    1  otherwise (or on hard failure)
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

# Force the fast, deterministic recall path before importing the store.
os.environ.setdefault("ENGRAM_SKIP_VECTOR", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from engram_router.store import MemoryRecord, MemoryStore  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class EvalCase:
    """A single query judgement.

    ``expected_ids_or_texts`` / ``forbidden_ids_or_texts`` are matched as
    substrings against ``MemoryRecord.raw_text`` (or exact id if the entry
    looks like a memory id).  For ``expected`` the semantics are "memory that
    contains this substring is a target"; for ``forbidden`` it is "memory
    that contains this substring must NOT appear in top-k".

    ``relevance_grades`` may map a specific memory text (or id) to an int
    grade in {0, 1, 2}. When absent, grade is derived from expected/forbidden:
    a memory containing every expected substring is grade 2, one containing
    at least one is grade 1, one containing a forbidden substring is 0.
    """

    query: str
    expected_ids_or_texts: list[str] = field(default_factory=list)
    forbidden_ids_or_texts: list[str] = field(default_factory=list)
    is_negative: bool = False
    relevance_grades: dict[str, int] = field(default_factory=dict)
    # Bookkeeping only — never affects scoring:
    id: str = ""
    description: str = ""
    old_passed: bool | None = None  # what the legacy reports said


@dataclass
class EvalScenario:
    id: str
    source: str            # "semantic_audit" | "multi_angle"
    memories: list[str]    # ingested via store.save() in this order
    cases: list[EvalCase]


# ─────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────

def _text_of(record: MemoryRecord) -> str:
    return record.raw_text or ""


def _matches(record: MemoryRecord, needle: str) -> bool:
    """A record matches a needle if the needle is a substring of its raw text,
    the raw text is a substring of the needle (full-memory match), or the id
    equals the needle."""
    if not needle:
        return False
    txt = _text_of(record)
    if needle == record.id:
        return True
    return needle in txt or txt in needle


def _rank_of_first_target(
    records: list[MemoryRecord], targets: Iterable[str]
) -> int:
    """1-indexed rank of the first record matching any target, or 0 if none."""
    targets = list(targets)
    for i, r in enumerate(records, start=1):
        if any(_matches(r, t) for t in targets):
            return i
    return 0


def _grade(record: MemoryRecord, case: EvalCase) -> int:
    """Relevance grade in {0, 1, 2} for a single record under a case."""
    txt = _text_of(record)
    # Explicit override wins.
    if case.relevance_grades:
        if record.id in case.relevance_grades:
            return int(case.relevance_grades[record.id])
        for key, grade in case.relevance_grades.items():
            if key and (key in txt or txt in key):
                return int(grade)
    # Forbidden always zero.
    if any(_matches(record, f) for f in case.forbidden_ids_or_texts):
        return 0
    if not case.expected_ids_or_texts:
        return 0
    hits = [t for t in case.expected_ids_or_texts if _matches(record, t)]
    if len(hits) == len(case.expected_ids_or_texts):
        return 2
    if hits:
        return 1
    return 0


# ─────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────

def precision_at_1(records: list[MemoryRecord], case: EvalCase) -> float:
    if not records or not case.expected_ids_or_texts:
        return 0.0
    top = records[0]
    return 1.0 if any(_matches(top, t) for t in case.expected_ids_or_texts) else 0.0


def precision_at_k(records: list[MemoryRecord], case: EvalCase, k: int = 3) -> float:
    if not case.expected_ids_or_texts:
        return 0.0
    for r in records[:k]:
        if any(_matches(r, t) for t in case.expected_ids_or_texts):
            return 1.0
    return 0.0


def reciprocal_rank(records: list[MemoryRecord], case: EvalCase) -> float:
    if not case.expected_ids_or_texts:
        return 0.0
    rank = _rank_of_first_target(records, case.expected_ids_or_texts)
    return 1.0 / rank if rank > 0 else 0.0


def ndcg_at_k(records: list[MemoryRecord], case: EvalCase, k: int = 5) -> float:
    if not case.expected_ids_or_texts:
        return 0.0
    gains = [_grade(r, case) for r in records[:k]]

    def _dcg(gs: list[int]) -> float:
        total = 0.0
        for i, g in enumerate(gs, start=1):
            if g <= 0:
                continue
            # Standard DCG with log2(i+1) discount.
            total += (2**g - 1) / math.log2(i + 1)
        return total

    dcg = _dcg(gains)
    # Ideal ordering: all grade-2 memories first, then grade-1, up to k slots.
    n_grade2 = len(case.expected_ids_or_texts)
    ideal = sorted(gains, reverse=True)
    # Ensure the ideal has at least the expected count of grade-2 targets.
    if ideal.count(2) < n_grade2:
        ideal = [2] * n_grade2 + [g for g in ideal if g < 2]
        ideal = ideal[:k]
    idcg = _dcg(ideal)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def recall_at_k(records: list[MemoryRecord], case: EvalCase, k: int = 5) -> float:
    if not case.expected_ids_or_texts:
        return 0.0
    top = records[:k]
    hit = 0
    for target in case.expected_ids_or_texts:
        if any(_matches(r, target) for r in top):
            hit += 1
    return hit / len(case.expected_ids_or_texts)


def contamination_at_k(
    records: list[MemoryRecord], case: EvalCase, k: int = 3
) -> float:
    if not case.forbidden_ids_or_texts:
        return 0.0
    top = records[:k]
    if not top:
        return 0.0
    bad = sum(
        1 for r in top
        if any(_matches(r, f) for f in case.forbidden_ids_or_texts)
    )
    return bad / len(top)


def rejection_correct(
    records: list[MemoryRecord], case: EvalCase, threshold: float = 1.0
) -> float:
    """1.0 if the negative case is correctly rejected (empty result or top-1
    score below threshold), else 0.0. Meaningful only when case.is_negative.
    """
    if not case.is_negative:
        return 0.0
    if not records:
        return 1.0
    return 1.0 if records[0].score < threshold else 0.0


# ─────────────────────────────────────────────────────────────────────────
# Scenario loading — replay the fixtures from the two legacy tests without
# importing them by name (they have no __init__.py and side-effects in main).
# ─────────────────────────────────────────────────────────────────────────

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _old_pass_status_semantic_audit() -> dict[str, bool]:
    """Read the legacy per-scenario pass verdict from
    docs/semantic_audit_report.json so we can compute regressions."""
    p = ROOT / "docs" / "semantic_audit_report.json"
    if not p.exists():
        return {}
    with p.open() as f:
        data = json.load(f)
    return {s["id"]: bool(s.get("passed", False)) for s in data.get("scenarios", [])}


def _old_pass_status_multi_angle(mod) -> dict[str, bool]:
    """Reproduce the legacy `contains in top-3 joined` rule per query, so we
    can flag regressions even though the old report only kept aggregates."""
    out: dict[str, bool] = {}
    for scenario in mod.SCENARIOS:
        for i, q in enumerate(scenario.queries):
            case_id = f"MA-{scenario.tag}-q{i}"
            expects = q.get("expect") or []
            forbids = q.get("not_expect") or []
            # In the legacy code, empty `expect` always yields contains_ok=True.
            # So a negative case with empty expect was "passed" by definition.
            # We record that verdict here to expose it as a regression later.
            out[case_id] = True if not expects else None  # type: ignore[assignment]
            # Note: the real recall-dependent verdict is filled in after we run.
    return out


def load_scenarios() -> list[EvalScenario]:
    tests_dir = ROOT / "tests"
    sa_mod = _load_module("_legacy_semantic_audit", tests_dir / "semantic_audit.py")
    ma_mod = _load_module("_legacy_multi_angle", tests_dir / "multi_angle_eval.py")

    old_sa = _old_pass_status_semantic_audit()

    scenarios: list[EvalScenario] = []

    # ── semantic_audit: 1 case per scenario, 3-memory store, exact target
    for s in sa_mod.SCENARIOS:
        expected = [s.store_memories[i] for i in s.expected_memory_idx]
        forbidden = [s.store_memories[i] for i in s.forbidden_memory_idx]
        # Grade 2 for each expected memory, everything else 0 (implicit).
        grades = {mem: 2 for mem in expected}
        case = EvalCase(
            id=f"SA-{s.id}",
            query=s.query,
            expected_ids_or_texts=expected,
            forbidden_ids_or_texts=forbidden,
            is_negative=False,
            relevance_grades=grades,
            description=f"[semantic_audit] {s.name}",
            old_passed=old_sa.get(s.id),
        )
        scenarios.append(EvalScenario(
            id=f"SA-{s.id}",
            source="semantic_audit",
            memories=list(s.store_memories),
            cases=[case],
        ))

    # ── multi_angle: N cases per scenario, sharing one conversation store
    for s in ma_mod.SCENARIOS:
        cases: list[EvalCase] = []
        for i, q in enumerate(s.queries):
            expects: list[str] = list(q.get("expect") or [])
            forbids: list[str] = list(q.get("not_expect") or [])
            is_neg = (s.category == "negative") and not expects
            # The legacy contains-rule for empty expect trivially passes,
            # so we mark it as old_passed=True for negatives to surface any
            # cases where the model actually returned a high-confidence hit.
            old_passed_legacy: bool | None = None
            if not expects:
                old_passed_legacy = True

            cases.append(EvalCase(
                id=f"MA-{s.tag}-q{i}",
                query=q["q"],
                expected_ids_or_texts=expects,
                forbidden_ids_or_texts=forbids,
                is_negative=is_neg,
                description=f"[{s.category}] {q.get('desc', '')}",
                old_passed=old_passed_legacy,
            ))
        scenarios.append(EvalScenario(
            id=f"MA-{s.tag}",
            source="multi_angle",
            memories=list(s.conversation),
            cases=cases,
        ))

    return scenarios


# ─────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────

def default_judge(case: EvalCase, store: MemoryStore) -> list[MemoryRecord]:
    return store.recall(case.query, top_k=10)


def _legacy_contains_verdict(records: list[MemoryRecord], case: EvalCase) -> bool:
    """Reproduce the old `joined-top3.lower()` rule exactly, so we can compare
    old-vs-new verdict for cases where the old report never wrote a per-query
    pass field (multi_angle)."""
    joined = " ".join(_text_of(r) for r in records[:3]).lower()
    contains_ok = True
    if case.expected_ids_or_texts:
        contains_ok = all(e.lower() in joined for e in case.expected_ids_or_texts)
    excludes_ok = all(
        f.lower() not in joined for f in case.forbidden_ids_or_texts
    ) if case.forbidden_ids_or_texts else True
    return contains_ok and excludes_ok


def run_eval_v2(
    scenarios: list[EvalScenario],
    judge_fn: Callable[[EvalCase, MemoryStore], list[MemoryRecord]] = default_judge,
    rejection_threshold: float = 1.0,
    hyde_enabled: bool | None = None,
) -> dict[str, Any]:
    """Execute every case, compute per-case and aggregate metrics, return
    a report dict (also suitable for JSON serialisation).

    ``hyde_enabled`` overrides ``RecallWeights.hyde_enabled`` for this run.
    None keeps the default (currently off) — set True from the driver when
    doing an A/B against HyDE. Requires ENGRAM_ALLOW_CLOUD_LLM=1 in that
    case so the LLM path is unblocked.
    """
    from engram_router.store import RecallWeights

    tmp_root = Path(tempfile.mkdtemp(prefix="eval_v2_"))
    per_case_rows: list[dict[str, Any]] = []

    for scen in scenarios:
        db_path = tmp_root / f"{scen.id}.db"
        if hyde_enabled is not None:
            weights = RecallWeights(hyde_enabled=hyde_enabled)
            store = MemoryStore(path=db_path, weights=weights)
        else:
            store = MemoryStore(path=db_path)
        try:
            for mem in scen.memories:
                store.save(mem)

            for case in scen.cases:
                records = judge_fn(case, store)
                top5 = [
                    {
                        "rank": i + 1,
                        "id": r.id,
                        "score": round(float(r.score), 4),
                        "text": _text_of(r)[:120],
                    }
                    for i, r in enumerate(records[:5])
                ]
                rank_of_target = (
                    _rank_of_first_target(records, case.expected_ids_or_texts)
                    if case.expected_ids_or_texts else 0
                )

                p1 = precision_at_1(records, case)
                p3 = precision_at_k(records, case, 3)
                p5 = precision_at_k(records, case, 5)
                rr = reciprocal_rank(records, case)
                ndcg5 = ndcg_at_k(records, case, 5)
                r5 = recall_at_k(records, case, 5)
                cont3 = contamination_at_k(records, case, 3)
                rej = (
                    rejection_correct(records, case, rejection_threshold)
                    if case.is_negative else None
                )
                new_pass = (
                    (rej == 1.0) if case.is_negative
                    else (p1 == 1.0 and cont3 == 0.0)
                )

                legacy_pass = _legacy_contains_verdict(records, case)
                # Prefer the legacy pass status recorded in the old report
                # when available; otherwise use the recomputed verdict.
                old_pass = case.old_passed if case.old_passed is not None else legacy_pass

                per_case_rows.append({
                    "scenario_id": scen.id,
                    "source": scen.source,
                    "case_id": case.id,
                    "query": case.query,
                    "description": case.description,
                    "is_negative": case.is_negative,
                    "expected": case.expected_ids_or_texts,
                    "forbidden": case.forbidden_ids_or_texts,
                    "rank_of_target": rank_of_target,
                    "p_at_1": p1,
                    "p_at_3": p3,
                    "p_at_5": p5,
                    "reciprocal_rank": round(rr, 4),
                    "ndcg_at_5": round(ndcg5, 4),
                    "recall_at_5": round(r5, 4),
                    "contamination_at_3": round(cont3, 4),
                    "rejection_correct": rej,
                    "top_5": top5,
                    "old_passed": old_pass,
                    "legacy_pass_recomputed": legacy_pass,
                    "new_passed": new_pass,
                })
        finally:
            store.close()

    # ── aggregate ────────────────────────────────────────────────────────
    def _avg(key: str, only_positive: bool = True) -> float:
        vals: list[float] = []
        for row in per_case_rows:
            if only_positive and row["is_negative"]:
                continue
            v = row.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return sum(vals) / len(vals) if vals else 0.0

    positive_rows = [r for r in per_case_rows if not r["is_negative"]]
    negative_rows = [r for r in per_case_rows if r["is_negative"]]

    aggregate = {
        "positive_cases": len(positive_rows),
        "negative_cases": len(negative_rows),
        "p_at_1": round(_avg("p_at_1"), 4),
        "p_at_3": round(_avg("p_at_3"), 4),
        "p_at_5": round(_avg("p_at_5"), 4),
        "mrr": round(_avg("reciprocal_rank"), 4),
        "ndcg_at_5": round(_avg("ndcg_at_5"), 4),
        "recall_at_5": round(_avg("recall_at_5"), 4),
        "contamination_at_3": round(_avg("contamination_at_3", only_positive=False), 4),
        "rejection_accuracy": round(
            (sum(r["rejection_correct"] for r in negative_rows) / len(negative_rows))
            if negative_rows else 0.0, 4,
        ),
    }

    # ── regressions: old said pass, new says fail (on positive cases) ────
    regressions: list[str] = []
    for row in per_case_rows:
        if row["is_negative"]:
            # Legacy trivially passed empty-expect negatives → any case where
            # the store returned a high-scoring top-1 is a hidden regression.
            if row["old_passed"] and row["rejection_correct"] == 0.0:
                regressions.append(
                    f"{row['case_id']} was reported passed but negative case leaked "
                    f"top-1 score {row['top_5'][0]['score'] if row['top_5'] else 'n/a'}"
                )
            continue
        if row["old_passed"] and row["p_at_1"] == 0.0:
            regressions.append(
                f"{row['case_id']} was reported passed but P@1=0 "
                f"(target at rank {row['rank_of_target'] or 'not-in-top-10'})"
            )

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rejection_threshold": rejection_threshold,
        "aggregate": aggregate,
        "per_scenario": per_case_rows,
        "regressions_vs_old_report": regressions,
    }
    return report


# ─────────────────────────────────────────────────────────────────────────
# Terminal presentation
# ─────────────────────────────────────────────────────────────────────────

def _fmt_cell(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def print_diff_table(report: dict[str, Any]) -> None:
    rows = report["per_scenario"]
    headers = [
        "case_id", "source", "old", "new", "P@1", "rank", "MRR", "nDCG@5",
        "cont@3", "rej",
    ]
    widths = [len(h) for h in headers]
    packed: list[list[str]] = []
    for row in rows:
        cells = [
            row["case_id"],
            row["source"],
            _fmt_cell(row["old_passed"]),
            _fmt_cell(row["new_passed"]),
            _fmt_cell(row["p_at_1"]),
            _fmt_cell(row["rank_of_target"]) if row["rank_of_target"] else "-",
            _fmt_cell(row["reciprocal_rank"]),
            _fmt_cell(row["ndcg_at_5"]),
            _fmt_cell(row["contamination_at_3"]),
            _fmt_cell(row["rejection_correct"]),
        ]
        packed.append(cells)
        for i, c in enumerate(cells):
            widths[i] = max(widths[i], len(c))

    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "-" * len(line)
    print(sep)
    print(line)
    print(sep)
    for cells in packed:
        marker = ""
        # Highlight regressions (old pass, new fail):
        if cells[2] == "PASS" and cells[3] == "FAIL":
            marker = "  <-- regression"
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + marker)
    print(sep)


def print_summary(report: dict[str, Any]) -> None:
    agg = report["aggregate"]
    print()
    print("=" * 72)
    print("  eval_v2 aggregate")
    print("=" * 72)
    for k in ["positive_cases", "negative_cases", "p_at_1", "p_at_3", "p_at_5",
              "mrr", "ndcg_at_5", "recall_at_5", "contamination_at_3",
              "rejection_accuracy"]:
        print(f"  {k:<22} {agg[k]}")
    print()
    regs = report["regressions_vs_old_report"]
    print(f"  regressions_vs_old_report: {len(regs)}")
    for r in regs:
        print(f"    - {r}")
    print()


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    scenarios = load_scenarios()
    # ENGRAM_EVAL_HYDE=1 enables HyDE; requires ENGRAM_ALLOW_CLOUD_LLM=1 and
    # ENGRAM_FORCE_CE=1 (else CE would be off in the SKIP_VECTOR pathway
    # eval_v2 defaults to). This is the switch used for the A/B report
    # against the CE-only baseline.
    hyde_flag = os.environ.get("ENGRAM_EVAL_HYDE") == "1"
    report = run_eval_v2(scenarios, hyde_enabled=hyde_flag if hyde_flag else None)

    print_diff_table(report)
    print_summary(report)

    out_path = ROOT / "docs" / "eval_v2_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  report saved: {out_path}")

    # Gate: aggregate P@1 >= 0.8 → exit 0.
    return 0 if report["aggregate"]["p_at_1"] >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
