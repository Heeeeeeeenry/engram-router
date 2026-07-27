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

import argparse
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
# Data loaders for L0.3 datasets
# ─────────────────────────────────────────────────────────────────────────

def load_adversarial_cases(path: Path | None = None) -> list[dict[str, Any]]:
    """Load adversarial near-miss evaluation cases from JSON.

    Returns a list of dicts, each with keys:
        id, query, expected, near_misses, memory_texts, difficulty, confusion_type
    """
    if path is None:
        path = ROOT / "tests" / "data" / "adversarial_cases.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_multilingual_cases(path: Path | None = None) -> list[dict[str, Any]]:
    """Load bilingual (Zh/En) evaluation cases from JSON.

    Returns a list of dicts, each with keys:
        id, query_zh, query_en, expected, memory_texts
    """
    if path is None:
        path = ROOT / "tests" / "data" / "multilingual_cases.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_held_out_cases(path: Path | None = None) -> list[dict[str, Any]]:
    """Load held-out evaluation cases from JSON.

    Returns a list of dicts matching the EvalCase + memories schema.
    """
    if path is None:
        path = ROOT / "tests" / "data" / "held_out_cases.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def held_out_cases_to_scenarios(cases: list[dict[str, Any]]) -> list[EvalScenario]:
    """Convert held-out JSON cases into EvalScenario objects.

    Groups cases by their source scenario, inferring a scenario ID.
    Each case dict must have: id, query, expected_ids_or_texts,
    forbidden_ids_or_texts, is_negative, memories.
    """
    # Group by scenario: for SA- prefixed, each is its own scenario.
    # For MA- prefixed, group by tag (first two segments).
    scenarios_map: dict[str, dict[str, Any]] = {}

    for c in cases:
        cid: str = c["id"]
        if cid.startswith("SA-"):
            scen_id = cid  # each SA case is standalone
        elif cid.startswith("MA-"):
            parts = cid.split("-")
            if len(parts) >= 2:
                scen_id = f"MA-{parts[1]}"  # group by tag
            else:
                scen_id = cid
        else:
            scen_id = cid

        if scen_id not in scenarios_map:
            scenarios_map[scen_id] = {
                "id": scen_id,
                "source": c.get("source", "held_out"),
                "memories": list(c.get("memories", [])),
                "cases": [],
            }
        scenarios_map[scen_id]["cases"].append(c)

    scenarios: list[EvalScenario] = []
    for sid, sdata in scenarios_map.items():
        eval_cases: list[EvalCase] = []
        for c in sdata["cases"]:
            eval_cases.append(EvalCase(
                id=c["id"],
                query=c["query"],
                expected_ids_or_texts=list(c.get("expected_ids_or_texts", [])),
                forbidden_ids_or_texts=list(c.get("forbidden_ids_or_texts", [])),
                is_negative=bool(c.get("is_negative", False)),
                description=c.get("description", ""),
            ))
        scenarios.append(EvalScenario(
            id=sid,
            source=sdata["source"],
            memories=sdata["memories"],
            cases=eval_cases,
        ))

    return scenarios


# ─────────────────────────────────────────────────────────────────────────
# CI gate & PR shadow diff (L0.5)
# ─────────────────────────────────────────────────────────────────────────

# Default CI gate thresholds (all configurable).
DEFAULT_CI_GATE_P1 = 0.8
DEFAULT_CI_GATE_CONTAMINATION = 0.05
DEFAULT_CI_GATE_REJECTION = 0.9


def check_ci_gates(
    report: dict[str, Any],
    p1_threshold: float = DEFAULT_CI_GATE_P1,
    contamination_threshold: float = DEFAULT_CI_GATE_CONTAMINATION,
    rejection_threshold: float = DEFAULT_CI_GATE_REJECTION,
) -> dict[str, Any]:
    """Check CI gate thresholds against an eval report.

    Returns a dict with per-gate pass/fail verdicts and an overall 'passed'.
    """
    agg = report.get("aggregate", {})
    p1 = agg.get("p_at_1", 0.0)
    cont = agg.get("contamination_at_3", 0.0)
    rej = agg.get("rejection_accuracy", 0.0)

    gates = {
        "p_at_1": {
            "value": p1,
            "threshold": p1_threshold,
            "passed": p1 >= p1_threshold,
        },
        "contamination_at_3": {
            "value": cont,
            "threshold": contamination_threshold,
            "passed": cont <= contamination_threshold,
        },
        "rejection_accuracy": {
            "value": rej,
            "threshold": rejection_threshold,
            "passed": rej >= rejection_threshold,
        },
    }

    all_passed = all(g["passed"] for g in gates.values())
    return {
        "gates": gates,
        "all_passed": all_passed,
    }


def pr_shadow_diff(
    main_report: dict[str, Any],
    branch_report: dict[str, Any],
) -> dict[str, Any]:
    """Compare two eval reports and flag regressions.

    Returns a dict with:
        p1_delta: change in P@1 (branch - main, positive = improvement)
        mr_delta: change in MRR
        contamination_delta: change in contamination (positive = worse)
        rejection_delta: change in rejection accuracy
        regressions: list of case_ids where branch P@1=0 while main P@1=1
        improvements: list of case_ids where branch P@1=1 while main P@1=0
        verdict: "pass" | "fail"
    """
    main_agg = main_report.get("aggregate", {})
    branch_agg = branch_report.get("aggregate", {})

    main_p1 = main_agg.get("p_at_1", 0.0)
    branch_p1 = branch_agg.get("p_at_1", 0.0)
    main_mrr = main_agg.get("mrr", 0.0)
    branch_mrr = branch_agg.get("mrr", 0.0)
    main_cont = main_agg.get("contamination_at_3", 0.0)
    branch_cont = branch_agg.get("contamination_at_3", 0.0)
    main_rej = main_agg.get("rejection_accuracy", 0.0)
    branch_rej = branch_agg.get("rejection_accuracy", 0.0)

    # Per-case comparison.
    main_cases: dict[str, dict] = {
        r["case_id"]: r for r in main_report.get("per_scenario", [])
    }
    branch_cases: dict[str, dict] = {
        r["case_id"]: r for r in branch_report.get("per_scenario", [])
    }

    regressions: list[str] = []
    improvements: list[str] = []

    for cid, mc in main_cases.items():
        bc = branch_cases.get(cid)
        if bc is None:
            continue
        m_pass = mc.get("new_passed", False)
        b_pass = bc.get("new_passed", False)
        if m_pass and not b_pass:
            regressions.append(cid)
        elif not m_pass and b_pass:
            improvements.append(cid)

    # Verdict: fail if P@1 drops by >= 0.05 or contamination rises by >= 0.02
    # or any case-level regressions exist.
    p1_delta = branch_p1 - main_p1
    cont_delta = branch_cont - main_cont

    verdict = "pass"
    if p1_delta < -0.05:
        verdict = "fail"
    if cont_delta > 0.02:
        verdict = "fail"
    if len(regressions) > 0:
        verdict = "fail"

    return {
        "p1_delta": round(p1_delta, 4),
        "mrr_delta": round(branch_mrr - main_mrr, 4),
        "contamination_delta": round(cont_delta, 4),
        "rejection_delta": round(branch_rej - main_rej, 4),
        "regressions": regressions,
        "improvements": improvements,
        "verdict": verdict,
    }


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
# Stability metrics (L0.1)
# ─────────────────────────────────────────────────────────────────────────

def compute_stability(
    store: MemoryStore,
    query: str,
    target_ids: set[str],
    n_runs: int = 20,
    top_k: int = 5,
) -> dict:
    """Run recall n_runs times and compute stability metrics.

    Measures how consistently the store returns the same ranking for a given
    query.  All metrics are deterministic when the store is in a fixed state
    (no random seeds, no clock drift).

    Returns:
        {
            "jaccard_p1": float,      # mean pairwise Jaccard of top-1 across runs
            "jaccard_p3": float,      # mean pairwise Jaccard of top-3 across runs
            "rank_variance": float,   # mean pop-variance of target_id ranks (lower=better)
            "rank_stability": float,  # fraction of runs where target holds its most-common rank
            "target_found_rate": float,  # fraction of runs where target appears in top-k
        }
    """
    all_ranks: list[list[MemoryRecord]] = []
    for _ in range(n_runs):
        records = store.recall(query, top_k=top_k)
        all_ranks.append(records)

    # --- Jaccard @ 1 (pairwise) ---
    p1_sets = [{records[0].id} if records else set() for records in all_ranks]
    j1_sum = 0.0
    j1_count = 0
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            if p1_sets[i] or p1_sets[j]:
                j1_sum += len(p1_sets[i] & p1_sets[j]) / len(p1_sets[i] | p1_sets[j])
            else:
                j1_sum += 1.0  # both empty → agreement
            j1_count += 1
    jaccard_p1 = j1_sum / j1_count if j1_count else 0.0

    # --- Jaccard @ 3 (pairwise) ---
    p3_sets = [{r.id for r in records[:3]} for records in all_ranks]
    j3_sum = 0.0
    j3_count = 0
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            if p3_sets[i] or p3_sets[j]:
                j3_sum += len(p3_sets[i] & p3_sets[j]) / len(p3_sets[i] | p3_sets[j])
            else:
                j3_sum += 1.0
            j3_count += 1
    jaccard_p3 = j3_sum / j3_count if j3_count else 0.0

    # --- Per-target rank variance & stability ---
    rank_variances: list[float] = []
    rank_stabilities: list[float] = []
    found_counts: list[int] = []

    for tid in target_ids:
        ranks: list[int] = []
        for records in all_ranks:
            found = False
            for pos, r in enumerate(records, start=1):
                if r.id == tid or tid in r.raw_text:
                    ranks.append(pos)
                    found = True
                    break
            if not found:
                ranks.append(top_k + 1)  # sentinel: not in top-k

        found_count = sum(1 for r in ranks if r <= top_k)
        found_counts.append(found_count)

        # Population variance of ranks.
        mean_r = sum(ranks) / len(ranks)
        var_r = sum((r - mean_r) ** 2 for r in ranks) / len(ranks)
        rank_variances.append(var_r)

        # Rank stability: fraction of runs at the most-common rank.
        from collections import Counter
        rank_counts = Counter(ranks)
        most_common_count = rank_counts.most_common(1)[0][1]
        rank_stabilities.append(most_common_count / n_runs)

    rank_variance = sum(rank_variances) / len(rank_variances) if rank_variances else 0.0
    rank_stability = sum(rank_stabilities) / len(rank_stabilities) if rank_stabilities else 0.0
    target_found_rate = (
        sum(found_counts) / (len(target_ids) * n_runs) if target_ids else 0.0
    )

    return {
        "jaccard_p1": round(jaccard_p1, 4),
        "jaccard_p3": round(jaccard_p3, 4),
        "rank_variance": round(rank_variance, 4),
        "rank_stability": round(rank_stability, 4),
        "target_found_rate": round(target_found_rate, 4),
    }


def stability_report(
    scenarios: list[EvalScenario],
    store_factory=None,
    n_runs: int = 20,
    top_k: int = 5,
) -> list[dict]:
    """Run stability evaluation across all scenarios and cases.

    Args:
        scenarios: loaded EvalScenario list.
        store_factory: callable ``(scenario_id) -> MemoryStore``; uses a fresh
            temp-db store with ingested memories when None.
        n_runs: number of recall repetitions per case.
        top_k: recall depth.

    Returns:
        A list of per-case stability dicts suitable for printing or JSON export.
    """
    import tempfile

    rows: list[dict] = []
    for scen in scenarios:
        if store_factory is not None:
            store = store_factory(scen.id)
        else:
            db_path = Path(tempfile.mkdtemp(prefix="eval_stability_")) / f"{scen.id}.db"
            store = MemoryStore(path=db_path)
            for mem in scen.memories:
                store.save(mem)

        try:
            for case in scen.cases:
                target_ids = set(case.expected_ids_or_texts)
                if not target_ids:
                    continue
                metrics = compute_stability(
                    store, case.query, target_ids,
                    n_runs=n_runs, top_k=top_k,
                )
                rows.append({
                    "scenario_id": scen.id,
                    "case_id": case.id,
                    "query": case.query,
                    "n_runs": n_runs,
                    "top_k": top_k,
                    **metrics,
                })
        finally:
            store.close()

    return rows


def print_stability_table(rows: list[dict]) -> None:
    """Print a formatted stability report table to stdout."""
    if not rows:
        print("  (no stability data)")
        return

    headers = [
        "case_id", "J@1", "J@3", "rank_var", "rank_stab", "found_rate",
    ]
    widths = [len(h) for h in headers]
    cells_rows: list[list[str]] = []
    for row in rows:
        cells = [
            row["case_id"],
            f"{row['jaccard_p1']:.3f}",
            f"{row['jaccard_p3']:.3f}",
            f"{row['rank_variance']:.3f}",
            f"{row['rank_stability']:.3f}",
            f"{row['target_found_rate']:.3f}",
        ]
        cells_rows.append(cells)
        for i, c in enumerate(cells):
            widths[i] = max(widths[i], len(c))

    sep = "-" * (sum(widths) + 2 * (len(headers) - 1))
    print()
    print("  Stability report")
    print(sep)
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print(sep)
    for cells in cells_rows:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(cells)))
    print(sep)
    print()


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="eval_v2 — rank-aware evaluator")
    parser.add_argument(
        "--stability", action="store_true",
        help="Run stability metrics (L0.1) and print report, then exit.",
    )
    parser.add_argument(
        "--stability-runs", type=int, default=20,
        help="Number of recall repetitions for stability (default: 20).",
    )
    parser.add_argument(
        "--stability-topk", type=int, default=5,
        help="Recall depth for stability metrics (default: 5).",
    )
    # L0.3 held-out split
    parser.add_argument(
        "--held-out", type=str, default=None,
        help="Path to held-out cases JSON. Separates dev vs held-out reporting. "
             "Only use once before release.",
    )
    parser.add_argument(
        "--held-out-only", action="store_true",
        help="Run only the held-out split (implies --held-out with default path).",
    )
    # L0.5 CI gate
    parser.add_argument(
        "--ci", action="store_true",
        help="Enforce CI gate thresholds (P@1>=0.8, Contamination@3<=0.05, Rejection>=0.9). "
             "Exit code 0 when all gates pass, non-zero otherwise.",
    )
    parser.add_argument(
        "--ci-p1", type=float, default=DEFAULT_CI_GATE_P1,
        help=f"CI gate P@1 threshold (default: {DEFAULT_CI_GATE_P1}).",
    )
    parser.add_argument(
        "--ci-contamination", type=float, default=DEFAULT_CI_GATE_CONTAMINATION,
        help=f"CI gate contamination threshold (default: {DEFAULT_CI_GATE_CONTAMINATION}).",
    )
    parser.add_argument(
        "--ci-rejection", type=float, default=DEFAULT_CI_GATE_REJECTION,
        help=f"CI gate rejection accuracy threshold (default: {DEFAULT_CI_GATE_REJECTION}).",
    )
    # PR shadow diff
    parser.add_argument(
        "--pr-shadow-diff-main", type=str, default=None,
        help="Path to main branch eval report JSON for shadow diff comparison.",
    )
    parser.add_argument(
        "--pr-shadow-diff-branch", type=str, default=None,
        help="Path to branch eval report JSON for shadow diff comparison.",
    )
    # Multi-language eval
    parser.add_argument(
        "--multilingual", action="store_true",
        help="Run bilingual evaluation with multilingual_cases.json.",
    )
    # Adversarial eval
    parser.add_argument(
        "--adversarial", action="store_true",
        help="Run adversarial near-miss evaluation with adversarial_cases.json.",
    )
    # Output
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to write the eval report JSON (default: docs/eval_v2_report.json).",
    )
    args = parser.parse_args()

    # ── Stability mode ─────────────────────────────────────────────────
    if args.stability:
        scenarios = load_scenarios()
        rows = stability_report(
            scenarios, n_runs=args.stability_runs, top_k=args.stability_topk,
        )
        print_stability_table(rows)
        return 0

    # ── PR shadow diff mode ────────────────────────────────────────────
    if args.pr_shadow_diff_main or args.pr_shadow_diff_branch:
        main_path = Path(args.pr_shadow_diff_main) if args.pr_shadow_diff_main else ROOT / "docs" / "eval_v2_report.json"
        branch_path = Path(args.pr_shadow_diff_branch) if args.pr_shadow_diff_branch else Path(tempfile.mkdtemp(prefix="eval_v2_")) / "branch_report.json"

        # If only main given, run branch eval first.
        if args.pr_shadow_diff_main and not args.pr_shadow_diff_branch:
            scenarios = load_scenarios()
            hyde_flag = os.environ.get("ENGRAM_EVAL_HYDE") == "1"
            branch_report = run_eval_v2(scenarios, hyde_enabled=hyde_flag if hyde_flag else None)
            with branch_path.open("w") as f:
                json.dump(branch_report, f, ensure_ascii=False, indent=2)

        with main_path.open("r") as f:
            main_report = json.load(f)
        with branch_path.open("r") as f:
            branch_report = json.load(f)

        diff = pr_shadow_diff(main_report, branch_report)
        print("\n  PR Shadow Diff Report")
        print("  ======================")
        print(f"  P@1 delta:          {diff['p1_delta']:+.4f}")
        print(f"  MRR delta:          {diff['mrr_delta']:+.4f}")
        print(f"  Contamination delta:{diff['contamination_delta']:+.4f}")
        print(f"  Rejection delta:    {diff['rejection_delta']:+.4f}")
        print(f"  Regressions:        {len(diff['regressions'])}")
        for r in diff["regressions"]:
            print(f"    - {r}")
        print(f"  Improvements:       {len(diff['improvements'])}")
        for imp in diff["improvements"]:
            print(f"    - {imp}")
        print(f"  Verdict:            {diff['verdict'].upper()}")
        print()
        return 0 if diff["verdict"] == "pass" else 1

    # ── Load scenarios ─────────────────────────────────────────────────
    dev_scenarios: list[EvalScenario] = []
    held_out_scenarios: list[EvalScenario] = []

    if args.held_out_only or args.held_out:
        held_out_path = Path(args.held_out) if args.held_out else ROOT / "tests" / "data" / "held_out_cases.json"
        ho_cases = load_held_out_cases(held_out_path)
        held_out_scenarios = held_out_cases_to_scenarios(ho_cases)
        if args.held_out_only:
            scenarios = held_out_scenarios
        else:
            dev_scenarios = load_scenarios()
            scenarios = dev_scenarios + held_out_scenarios
    elif args.multilingual:
        ml_cases = load_multilingual_cases()
        scenarios = _build_scenarios_from_json(ml_cases, "multilingual")
    elif args.adversarial:
        adv_cases = load_adversarial_cases()
        scenarios = _build_scenarios_from_json(adv_cases, "adversarial")
    else:
        dev_scenarios = load_scenarios()
        scenarios = dev_scenarios

    # ── Run eval ───────────────────────────────────────────────────────
    hyde_flag = os.environ.get("ENGRAM_EVAL_HYDE") == "1"
    report = run_eval_v2(scenarios, hyde_enabled=hyde_flag if hyde_flag else None)

    # When both dev and held-out present, compute separate aggregates.
    if dev_scenarios and held_out_scenarios:
        dev_report = run_eval_v2(dev_scenarios, hyde_enabled=hyde_flag if hyde_flag else None)
        ho_report = run_eval_v2(held_out_scenarios, hyde_enabled=hyde_flag if hyde_flag else None)

        print_diff_table(report)
        print_summary(report)

        print("\n" + "=" * 72)
        print("  DEVELOPMENT SET")
        print("=" * 72)
        print_summary(dev_report)

        print("\n" + "=" * 72)
        print("  HELD-OUT SET (run once before release)")
        print("=" * 72)
        print_summary(ho_report)

        report["dev_aggregate"] = dev_report["aggregate"]
        report["held_out_aggregate"] = ho_report["aggregate"]
    else:
        print_diff_table(report)
        print_summary(report)

    out_path = Path(args.output) if args.output else ROOT / "docs" / "eval_v2_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  report saved: {out_path}")

    # ── CI gate ────────────────────────────────────────────────────────
    if args.ci:
        gate_result = check_ci_gates(
            report,
            p1_threshold=args.ci_p1,
            contamination_threshold=args.ci_contamination,
            rejection_threshold=args.ci_rejection,
        )
        print("\n  CI Gate Check")
        print("  =============")
        for gate_name, gate in gate_result["gates"].items():
            status = "PASS" if gate["passed"] else "FAIL"
            print(f"  {gate_name}: {gate['value']:.4f} vs threshold {gate['threshold']:.4f}  [{status}]")
        print(f"  Overall: {'PASS' if gate_result['all_passed'] else 'FAIL'}")
        print()
        if not gate_result["all_passed"]:
            return 1
        return 0

    # Default gate: P@1 >= 0.8 → exit 0.
    return 0 if report["aggregate"]["p_at_1"] >= 0.8 else 1


def _build_scenarios_from_json(cases: list[dict[str, Any]], source: str) -> list[EvalScenario]:
    """Convert generic JSON cases (adversarial or multilingual) into EvalScenario objects."""
    if not cases:
        return []

    scenarios: list[EvalScenario] = []
    # Group cases by some heuristic — for adversarial/multilingual, each case
    # is a self-contained scenario with its own memories.
    for c in cases:
        cid = c.get("id", f"{source}-{len(scenarios)}")
        # Determine expected texts based on source type.
        if source == "adversarial":
            expected = [c.get("expected", "")]
            forbidden = list(c.get("near_misses", []))
            memories = list(c.get("memory_texts", []))
            query = c.get("query", "")
        elif source == "multilingual":
            expected = [c.get("expected", "")]
            forbidden = []
            memories = list(c.get("memory_texts", []))
            query = c.get("query_zh", c.get("query_en", ""))
        else:
            expected = list(c.get("expected_ids_or_texts", []))
            forbidden = list(c.get("forbidden_ids_or_texts", []))
            memories = list(c.get("memories", []))
            query = c.get("query", "")

        if not query:
            continue

        ec = EvalCase(
            id=cid,
            query=query,
            expected_ids_or_texts=expected,
            forbidden_ids_or_texts=forbidden,
            is_negative=False,
            description=c.get("description", c.get("confusion_type", "")),
        )
        scenarios.append(EvalScenario(
            id=f"{source}-{cid}",
            source=source,
            memories=memories,
            cases=[ec],
        ))

    # For multilingual: also append the English query variant as additional cases.
    if source == "multilingual":
        extra_cases: list[tuple[str, EvalCase]] = []
        for i, sc in enumerate(scenarios):
            c = cases[i]
            en_query = c.get("query_en", "")
            if en_query:
                ec_en = EvalCase(
                    id=f"{c.get('id', sc.id)}-en",
                    query=en_query,
                    expected_ids_or_texts=[c.get("expected", "")],
                    forbidden_ids_or_texts=[],
                    is_negative=False,
                    description=f"[en] {c.get('description', '')}",
                )
                extra_cases.append((sc.id, ec_en))
        for sid, ec in extra_cases:
            for sc in scenarios:
                if sc.id == sid:
                    sc.cases.append(ec)
                    break

    return scenarios


if __name__ == "__main__":
    sys.exit(main())
