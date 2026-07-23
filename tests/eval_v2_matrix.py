#!/usr/bin/env python3
"""Cross-provider evaluation matrix.

Runs the same scenario set that ``tests/eval_v2.py`` builds against every
enabled provider and prints a side-by-side table. Providers that fail to
open (missing dependency, missing API key, etc.) are skipped with a clear
reason so the matrix still produces a report for the reachable ones.

Usage::

    ENGRAM_ALLOW_CLOUD_LLM=1 python tests/eval_v2_matrix.py

Env switches:
    ENGRAM_MATRIX_PROVIDERS = comma-separated subset of provider names
        (default: engram, engram-hyde, naive-vector, mem0)
    ENGRAM_MATRIX_LIMIT     = only run the first N scenarios (dev speed-up)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

# Import metric helpers from eval_v2 so ranking rules stay identical.
from eval_v2 import (  # type: ignore[import-not-found]
    EvalCase,
    EvalScenario,
    contamination_at_k,
    load_scenarios,
    ndcg_at_k,
    precision_at_1,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
    rejection_correct,
)
from engram_router.store import MemoryRecord

from providers.base import MemoryProvider, ProviderRecord
from providers.engram_provider import EngramProvider
from providers.long_context import LongContextProvider
from providers.mem0_provider import Mem0Provider
from providers.naive_vector import NaiveVectorProvider

# ─────────────────────────────────────────────────────────────────────────
# Adapter: provider records → MemoryRecord (what eval_v2 metrics expect)
# ─────────────────────────────────────────────────────────────────────────

def _to_memory_record(pr: ProviderRecord) -> MemoryRecord:
    return MemoryRecord(
        id=pr.id,
        raw_text=pr.text,
        summary=pr.text[:120],
        confidence=1.0,
        metadata=dict(pr.metadata),
        evidence_refs=[],
        score=float(pr.score),
        match_reason="",
    )


# ─────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────

def _build_registry() -> dict[str, Any]:
    """Provider factories. Late-bound so we surface import errors as skips."""
    def _engram_with_weight(w: float) -> Any:
        return lambda: EngramProvider(ce_enabled=True, hyde_enabled=False,
                                       ce_weight=w)

    return {
        "engram": lambda: EngramProvider(ce_enabled=True, hyde_enabled=False),
        "engram-w0.6": _engram_with_weight(0.6),
        "engram-w0.75": _engram_with_weight(0.75),
        "engram-w0.85": _engram_with_weight(0.85),
        "engram-w1.0": _engram_with_weight(1.0),
        "engram-hyde": lambda: EngramProvider(ce_enabled=True, hyde_enabled=True),
        "naive-vector": lambda: NaiveVectorProvider(),
        "mem0": lambda: Mem0Provider(),
        "long-context": lambda: LongContextProvider(),
    }


def _selected_providers() -> list[str]:
    raw = os.environ.get("ENGRAM_MATRIX_PROVIDERS")
    if raw:
        return [p.strip() for p in raw.split(",") if p.strip()]
    return ["engram", "naive-vector", "mem0"]


def _scenario_limit() -> int | None:
    v = os.environ.get("ENGRAM_MATRIX_LIMIT")
    return int(v) if v else None


def evaluate_provider(
    provider: MemoryProvider,
    scenarios: list[EvalScenario],
    workspace_root: Path,
    rejection_threshold: float = 1.0,
) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    latencies: list[float] = []

    for scen in scenarios:
        ws = workspace_root / scen.id
        ws.mkdir(parents=True, exist_ok=True)
        provider.open(ws)
        try:
            for mem in scen.memories:
                provider.save(mem)

            for case in scen.cases:
                t0 = time.perf_counter()
                hits = provider.recall(case.query, top_k=10)
                dt_ms = (time.perf_counter() - t0) * 1000
                latencies.append(dt_ms)

                records = [_to_memory_record(h) for h in hits]

                rej = (rejection_correct(records, case, rejection_threshold)
                       if case.is_negative else None)
                per_case.append({
                    "scenario_id": scen.id,
                    "case_id": case.id,
                    "query": case.query,
                    "is_negative": case.is_negative,
                    "p_at_1": precision_at_1(records, case),
                    "p_at_3": precision_at_k(records, case, 3),
                    "p_at_5": precision_at_k(records, case, 5),
                    "reciprocal_rank": reciprocal_rank(records, case),
                    "ndcg_at_5": ndcg_at_k(records, case, 5),
                    "recall_at_5": recall_at_k(records, case, 5),
                    "contamination_at_3": contamination_at_k(records, case, 3),
                    "rejection_correct": rej,
                    "latency_ms": round(dt_ms, 1),
                    "top_3": [
                        {"id": h.id, "score": round(h.score, 4),
                         "text": h.text[:100]}
                        for h in hits[:3]
                    ],
                })
        finally:
            provider.close()

    def _avg(key: str, positive_only: bool = True) -> float:
        vals: list[float] = []
        for row in per_case:
            if positive_only and row["is_negative"]:
                continue
            v = row.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return sum(vals) / len(vals) if vals else 0.0

    negatives = [r for r in per_case if r["is_negative"]]
    positives = [r for r in per_case if not r["is_negative"]]

    aggregate = {
        "positive_cases": len(positives),
        "negative_cases": len(negatives),
        "p_at_1": round(_avg("p_at_1"), 4),
        "p_at_3": round(_avg("p_at_3"), 4),
        "p_at_5": round(_avg("p_at_5"), 4),
        "mrr": round(_avg("reciprocal_rank"), 4),
        "ndcg_at_5": round(_avg("ndcg_at_5"), 4),
        "recall_at_5": round(_avg("recall_at_5"), 4),
        "contamination_at_3": round(_avg("contamination_at_3", positive_only=False), 4),
        "rejection_accuracy": round(
            (sum(r["rejection_correct"] for r in negatives) / len(negatives))
            if negatives else 0.0, 4,
        ),
        "latency_ms_avg": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "latency_ms_p95": round(sorted(latencies)[int(len(latencies) * 0.95)], 1)
                          if latencies else 0.0,
    }
    return {
        "provider": provider.name,
        "aggregate": aggregate,
        "per_case": per_case,
    }


# ─────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────

def print_matrix(results: dict[str, dict[str, Any]]) -> None:
    metrics = [
        "p_at_1", "p_at_3", "p_at_5",
        "mrr", "ndcg_at_5", "recall_at_5",
        "contamination_at_3", "rejection_accuracy",
        "latency_ms_avg", "latency_ms_p95",
    ]
    names = list(results.keys())
    if not names:
        print("no providers ran")
        return

    print()
    print("=" * 100)
    print("  eval_v2 matrix — same scenarios, multiple providers")
    print("=" * 100)
    header = f"{'metric':<22}" + "".join(f"{n:>18}" for n in names)
    print(header)
    print("-" * len(header))
    for m in metrics:
        row = f"{m:<22}"
        for n in names:
            v = results[n]["aggregate"].get(m, "-")
            row += f"{v:>18.4f}" if isinstance(v, float) else f"{v!s:>18}"
        print(row)
    print("-" * len(header))


def main() -> int:
    scenarios = load_scenarios()
    limit = _scenario_limit()
    if limit is not None:
        scenarios = scenarios[:limit]
        print(f"[matrix] scenario limit → {limit}")

    picked = _selected_providers()
    print(f"[matrix] providers → {picked}")

    registry = _build_registry()
    workspace_root = Path(tempfile.mkdtemp(prefix="eval_v2_matrix_"))
    results: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}

    for name in picked:
        if name not in registry:
            skipped[name] = f"unknown provider ({name!r})"
            continue
        try:
            provider = registry[name]()
        except Exception as exc:
            skipped[name] = f"construction failed: {exc}"
            traceback.print_exc()
            continue
        try:
            result = evaluate_provider(provider, scenarios, workspace_root)
            results[result["provider"]] = result
        except Exception as exc:
            skipped[name] = f"evaluation failed: {exc}"
            traceback.print_exc()

    print_matrix(results)

    if skipped:
        print()
        print("skipped providers:")
        for n, why in skipped.items():
            print(f"  - {n}: {why}")

    out = ROOT / "docs" / "eval_v2_matrix.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "workspace": str(workspace_root),
        "results": {n: r for n, r in results.items()},
        "skipped": skipped,
    }
    with out.open("w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  report saved: {out}")

    engram = results.get("engram/ce+/hyde-", {}).get("aggregate", {}) if results else {}
    return 0 if engram.get("p_at_1", 0.0) >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
