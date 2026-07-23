#!/usr/bin/env python3
"""LongMemEval evaluation driver — the first non-toy benchmark run.

Runs LMECase items (see ``longmemeval_loader.py``) through one or more
:class:`MemoryProvider` adapters and reports P@1/MRR/nDCG using the exact
same metric functions as ``eval_v2.py`` — so numbers are comparable across
the toy semantic_audit/multi_angle suite and this real corpus.

IMPORTANT CAVEAT (read before trusting the numbers):
LongMemEval's own paper scores answer *correctness* via a GPT-4 judge on the
system's free-text answer, generated from whatever context the retriever
handed it. This driver does NOT run that generation+judge step — it only
measures retrieval quality directly: "does the top-k contain a memory whose
text contains the reference answer string?" That is a stricter, retrieval-only
proxy. It will under-count cases where the answer is phrased differently
across the question and the source turn (paraphrase, synonym, number format).
Treat these numbers as a *lower bound* on what a full pipeline (retrieval +
answer generation) would score, not as the paper's headline metric.

Usage:
    ENGRAM_ALLOW_CLOUD_LLM=1 python tests/eval_v2_longmemeval.py \\
        --split oracle --limit 100 --providers engram,naive-vector
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from eval_v2 import (  # type: ignore[import-not-found]
    EvalCase,
    contamination_at_k,
    ndcg_at_k,
    precision_at_1,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
)
from engram_router.store import MemoryRecord
from longmemeval_loader import LMECase, dataset_stats, load_longmemeval
from lme_judge import generate_answer, judge_correctness
from providers.base import MemoryProvider, ProviderRecord
from providers.engram_provider import EngramProvider
from providers.naive_vector import NaiveVectorProvider


def _to_memory_record(pr: ProviderRecord) -> MemoryRecord:
    return MemoryRecord(
        id=pr.id, raw_text=pr.text, summary=pr.text[:120], confidence=1.0,
        metadata=dict(pr.metadata), evidence_refs=[], score=float(pr.score),
        match_reason="",
    )


def _case_from_lme(item: LMECase) -> EvalCase:
    return EvalCase(
        query=item.question,
        expected_ids_or_texts=[item.answer] if item.answer else [],
        forbidden_ids_or_texts=[],
        is_negative=False,
        id=item.question_id,
        description=f"[longmemeval:{item.question_type}]",
    )


def _build_registry() -> dict[str, Any]:
    return {
        "engram": lambda: EngramProvider(ce_enabled=True, hyde_enabled=False),
        "naive-vector": lambda: NaiveVectorProvider(),
    }


def evaluate_provider(
    provider: MemoryProvider,
    items: list[LMECase],
    workspace_root: Path,
    top_k: int = 10,
    judge_client: Any | None = None,
) -> dict[str, Any]:
    """Score one provider over a list of LongMemEval items.

    When ``judge_client`` is None (default): retrieval-only substring proxy —
    what the driver did before. When a client is supplied: generate an
    answer from the retrieved top-k, then LLM-judge it against the reference.
    The judged score (0/1) becomes ``judge_correct`` on each row and the
    aggregate ``judge_accuracy`` field. This is the metric LongMemEval's own
    paper reports.
    """
    per_case: list[dict[str, Any]] = []
    per_type: dict[str, list[dict[str, Any]]] = {}
    latencies: list[float] = []
    judge_latencies: list[float] = []

    for idx, item in enumerate(items):
        case = _case_from_lme(item)
        ws = workspace_root / f"lme_{idx}"
        ws.mkdir(parents=True, exist_ok=True)
        provider.open(ws)
        try:
            for mem in item.memories:
                try:
                    provider.save(mem)
                except Exception as exc:
                    # A single bad turn shouldn't sink the whole item.
                    print(f"  [warn] save failed for item {item.question_id}: {exc}",
                          file=sys.stderr)

            if not case.expected_ids_or_texts:
                provider.close()
                continue

            t0 = time.perf_counter()
            hits = provider.recall(case.query, top_k=top_k)
            dt_ms = (time.perf_counter() - t0) * 1000
            latencies.append(dt_ms)

            records = [_to_memory_record(h) for h in hits]
            row = {
                "question_id": item.question_id,
                "question_type": item.question_type,
                "query": case.query,
                "answer": item.answer,
                "num_memories": len(item.memories),
                "p_at_1": precision_at_1(records, case),
                "p_at_3": precision_at_k(records, case, 3),
                "p_at_5": precision_at_k(records, case, 5),
                "reciprocal_rank": reciprocal_rank(records, case),
                "ndcg_at_5": ndcg_at_k(records, case, 5),
                "recall_at_5": recall_at_k(records, case, 5),
                "latency_ms": round(dt_ms, 1),
                "top_3": [{"score": round(h.score, 4), "text": h.text[:100]}
                          for h in hits[:3]],
            }

            # ── Optional LLM judge branch ─────────────────────────────
            if judge_client is not None:
                jt0 = time.perf_counter()
                try:
                    memory_texts = [h.text for h in hits[:top_k]]
                    # Extract timestamps from provider metadata (step-L: conflict resolution)
                    memory_timestamps = [
                        h.metadata.get("created_at", "") for h in hits[:top_k]
                    ]
                    # Only pass timestamps if we have them (non-empty).
                    if not any(memory_timestamps):
                        memory_timestamps = None
                    ans = generate_answer(
                        judge_client, case.query, memory_texts,
                        memory_timestamps=memory_timestamps,
                    )
                    verdict = judge_correctness(judge_client, case.query,
                                                item.answer, ans.text)
                    row["generated_answer"] = ans.text[:500]
                    row["judge_correct"] = int(verdict.correct)
                    row["judge_reason"] = verdict.reason
                except Exception as exc:
                    row["judge_correct"] = 0
                    row["judge_reason"] = f"pipeline error: {exc}"
                judge_latencies.append((time.perf_counter() - jt0) * 1000)

            per_case.append(row)
            per_type.setdefault(item.question_type, []).append(row)
        finally:
            provider.close()

    def _avg(rows: list[dict[str, Any]], key: str) -> float:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    aggregate = {
        "total_cases": len(per_case),
        "p_at_1": _avg(per_case, "p_at_1"),
        "p_at_3": _avg(per_case, "p_at_3"),
        "p_at_5": _avg(per_case, "p_at_5"),
        "mrr": _avg(per_case, "reciprocal_rank"),
        "ndcg_at_5": _avg(per_case, "ndcg_at_5"),
        "recall_at_5": _avg(per_case, "recall_at_5"),
        "latency_ms_avg": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "latency_ms_p95": round(sorted(latencies)[int(len(latencies) * 0.95)], 1)
                          if latencies else 0.0,
    }
    if judge_client is not None:
        aggregate["judge_accuracy"] = _avg(per_case, "judge_correct")
        aggregate["judge_latency_ms_avg"] = round(
            sum(judge_latencies) / len(judge_latencies), 1) if judge_latencies else 0.0
    by_type = {
        qt: {"n": len(rows), "p_at_1": _avg(rows, "p_at_1"),
             "mrr": _avg(rows, "reciprocal_rank"),
             "judge_accuracy": _avg(rows, "judge_correct")
             if judge_client is not None else None}
        for qt, rows in per_type.items()
    }

    return {
        "provider": provider.name,
        "aggregate": aggregate,
        "by_question_type": by_type,
        "per_case": per_case,
    }


def print_report(results: dict[str, dict[str, Any]]) -> None:
    names = list(results.keys())
    if not names:
        print("no providers ran")
        return
    metrics = ["p_at_1", "p_at_3", "p_at_5", "mrr", "ndcg_at_5", "recall_at_5",
               "judge_accuracy", "latency_ms_avg", "latency_ms_p95",
               "judge_latency_ms_avg"]
    print()
    print("=" * 90)
    print("  LongMemEval evaluation")
    print("=" * 90)
    header = f"{'metric':<25}" + "".join(f"{n:>20}" for n in names)
    print(header)
    print("-" * len(header))
    for m in metrics:
        row = f"{m:<25}"
        for n in names:
            v = results[n]["aggregate"].get(m)
            if v is None:
                row += f"{'-':>20}"
            elif isinstance(v, float):
                row += f"{v:>20.4f}"
            else:
                row += f"{v!s:>20}"
        print(row)
    print("-" * len(header))

    print()
    print("  by question_type (p_at_1 / judge_acc):")
    all_types = sorted({qt for r in results.values() for qt in r["by_question_type"]})
    for qt in all_types:
        cells = []
        for n in names:
            bt = results[n]["by_question_type"].get(qt)
            if not bt:
                cells.append("-")
                continue
            ja = bt.get("judge_accuracy")
            ja_s = f"{ja:.2f}" if ja is not None else "-"
            cells.append(f"P@1={bt['p_at_1']:.2f} judge={ja_s} (n={bt['n']})")
        print(f"    {qt:<28}" + "  ".join(cells))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="oracle", choices=["oracle", "s", "m"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--providers", default="engram,naive-vector")
    ap.add_argument("--question-types", default=None,
                     help="comma-separated subset, default all")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--use-judge", action="store_true",
                     help="Enable LLM judge (LongMemEval's official metric).")
    args = ap.parse_args()

    qtypes = set(args.question_types.split(",")) if args.question_types else None
    items = load_longmemeval(split=args.split, limit=args.limit, question_types=qtypes)
    print(f"[lme] loaded {len(items)} items from split={args.split!r}")
    print(f"[lme] stats: {json.dumps(dataset_stats(items))}")

    registry = _build_registry()
    picked = [p.strip() for p in args.providers.split(",") if p.strip()]
    print(f"[lme] providers → {picked}")

    judge_client = None
    if args.use_judge:
        from lme_judge import _make_client

        judge_client = _make_client()
        print(f"[lme] LLM judge enabled (client available={judge_client.available})")

    workspace_root = Path(tempfile.mkdtemp(prefix="eval_lme_"))
    results: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}

    for name in picked:
        if name not in registry:
            skipped[name] = f"unknown provider {name!r}"
            continue
        try:
            provider = registry[name]()
            result = evaluate_provider(provider, items, workspace_root,
                                       top_k=args.top_k, judge_client=judge_client)
            results[result["provider"]] = result
        except Exception as exc:
            skipped[name] = f"failed: {exc}"
            traceback.print_exc()

    print_report(results)
    if skipped:
        print("\nskipped:")
        for n, why in skipped.items():
            print(f"  - {n}: {why}")

    suffix = "_judge" if args.use_judge else ""
    out = ROOT / "docs" / f"eval_v2_longmemeval_{args.split}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "split": args.split,
            "limit": args.limit,
            "question_types": sorted(qtypes) if qtypes else None,
            "use_judge": args.use_judge,
            "results": results,
            "skipped": skipped,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  report saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
