"""L4 soak test — hammer the store with synthetic data and measure latency.

Design:
  - Insert N memories of pseudo-realistic Chinese text (generated from
    templates, no LLM/vector/CE overhead).
  - Run M random recall queries and record P50 / P95 / P99 latency.
  - Report memory usage (RSS) at start and end.

Usage:
    python tests/soak_l4.py              # default: 10K mem, 500 queries
    python tests/soak_l4.py --n 50000    # 50K memories
    python tests/soak_l4.py --n 100000 --queries 1000   # production-scale

Dependencies: none (stdlib only — psutil optional for RSS).
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

# Add src to path so we can import engram_router
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from engram_router import MemoryStore  # noqa: E402


# ── Template-based synthetic Chinese text generator ──────────────────────

_PERSONS = ["张三", "李四", "王五", "赵六", "老张", "小李", "小王", "陈总", "刘姐"]
_VERBS = ["在", "去了", "买了", "吃了", "学了", "看了", "做了", "发布了", "完成了", "面试了"]
_OBJECTS = [
    "北京", "上海", "深圳", "火锅", "电脑", "一本书", "新项目", "前端框架",
    "设计稿", "数据库", "部署方案", "算法题", "咖啡", "健身房", "会议室",
]
_PLACES = ["在公司", "在家", "在咖啡厅", "在图书馆", "在地铁上"]
_TIMES = ["今天", "昨天", "上周", "刚才", "周末", "", "", "", ""]  # 5/9 chance of time expr
_EVALS = [
    "挺不错的", "有点累", "很开心", "学到了很多", "效率很高",
    "遇到了问题", "成功上线", "被否决了", "延期了", "很顺利",
]

# Additional memory templates for variety
_TEMPLATES = [
    "{person}{time}{verb}{obj}，{eval}",
    "{time}{person}在{place}{verb}{obj}",
    "{person}{verb}{obj}，感觉{eval}",
    "今天{person}{verb}{obj}，{eval}",
    "{person}和同事{verb}{obj}，{eval}",
    "{time}的会议{person}{verb}{obj}",
    "{person}说{verb}{obj}，{eval}",
    "{person}在{place}{verb}{obj}，{time}的成果",
]


def _rand_choice(choices: list[str]) -> str:
    return random.choice(choices)


def generate_memory() -> str:
    """Return one synthetic Chinese-language memory text."""
    template = _rand_choice(_TEMPLATES)
    return template.format(
        person=_rand_choice(_PERSONS),
        time=_rand_choice(_TIMES),
        verb=_rand_choice(_VERBS),
        obj=_rand_choice(_OBJECTS),
        place=_rand_choice(_PLACES),
        eval=_rand_choice(_EVALS),
    )


def generate_query() -> str:
    """Return one synthetic recall query."""
    templates = [
        "{person}做了什么",
        "{person}在{place}干嘛",
        "{person}{verb}了什么",
        "最近{person}怎么样",
        "关于{obj}的信息",
        "{person}在哪里",
        "{person}感觉如何",
    ]
    template = _rand_choice(templates)
    return template.format(
        person=_rand_choice(_PERSONS),
        verb=_rand_choice(_VERBS),
        obj=_rand_choice(_OBJECTS),
        place=_rand_choice(_PLACES),
    )


# ── RSS helper (optional) ────────────────────────────────────────────────

def _rss_mb() -> float | None:
    try:
        import resource  # Unix-only
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS returns bytes, Linux returns KB
        if sys.platform == "darwin":
            return rss / (1024 * 1024)
        return rss / 1024.0
    except ImportError:
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            return None


# ── Main ──────────────────────────────────────────────────────────────────

def run_soak(
    n_memories: int = 10_000,
    n_queries: int = 500,
    batch_size: int = 100,
    seed: int = 42,
) -> dict:
    """Run the soak test and return a report dict."""
    random.seed(seed)

    rss_start = _rss_mb()

    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        t0 = time.perf_counter()
        store = MemoryStore(path=tmp.name, enable_vector=False)

        # ── Phase 1: Bulk insert ──
        save_latencies: list[float] = []
        items: list[tuple[str, str, dict | None]] = []

        for i in range(1, n_memories + 1):
            text = generate_memory()
            items.append((text, "conversation", None))

            if len(items) >= batch_size or i == n_memories:
                t_batch = time.perf_counter()
                store.save_batch(items, namespace="soak")
                elapsed = (time.perf_counter() - t_batch) * 1000
                save_latencies.append(elapsed / len(items))  # per-item amortized
                items = []

            if i % (n_memories // 10) == 0:
                elapsed_total = time.perf_counter() - t0
                print(f"  [{i}/{n_memories}] inserted in {elapsed_total:.1f}s", file=sys.stderr)

        insert_total_s = time.perf_counter() - t0
        store.conn.commit()

        # ── Phase 2: Random queries ──
        query_latencies: list[float] = []
        t_query_start = time.perf_counter()

        for i in range(n_queries):
            q = generate_query()
            t_q = time.perf_counter()
            results = store.recall(q, top_k=5, namespace="soak")
            elapsed = (time.perf_counter() - t_q) * 1000
            query_latencies.append(elapsed)

            if (i + 1) % (n_queries // 5) == 0:
                print(f"  [{i+1}/{n_queries}] queries run", file=sys.stderr)

        query_total_s = time.perf_counter() - t_query_start

        store.close()

    rss_end = _rss_mb()

    # ── Report ──
    def quantile(data: list[float], q: float) -> float:
        s = sorted(data)
        idx = int(len(s) * q)
        return s[min(idx, len(s) - 1)]

    return {
        "n_memories": n_memories,
        "n_queries": n_queries,
        "batch_size": batch_size,
        "insert_total_s": round(insert_total_s, 2),
        "insert_rate_mem_per_s": round(n_memories / insert_total_s, 1),
        "insert_p50_ms": round(quantile(save_latencies, 0.50), 3),
        "insert_p95_ms": round(quantile(save_latencies, 0.95), 3),
        "insert_p99_ms": round(quantile(save_latencies, 0.99), 3),
        "query_total_s": round(query_total_s, 2),
        "query_rate_qps": round(n_queries / query_total_s, 1),
        "query_p50_ms": round(quantile(query_latencies, 0.50), 3),
        "query_p95_ms": round(quantile(query_latencies, 0.95), 3),
        "query_p99_ms": round(quantile(query_latencies, 0.99), 3),
        "query_mean_ms": round(statistics.mean(query_latencies), 3),
        "rss_start_mb": round(rss_start, 1) if rss_start else None,
        "rss_end_mb": round(rss_end, 1) if rss_end else None,
    }


def print_report(report: dict) -> None:
    print("=" * 60)
    print("EngramRouter L4 Soak Test Report")
    print("=" * 60)
    print(f"  Memories inserted : {report['n_memories']:,}")
    print(f"  Queries executed   : {report['n_queries']:,}")
    print(f"  Batch size         : {report['batch_size']}")
    print()
    print("  ── Insert ──")
    print(f"  Total time         : {report['insert_total_s']:.1f}s")
    print(f"  Rate               : {report['insert_rate_mem_per_s']:.1f} mem/s")
    print(f"  Per-item P50       : {report['insert_p50_ms']:.3f}ms")
    print(f"  Per-item P95       : {report['insert_p95_ms']:.3f}ms")
    print(f"  Per-item P99       : {report['insert_p99_ms']:.3f}ms")
    print()
    print("  ── Recall ──")
    print(f"  Total time         : {report['query_total_s']:.1f}s")
    print(f"  Rate               : {report['query_rate_qps']:.1f} qps")
    print(f"  P50 latency        : {report['query_p50_ms']:.3f}ms")
    print(f"  P95 latency        : {report['query_p95_ms']:.3f}ms")
    print(f"  P99 latency        : {report['query_p99_ms']:.3f}ms")
    print(f"  Mean latency       : {report['query_mean_ms']:.3f}ms")
    print()
    if report["rss_end_mb"]:
        print("  ── Memory ──")
        print(f"  RSS start          : {report['rss_start_mb']:.1f}MB")
        print(f"  RSS end            : {report['rss_end_mb']:.1f}MB")
        print(f"  RSS delta          : {report['rss_end_mb'] - report['rss_start_mb']:.1f}MB")
    print("=" * 60)

    # Quality check
    p99 = report["query_p99_ms"]
    if p99 < 10:
        print("✅ P99 < 10ms — excellent (well within interactive budget)")
    elif p99 < 50:
        print("✅ P99 < 50ms — good (acceptable for interactive use)")
    elif p99 < 200:
        print("⚠️  P99 < 200ms — acceptable (noticeable lag, consider tuning)")
    else:
        print("❌ P99 ≥ 200ms — needs investigation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EngramRouter L4 soak test")
    parser.add_argument("--n", type=int, default=10_000, help="Number of memories to insert")
    parser.add_argument("--queries", type=int, default=500, help="Number of recall queries")
    parser.add_argument("--batch-size", type=int, default=100, help="Save batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Starting soak test: {args.n:,} memories, {args.queries} queries...", file=sys.stderr)
    report = run_soak(
        n_memories=args.n,
        n_queries=args.queries,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print_report(report)
