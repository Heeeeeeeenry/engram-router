"""L4 microbenchmarks for core operations.

Run:
    pytest tests/test_bench_l4.py -v          # full benchmark
    pytest tests/test_bench_l4.py -v -k save  # save only

Requires: pytest-benchmark (pip install pytest-benchmark)
"""

from __future__ import annotations

import tempfile

import pytest

from engram_router import MemoryStore

# Try importing pytest-benchmark; gracefully skip if absent
try:
    import pytest_benchmark  # noqa: F401
    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False

pytestmark = pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")


@pytest.fixture
def store():
    """In-memory store with vector disabled."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        s = MemoryStore(path=tmp.name, enable_vector=False)
        yield s
        s.close()


@pytest.fixture
def populated_store(store):
    """Store with 100 memories pre-loaded."""
    for i in range(100):
        store.save(f"张三{i%3}在北京工作，喜欢钓鱼和爬山，最近在学Python开发", namespace="bench")
    store.conn.commit()
    return store


# ── Save ──────────────────────────────────────────────────────────────

def test_bench_save_single(benchmark, store):
    """Benchmark single save() call."""
    result = benchmark(
        store.save,
        "张三今年30岁在北京工作，是个程序员",
        namespace="bench",
    )
    assert result.startswith("mem_")


def test_bench_save_batch_10(benchmark, store):
    """Benchmark save_batch with 10 items."""
    items = [(
        f"张三{i}在北京工作，喜欢钓鱼爬山",
        "conversation",
        None,
    ) for i in range(10)]

    result = benchmark(store.save_batch, items, namespace="bench")
    assert len(result) == 10


# ── Recall ────────────────────────────────────────────────────────────

def test_bench_recall_simple(benchmark, populated_store):
    """Benchmark recall with simple query."""
    result = benchmark(
        populated_store.recall,
        "张三的爱好",
        top_k=5,
        namespace="bench",
    )
    assert len(result) > 0


def test_bench_recall_complex(benchmark, populated_store):
    """Benchmark recall with multi-term query."""
    result = benchmark(
        populated_store.recall,
        "张三在北京做什么工作喜欢什么",
        top_k=10,
        namespace="bench",
    )
    assert len(result) > 0


# ── Delete ────────────────────────────────────────────────────────────

def test_bench_delete(benchmark, store):
    """Benchmark delete."""
    mid = store.save("要删除的记忆", namespace="bench")
    result = benchmark(store.delete, mid, namespace="bench")
    assert result is True


# ── Consolidate ───────────────────────────────────────────────────────

def test_bench_consolidate(benchmark, populated_store):
    """Benchmark entity consolidation."""
    result = benchmark(populated_store.consolidate, namespace="bench")
    assert "merged_entities" in result


# ── End-to-end ────────────────────────────────────────────────────────

def test_bench_e2e_save_recall_delete(benchmark, store):
    """Benchmark full save→recall→delete cycle."""
    def cycle():
        mid = store.save("张三在北京工作", namespace="bench")
        results = store.recall("张三在哪", top_k=3, namespace="bench")
        store.delete(mid, namespace="bench")
        return len(results) > 0

    result = benchmark(cycle)
    assert result is True


def test_bench_maintenance(benchmark, populated_store):
    """Benchmark run_maintenance()."""
    result = benchmark(populated_store.run_maintenance, namespace="bench")
    assert isinstance(result, dict)
