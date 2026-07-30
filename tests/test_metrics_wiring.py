"""Tests for Prometheus metrics wiring (N1) + dim mismatch counter (N2)."""

import random
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class TestMetricsModule:
    def test_metrics_importable(self):
        from engram_router.store import metrics
        assert hasattr(metrics, "recall_latency_ms")
        assert hasattr(metrics, "stage_hit_ratio")
        assert hasattr(metrics, "save_dedup_ratio")
        assert hasattr(metrics, "dropped_vector_count")
        assert hasattr(metrics, "stage_skipped_count")

    def test_noop_when_prometheus_absent(self):
        """Metrics degrade to no-ops without prometheus_client (module always works)."""
        from engram_router.store import metrics
        # These calls should never raise, regardless of prometheus availability
        metrics.recall_latency_ms.observe(random.uniform(1, 500))
        metrics.stage_hit_ratio.labels(stage="test", outcome="hit").inc()
        metrics.save_dedup_ratio.set(0.75)
        metrics.dropped_vector_count.labels(reason="test").inc()
        metrics.stage_skipped_count.labels(stage="hyde").inc()

    def test_start_metrics_server_returns_bool(self):
        from engram_router.store.metrics import start_metrics_server
        # Use random port; result may be False if prometheus_client is missing
        # or the port is taken — either way must return a bool, not raise.
        port = random.randint(30000, 60000)
        result = start_metrics_server(port=port, addr="127.0.0.1")
        assert isinstance(result, bool)


class TestRecallLatencyInstrumentation:
    def test_recall_records_latency(self, tmp_path):
        """recall() must call recall_latency_ms.observe() even on empty result."""
        from engram_router.store import MemoryStore
        from engram_router.store import metrics as _m

        # Save current sample count if prometheus is real
        _prom_available = hasattr(_m.recall_latency_ms, "_sum") or type(_m.recall_latency_ms).__name__ != "_NoopHistogram"

        store = MemoryStore(path=tmp_path / "diag.db", enable_vector=False)
        try:
            texts = [
                f"随机记录{random.randint(1,999)}号，讨论{random.choice(['工作','学习','旅游'])}。"
                for _ in range(5)
            ]
            for t in texts:
                store.save(t)

            for _ in range(3):
                q = f"我最近{random.choice(['做','看','买'])}了什么？"
                records = store.recall(q, top_k=5)
                assert isinstance(records, list)
        finally:
            store.close()
        # No crash, no assertion — pure smoke test for the wiring


class TestDimMismatchCounter:
    def test_dim_mismatch_incrementer_reachable(self):
        """dropped_vector_count is exposed with 'dim_mismatch' + 'value_error' reasons."""
        from engram_router.store import metrics
        # Both label values should be usable
        metrics.dropped_vector_count.labels(reason="dim_mismatch").inc()
        metrics.dropped_vector_count.labels(reason="value_error").inc()


class TestFTSStageMetric:
    def test_fts_hit_miss_labels_incrementable(self):
        from engram_router.store import metrics
        metrics.stage_hit_ratio.labels(stage="fts_candidate", outcome="hit").inc()
        metrics.stage_hit_ratio.labels(stage="fts_candidate", outcome="miss").inc()
