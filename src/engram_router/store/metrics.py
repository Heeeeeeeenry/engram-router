"""Prometheus metrics stubs for EngramRouter store pipeline.

All counters/histograms degrade gracefully when ``prometheus_client`` is
not installed.  When the library IS installed, real metrics are registered
and accessible via the default collector registry.

Usage::

    from engram_router.store.metrics import recall_latency_ms, stage_hit_ratio
    recall_latency_ms.observe(12.5)
    stage_hit_ratio.labels(stage="fts_candidate").inc()
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful import
# ---------------------------------------------------------------------------

_prom_available = False
try:
    import prometheus_client  # noqa: F401

    _prom_available = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


class _NoopMetric:
    """Metric stub that silently eats all calls."""

    def observe(self, amount: float, **kwargs: Any) -> None:  # noqa: D401
        pass

    def inc(self, amount: float = 1, **kwargs: Any) -> None:
        pass

    def set(self, value: float, **kwargs: Any) -> None:
        pass

    def labels(self, **kwargs: Any) -> "_NoopMetric":
        return self

    def time(self) -> "_NoopTimer":
        return _NoopTimer()


class _NoopHistogram(_NoopMetric):
    pass


class _NoopCounter(_NoopMetric):
    pass


class _NoopGauge(_NoopMetric):
    pass


class _NoopTimer:
    def __enter__(self) -> "_NoopTimer":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if _prom_available:
    import prometheus_client

    recall_latency_ms: Any = prometheus_client.Histogram(
        "engram_recall_latency_ms",
        "Total recall pipeline latency in milliseconds",
        buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500),
    )

    stage_hit_ratio: Any = prometheus_client.Counter(
        "engram_stage_hit_count",
        "Per-stage item counts (candidates returned, scored, etc.)",
        ["stage", "outcome"],
    )

    save_dedup_ratio: Any = prometheus_client.Gauge(
        "engram_save_dedup_ratio",
        "Ratio of deduplicated saves (1.0 = all unique, 0 = all dupes)",
    )

    # N2: dim mismatch counter for silent-data-loss telemetry
    dropped_vector_count: Any = prometheus_client.Counter(
        "engram_dropped_vector_count",
        "Vectors rejected due to dim mismatch or index error",
        ["reason"],
    )

    # N7: stage skipped counter (deadline budget exceeded)
    stage_skipped_count: Any = prometheus_client.Counter(
        "engram_stage_skipped_count",
        "Recall stages skipped due to deadline budget",
        ["stage"],
    )
else:
    recall_latency_ms = _NoopHistogram()
    stage_hit_ratio = _NoopCounter()
    save_dedup_ratio = _NoopGauge()
    dropped_vector_count = _NoopCounter()
    stage_skipped_count = _NoopCounter()

    logger.debug(
        "prometheus_client not installed; metrics are no-ops. "
        "Install with: pip install prometheus-client"
    )


# ---------------------------------------------------------------------------
# HTTP exposition (N1)
# ---------------------------------------------------------------------------

def start_metrics_server(port: int = 9090, addr: str = "0.0.0.0") -> bool:
    """Start prometheus /metrics HTTP endpoint. Returns True on success.

    Called from cli.py (server mode) and mcp_server.py at startup. Silently
    no-ops if prometheus_client is not installed or the port is taken.
    """
    if not _prom_available:
        logger.info("Metrics HTTP disabled (prometheus_client not installed)")
        return False
    try:
        import prometheus_client
        prometheus_client.start_http_server(port, addr=addr)
        logger.info("Prometheus /metrics listening on %s:%d", addr, port)
        return True
    except OSError as exc:
        # Port taken / permission — fail-open (metrics still collected in-proc)
        logger.warning("Metrics HTTP failed to bind %s:%d — %s", addr, port, exc)
        return False
    except Exception as exc:
        logger.warning("Metrics HTTP startup failed: %s", exc)
        return False
