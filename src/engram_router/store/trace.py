"""Recall debug trace: per-stage input/output/top5/latency.

Wraps pipeline execution with structured trace recording for
observability and debugging.  Each stage produces a ``StageTrace``
with timing, summary counts, and a sample of top items.
"""

from __future__ import annotations

import time as _time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator


@dataclass
class StageTrace:
    """Snapshot of one pipeline stage's execution."""

    stage_name: str
    elapsed_ms: float = 0.0
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    top_items: list[Any] = field(default_factory=list)


class RecallTracer:
    """Wraps pipeline execution with per-stage tracing.

    Usage::

        tracer = RecallTracer()
        # Before/after each stage, call tracer or use the context manager.
    """

    def __init__(self) -> None:
        self._traces: list[StageTrace] = []
        self._stage_start: float = 0.0

    @contextmanager
    def trace_stage(self, stage_name: str) -> Generator[StageTrace, None, None]:
        """Context manager that records timing and returns the trace entry.

        The caller should populate ``input_summary``, ``output_summary``,
        and ``top_items`` on the yielded ``StageTrace`` before exiting the
        block.
        """
        trace = StageTrace(stage_name=stage_name)
        self._stage_start = _time.perf_counter()
        try:
            yield trace
        finally:
            trace.elapsed_ms = round(
                (_time.perf_counter() - self._stage_start) * 1000, 3,
            )
            self._traces.append(trace)

    def snapshot(self) -> list[StageTrace]:
        """Return a copy of all recorded stage traces."""
        return list(self._traces)

    def to_dict(self) -> dict[str, Any]:
        """Serializable representation of all traces."""
        total_ms = sum(t.elapsed_ms for t in self._traces)
        return {
            "total_elapsed_ms": round(total_ms, 3),
            "stages": [
                {
                    "name": t.stage_name,
                    "elapsed_ms": t.elapsed_ms,
                    "input": t.input_summary,
                    "output": t.output_summary,
                    "top_items": t.top_items,
                }
                for t in self._traces
            ],
        }

    def reset(self) -> None:
        """Clear all recorded traces."""
        self._traces.clear()
