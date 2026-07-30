"""Recall pipeline: RecallContext + RecallStage Protocol + stage implementations.

The pipeline module provides a composable, traceable wrapper around the
existing recall free functions in ``recall.py``, ``scoring.py``,
``candidates.py``, and ``graph.py``.  Each stage implements the
``RecallStage`` Protocol and receives a ``RecallContext`` that accumulates
state as it flows through the chain.

Strategies:
  - ``"standard"`` -- QueryPrep + FTS + Edge + Scoring + Response
  - ``"rrf"`` -- QueryPrep + FTS + Edge + Scoring + Response
  - ``"fallback"`` -- QueryPrep + FTS + Scoring + Response (no edge expansion)
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from . import candidates
from . import graph
from . import recall as _recall
from . import scoring
from ..entities import extract_entities
from .records import MemoryRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RecallContext
# ---------------------------------------------------------------------------


@dataclass
class RecallContext:
    """Mutable pipeline state shared across stages."""

    query: str
    top_k: int = 5
    namespace: str = "default"
    strategy: str = "standard"

    # Query prep outputs
    terms: list[str] = field(default_factory=list)
    query_entity_objs: list[dict[str, Any]] = field(default_factory=list)
    query_entities: set[str] = field(default_factory=set)
    query_topics: set[str] = field(default_factory=set)
    query_identity_subjects: set[str] = field(default_factory=set)
    corrected_ids: set[str] = field(default_factory=set)

    # FTS outputs
    fts_ids: set[str] | None = None
    rows: list[Any] = field(default_factory=list)
    entity_map: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    # Edge expansion
    edge_bonus: dict[str, tuple[float, str]] = field(default_factory=dict)

    # Scored candidates (raw)
    scored: list[tuple[float, str, Any]] = field(default_factory=list)

    # Final results
    records: list[MemoryRecord] = field(default_factory=list)

    # Debug / trace
    debug: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RecallStage Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class RecallStage(Protocol):
    """A single stage in the recall pipeline.

    Each stage takes a ``MemoryStore`` (or compatible object) and a
    ``RecallContext``, mutates the context in place, and returns it.
    """

    def __call__(
        self, store: Any, ctx: RecallContext
    ) -> RecallContext:
        ...


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


class QueryPrepStage:
    """Tokenise the query and extract entities and intent signals."""

    @property
    def name(self) -> str:
        return "query_prep"

    def __call__(
        self, store: Any, ctx: RecallContext
    ) -> RecallContext:
        ctx.terms = list(dict.fromkeys(scoring.terms(ctx.query)))
        ctx.query_entity_objs = [
            e
            for e in extract_entities(ctx.query)
            if e.get("kind") != "cjk_ngram"
        ]
        ctx.query_entities = {e["name"] for e in ctx.query_entity_objs}
        ctx.query_topics = {
            e["name"]
            for e in ctx.query_entity_objs
            if e["kind"] == "topic"
        }
        ctx.query_identity_subjects = _recall._identity_subjects(
            ctx.query_entity_objs
        )
        ctx.corrected_ids = _recall._get_corrected_ids(store.conn)
        ctx.debug.setdefault("stages", []).append({
            "stage": self.name,
            "n_terms": len(ctx.terms),
            "n_entities": len(ctx.query_entity_objs),
            "elapsed_ms": 0.0,
        })
        return ctx


class FTSCandidateStage:
    """FTS5 trigram candidate selection + memory row fetch + entity map."""

    @property
    def name(self) -> str:
        return "fts_candidate"

    def __call__(
        self, store: Any, ctx: RecallContext
    ) -> RecallContext:
        fts_enabled = getattr(store, "_fts_enabled", True)
        ctx.fts_ids = candidates.fts_candidates(
            store.conn, fts_enabled, ctx.query, ctx.terms, ctx.namespace,
        )
        ctx.rows = candidates.memory_rows(
            store.conn, store.weights, ctx.fts_ids, ctx.namespace,
        )
        if ctx.rows:
            ctx.entity_map = candidates.entities_for_memories(
                store.conn, [r["id"] for r in ctx.rows],
            )
        ctx.debug.setdefault("stages", []).append({
            "stage": self.name,
            "n_candidates": len(ctx.rows),
            "fts_ids_count": len(ctx.fts_ids) if ctx.fts_ids else 0,
            "elapsed_ms": 0.0,
        })
        return ctx


class EdgeExpansionStage:
    """One-hop edge expansion using the typed edges table."""

    @property
    def name(self) -> str:
        return "edge_expansion"

    def __call__(
        self, store: Any, ctx: RecallContext
    ) -> RecallContext:
        if not ctx.rows:
            ctx.debug.setdefault("stages", []).append({
                "stage": self.name,
                "n_expanded": 0,
                "elapsed_ms": 0.0,
            })
            return ctx
        ctx.edge_bonus = graph.edge_expansion(
            store.conn, weights=store.weights,
            query=ctx.query, terms=ctx.terms, rows=ctx.rows,
            entity_map=ctx.entity_map, namespace=ctx.namespace,
        )
        missing_edge_ids = sorted(
            set(ctx.edge_bonus) - {r["id"] for r in ctx.rows}
        )
        if missing_edge_ids:
            ctx.rows.extend(
                candidates.rows_by_ids(
                    store.conn, missing_edge_ids, namespace=ctx.namespace,
                )
            )
            ctx.entity_map = candidates.entities_for_memories(
                store.conn, [r["id"] for r in ctx.rows],
            )
        ctx.debug.setdefault("stages", []).append({
            "stage": self.name,
            "n_expanded": len(ctx.edge_bonus),
            "n_missing": len(missing_edge_ids),
            "elapsed_ms": 0.0,
        })
        return ctx


class ScoringStage:
    """Score every candidate through the composable scoring pipeline."""

    @property
    def name(self) -> str:
        return "scoring"

    def __call__(
        self, store: Any, ctx: RecallContext
    ) -> RecallContext:
        if not ctx.rows:
            ctx.debug.setdefault("stages", []).append({
                "stage": self.name,
                "n_scored": 0,
                "elapsed_ms": 0.0,
            })
            return ctx
        ctx.scored = _recall.build_scored_candidates(
            store, query=ctx.query, terms=ctx.terms, rows=ctx.rows,
            entity_map=ctx.entity_map, edge_bonus=ctx.edge_bonus,
            fts_ids=ctx.fts_ids, corrected_ids=ctx.corrected_ids,
            query_entities=ctx.query_entities, query_topics=ctx.query_topics,
            query_identity_subjects=ctx.query_identity_subjects,
            query_entity_objs=ctx.query_entity_objs,
        )
        ctx.debug.setdefault("stages", []).append({
            "stage": self.name,
            "n_scored": len(ctx.scored),
            "elapsed_ms": 0.0,
        })
        return ctx


class ResponseStage:
    """Build the final MemoryRecord list."""

    @property
    def name(self) -> str:
        return "response"

    def __call__(
        self, store: Any, ctx: RecallContext
    ) -> RecallContext:
        ctx.records = _recall.build_recall_response(
            store, ctx.scored, ctx.top_k, ctx.query,
            namespace=ctx.namespace,
        )
        ctx.debug.setdefault("stages", []).append({
            "stage": self.name,
            "n_returned": len(ctx.records),
            "elapsed_ms": 0.0,
        })
        return ctx


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

_STANDARD_PIPELINE: list[RecallStage] = [
    QueryPrepStage(),
    FTSCandidateStage(),
    EdgeExpansionStage(),
    ScoringStage(),
    ResponseStage(),
]

# _RRF_PIPELINE is intentionally identical to _STANDARD_PIPELINE at present:
# the RRF fusion logic lives INSIDE build_recall_response(store, ...) rather
# than as a separate stage.  The alias exists so callers can request an "rrf"
# strategy explicitly (semantic hint) even though the current stage list is
# the same.  A future refactor may extract a dedicated FusionStage; keeping
# the alias reserves the name and avoids call-site churn later.
_RRF_PIPELINE: list[RecallStage] = _STANDARD_PIPELINE

_FALLBACK_PIPELINE: list[RecallStage] = [
    QueryPrepStage(),
    FTSCandidateStage(),
    ScoringStage(),
    ResponseStage(),
]


def build_pipeline(strategy: str = "standard") -> list[RecallStage]:
    """Build a recall pipeline for the given strategy.

    Args:
        strategy: One of ``"standard"``, ``"rrf"``, or ``"fallback"``.

    Returns:
        An ordered list of stages implementing ``RecallStage``.
    """
    if strategy == "standard":
        return [s for s in _STANDARD_PIPELINE]
    if strategy == "rrf":
        return [s for s in _RRF_PIPELINE]
    if strategy == "fallback":
        return [s for s in _FALLBACK_PIPELINE]
    raise ValueError(
        f"Unknown pipeline strategy: {strategy!r}. "
        f"Valid strategies: standard, rrf, fallback."
    )


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


def run_pipeline(
    store: Any,
    query: str,
    top_k: int = 5,
    namespace: str = "default",
    strategy: str = "standard",
) -> RecallContext:
    """Run a full pipeline end-to-end and return the final context."""
    stages = build_pipeline(strategy)
    ctx = RecallContext(
        query=query, top_k=top_k, namespace=namespace, strategy=strategy,
    )
    start = _time.perf_counter()
    for stage in stages:
        t0 = _time.perf_counter()
        ctx = stage(store, ctx)
        elapsed = (_time.perf_counter() - t0) * 1000
        # Update elapsed in the last debug entry
        if ctx.debug.get("stages"):
            ctx.debug["stages"][-1]["elapsed_ms"] = round(elapsed, 3)
    total_ms = (_time.perf_counter() - start) * 1000
    ctx.debug["total_elapsed_ms"] = round(total_ms, 3)
    return ctx
