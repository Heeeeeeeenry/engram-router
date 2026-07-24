"""Three-channel candidate retrieval for EngramRouter (L1.1).

Each channel retrieves candidates independently. Results are fused
downstream via one of the fusion methods.

Channel summary:
- ``fts_channel``: SQLite FTS5 trigram search + shared-entity + edge bonus
- ``vector_channel``: bi-encoder vector search (FAISS)
- ``hyde_channel``: HyDE (Hypothetical Document Embeddings) search
- ``run_channels``: runs all enabled channels and returns per-channel results
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def fts_channel(
    store: Any,
    query: str,
    top_n: int = 50,
) -> list[tuple[str, float, str, dict]]:
    """FTS + entity + edge channel: keyword-aware structured recall.

    Performs FTS5 trigram candidate retrieval, augments with shared-entity
    lookups and edge-expansion bonuses, then scores and ranks.

    Returns:
        List of ``(id, score, reason, metadata)`` tuples sorted by score DESC.
    """
    from . import candidates
    from . import graph
    from . import scoring
    from ..entities import extract_entities

    records: list[tuple[str, float, str, dict]] = []
    try:
        terms = scoring.terms(query)
        query_entity_objs = [
            e for e in extract_entities(query)
            if e.get("kind") != "cjk_ngram"
        ]
        namespace = "default"
        conn = store.conn
        _fts_enabled = getattr(store, '_fts_enabled', True)

        fts_ids = candidates.fts_candidates(
            conn, _fts_enabled, query, terms, namespace)

        rows = candidates.memory_rows(conn, store.weights, fts_ids, namespace)
        if not rows:
            return []

        entity_map = candidates.entities_for_memories(
            conn, [r["id"] for r in rows])
        edge_bonus = graph.edge_expansion(
            conn, weights=store.weights,
            query=query, terms=terms, rows=rows,
            entity_map=entity_map, namespace=namespace,
        )

        # Score each row.
        for row in rows:
            raw_text = row["raw_text"]
            summary = row["summary"] if row["summary"] else ""
            haystack = f"{raw_text} {summary}".lower()
            base = scoring.base_score(
                query, terms, haystack,
                store.weights, getattr(store, 'STOP_CHARS', frozenset()),
                getattr(store, 'REASON_MARKERS', ()),
                ranker=getattr(store, 'ranker', None),
                store=store,
            )
            reason = scoring.match_reason(terms, haystack, base)
            score = base

            # FTS provenance.
            if row["id"] in fts_ids:
                score += store.weights.fts_boost

            # Shared entities.
            mem_entity_objs = entity_map.get(row["id"], [])
            query_ents_set = {e["name"] for e in query_entity_objs
                              if e.get("kind") != "cjk_ngram"}
            mem_ents_set = {
                e["name"] for e in mem_entity_objs
                if e.get("kind") != "cjk_ngram"
            }
            shared = query_ents_set & mem_ents_set
            if shared:
                score += store.weights.shared_entity_multiplier * len(shared)
                reason += "; shared entities: " + ", ".join(sorted(shared))

            # Edge bonus.
            bonus = edge_bonus.get(row["id"])
            if bonus is not None:
                score += bonus[0] * store.weights.edge_assoc_boost

            if score > 0:
                metadata = {
                    "raw_text": raw_text[:200],
                    "entity_count": len(mem_entity_objs),
                }
                records.append((row["id"], score, reason.strip("; "), metadata))

        records.sort(key=lambda x: x[1], reverse=True)
    except Exception as exc:
        logger.debug("fts_channel error: %s", exc)

    return records[:top_n]


def vector_channel(
    store: Any,
    query: str,
    top_n: int = 50,
) -> list[tuple[str, float, str, dict]]:
    """Vector bi-encoder channel: dense semantic retrieval.

    Uses the store's embedding engine and FAISS vector index for pure
    semantic (embedding-space) nearest-neighbour search.

    Returns:
        List of ``(id, score, reason, metadata)`` tuples sorted by score DESC.
    """
    records: list[tuple[str, float, str, dict]] = []
    try:
        embedding_engine = getattr(store, 'embedding_engine', None)
        vector_index = getattr(store, 'vector_index', None)
        if embedding_engine is None or vector_index is None:
            return []

        vec = embedding_engine.encode(query)
        if vec is None:
            return []

        results = vector_index.search(vec, k=top_n)
        for mid, sim in results:
            metadata = {"similarity": float(sim)}
            records.append((mid, float(sim), "vector-semantic", metadata))

        records.sort(key=lambda x: x[1], reverse=True)
    except Exception as exc:
        logger.debug("vector_channel error: %s", exc)

    return records[:top_n]


def hyde_channel(
    store: Any,
    query: str,
    top_n: int = 50,
) -> list[tuple[str, float, str, dict]]:
    """HyDE channel: hypothetical-document retrieval.

    Generates hypothetical answer documents via an LLM, then encodes them
    and searches the vector index. Falls back gracefully when HyDE is
    disabled or unavailable.

    Returns:
        List of ``(id, score, reason, metadata)`` tuples sorted by score DESC.
    """
    records: list[tuple[str, float, str, dict]] = []
    try:
        hyde = getattr(store, 'hyde', None)
        embedding_engine = getattr(store, 'embedding_engine', None)
        vector_index = getattr(store, 'vector_index', None)

        if hyde is None or not getattr(hyde, "available", False):
            return []
        if embedding_engine is None or vector_index is None:
            return []

        hyde_list, hyde_result = hyde.expand_and_recall(
            query, embedding_engine, vector_index, k=top_n,
        )

        if hyde_result and hyde_result.hypotheses:
            for mid, score in hyde_list:
                records.append((
                    mid,
                    float(score),
                    "hyde-hypothetical",
                    {"source": hyde_result.source},
                ))

        records.sort(key=lambda x: x[1], reverse=True)
    except Exception as exc:
        logger.debug("hyde_channel error: %s", exc)

    return records[:top_n]


def run_channels(
    store: Any,
    query: str,
    top_n: int = 50,
) -> list[tuple[str, list[tuple[str, float, str, dict]]]]:
    """Run all enabled channels and return per-channel results.

    Args:
        store: ``MemoryStore`` instance.
        query: Query string.
        top_n: Max candidates per channel (default 50).

    Returns:
        List of ``(channel_name, candidates)`` tuples, one per channel that
        returned results. Each ``candidates`` is a list of
        ``(id, score, reason, metadata)``.
    """
    results: list[tuple[str, list[tuple[str, float, str, dict]]]] = []

    fts = fts_channel(store, query, top_n=top_n)
    if fts:
        results.append(("fts", fts))

    vec = vector_channel(store, query, top_n=top_n)
    if vec:
        results.append(("vector", vec))

    hyde = hyde_channel(store, query, top_n=top_n)
    if hyde:
        results.append(("hyde", hyde))

    return results
