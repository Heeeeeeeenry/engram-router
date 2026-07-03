"""Reciprocal Rank Fusion (RRF) for multi-path retrieval merging.

RRF combines ranked results from multiple retrieval paths (vector, keyword,
graph) into a single ranking, without requiring score calibration.

Formula: RRF(doc) = Σ 1 / (k + rank_i(doc))

Where:
  - k: constant (default 60),
  - rank_i(doc): 1-based rank of doc in result list i
"""

from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[tuple[str, float]]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Merge multiple ranked result lists using RRF.

    Args:
        result_lists: Each list is [(id, score), ...] in order of relevance.
                      Scores are unused in RRF — only ranks matter.
        k: Damping constant. Higher k reduces the effect of high ranks.
           Typical values: 60 (default), 0 (no damping).
        weights: Optional per-list weight multipliers. Default: all 1.0.

    Returns:
        Merged list of (id, rrf_score) sorted by score DESC.
    """
    if weights is None:
        weights = [1.0] * len(result_lists)

    # Accumulate RRF scores
    scores: dict[str, float] = {}
    for w, results in zip(weights, result_lists):
        for rank, (doc_id, _) in enumerate(results):
            rrf = w / (k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf

    # Sort by score descending
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged


def weighted_score_fusion(
    keyword_results: list[tuple[str, float]],
    vector_results: list[tuple[str, float]],
    keyword_weight: float = 0.4,
    vector_weight: float = 0.6,
) -> list[tuple[str, float]]:
    """Fuse two result sets by weighted score sum.

    Unlike RRF (which ignores scores), this method uses the actual scores
    from each retrieval path, normalizes them, and computes a weighted sum.

    Better when both paths produce meaningful, comparable scores.
    """
    # Normalize keyword scores to [0, 1]
    kw_scores = dict(keyword_results)
    vec_scores = dict(vector_results)

    kw_max = max(kw_scores.values()) if kw_scores else 1.0
    vec_max = max(vec_scores.values()) if vec_scores else 1.0

    merged: dict[str, float] = {}
    for doc_id in set(kw_scores) | set(vec_scores):
        kw_norm = kw_scores.get(doc_id, 0.0) / max(kw_max, 0.001)
        vec_norm = vec_scores.get(doc_id, 0.0) / max(vec_max, 0.001)
        merged[doc_id] = keyword_weight * kw_norm + vector_weight * vec_norm

    return sorted(merged.items(), key=lambda x: x[1], reverse=True)
