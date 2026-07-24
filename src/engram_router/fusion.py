"""Fusion methods for multi-path retrieval merging.

Provides Reciprocal Rank Fusion (RRF), Inverse Score Fusion (ISF),
Reciprocal Best Fusion (RBF), weighted sum, and learned fusion weight
optimization — all accessible through a unified `fuse()` dispatcher.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Sequence

logger = logging.getLogger(__name__)


class FusionMethod(str, Enum):
    """Supported fusion strategy identifiers."""

    RRF = "rrf"
    ISF = "isf"
    RBF = "rbf"
    WEIGHTED_SUM = "weighted_sum"
    LEARNED_ISF = "learned_isf"


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


def _minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Normalize scores to [0, 1] using min-max scaling."""
    vals = list(scores.values())
    if not vals:
        return scores
    mn, mx = min(vals), max(vals)
    denom = mx - mn if mx != mn else 1.0
    return {k: (v - mn) / denom for k, v in scores.items()}


def _softmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Normalize scores to [0, 1] using softmax."""
    import math

    vals = list(scores.values())
    if not vals:
        return scores
    exp_vals = {k: math.exp(v) for k, v in scores.items()}
    total = sum(exp_vals.values()) or 1.0
    return {k: v / total for k, v in exp_vals.items()}


def inverse_score_fusion(
    result_lists: Sequence[Sequence[tuple[str, float]]],
    weights: Sequence[float] | None = None,
    norm: str = "minmax",
) -> list[tuple[str, float]]:
    """Inverse Score Fusion — uses actual scores, not just ranks.

    Unlike RRF which ignores score magnitudes, ISF preserves relative
    confidence by normalising per-channel scores then summing weighted
    contributions.

    Args:
        result_lists: Each list is [(id, score), ...] — order is ignored,
                      only the scores matter.
        weights: Optional per-list weight multipliers. Default: all 1.0.
        norm: Normalisation type — ``"minmax"`` or ``"softmax"``.

    Returns:
        Merged list of (id, fused_score) sorted by score DESC.
    """
    if not result_lists:
        return []

    if weights is None:
        weights = [1.0] * len(result_lists)

    _normalize = _softmax_normalize if norm == "softmax" else _minmax_normalize

    fused: dict[str, float] = {}
    for w, results in zip(weights, result_lists):
        raw = dict(results)
        normed = _normalize(raw)
        for doc_id, ns in normed.items():
            fused[doc_id] = fused.get(doc_id, 0.0) + w * ns

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def rbf_fusion(
    result_lists: Sequence[Sequence[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Best Fusion — takes the *max* RRF score per doc.

    Instead of summing RRF contributions across channels (which favours
    recall), RBF takes the single best per-document RRF score.  This
    favours precision — a document ranks highly only if at least one
    channel ranked it highly.

    Args:
        result_lists: Each list is [(id, score), ...] in rank order.
        k: RRF damping constant (default 60).

    Returns:
        Merged list of (id, rbf_score) sorted by score DESC.
    """
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, (doc_id, _) in enumerate(results):
            rrf = 1.0 / (k + rank + 1)
            if doc_id not in scores or rrf > scores[doc_id]:
                scores[doc_id] = rrf

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def learned_fusion_weights(
    training_data: list[dict],
) -> tuple[list[float], float]:
    """Learn per-channel fusion weights via coordinate-ascent grid search.

    Iterates over each channel and tries candidate weight values, picking
    the one that maximises mean Precision@1 across the training set.

    Args:
        training_data: List of dicts, each with::

            {
                "result_lists": [[(id, score), ...], ...],  # per channel
                "ground_truth_ids": set[str],  # correct memory IDs
            }

    Returns:
        ``(weights, best_p1)`` where *weights* is a list of float weights
        (one per channel) and *best_p1* is the achieved mean P@1.
    """
    if not training_data:
        return [], 0.0

    candidates = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0]
    n_channels = len(training_data[0].get("result_lists", []))
    if n_channels == 0:
        return [], 0.0

    weights = [1.0] * n_channels

    def _p1(example: dict, w: list[float]) -> float:
        fused = inverse_score_fusion(example["result_lists"], weights=w, norm="minmax")
        if not fused:
            return 0.0
        return 1.0 if fused[0][0] in example["ground_truth_ids"] else 0.0

    def _mean_p1(w: list[float]) -> float:
        return sum(_p1(ex, w) for ex in training_data) / len(training_data)

    best_weights = list(weights)
    best_score = _mean_p1(best_weights)

    improved = True
    max_iter = 5
    while improved and max_iter > 0:
        improved = False
        for j in range(n_channels):
            best_w_j = weights[j]
            best_p1_j = _mean_p1(weights)
            for c in candidates:
                candidate_weights = list(weights)
                candidate_weights[j] = c
                p1 = _mean_p1(candidate_weights)
                if p1 > best_p1_j:
                    best_w_j = c
                    best_p1_j = p1
            if best_w_j != weights[j]:
                weights[j] = best_w_j
                improved = True
        best_score = _mean_p1(weights)
        max_iter -= 1

    return weights, best_score


def fuse(
    method: FusionMethod | str,
    result_lists: Sequence[Sequence[tuple[str, float]]],
    weights: Sequence[float] | None = None,
    **kwargs,
) -> list[tuple[str, float]]:
    """Unified fusion dispatcher.

    Args:
        method: Fusion strategy — see :class:`FusionMethod`.
        result_lists: Per-channel ranked result lists.
        weights: Optional per-channel weight multipliers (used by RRF,
                 ISF, and WEIGHTED_SUM).
        **kwargs: Method-specific options (``k`` for RRF/RBF,
                  ``norm`` for ISF, ``keyword_weight`` / ``vector_weight``
                  for WEIGHTED_SUM, ``kw_results`` / ``vec_results``).

    Returns:
        Merged list of ``(id, score)`` sorted by score DESC.
    """
    method = FusionMethod(method)

    if method == FusionMethod.RRF:
        return reciprocal_rank_fusion(
            result_lists, k=kwargs.get("k", 60), weights=weights,
        )
    if method == FusionMethod.ISF:
        return inverse_score_fusion(
            result_lists, weights=weights, norm=kwargs.get("norm", "minmax"),
        )
    if method == FusionMethod.RBF:
        return rbf_fusion(
            result_lists, k=kwargs.get("k", 60),
        )
    if method == FusionMethod.WEIGHTED_SUM:
        kw_results = kwargs.get("kw_results", result_lists[0] if result_lists else [])
        vec_results = kwargs.get("vec_results", result_lists[1] if len(result_lists) > 1 else [])
        return weighted_score_fusion(
            kw_results,
            vec_results,
            keyword_weight=kwargs.get("keyword_weight", 0.4),
            vector_weight=kwargs.get("vector_weight", 0.6),
        )
    if method == FusionMethod.LEARNED_ISF:
        learned_w: list[float] | None = kwargs.get("learned_weights")
        if learned_w is None:
            raise ValueError(
                "LEARNED_ISF requires `learned_weights` kwarg "
                "(see `learned_fusion_weights()`)."
            )
        return inverse_score_fusion(
            result_lists, weights=learned_w, norm=kwargs.get("norm", "minmax"),
        )
    raise ValueError(f"Unknown fusion method: {method}")
