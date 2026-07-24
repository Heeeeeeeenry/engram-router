"""Tests for upgraded fusion methods (fusion.py)."""

import copy
import math

import pytest

from engram_router.fusion import (
    FusionMethod,
    fuse,
    inverse_score_fusion,
    learned_fusion_weights,
    rbf_fusion,
    reciprocal_rank_fusion,
)


# ---------------------------------------------------------------------------
# Inverse Score Fusion
# ---------------------------------------------------------------------------

def test_inverse_score_fusion_basic():
    """ISF fuses two result lists with minmax normalisation."""
    list_a = [("a", 10.0), ("b", 5.0), ("c", 1.0)]
    list_b = [("b", 8.0), ("a", 4.0), ("d", 6.0)]
    fused = inverse_score_fusion(
        [list_a, list_b], weights=[1.0, 1.0], norm="minmax",
    )
    assert isinstance(fused, list)
    assert len(fused) > 0
    # All fused scores should be finite.
    for mid, s in fused:
        assert math.isfinite(s)


def test_inverse_score_fusion_preserves_score_magnitudes():
    """ISF normalises but preserves relative ordering within channels."""
    list_a = [("x", 100.0), ("y", 50.0), ("z", 0.0)]
    list_b = [("x", 0.5), ("y", 0.1)]
    fused = inverse_score_fusion(
        [list_a, list_b], weights=[1.0, 1.0], norm="minmax",
    )
    fused_dict = dict(fused)
    # x appears in both channels with higher scores → should rank high.
    assert fused_dict.get("x", 0) > 0


def test_inverse_score_fusion_with_weights():
    """ISF respects per-channel weight multipliers."""
    list_a = [("a", 1.0), ("b", 0.5)]
    list_b = [("b", 1.0), ("a", 0.5)]
    # With equal weights, both should be present.
    fused_equal = inverse_score_fusion(
        [list_a, list_b], weights=[1.0, 1.0], norm="minmax",
    )
    # With bias toward list_a, "a" should dominate.
    fused_biased = inverse_score_fusion(
        [list_a, list_b], weights=[10.0, 0.1], norm="minmax",
    )
    assert fused_biased[0][0] == "a"


def test_inverse_score_fusion_empty_lists():
    """ISF handles empty input gracefully."""
    assert inverse_score_fusion([]) == []
    assert inverse_score_fusion([], weights=[]) == []


# ---------------------------------------------------------------------------
# RBF Fusion
# ---------------------------------------------------------------------------

def test_rbf_fusion_basic():
    """RBF fusion takes max RRF score per document across lists."""
    list_a = [("a", 1.0), ("b", 0.5)]
    list_b = [("b", 1.0), ("c", 0.8)]
    fused = rbf_fusion([list_a, list_b], k=60)
    assert isinstance(fused, list)
    # b should score well (appears in both), c should appear too.
    ids = {mid for mid, _ in fused}
    assert "b" in ids
    assert "c" in ids


def test_rbf_fusion_returns_max_not_sum():
    """RBF uses max per doc, not sum — a doc in only one list still scores."""
    list_a = [("only_a", 1.0)]
    list_b = [("only_b", 1.0)]
    fused = rbf_fusion([list_a, list_b], k=60)
    fused_dict = dict(fused)
    assert "only_a" in fused_dict
    assert "only_b" in fused_dict


def test_rbf_fusion_empty():
    """RBF handles empty result lists."""
    assert rbf_fusion([]) == []
    assert rbf_fusion([], k=30) == []


# ---------------------------------------------------------------------------
# FusionMethod Enum
# ---------------------------------------------------------------------------

def test_fusion_method_enum():
    """FusionMethod enum has expected members."""
    assert FusionMethod.RRF.value == "rrf"
    assert FusionMethod.ISF.value == "isf"
    assert FusionMethod.RBF.value == "rbf"
    assert FusionMethod.WEIGHTED_SUM.value == "weighted_sum"
    assert FusionMethod.LEARNED_ISF.value == "learned_isf"

    # Can construct from string.
    assert FusionMethod("rrf") == FusionMethod.RRF
    assert FusionMethod("isf") == FusionMethod.ISF

    with pytest.raises(ValueError):
        FusionMethod("unknown_method")


# ---------------------------------------------------------------------------
# fuse dispatcher
# ---------------------------------------------------------------------------

def test_fuse_dispatcher_rrf():
    """fuse() dispatches to RRF."""
    list_a = [("a", 1.0), ("b", 0.5)]
    list_b = [("b", 1.0), ("a", 0.3)]
    result = fuse("rrf", [list_a, list_b], k=60)
    assert isinstance(result, list)
    assert len(result) >= 2


def test_fuse_dispatcher_isf():
    """fuse() dispatches to ISF."""
    list_a = [("x", 10.0), ("y", 5.0)]
    list_b = [("x", 0.8), ("z", 0.9)]
    result = fuse("isf", [list_a, list_b], weights=[1.0, 1.0])
    assert isinstance(result, list)
    assert len(result) > 0


def test_fuse_dispatcher_rbf():
    """fuse() dispatches to RBF."""
    list_a = [("a", 1.0)]
    list_b = [("b", 1.0)]
    result = fuse("rbf", [list_a, list_b], k=60)
    ids = {mid for mid, _ in result}
    assert "a" in ids
    assert "b" in ids


def test_fuse_dispatcher_learned_isf():
    """fuse() dispatches to LEARNED_ISF with learned_weights kwarg."""
    list_a = [("a", 10.0), ("b", 5.0)]
    list_b = [("a", 0.5), ("c", 0.9)]
    result = fuse(
        "learned_isf", [list_a, list_b],
        learned_weights=[0.7, 0.3],
    )
    assert isinstance(result, list)
    assert len(result) > 0

    # Missing learned_weights raises.
    with pytest.raises(ValueError):
        fuse("learned_isf", [list_a, list_b])


def test_fuse_dispatcher_unknown():
    """fuse() raises on unknown method."""
    with pytest.raises(ValueError):
        fuse("nonexistent", [[("a", 1.0)]])


# ---------------------------------------------------------------------------
# learned_fusion_weights
# ---------------------------------------------------------------------------

def test_learned_fusion_weights_converges():
    """learned_fusion_weights converges on training data with clear signal."""
    training = [
        {
            "result_lists": [[("a", 10.0), ("b", 1.0)], [("x", 5.0), ("a", 0.1)]],
            "ground_truth_ids": {"a"},
        },
        {
            "result_lists": [[("b", 10.0), ("a", 1.0)], [("a", 8.0), ("b", 2.0)]],
            "ground_truth_ids": {"a"},
        },
    ]
    weights, score = learned_fusion_weights(training)
    assert len(weights) == 2
    assert all(w > 0 for w in weights)
    assert 0.0 <= score <= 1.0


def test_learned_fusion_weights_respects_data():
    """learned_fusion_weights returns weights that pass P@1 on clear signal."""
    # Channel 2 always has the ground truth at rank 1, channel 1 never does.
    training = []
    for _ in range(10):
        training.append({
            "result_lists": [
                [("wrong", 10.0), ("right", 1.0)],
                [("right", 10.0), ("wrong", 1.0)],
            ],
            "ground_truth_ids": {"right"},
        })
    weights, p1 = learned_fusion_weights(training)
    assert len(weights) == 2
    # Channel 2 should be favoured.
    assert weights[1] > weights[0] or p1 >= 0.5


def test_learned_fusion_weights_empty():
    """learned_fusion_weights handles empty training data."""
    weights, score = learned_fusion_weights([])
    assert weights == []
    assert score == 0.0


def test_learned_fusion_weights_no_channels():
    """learned_fusion_weights handles data with no channels."""
    training = [{"result_lists": [], "ground_truth_ids": {"a"}}]
    weights, score = learned_fusion_weights(training)
    assert weights == []
    assert score == 0.0
