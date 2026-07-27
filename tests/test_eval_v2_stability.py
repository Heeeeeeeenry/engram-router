"""Tests for L0.1 stability metrics in eval_v2.py."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Force deterministic recall path.
os.environ["ENGRAM_SKIP_VECTOR"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from engram_router.store import MemoryStore, MemoryRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(memories: list[str]) -> MemoryStore:
    """Create a temp-file store pre-populated with *memories*."""
    db_path = Path(tempfile.mkdtemp(prefix="test_stability_")) / "store.db"
    store = MemoryStore(path=db_path)
    for mem in memories:
        store.save(mem)
    return store


# ---------------------------------------------------------------------------
# compute_stability
# ---------------------------------------------------------------------------

def test_compute_stability_deterministic_store():
    """Stability should be high (near 1.0) for a deterministic store."""
    from eval_v2 import compute_stability

    store = _make_store([
        "The user's cat is named Whiskers.",
        "The user bought a blue bicycle in March.",
        "The user went to Tokyo in 2023.",
    ])

    query = "What is the user's cat's name?"
    target_ids = {"Whiskers"}

    result = compute_stability(store, query, target_ids, n_runs=10, top_k=5)

    # For a deterministic store, rank variance should be 0 and stability high.
    assert result["rank_variance"] == 0.0, f"Expected 0 variance, got {result}"
    assert result["rank_stability"] == 1.0, f"Expected rank_stability=1, got {result}"
    assert result["target_found_rate"] == 1.0, f"Expected found_rate=1, got {result}"
    assert result["jaccard_p1"] == 1.0, f"Expected J@1=1, got {result}"

    store.close()


def test_jaccard_similarity_empty_store():
    """Jaccard should be 1.0 when all runs return empty (trivial agreement)."""
    from eval_v2 import compute_stability

    store = _make_store([])
    result = compute_stability(store, "any query", {"nothing"}, n_runs=5, top_k=5)
    # jaccard_p1 should be 1.0 (empty vs empty = 1.0 agreement).
    assert result["jaccard_p1"] == 1.0, f"Got {result['jaccard_p1']}"
    store.close()


def test_rank_variance_on_unstable_results():
    """Simulate unstable results by patching store.recall to return varying ranks."""
    from eval_v2 import compute_stability

    store = _make_store(["memory A", "memory B", "memory C"])

    # Build a sequence of responses where target "A" moves around.
    responses = [
        # Run 0-1: A at pos 1
        [MockMem("mem0", "memory A", 0.9), MockMem("mem1", "memory B", 0.8), MockMem("mem2", "memory C", 0.7)],
        [MockMem("mem0", "memory A", 0.9), MockMem("mem1", "memory B", 0.8), MockMem("mem2", "memory C", 0.7)],
        # Run 2: A at pos 2
        [MockMem("mem1", "memory B", 0.9), MockMem("mem0", "memory A", 0.8), MockMem("mem2", "memory C", 0.7)],
        # Run 3: A at pos 2
        [MockMem("mem1", "memory B", 0.9), MockMem("mem0", "memory A", 0.8), MockMem("mem2", "memory C", 0.7)],
        # Run 4-6: A at pos 1
        [MockMem("mem0", "memory A", 0.9), MockMem("mem1", "memory B", 0.8), MockMem("mem2", "memory C", 0.7)],
        [MockMem("mem0", "memory A", 0.9), MockMem("mem1", "memory B", 0.8), MockMem("mem2", "memory C", 0.7)],
        [MockMem("mem0", "memory A", 0.9), MockMem("mem1", "memory B", 0.8), MockMem("mem2", "memory C", 0.7)],
        # Run 7: A not in top-5
        [MockMem("mem1", "memory B", 0.9), MockMem("mem2", "memory C", 0.8), MockMem("mem9", "other", 0.1)],
        # Run 8-9: A at pos 1
        [MockMem("mem0", "memory A", 0.9), MockMem("mem1", "memory B", 0.8), MockMem("mem2", "memory C", 0.7)],
        [MockMem("mem0", "memory A", 0.9), MockMem("mem1", "memory B", 0.8), MockMem("mem2", "memory C", 0.7)],
    ]

    call_count = [0]

    def fake_recall(query, top_k=5):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        return responses[idx]

    with patch.object(store, "recall", fake_recall):
        result = compute_stability(store, "query", {"memory A"}, n_runs=10, top_k=5)

    # A is at rank 1 in 7/10 runs (mem0 at pos 1), mem1 at pos 1 in 3/10.
    # rank_variance > 0, rank_stability < 1, target_found_rate = 9/10 = 0.9.
    assert result["target_found_rate"] == 0.9, f"Got {result}"
    assert result["rank_variance"] > 0, f"Expected positive variance, got {result}"
    assert result["rank_stability"] < 1.0, f"Expected < 1, got {result}"
    # Jaccard@1 is pairwise agreement on top-1 id.
    # 7 runs have "mem0" at top-1, 3 runs have "mem1" at top-1.
    # Agreements: 7C2=21 (mem0), 3C2=3 (mem1). Total pairs = 10C2 = 45.
    # J@1 = 24/45 ≈ 0.5333.
    expected_j1 = 24 / 45
    assert abs(result["jaccard_p1"] - expected_j1) < 0.01, (
        f"expected ~{expected_j1:.4f}, got {result['jaccard_p1']}"
    )

    store.close()


def test_empty_target_ids():
    """With no target IDs, stability metrics should all be zero / default."""
    from eval_v2 import compute_stability

    store = _make_store(["memory A", "memory B"])
    result = compute_stability(store, "query", set(), n_runs=5, top_k=5)
    assert result["rank_variance"] == 0.0
    assert result["rank_stability"] == 0.0
    assert result["target_found_rate"] == 0.0
    store.close()


def test_multiple_target_ids():
    """Test stability with multiple target IDs."""
    from eval_v2 import compute_stability

    store = _make_store([
        "The user likes apples.",
        "The user likes bananas.",
        "The user likes cherries.",
    ])

    result = compute_stability(
        store, "fruit preferences", {"bananas", "cherries"}, n_runs=5, top_k=5
    )
    assert 0.0 <= result["target_found_rate"] <= 1.0
    assert result["rank_variance"] >= 0.0
    store.close()


# ---------------------------------------------------------------------------
# Mock record for testing
# ---------------------------------------------------------------------------

class MockMem:
    """Minimal mock mimicking MemoryRecord for stability tests."""

    def __init__(self, id_: str, raw_text: str, score: float):
        self.id = id_
        self.raw_text = raw_text
        self.score = score
