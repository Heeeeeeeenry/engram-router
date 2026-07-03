"""Pytest-parameterized semantic audit tests.

Converts the standalone tests/semantic_audit.py scenarios into pytest
parametrized tests so they run as part of the normal test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engram_router.store import MemoryStore  # noqa: E402

# Reuse the SCENARIOS list from the existing audit script.
from tests.semantic_audit import SCENARIOS  # noqa: E402


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_semantic_scenario(scenario, tmp_path):
    store = MemoryStore(path=str(tmp_path / f"{scenario.id}.db"))
    for mem in scenario.store_memories:
        store.save(mem)

    # Verify top-3 recall
    results = store.recall(scenario.query, top_k=3)
    recalled = {r.raw_text for r in results}

    # At least one expected memory in top-3
    expected_texts = {scenario.store_memories[i] for i in scenario.expected_memory_idx}
    assert recalled & expected_texts, f"{scenario.id}: 期望记忆未召回"

    # Forbidden memories must not leak into top-3
    if hasattr(scenario, "forbidden_memory_idx") and scenario.forbidden_memory_idx:
        forbidden_texts = {scenario.store_memories[i] for i in scenario.forbidden_memory_idx}
        assert not (recalled & forbidden_texts), f"{scenario.id}: 禁止记忆泄漏"
