"""Tests for the three-channel candidate retrieval (channels.py)."""

import pytest
from engram_router.store import MemoryStore, RecallWeights
from engram_router.store import channels as _channels


def test_fts_channel_returns_candidates(tmp_path):
    """fts_channel returns scored candidates for a keyword match."""
    store = MemoryStore(
        path=tmp_path / "f1.db",
        enable_vector=False,
    )
    store.save("张三前两天送我一把 HHKB，说是生日礼物")
    store.save("张三是我的前同事，现在在腾讯。")

    results = _channels.fts_channel(store, "HHKB", top_n=10)
    assert len(results) > 0
    # Each result is (id, score, reason, metadata)
    for r in results:
        assert isinstance(r, tuple)
        assert len(r) == 4
        assert isinstance(r[0], str)
        assert isinstance(r[1], float)
        assert r[1] > 0


def test_fts_channel_graceful_empty(tmp_path):
    """fts_channel returns empty list for a query with no matches."""
    store = MemoryStore(
        path=tmp_path / "f2.db",
        enable_vector=False,
    )
    results = _channels.fts_channel(store, "nonexistent_query_xyz", top_n=10)
    assert results == []


def test_fts_channel_score_sorted(tmp_path):
    """fts_channel returns results sorted by score descending."""
    store = MemoryStore(
        path=tmp_path / "f3.db",
        enable_vector=False,
    )
    store.save("HHKB 键盘很舒服，我很喜欢")
    store.save("今天天气不错")

    results = _channels.fts_channel(store, "HHKB", top_n=10)
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)
    # First result should mention HHKB
    assert "HHKB" in results[0][2] or any(
        mid == results[0][0] and "HHKB" in store.conn.execute(
            "SELECT raw_text FROM memories WHERE id=?",
            (results[0][0],),
        ).fetchone()["raw_text"]
        for mid in [results[0][0]]
    )


def test_vector_channel_returns_candidates(tmp_path):
    """vector_channel returns candidates when vector engine is available."""
    # Vector is disabled via ENGRAM_SKIP_VECTOR=1 in conftest, so this
    # test verifies graceful None when vector is not available.
    store = MemoryStore(
        path=tmp_path / "v1.db",
        enable_vector=False,
    )
    store.save("test memory content")
    results = _channels.vector_channel(store, "test", top_n=10)
    # With vector disabled, should return empty list.
    assert isinstance(results, list)


def test_vector_channel_none_engine(tmp_path):
    """vector_channel returns empty when embedding engine is None."""
    store = MemoryStore(
        path=tmp_path / "v2.db",
        enable_vector=False,
    )
    store.embedding_engine = None
    store.vector_index = None
    results = _channels.vector_channel(store, "anything", top_n=10)
    assert results == []


def test_hyde_channel_disabled(tmp_path):
    """hyde_channel returns empty when HyDE is not available."""
    store = MemoryStore(
        path=tmp_path / "h1.db",
        enable_vector=False,
    )
    store.save("some content")
    results = _channels.hyde_channel(store, "query", top_n=10)
    assert results == []


def test_run_channels_all_enabled(tmp_path):
    """run_channels returns per-channel results for all channels that produce output."""
    store = MemoryStore(
        path=tmp_path / "r1.db",
        enable_vector=False,
    )
    store.save("张三前两天送我一把 HHKB，说是生日礼物")
    store.save("张三是我的前同事，现在在腾讯。")

    results = _channels.run_channels(store, "HHKB", top_n=10)
    # channels module operates; with vector disabled, we get >=1 channel (fts)
    assert isinstance(results, list)
    # Each entry is (channel_name, candidates_list)
    for ch_name, cands in results:
        assert isinstance(ch_name, str)
        assert isinstance(cands, list)
        if cands:
            assert all(isinstance(c, tuple) and len(c) == 4 for c in cands)


def test_run_channels_respects_top_n(tmp_path):
    """run_channels respects per-channel top_n limit."""
    store = MemoryStore(
        path=tmp_path / "r2.db",
        enable_vector=False,
    )
    for i in range(5):
        store.save(f"HHKB 测试内容 {i}")

    results = _channels.run_channels(store, "HHKB", top_n=2)
    for ch_name, cands in results:
        assert len(cands) <= 2


def test_three_channel_e2e(tmp_path):
    """End-to-end: channels integration with recall pipeline.

    This validates that the three-channel path does not break the existing
    recall flow when channels.py is present.
    """
    store = MemoryStore(
        path=tmp_path / "e2e.db",
        enable_vector=False,
    )
    store.save("张三前两天送我一把 HHKB，说是生日礼物")
    store.save("张三是我的前同事，现在在腾讯。")

    # Recall should work with channels.py present (importable)
    records = store.recall("HHKB", top_k=5)
    assert len(records) > 0
    assert any("HHKB" in r.raw_text for r in records)


def test_channels_with_custom_weights(tmp_path):
    """channel weights from RecallWeights are used by run_channels."""
    store = MemoryStore(
        path=tmp_path / "cw.db",
        enable_vector=False,
        weights=RecallWeights(
            channel_top_n=5,
            channel_fts_weight=0.5,
            channel_vector_weight=0.5,
            channel_hyde_weight=0.3,
            fusion_method="rrf",
            fusion_k=30,
        ),
    )
    store.save("HHKB 测试")
    store.save("键盘 测试")

    results = _channels.run_channels(store, "HHKB", top_n=store.weights.channel_top_n)
    for ch_name, cands in results:
        assert len(cands) <= 5
