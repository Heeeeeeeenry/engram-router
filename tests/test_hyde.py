"""Tests for HyDEExpander (Phase 2 rerank_and_hyde.md).

Fast/offline by default — HyDE is behind an LLM key so the real path is
gated behind ``ENGRAM_TEST_REAL_HYDE=1``. The routine unit tests use a
stub LLM client and a stub vector index to exercise every branch.
"""

from __future__ import annotations

import os
import pytest

from engram_router.hyde import (
    HyDEExpander,
    HyDEResult,
    _looks_negative,
    _parse_hypotheses,
)


# ── pure-function tests ────────────────────────────────────────────────

def test_looks_negative_polar_questions():
    assert _looks_negative("李四买特斯拉了吗")
    assert _looks_negative("张三是不是走了")
    assert _looks_negative("我有没有说过这句话")
    assert _looks_negative("这个方案不错嘛")


def test_looks_negative_negation_words():
    # "有没有…" is caught by the 有没有 rule
    assert _looks_negative("这件事我有没有告诉你")
    # A polar 不…吗 pair triggers via the negation-then-polar rule.
    assert _looks_negative("你不想去吗")


def test_looks_negative_positive_stays_positive():
    # "心情不好" is a fixed idiom for "sad" — HyDE should still fire on it.
    assert not _looks_negative("我为什么心情不好")
    assert not _looks_negative("张三送我的键盘是什么牌子")
    assert not _looks_negative("最近哪次去旅游")
    # "什么" ending must NOT trigger — it's info-seeking, not polar.
    assert not _looks_negative("我要吃什么")


def test_parse_hypotheses_direct_array():
    raw = '["一段", "二段", "三段"]'
    assert _parse_hypotheses(raw) == ["一段", "二段", "三段"]


def test_parse_hypotheses_markdown_fence():
    raw = "```json\n[\"a\", \"b\"]\n```"
    assert _parse_hypotheses(raw) == ["a", "b"]


def test_parse_hypotheses_prefix_prose():
    raw = "好的,以下是假想答案:\n[\"one\", \"two\"] end"
    assert _parse_hypotheses(raw) == ["one", "two"]


def test_parse_hypotheses_bad_input_returns_empty():
    assert _parse_hypotheses("") == []
    assert _parse_hypotheses("no json here") == []


# ── HyDEExpander unit tests ────────────────────────────────────────────

class _StubClient:
    """Minimal LLM client stub — mirrors LLMClient's public surface."""

    available = True

    def __init__(self, response: str = ""):
        self.response = response
        self.call_count = 0

    def chat(self, messages, temperature=0.0, max_tokens=400):
        self.call_count += 1
        return self.response


def _expander(response: str = '["A", "B", "C"]', **kw):
    ex = HyDEExpander(client=_StubClient(response), allow_cloud=True, **kw)
    return ex


def test_construction_defaults():
    ex = _expander()
    assert ex.available
    assert ex.cache_size == 0


def test_invalid_num_hypotheses_rejected():
    with pytest.raises(ValueError):
        HyDEExpander(client=_StubClient(), num_hypotheses=0)
    with pytest.raises(ValueError):
        HyDEExpander(client=_StubClient(), max_len=4)


def test_available_false_without_client():
    class _NoAvail:
        available = False

        def chat(self, *_, **__):
            raise AssertionError("must not call chat when unavailable")

    ex = HyDEExpander(client=_NoAvail(), allow_cloud=True)
    assert ex.available is False
    result = ex.generate("我为什么心情不好呀今天")
    assert result.source == "skipped"
    assert "LLM" in result.skip_reason


def test_available_false_when_cloud_disabled():
    ex = HyDEExpander(client=_StubClient(), allow_cloud=False)
    assert ex.available is False


def test_short_query_skipped():
    ex = _expander(min_query_chars=10)
    r = ex.generate("你好")
    assert r.source == "skipped"
    assert "chars" in r.skip_reason


def test_negative_query_skipped():
    ex = _expander()
    r = ex.generate("李四这个月买特斯拉了吗")
    assert r.source == "skipped"
    assert "negative" in r.skip_reason


def test_should_run_predicate_skip():
    ex = HyDEExpander(client=_StubClient(), allow_cloud=True,
                      should_run=lambda q: False)
    r = ex.generate("我为什么心情不好呀今天")
    assert r.source == "skipped"
    assert "should_run" in r.skip_reason


def test_generate_fresh_and_truncate():
    ex = _expander(response='["short", "' + "x" * 200 + '"]', max_len=20)
    r = ex.generate("我为什么心情不好呀今天")
    assert r.source == "fresh"
    assert len(r.hypotheses) == 2
    assert all(len(h) <= 20 for h in r.hypotheses)


def test_cache_hit_second_call():
    client = _StubClient(response='["a", "b", "c"]')
    ex = HyDEExpander(client=client, allow_cloud=True)
    q = "我为什么心情不好呀今天"
    r1 = ex.generate(q)
    assert r1.source == "fresh"
    assert client.call_count == 1
    r2 = ex.generate(q)
    assert r2.source == "cache"
    assert client.call_count == 1  # no new call


def test_cache_negative_result_stays_cached():
    """A parse failure should still be cached so we don't re-hit the LLM."""
    client = _StubClient(response="garbage no json")
    ex = HyDEExpander(client=client, allow_cloud=True)
    q = "这是一个复杂的问题需要多个字符"
    r1 = ex.generate(q)
    assert r1.hypotheses == []
    assert r1.source == "fresh"
    r2 = ex.generate(q)
    assert r2.hypotheses == []
    assert r2.source == "cache"
    assert client.call_count == 1


def test_llm_error_does_not_leak():
    class BoomClient:
        available = True

        def chat(self, *_, **__):
            raise RuntimeError("network down")

    ex = HyDEExpander(client=BoomClient(), allow_cloud=True)
    r = ex.generate("我为什么心情不好呀今天")
    assert r.hypotheses == []
    assert r.source == "fresh"  # attempted, no exception surfaced


# ── expand_and_recall integration with stub vector index ──────────────

class _StubEmbed:
    def encode(self, text):
        # Deterministic embed: hash → single-dim vector; sufficient for
        # exercising the wiring, not for meaningful similarity.
        return [len(text) % 7]


class _StubVectorIndex:
    def __init__(self, table: dict[str, list[tuple[str, float]]]):
        self.table = table
        self.queries: list[list] = []

    def search(self, vec, k=20):
        self.queries.append(vec)
        # Return the same list keyed by the embed value; falls back to empty.
        return list(self.table.get(str(vec[0]), []))[:k]


def test_expand_and_recall_merges_and_deduplicates():
    # Two hypotheses embed to the same "vector" here (same length modulo 7),
    # but the third pulls in a fresh memory.
    ex = _expander(response='["aa", "bb", "cc"]')
    embed = _StubEmbed()
    # All three hypotheses have length 2 → same vec key "2" → same 2 hits.
    vidx = _StubVectorIndex({
        "2": [("mem_1", 0.9), ("mem_2", 0.5)],
    })
    q = "我为什么心情不好呀今天"
    merged, result = ex.expand_and_recall(q, embed, vidx, k=5)
    assert result.source == "fresh"
    assert result.hypotheses == ["aa", "bb", "cc"]
    assert merged == [("mem_1", 0.9), ("mem_2", 0.5)]
    # Three encode calls, three search calls (once per hypothesis).
    assert len(vidx.queries) == 3


def test_expand_and_recall_no_hypotheses_returns_empty():
    ex = _expander(response="")  # LLM returns empty → parse gives []
    merged, result = ex.expand_and_recall(
        "我为什么心情不好呀今天", _StubEmbed(), _StubVectorIndex({}))
    assert merged == []
    assert result.hypotheses == []


def test_expand_and_recall_missing_vector_index():
    ex = _expander(response='["one"]')
    merged, result = ex.expand_and_recall(
        "我为什么心情不好呀今天", _StubEmbed(), None)
    assert merged == []
    assert result.hypotheses == ["one"]


# ── store wiring ───────────────────────────────────────────────────────

def test_store_hyde_defaults_off(monkeypatch, tmp_path):
    monkeypatch.setenv("ENGRAM_SKIP_VECTOR", "1")
    from engram_router.store import MemoryStore

    s = MemoryStore(path=tmp_path / "hyde.db")
    assert s.hyde is None


def test_store_hyde_opt_in(monkeypatch, tmp_path):
    monkeypatch.setenv("ENGRAM_SKIP_VECTOR", "1")
    from engram_router.store import MemoryStore, RecallWeights

    w = RecallWeights(hyde_enabled=True)
    s = MemoryStore(path=tmp_path / "hyde.db", weights=w)
    # HyDEExpander is instantiated even without an API key; its .available
    # will just be False so recall() silently skips it.
    assert s.hyde is not None


def test_store_hyde_injection(monkeypatch, tmp_path):
    """Caller-provided HyDE bypasses env/weights gating."""
    monkeypatch.setenv("ENGRAM_SKIP_VECTOR", "1")
    from engram_router.store import MemoryStore

    class StubHyDE:
        available = False

        def expand_and_recall(self, *a, **k):
            return [], None

    s = MemoryStore(path=tmp_path / "hyde.db", hyde=StubHyDE())
    assert isinstance(s.hyde, StubHyDE)


# ── real-model integration ─────────────────────────────────────────────

_REAL_HYDE_ENV = "ENGRAM_TEST_REAL_HYDE"


@pytest.mark.skipif(
    os.environ.get(_REAL_HYDE_ENV) != "1",
    reason=f"Set {_REAL_HYDE_ENV}=1 and configure LLM key to run this",
)
def test_real_hyde_generates_plausible_hypotheses():
    """End-to-end with a live LLM — smokes the prompt & parser."""
    ex = HyDEExpander(allow_cloud=True)
    if not ex.available:
        pytest.skip("LLM key not configured")
    # min_query_chars default is 10; use a query longer than that so the
    # gate doesn't short-circuit before we hit the LLM.
    r = ex.generate("我最近为什么心情不好呢感觉不太开心")
    assert r.hypotheses, r
    assert all(isinstance(h, str) and h for h in r.hypotheses)
