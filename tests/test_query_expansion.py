"""Tests for query_expansion module."""

import pytest

from engram_router.query_expansion import (
    ExpandedQuery,
    ExpansionCache,
    ExpansionStats,
    LLMQueryRewriter,
    QueryExpander,
    RewriteResult,
    SynonymTable,
)


# ═══════════════════════════════════════════════════════════════════════════
# SynonymTable tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSynonymTable:
    def test_exact_match(self):
        st = SynonymTable()
        result = st.expand("我的HHKB键盘坏了")
        assert "HHKB" in result
        assert "机械键盘" in result["HHKB"]
        assert "键盘" in result["HHKB"]

    def test_long_match_first(self):
        """Verify long-key-first matching: '机械键盘' not eaten by '键盘'."""
        st = SynonymTable()
        result = st.expand("机械键盘")
        assert "机械键盘" in result
        assert "键盘" in result  # Short key also hits (different entry)

    def test_no_match(self):
        st = SynonymTable()
        result = st.expand("今天天气真好")
        assert result == {}

    def test_dynamic_add(self):
        st = SynonymTable()
        st.add("AB测试", ["A/B测试", "灰度实验"])
        result = st.expand("我们做了AB测试")
        assert "AB测试" in result
        assert "A/B测试" in result["AB测试"]

    def test_dynamic_remove(self):
        st = SynonymTable()
        assert st.remove("HHKB") is True
        result = st.expand("HHKB键盘")
        assert "HHKB" not in result

    def test_remove_nonexistent(self):
        st = SynonymTable()
        assert st.remove("nonexistent") is False

    def test_custom_synonyms(self):
        """User-provided synonyms should merge with built-in defaults."""
        st = SynonymTable(extra_synonyms={"Mac": ["苹果电脑", "MacBook"]})
        result = st.expand("Mac")
        assert "Mac" in result
        assert "苹果电脑" in result["Mac"]
        # Built-in still works.
        result2 = st.expand("HHKB")
        assert "HHKB" in result2

    def test_size(self):
        st = SynonymTable()
        initial = st.size
        assert initial > 0
        st.add("custom", ["custom_syn"])
        assert st.size == initial + 1

    def test_overwrite_via_add(self):
        st = SynonymTable()
        st.add("HHKB", ["自定义键盘"])
        result = st.expand("HHKB")
        assert result["HHKB"] == ["自定义键盘"]


# ═══════════════════════════════════════════════════════════════════════════
# ExpansionCache tests
# ═══════════════════════════════════════════════════════════════════════════


class TestExpansionCache:
    def test_put_and_get(self):
        cache = ExpansionCache(max_size=10)
        eq = ExpandedQuery(original="测试", variants=["v1"])
        cache.put("测试", eq)
        cached = cache.get("测试")
        assert cached is not None
        assert cached.variants == ["v1"]
        assert cached.original == "测试"

    def test_lru_eviction(self):
        cache = ExpansionCache(max_size=2)
        for i in range(5):
            cache.put(f"query_{i}", ExpandedQuery(original=f"query_{i}"))
        assert cache.size == 2
        # Oldest entries should be evicted.
        assert cache.get("query_0") is None
        assert cache.get("query_1") is None
        assert cache.get("query_2") is None
        # Most recent two should remain.
        assert cache.get("query_3") is not None
        assert cache.get("query_4") is not None

    def test_normalization(self):
        cache = ExpansionCache()
        eq = ExpandedQuery(original="Hello    World")
        cache.put("  Hello    World ", eq)
        assert cache.get("hello world") is not None

    def test_miss_returns_none(self):
        cache = ExpansionCache()
        assert cache.get("nonexistent") is None

    def test_update_existing(self):
        cache = ExpansionCache()
        eq = ExpandedQuery(original="test", source="synonym-only")
        cache.put("test", eq)

        updated = cache.update("test", variants=["v1", "v2"], entities=[{"name": "e1"}])
        assert updated is True
        cached = cache.get("test")
        assert cached.variants == ["v1", "v2"]
        assert cached.extra_entities == [{"name": "e1"}]
        assert cached.source == "llm-cached"

    def test_update_missing(self):
        cache = ExpansionCache()
        updated = cache.update("not_there", variants=["v1"], entities=[])
        assert updated is False

    def test_clear(self):
        cache = ExpansionCache()
        cache.put("a", ExpandedQuery(original="a"))
        cache.put("b", ExpandedQuery(original="b"))
        assert cache.size == 2
        cache.clear()
        assert cache.size == 0

    def test_len(self):
        cache = ExpansionCache()
        assert len(cache) == 0
        cache.put("a", ExpandedQuery(original="a"))
        assert len(cache) == 1

    def test_recently_accessed_not_evicted(self):
        cache = ExpansionCache(max_size=3)
        cache.put("a", ExpandedQuery(original="a"))
        cache.put("b", ExpandedQuery(original="b"))
        cache.put("c", ExpandedQuery(original="c"))
        # Access 'a' to move it to the end.
        cache.get("a")
        # Insert 'd' — should evict 'b' (now the oldest accessed).
        cache.put("d", ExpandedQuery(original="d"))
        assert cache.size == 3
        assert cache.get("a") is not None
        assert cache.get("c") is not None
        assert cache.get("d") is not None
        assert cache.get("b") is None


# ═══════════════════════════════════════════════════════════════════════════
# ExpandedQuery tests
# ═══════════════════════════════════════════════════════════════════════════


class TestExpandedQuery:
    def test_defaults(self):
        eq = ExpandedQuery(original="hello")
        assert eq.original == "hello"
        assert eq.variants == []
        assert eq.extra_entities == []
        assert eq.synonyms == {}
        assert eq.source == "none"
        assert eq.latency_ms == 0.0

    def test_full_construction(self):
        eq = ExpandedQuery(
            original="测试查询",
            variants=["变体1", "变体2"],
            extra_entities=[{"name": "实体", "kind": "object"}],
            synonyms={"测试": ["test"]},
            source="llm-fresh",
            latency_ms=500.0,
        )
        assert len(eq.variants) == 2
        assert len(eq.extra_entities) == 1
        assert eq.synonyms["测试"] == ["test"]


# ═══════════════════════════════════════════════════════════════════════════
# QueryExpander tests
# ═══════════════════════════════════════════════════════════════════════════


class TestQueryExpander:
    @pytest.fixture
    def expander_no_llm(self):
        """Expander with LLM disabled (synonym-only mode)."""
        return QueryExpander(enable_llm=False)

    @pytest.fixture
    def expander(self):
        """Default expander (LLM enabled but may not have API key)."""
        return QueryExpander()

    def test_synonym_only(self, expander_no_llm):
        """Without LLM, should only use synonym path."""
        eq = expander_no_llm.expand("我的HHKB键盘")
        assert eq.source == "synonym-only"
        assert "HHKB" in eq.synonyms
        assert eq.latency_ms < 100  # Should be very fast

    def test_cache_hit(self, expander_no_llm):
        """Second identical query should hit cache."""
        eq1 = expander_no_llm.expand("同事送的键盘")
        eq2 = expander_no_llm.expand("同事送的键盘")
        # Cache hit: second call returns same result instantly.
        assert eq2.latency_ms < 10

    def test_stats_tracking(self, expander_no_llm):
        """Verify stats are updated correctly."""
        expander_no_llm.reset_stats()
        expander_no_llm.expand("HHKB键盘")
        expander_no_llm.expand("特斯拉")
        stats = expander_no_llm.stats
        assert stats.total_calls >= 2
        assert stats.synonym_only >= 1

    def test_prewarm(self, expander_no_llm):
        """Prewarm should trigger async cache population for new queries."""
        # Prewarm doesn't error out even without LLM.
        expander_no_llm.prewarm(["测试查询1", "测试查询2"])
        # Just verifying no exception.

    def test_expand_sync(self, expander_no_llm):
        """expand_sync gives same result as expand for synonym-only."""
        eq = expander_no_llm.expand_sync("HHKB键盘坏了")
        assert eq.original == "HHKB键盘坏了"

    def test_custom_synonym_table(self):
        """QueryExpander should use a custom SynonymTable if provided."""
        st = SynonymTable(extra_synonyms={"自定义": ["custom"]})
        expander = QueryExpander(synonym_table=st, enable_llm=False)
        eq = expander.expand("我的自定义查询")
        assert "自定义" in eq.synonyms

    def test_no_synonym_match(self, expander_no_llm):
        """Query with no matching synonyms should still return valid result."""
        eq = expander_no_llm.expand("今天天气真好阳光明媚")
        assert eq.synonyms == {}
        assert eq.source == "synonym-only"
        assert eq.original == "今天天气真好阳光明媚"


# ═══════════════════════════════════════════════════════════════════════════
# LLMQueryRewriter tests (no actual LLM calls)
# ═══════════════════════════════════════════════════════════════════════════


class TestLLMQueryRewriter:
    def test_short_query_skipped(self):
        """Queries under 6 characters should return empty result."""
        rewriter = LLMQueryRewriter()
        # We can't actually test rewrite() without an API key,
        # but the logic path for short queries should work.
        result = rewriter.rewrite("短")
        assert result.variants == []
        assert result.entities == []

    def test_available_property(self):
        """available should reflect whether API key is set."""
        rewriter = LLMQueryRewriter()
        # Without DEEPSEEK_API_KEY, available should be False.
        # But we just check it doesn't raise.
        is_avail = rewriter.available
        assert isinstance(is_avail, bool)


# ═══════════════════════════════════════════════════════════════════════════
# ExpansionStats tests
# ═══════════════════════════════════════════════════════════════════════════


class TestExpansionStats:
    def test_defaults(self):
        stats = ExpansionStats()
        assert stats.total_calls == 0
        assert stats.cache_hits == 0
        assert stats.avg_latency_ms == 0.0

    def test_mutable(self):
        stats = ExpansionStats()
        stats.total_calls = 5
        stats.synonym_only = 3
        assert stats.total_calls == 5


# ═══════════════════════════════════════════════════════════════════════════
# RewriteResult tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRewriteResult:
    def test_defaults(self):
        rr = RewriteResult()
        assert rr.variants == []
        assert rr.entities == []

    def test_with_data(self):
        rr = RewriteResult(
            variants=["改写1", "改写2"],
            entities=[{"name": "键盘", "kind": "object"}],
        )
        assert len(rr.variants) == 2
        assert rr.entities[0]["name"] == "键盘"
