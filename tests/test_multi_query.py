"""Tests for multi-query expansion (Phase 3 of rerank_and_hyde.md).

Multi-query expands a single user query into 3-5 semantically-equivalent
variants, runs independent recall on each variant, and RRF-fuses the results.
This improves recall by capturing different phrasings of the same information
need.

Run with ENGRAM_SKIP_VECTOR=1 for fast tests (no embedding model needed):
    ENGRAM_SKIP_VECTOR=1 python -m pytest tests/test_multi_query.py -q
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from engram_router.query_expansion import QueryExpander, ExpandedQuery, SynonymTable
from engram_router.store import MemoryStore, RecallWeights


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests: expand_multi_query
# ═══════════════════════════════════════════════════════════════════════════


class TestExpandMultiQuery:
    """Test QueryExpander.expand_multi_query in isolation."""

    @pytest.fixture
    def expander_with_mock_llm(self):
        """Expander with a mocked LLM client that returns a JSON array."""
        mock_client = MagicMock()
        mock_client.available = True
        expander = QueryExpander(enable_llm=True, allow_cloud_llm=True)
        expander._rewriter._client = mock_client
        # QueryExpander.__init__ already checks available for _enable_llm.
        # Replace the client to force available=True for the mock.
        expander._rewriter._client = mock_client
        return expander, mock_client

    # Test queries need >= 4 chars to pass the short-query guard.
    _Q = "测试查询内容"  # 6 chars, enough to pass the guard

    def test_returns_list_of_strings(self, expander_with_mock_llm):
        """expand_multi_query returns a list of string variants."""
        expander, mock_client = expander_with_mock_llm
        mock_client.chat.return_value = json.dumps([
            "同事送给我的键盘是什么品牌",
            "我同事送了哪种键盘给我",
            "键盘的品牌和型号是多少",
        ])
        variants = expander.expand_multi_query("我同事送我的键盘什么牌子", num_variants=3)
        assert isinstance(variants, list)
        assert all(isinstance(v, str) for v in variants)
        assert len(variants) == 3

    def test_deduplicates_identical_variants(self, expander_with_mock_llm):
        """expand_multi_query deduplicates case-insensitive identical variants."""
        expander, mock_client = expander_with_mock_llm
        mock_client.chat.return_value = json.dumps([
            "同事送给我的键盘是什么品牌",
            "同事送给我的键盘是什么品牌",  # duplicate
            "键盘的品牌和型号",
        ])
        variants = expander.expand_multi_query("键盘品牌", num_variants=3)
        assert len(variants) == 2

    def test_removes_original_query(self, expander_with_mock_llm):
        """Original query should not appear in variants."""
        expander, mock_client = expander_with_mock_llm
        mock_client.chat.return_value = json.dumps([
            "我同事送我的键盘什么牌子",  # same as original
            "键盘的品牌和型号",
        ])
        variants = expander.expand_multi_query("我同事送我的键盘什么牌子", num_variants=3)
        assert "我同事送我的键盘什么牌子" not in variants

    def test_num_variants_caps_output(self, expander_with_mock_llm):
        """Variants are capped at num_variants."""
        expander, mock_client = expander_with_mock_llm
        mock_client.chat.return_value = json.dumps([
            "变体1", "变体2", "变体3", "变体4", "变体5",
        ])
        variants = expander.expand_multi_query(self._Q, num_variants=3)
        assert len(variants) <= 3

    def test_empty_list_when_llm_unavailable(self):
        """Returns empty list when LLM is not available."""
        expander = QueryExpander(enable_llm=True, allow_cloud_llm=True)
        # Replace the client with one whose `available` property returns False.
        mock_client = MagicMock()
        type(mock_client).available = property(lambda self: False)
        expander._rewriter._client = mock_client
        variants = expander.expand_multi_query("测试查询内容")
        assert variants == []

    def test_empty_list_when_llm_disabled(self):
        """Returns empty list when enable_llm is False."""
        expander = QueryExpander(enable_llm=False)
        variants = expander.expand_multi_query("查询测试")
        assert variants == []

    def test_empty_list_when_query_too_short(self, expander_with_mock_llm):
        """Returns empty list when query is < 4 characters."""
        expander, mock_client = expander_with_mock_llm
        variants = expander.expand_multi_query("ab", num_variants=3)
        assert variants == []
        mock_client.chat.assert_not_called()

    def test_handles_llm_returning_object(self, expander_with_mock_llm):
        """Handles LLM response wrapped in an object with 'variants' key."""
        expander, mock_client = expander_with_mock_llm
        mock_client.chat.return_value = json.dumps({
            "variants": ["变体A", "变体B", "变体C"],
        })
        variants = expander.expand_multi_query(self._Q, num_variants=3)
        assert len(variants) == 3

    def test_handles_llm_returning_plain_array(self, expander_with_mock_llm):
        """Handles LLM response as a plain JSON array."""
        expander, mock_client = expander_with_mock_llm
        mock_client.chat.return_value = '["变体A", "变体B", "变体C"]'
        variants = expander.expand_multi_query(self._Q, num_variants=3)
        assert len(variants) == 3

    def test_empty_on_llm_error(self, expander_with_mock_llm):
        """Returns empty list when LLM raises an exception."""
        expander, mock_client = expander_with_mock_llm
        mock_client.chat.side_effect = RuntimeError("API error")
        variants = expander.expand_multi_query("查询测试")
        assert variants == []

    def test_empty_on_malformed_json(self, expander_with_mock_llm):
        """Returns empty list when LLM returns unparseable text."""
        expander, mock_client = expander_with_mock_llm
        mock_client.chat.return_value = "这不是 JSON"
        variants = expander.expand_multi_query("查询测试")
        assert variants == []

    def test_whitespace_only_variants_filtered(self, expander_with_mock_llm):
        """Whitespace-only or empty variants are filtered out."""
        expander, mock_client = expander_with_mock_llm
        mock_client.chat.return_value = json.dumps([
            "   ", "有效变体", "",
        ])
        variants = expander.expand_multi_query(self._Q, num_variants=3)
        assert len(variants) == 1
        assert variants[0] == "有效变体"


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests: RecallWeights multi-query fields
# ═══════════════════════════════════════════════════════════════════════════


class TestRecallWeightsMultiQuery:
    """Test RecallWeights multi-query configuration fields."""

    def test_default_values(self):
        """Default multi_query_enabled is False, safe defaults."""
        w = RecallWeights()
        assert w.multi_query_enabled is False
        assert w.multi_query_num_variants == 3
        assert w.multi_query_fusion_weight == 0.5

    def test_custom_values(self):
        """Custom values are accepted."""
        w = RecallWeights(
            multi_query_enabled=True,
            multi_query_num_variants=5,
            multi_query_fusion_weight=0.3,
        )
        assert w.multi_query_enabled is True
        assert w.multi_query_num_variants == 5
        assert w.multi_query_fusion_weight == 0.3

    def test_invalid_num_variants(self):
        """num_variants < 1 raises ValueError."""
        with pytest.raises(ValueError, match="multi_query_num_variants"):
            RecallWeights(multi_query_num_variants=0)

    def test_invalid_fusion_weight_negative(self):
        """fusion_weight < 0 raises ValueError."""
        with pytest.raises(ValueError, match="multi_query_fusion_weight"):
            RecallWeights(multi_query_fusion_weight=-0.1)

    def test_invalid_fusion_weight_too_high(self):
        """fusion_weight > 5 raises ValueError."""
        with pytest.raises(ValueError, match="multi_query_fusion_weight"):
            RecallWeights(multi_query_fusion_weight=5.1)


# ═══════════════════════════════════════════════════════════════════════════
# Integration tests: end-to-end with MemoryStore
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiQueryIntegration:
    """End-to-end tests: multi-query recall with MemoryStore."""

    @pytest.fixture
    def store_with_fake_llm(self, tmp_path):
        """Store with a mocked LLM client for multi-query.

        Uses ENGRAM_SKIP_VECTOR implicitly (set in conftest) for fast tests.
        """
        store = MemoryStore(
            path=tmp_path / "mq_store.db",
            enable_vector=False,
        )
        store.save("张三前两天送我一把 HHKB 键盘，说是生日礼物")
        store.save("张三是我的前同事，现在在腾讯工作。")
        store.save("上周去超市买了宫保鸡丁的食材，做了川菜。")
        store.save("我的手机是 iPhone 14 Pro，拍照效果很好。")

        # Mock the LLM client so expand_multi_query works.
        store.query_expander._rewriter._client = MagicMock()
        store.query_expander._rewriter._client.available = True
        store.query_expander._enable_llm = True
        return store

    def test_multi_query_disabled_does_nothing(self, store_with_fake_llm):
        """When multi_query_enabled=False, recall works normally (no variants)."""
        store = store_with_fake_llm
        store.weights = RecallWeights(multi_query_enabled=False)

        results = store.recall("HHKB 键盘什么牌子")
        assert len(results) > 0
        # Should find the HHKB memory.
        texts = [r.raw_text for r in results]
        assert any("HHKB" in t for t in texts)

    def test_multi_query_no_llm_skips_silently(self, store_with_fake_llm):
        """When LLM unavailable, multi-query falls through to single-query."""
        store = store_with_fake_llm
        store.query_expander._rewriter._client.available = False
        store.weights = RecallWeights(multi_query_enabled=True)

        # Should not crash — falls through to standard pipeline.
        results = store.recall("键盘")
        assert len(results) > 0
        texts = [r.raw_text for r in results]
        assert any("HHKB" in t or "键盘" in t for t in texts)

    def test_multi_query_returns_results(self, store_with_fake_llm):
        """Multi-query enabled with mocked LLM returns valid results."""
        store = store_with_fake_llm
        store.query_expander._rewriter._client.chat.return_value = json.dumps([
            "张三送我的HHKB是什么牌子",
            "HHKB的品牌和型号是什么",
            "同事给我的键盘什么牌子",
        ])
        store.weights = RecallWeights(
            multi_query_enabled=True,
            multi_query_num_variants=3,
        )

        results = store.recall("HHKB 键盘什么牌子")
        assert len(results) > 0
        # At least one result should relate to keyboard/HHKB.
        texts = [r.raw_text for r in results]
        assert any("HHKB" in t or "键盘" in t for t in texts)

    def test_multi_query_finds_results_single_misses(self, store_with_fake_llm):
        """Multi-query variants should surface results that single query misses.

        If we save a memory with a particular phrasing, a variant rephrasing
        that matches better should bring it up via RRF fusion.
        """
        store = store_with_fake_llm
        # Save a memory that uses "机械键盘" but query uses "外设".
        store.save("我最近购入了一款机械键盘，手感很不错。")

        # Mock LLM to produce a variant that matches better.
        store.query_expander._rewriter._client.chat.return_value = json.dumps([
            "我的机械键盘怎么样",
            "机械键盘的使用体验",
        ])
        store.weights = RecallWeights(
            multi_query_enabled=True,
            multi_query_num_variants=2,
            multi_query_fusion_weight=0.8,
        )

        results = store.recall("我的外设使用体验如何")
        texts = [r.raw_text for r in results]
        assert any("机械键盘" in t or "HHKB" in t for t in texts), \
            f"Multi-query should find keyboard memory. Got: {texts}"


# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════

import json
