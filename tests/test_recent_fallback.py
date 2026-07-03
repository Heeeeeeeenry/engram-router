"""Targeted tests for recent fallback mechanism in recall()."""

import pytest
from engram_router.store import MemoryStore


class TestRecentFallback:
    """Verify that recall() supplements with recent items when results < top_k."""

    def test_historical_records_returns_results(self, tmp_path):
        """'历史记录' query (no keyword match) must return top_k results via fallback."""
        store = MemoryStore(enable_vector=False, path=tmp_path / "history.db")
        store.save("张三送我一把 HHKB 键盘。")
        store.save("今天天气很好，适合散步。")
        store.save("昨天和同事讨论了项目进度。")
        store.save("妈妈做的红烧肉很好吃。")
        store.save("周末计划去爬山。")
        store.save("前几天买了一本新书。")
        store.save("咖啡馆的拿铁很不错。")

        results = store.recall("历史记录", top_k=5)
        assert len(results) == 5, f"Expected 5 results via fallback, got {len(results)}"
        for r in results:
            assert r.match_reason is not None, "Every record must have a match_reason"
        # At least some should be recent fallback since "历史记录" has no keyword overlap
        fallback_msgs = [r.match_reason for r in results if "fallback" in r.match_reason.lower()]
        assert len(fallback_msgs) > 0, "Expected at least one 'recent fallback' reason"

    def test_list_recent_conversations_returns_results(self, tmp_path):
        """'罗列一下最近对话' query must return top_k results."""
        store = MemoryStore(enable_vector=False, path=tmp_path / "recent_convo.db")
        store.save("张三送我一把 HHKB 键盘。")
        store.save("今天天气很好，适合散步。")
        store.save("昨天和同事讨论了项目进度。")
        store.save("妈妈做的红烧肉很好吃。")
        store.save("周末计划去爬山。")
        store.save("前几天买了一本新书。")
        store.save("咖啡馆的拿铁很不错。")

        results = store.recall("罗列一下最近对话", top_k=5)
        assert len(results) == 5, f"Expected 5 results via fallback, got {len(results)}"
        fallback_msgs = [r.match_reason for r in results if "fallback" in r.match_reason.lower()]
        assert len(fallback_msgs) > 0, "Expected at least one 'recent fallback' reason"

    def test_recent_chat_returns_results(self, tmp_path):
        """'最近的聊天' query must return top_k results (may match via trigram)."""
        store = MemoryStore(enable_vector=False, path=tmp_path / "recent_chat.db")
        store.save("张三送我一把 HHKB 键盘。")
        store.save("今天天气很好，适合散步。")
        store.save("昨天和同事讨论了项目进度。")
        store.save("妈妈做的红烧肉很好吃。")
        store.save("周末计划去爬山。")
        store.save("前几天买了一本新书。")
        store.save("咖啡馆的拿铁很不错。")

        results = store.recall("最近的聊天", top_k=5)
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        # '最近' matches trigram against content; results are keyword-matched, not fallback.
        # The key point: previously this returned 0-2, now returns 5.

    def test_normal_keyword_query_still_works(self, tmp_path):
        """'HHKB' query must return HHKB-related results, not just any recent items."""
        store = MemoryStore(enable_vector=False, path=tmp_path / "hhkb.db")
        store.save("张三送我一把 HHKB 键盘，是生日礼物。")
        store.save("今天天气很好，适合散步。")
        store.save("昨天和同事讨论了项目进度。")
        store.save("妈妈做的红烧肉很好吃。")
        store.save("周末计划去爬山。")

        results = store.recall("HHKB", top_k=5)
        assert len(results) == 5, "Should return 5 results (some fallback if needed)"
        # The HHKB memory should be in the results with a high score
        hhkb_results = [r for r in results if "HHKB" in r.raw_text]
        assert len(hhkb_results) >= 1, "HHKB memory must be in results"
        # HHKB memory should not be labeled as fallback
        for r in hhkb_results:
            assert "fallback" not in r.match_reason.lower(), \
                f"HHKB memory should not be labeled as fallback, got: {r.match_reason}"

    def test_normal_keyword_query_dominates_over_fallback(self, tmp_path):
        """Keyword-matched items should rank higher than fallback items."""
        store = MemoryStore(enable_vector=False, path=tmp_path / "rank.db")
        store.save("some filler data A")
        store.save("some filler data B")
        store.save("some filler data C")
        store.save("张三送我一把 HHKB 键盘。")
        store.save("some filler data D")
        store.save("some filler data E")
        store.save("HHKB 的键帽手感很好。")

        results = store.recall("HHKB", top_k=3)
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        # The first items should have higher scores (not fallback)
        assert "fallback" not in results[0].match_reason.lower(), \
            f"First result should be keyword-matched, got: {results[0].match_reason}"

    def test_empty_results_returns_recent_not_empty(self, tmp_path):
        """Query with zero keyword overlap must not return empty list."""
        store = MemoryStore(enable_vector=False, path=tmp_path / "no_keywords.db")
        store.save("张三送我一把 HHKB 键盘。")
        store.save("今天天气很好，适合散步。")
        store.save("妈妈做的红烧肉很好吃。")
        store.save("周末去爬山看日出。")
        store.save("咖啡馆里的拿铁很不错。")

        # Use a pure-English nonsense query with NO CJK character overlap
        results = store.recall("xyzzy_nonexistent_bogus_123", top_k=3)
        assert len(results) == 3, f"Expected 3 fallback results, got {len(results)}"
        for r in results:
            assert "fallback" in r.match_reason.lower(), \
                f"All results should be fallback for unrelated query, got: {r.match_reason}"

    def test_namespace_isolation_with_fallback(self, tmp_path):
        """Fallback must respect namespace isolation."""
        store = MemoryStore(enable_vector=False, path=tmp_path / "ns_fallback.db")
        store.save("work item 1", namespace="work")
        store.save("work item 2", namespace="work")
        store.save("work item 3", namespace="work")
        store.save("personal item 1", namespace="personal")
        store.save("personal item 2", namespace="personal")
        store.save("personal item 3", namespace="personal")

        # Query that won't keyword match anything — should fallback within namespace
        work_results = store.recall("xyzzy_keyword", namespace="work", top_k=3)
        assert len(work_results) == 3
        for r in work_results:
            assert "work" in r.raw_text, \
                f"Namespace isolation violated: {r.raw_text}"

        personal_results = store.recall("xyzzy_keyword", namespace="personal", top_k=3)
        assert len(personal_results) == 3
        for r in personal_results:
            assert "personal" in r.raw_text, \
                f"Namespace isolation violated: {r.raw_text}"

    def test_fallback_supplements_partial_keyword_results(self, tmp_path):
        """When keyword results are below top_k, fallback fills the gap."""
        store = MemoryStore(enable_vector=False, path=tmp_path / "partial.db")
        store.save("张三送我一把 HHKB 键盘。")
        store.save("some filler data A")
        store.save("some filler data B")
        store.save("some filler data C")
        store.save("some filler data D")
        store.save("some filler data E")

        # "HHKB" should match 1 memory, fallback should fill to top_k=5
        results = store.recall("HHKB", top_k=5)
        assert len(results) == 5, f"Expected 5 results (1 keyword + 4 fallback), got {len(results)}"
        hhkb_count = sum(1 for r in results if "HHKB" in r.raw_text)
        fallback_count = sum(1 for r in results if "fallback" in r.match_reason.lower())
        assert hhkb_count >= 1, "HHKB memory must be in results"
        assert hhkb_count + fallback_count == len(results), \
            f"All results should be either HHKB or fallback: hhkb={hhkb_count}, fallback={fallback_count}"


class TestEdgeExpansionStillWorks:
    """Verify edge expansion still works with recent fallback in place."""

    def test_edge_expansion_brings_related_memory(self, tmp_path):
        """A memory reached via edge hops should still appear."""
        store = MemoryStore(enable_vector=False, path=tmp_path / "edge_with_fallback.db")
        store.save("张三送我一把 HHKB 键盘。")
        store.save("张三现在在腾讯工作。")

        results = store.recall("HHKB", top_k=5)
        hhkb_texts = [r.raw_text for r in results]
        assert any("腾讯" in t for t in hhkb_texts), \
            "Edge-expanded memory about 腾讯 should appear in results"

    def test_edge_expansion_reported_in_match_reason(self, tmp_path):
        """When a record is pulled via edge hop, its match_reason should say so."""
        store = MemoryStore(enable_vector=False, path=tmp_path / "edge_reason_fb.db")
        store.save("张三送我一把 HHKB 键盘。")
        store.save("张三现在在腾讯工作。")

        results = store.recall("HHKB", top_k=5)
        hop = [r for r in results if "腾讯" in r.raw_text]
        assert hop, "Edge-reached memory should be in results"
        assert any("edge" in r.match_reason.lower() for r in hop), \
            f"Edge hop should be reported in match_reason: {hop[0].match_reason}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
