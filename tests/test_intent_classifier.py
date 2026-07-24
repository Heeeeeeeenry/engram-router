"""Tests for the LLM-based intent classifier (L1.3)."""

from __future__ import annotations

import pytest

from engram_router.intent_classifier import (
    IntentClassifier,
    IntentResult,
    classify_regex,
)
from engram_router.store import query_intent


# ═══════════════════════════════════════════════════════════════════════════
# IntentResult dataclass tests
# ═══════════════════════════════════════════════════════════════════════════


class TestIntentResult:
    def test_default_all_zeros(self):
        r = IntentResult()
        assert r.brand == 0.0
        assert r.identity == 0.0
        assert r.eval == 0.0
        assert r.reason == 0.0
        assert r.person == 0.0
        assert r.time == 0.0
        assert r.location == 0.0
        assert r.object == 0.0

    def test_dominant_default_threshold(self):
        r = IntentResult(brand=0.95, identity=0.1, eval=0.0, object=0.7)
        dom = r.dominant()
        assert set(dom) == {"brand", "object"}

    def test_dominant_custom_threshold(self):
        r = IntentResult(brand=0.95, identity=0.0, eval=0.0, object=0.7)
        assert r.dominant(threshold=0.8) == ["brand"]
        assert r.dominant(threshold=0.6) == ["brand", "object"]

    def test_dominant_none_above_threshold(self):
        r = IntentResult(brand=0.1, identity=0.2)
        assert r.dominant() == []
        assert r.dominant(threshold=0.05) == ["brand", "identity"]

    def test_is_polar_question_true(self):
        # All scores low → polar (yes/no) question
        r = IntentResult(brand=0.0, identity=0.0, eval=0.0)
        assert r.is_polar_question() is True

    def test_is_polar_question_false(self):
        r = IntentResult(brand=0.95, identity=0.0, eval=0.0)
        assert r.is_polar_question() is False

    def test_is_polar_question_borderline(self):
        # Exactly at 0.3 threshold — not dominant (must be > 0.3)
        r = IntentResult(brand=0.3)
        assert r.is_polar_question() is True

    def test_from_booleans(self):
        r = IntentResult.from_booleans(brand=True, eval_=True)
        assert r.brand == 1.0
        assert r.identity == 0.0
        assert r.eval == 1.0
        assert r.reason == 0.0

    def test_from_booleans_all_false(self):
        r = IntentResult.from_booleans()
        assert all(
            v == 0.0 for v in [
                r.brand, r.identity, r.eval, r.reason,
                r.person, r.time, r.location, r.object,
            ]
        )


# ═══════════════════════════════════════════════════════════════════════════
# Regex fallback tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRegexFallback:
    """Test that the regex fallback path produces correct hard booleans."""

    def test_classify_regex_brand(self):
        r = classify_regex("这个键盘是什么牌子")
        assert r.brand == 1.0
        assert r.identity == 0.0

    def test_classify_regex_identity(self):
        r = classify_regex("她今年多大了？")
        assert r.identity == 1.0
        assert r.brand == 0.0

    def test_classify_regex_eval(self):
        r = classify_regex("这家店的奶茶怎么样？")
        assert r.eval == 1.0

    def test_classify_regex_reason(self):
        r = classify_regex("为什么今天会下雨？")
        assert r.reason == 1.0

    def test_classify_regex_person(self):
        r = classify_regex("谁来帮我搬一下？")
        assert r.person == 1.0

    def test_classify_regex_time(self):
        r = classify_regex("我们什么时候出发？")
        assert r.time == 1.0

    def test_classify_regex_location(self):
        r = classify_regex("水杯在哪里？")
        assert r.location == 1.0

    def test_classify_regex_object(self):
        r = classify_regex("桌子上的是什么东西？")
        assert r.object == 1.0

    def test_classify_regex_multiple(self):
        r = classify_regex("为什么这个键盘的牌子好不好？")
        assert r.brand == 1.0
        assert r.reason == 1.0
        assert r.eval == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# IntentClassifier tests (without LLM)
# ═══════════════════════════════════════════════════════════════════════════


class TestIntentClassifierNoLLM:
    """Test IntentClassifier behaviour when LLM is unavailable."""

    def test_available_is_false_without_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("ENGRA_LLM_BASE_URL", raising=False)
        from engram_router.llm_extractor import LLMClient
        client = LLMClient(api_key=None)
        c = IntentClassifier(client=client, allow_cloud=False)
        assert c.available is False

    def test_classify_falls_back_to_regex(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        from engram_router.llm_extractor import LLMClient
        client = LLMClient(api_key=None)
        c = IntentClassifier(client=client, allow_cloud=False)

        r = c.classify("这个键盘是什么牌子")
        assert r.brand == 1.0
        assert r.identity == 0.0

    def test_classify_greeting_is_polar(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        from engram_router.llm_extractor import LLMClient
        client = LLMClient(api_key=None)
        c = IntentClassifier(client=client, allow_cloud=False)

        r = c.classify("你好")
        assert r.dominant() == []
        assert r.is_polar_question() is True

    def test_classify_returns_valid_probabilities(self, monkeypatch):
        """All probabilities must be in [0, 1]."""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        from engram_router.llm_extractor import LLMClient
        client = LLMClient(api_key=None)
        c = IntentClassifier(client=client, allow_cloud=False)

        queries = [
            "这个键盘是什么牌子",
            "她今年多大了",
            "这个好吃吗",
            "为什么今天下雨",
            "谁来过",
            "什么时候出发",
            "在哪里碰头",
            "桌上放的是什么东西",
        ]
        for q in queries:
            r = c.classify(q)
            for v in [r.brand, r.identity, r.eval, r.reason,
                      r.person, r.time, r.location, r.object]:
                assert 0.0 <= v <= 1.0, f"Probability out of range for '{q}': {v}"


# ═══════════════════════════════════════════════════════════════════════════
# _dict_to_result / _parse_intent_response tests
# ═══════════════════════════════════════════════════════════════════════════


class TestParseIntentResponse:
    def test_parse_valid_json(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        from engram_router.llm_extractor import LLMClient
        client = LLMClient(api_key=None)
        c = IntentClassifier(client=client, allow_cloud=False)

        raw = '{"brand": 0.95, "identity": 0.0, "eval": 0.1, "reason": 0.0, "person": 0.0, "time": 0.0, "location": 0.0, "object": 0.7}'
        r = c._parse_intent_response(raw)
        assert r.brand == 0.95
        assert r.identity == 0.0
        assert r.eval == 0.1
        assert r.object == 0.7

    def test_parse_json_with_markdown_fences(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        from engram_router.llm_extractor import LLMClient
        client = LLMClient(api_key=None)
        c = IntentClassifier(client=client, allow_cloud=False)

        raw = '```json\n{"brand": 0.8, "identity": 0.0, "eval": 0.0, "reason": 0.0, "person": 0.0, "time": 0.0, "location": 0.0, "object": 0.6}\n```'
        r = c._parse_intent_response(raw)
        assert r.brand == 0.8
        assert r.object == 0.6

    def test_parse_json_with_reasoning_prefix(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        from engram_router.llm_extractor import LLMClient
        client = LLMClient(api_key=None)
        c = IntentClassifier(client=client, allow_cloud=False)

        raw = 'The query is about brand identification.\n{"brand": 0.9, "identity": 0.0, "eval": 0.0, "reason": 0.0, "person": 0.0, "time": 0.0, "location": 0.0, "object": 0.5}'
        r = c._parse_intent_response(raw)
        assert r.brand == 0.9

    def test_parse_malformed_json_returns_zeros(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        from engram_router.llm_extractor import LLMClient
        client = LLMClient(api_key=None)
        c = IntentClassifier(client=client, allow_cloud=False)

        r = c._parse_intent_response("not json at all")
        assert r.brand == 0.0
        assert r.identity == 0.0
        assert r.dominant() == []

    def test_parse_json_with_missing_keys(self, monkeypatch):
        """Missing keys should default to 0.0."""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        from engram_router.llm_extractor import LLMClient
        client = LLMClient(api_key=None)
        c = IntentClassifier(client=client, allow_cloud=False)

        r = c._parse_intent_response('{"brand": 0.9, "eval": 0.3}')
        assert r.brand == 0.9
        assert r.eval == 0.3
        assert r.identity == 0.0  # missing → default

    def test_parse_json_values_clamped(self, monkeypatch):
        """Values outside [0, 1] should be clamped."""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        from engram_router.llm_extractor import LLMClient
        client = LLMClient(api_key=None)
        c = IntentClassifier(client=client, allow_cloud=False)

        r = c._parse_intent_response(
            '{"brand": 1.5, "identity": -0.3, "eval": 0.5, "reason": 0.0, '
            '"person": 0.0, "time": 0.0, "location": 0.0, "object": 0.0}'
        )
        assert r.brand == 1.0
        assert r.identity == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Integration with MemoryStore
# ═══════════════════════════════════════════════════════════════════════════


class TestIntentClassifierIntegration:
    """Test that classifier can be attached to MemoryStore and used in recall."""

    def test_classifier_attached_to_store(self, temp_db, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.setenv("ENGRAM_SKIP_VECTOR", "1")

        from engram_router.intent_classifier import IntentClassifier
        from engram_router.llm_extractor import LLMClient

        store = temp_db
        store.save("张三前两天送我一把 HHKB，说是生日礼物")
        store.save("张三是我的前同事，现在在腾讯")

        client = LLMClient(api_key=None)
        store.intent_classifier = IntentClassifier(client=client, allow_cloud=False)
        assert store.intent_classifier.available is False

        # Recall should still work (regex fallback).
        results = store.recall("这是什么键盘")
        assert len(results) > 0

    def test_intent_classifier_enabled_default_off(self):
        from engram_router.store.scoring import RecallWeights
        w = RecallWeights()
        assert w.intent_classifier_enabled is False
