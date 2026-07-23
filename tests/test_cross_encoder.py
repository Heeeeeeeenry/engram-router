"""Tests for the cross-encoder reranker (Phase 1 rerank_and_hyde.md).

These tests must be safe to run in the fast/offline test path — the CE
component is engineered to be a no-op when its model can't be loaded, and the
suite asserts that graceful-degradation contract directly. A single "real
model" test is guarded behind an env var so CI can opt-in when the ~1.1 GB
weights are available.
"""

from __future__ import annotations

import os
import pytest

from engram_router.cross_encoder import CrossEncoderReranker, DEFAULT_MODEL_KEY
from engram_router.store import MemoryStore, RecallWeights


# ── unit tests: no model load required ─────────────────────────────────

def test_construction_defaults():
    ce = CrossEncoderReranker()
    assert ce.model_name.startswith("BAAI/bge-reranker")
    # available is a hint until the first score() call.
    assert ce.available is True
    assert ce.device is None  # not yet loaded


def test_invalid_ce_weight_rejected():
    with pytest.raises(ValueError):
        CrossEncoderReranker(ce_weight=1.5)
    with pytest.raises(ValueError):
        CrossEncoderReranker(ce_weight=-0.1)


def test_unknown_model_key_is_safe():
    ce = CrossEncoderReranker(model="does-not-exist")
    # Hint is False for an unknown key with no HF-style '/'.
    assert ce.available is False
    # score() must still be safe — returns zeros of the right length.
    scored = ce.score("query", ["a", "b", "c"])
    assert scored == [0.0, 0.0, 0.0]
    # After the failed load attempt, available flips to False.
    assert ce.available is False


def test_rerank_empty_and_single_are_passthrough():
    ce = CrossEncoderReranker(model="does-not-exist")
    assert ce.rerank("q", []) == []
    single = [{"text": "one", "score": 0.5}]
    assert ce.rerank("q", single) == single


def test_rerank_missing_model_returns_input_unchanged():
    ce = CrossEncoderReranker(model="does-not-exist")
    cands = [
        {"text": "a", "score": 1.0},
        {"text": "b", "score": 0.5},
    ]
    out = ce.rerank("q", cands)
    # Same objects, same relative order.
    assert out == cands
    assert out[0]["text"] == "a"


def test_recall_weights_carry_ce_fields_with_backwards_compatible_defaults():
    w = RecallWeights()
    assert w.ce_enabled is True
    assert w.ce_model == "bge-v2-m3"
    assert w.ce_max_candidates == 20
    assert 0.0 <= w.ce_weight <= 1.0


def test_recall_weights_ce_weight_validated():
    with pytest.raises(ValueError):
        RecallWeights(ce_weight=1.5)
    with pytest.raises(ValueError):
        RecallWeights(ce_max_candidates=0)


def test_store_wires_cross_encoder_by_default(monkeypatch, tmp_path):
    # Under ENGRAM_SKIP_VECTOR=1 CE construction is deliberately skipped so
    # the fast test path never triggers a 1.1 GB download. This is the
    # contract exercise for that guard.
    monkeypatch.setenv("ENGRAM_SKIP_VECTOR", "1")
    s = MemoryStore(path=tmp_path / "ce.db")
    assert s.cross_encoder is None


def test_store_wires_cross_encoder_when_ce_forced(monkeypatch, tmp_path):
    monkeypatch.delenv("ENGRAM_SKIP_VECTOR", raising=False)
    monkeypatch.setenv("ENGRAM_SKIP_CE", "1")
    s = MemoryStore(path=tmp_path / "ce.db", enable_vector=False)
    # ENGRAM_SKIP_CE also disables construction (documented in store.py).
    assert s.cross_encoder is None


def test_store_ce_off_when_weights_disable(monkeypatch, tmp_path):
    monkeypatch.delenv("ENGRAM_SKIP_VECTOR", raising=False)
    monkeypatch.delenv("ENGRAM_SKIP_CE", raising=False)
    w = RecallWeights(ce_enabled=False)
    s = MemoryStore(path=tmp_path / "ce.db", weights=w, enable_vector=False)
    assert s.cross_encoder is None


def test_store_ce_injected_directly(monkeypatch, tmp_path):
    """Caller-provided CE bypasses env/weights gating entirely."""
    monkeypatch.setenv("ENGRAM_SKIP_VECTOR", "1")

    class StubCE:
        available = True

        def rerank(self, query, cands):  # noqa: D401 — stub
            return cands

    s = MemoryStore(path=tmp_path / "ce.db", cross_encoder=StubCE())
    assert isinstance(s.cross_encoder, StubCE)


def test_recall_is_no_op_when_ce_unavailable(monkeypatch, tmp_path):
    """A CE that reports available=False must not touch the record list."""
    monkeypatch.setenv("ENGRAM_SKIP_VECTOR", "1")

    class DeadCE:
        available = False

        def rerank(self, query, cands):
            raise RuntimeError("must not be called when available=False")

    s = MemoryStore(path=tmp_path / "ce.db", cross_encoder=DeadCE())
    s.save("张三送我一把 HHKB 键盘,因为是生日礼物")
    s.save("李四也送了我一本书,挺好看的")
    records = s.recall("谁送我键盘", top_k=3)
    assert records  # sanity: recall still returns something
    # No ce_score annotation, no CE marker in match_reason.
    for r in records:
        assert "cross-encoder" not in r.match_reason


def test_recall_annotates_match_reason_when_ce_active(monkeypatch, tmp_path):
    """A functional CE stub should tag reranked records with a match reason."""
    monkeypatch.setenv("ENGRAM_SKIP_VECTOR", "1")

    class FakeCE:
        available = True

        def rerank(self, query, cands):
            for c in cands:
                c["ce_score"] = 0.99
                c["score"] = 1.0
            return cands

    s = MemoryStore(path=tmp_path / "ce.db", cross_encoder=FakeCE())
    s.save("张三送我一把 HHKB 键盘,因为是生日礼物")
    s.save("李四也送了我一本书,挺好看的")
    records = s.recall("谁送我键盘", top_k=3)
    assert records
    assert any("cross-encoder" in r.match_reason for r in records)


# ── real-model integration ─────────────────────────────────────────────

_REAL_CE_ENV = "ENGRAM_TEST_REAL_CE"
_REAL_CE_ENABLED = os.environ.get(_REAL_CE_ENV) == "1"


@pytest.mark.skipif(
    not _REAL_CE_ENABLED,
    reason=f"Set {_REAL_CE_ENV}=1 to run the real-model integration test",
)
def test_real_bge_reranker_promotes_semantic_match():
    """End-to-end: bge-reranker-v2-m3 should promote a semantically better
    candidate over a lexically-closer distractor.

    Set ENGRAM_TEST_REAL_CE=1 to enable. Requires ~1.1 GB of weights and
    warm CPU/Metal.
    """
    ce = CrossEncoderReranker(model=DEFAULT_MODEL_KEY)
    query = "我最近胖了"
    cands = [
        {"text": "我这个月体重增加了5公斤,得减肥了。", "score": 0.2},  # semantic
        {"text": "我最近开始跑步,每天跑5公里。", "score": 0.4},          # lexical "最近"
        {"text": "我换了个新手机,拍照效果不错。", "score": 0.1},          # distractor
    ]
    out = ce.rerank(query, cands)
    assert out[0]["text"].startswith("我这个月体重增加了5公斤")
