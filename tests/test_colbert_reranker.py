"""Tests for ColBERT late-interaction reranker."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from engram_router.colbert_reranker import ColBERTReranker


class TestColBERTRerankerInit:
    def test_default_init(self):
        c = ColBERTReranker()
        assert c.dim == 1024
        assert c._max_candidates == 20
        assert c._colbert_weight == 0.4
        assert c.available is False  # model not loaded, but module is healthy

    def test_custom_config(self):
        c = ColBERTReranker(
            model="jina-colbert-v1",
            max_candidates=10,
            colbert_weight=0.3,
            device="cpu",
        )
        assert c.dim == 768
        assert c._max_candidates == 10
        assert c._colbert_weight == 0.3

    def test_enabled_false(self):
        c = ColBERTReranker(enabled=False)
        assert c.available is False

    def test_unknown_model(self):
        c = ColBERTReranker(model="nonexistent")
        assert c._model_info is None
        c._ensure_loaded()
        assert c.available is False
        assert c._init_error


class TestColBERTRerankerRerank:
    def test_skip_env(self, monkeypatch):
        monkeypatch.setenv("ENGRAM_SKIP_COLBERT", "1")
        c = ColBERTReranker()
        candidates = [
            {"text": "hello world", "score": 0.5, "id": "m1"},
            {"text": "foo bar", "score": 0.3, "id": "m2"},
        ]
        result = c.rerank("test", candidates)
        assert result == candidates  # unchanged when skipped

    def test_empty_candidates(self, monkeypatch):
        monkeypatch.setenv("ENGRAM_SKIP_COLBERT", "1")
        c = ColBERTReranker()
        assert c.rerank("test", []) == []
        assert c.rerank("test", [{"text": "a", "score": 1.0}]) == [{"text": "a", "score": 1.0}]

    def test_no_query(self):
        c = ColBERTReranker()
        result = c.score("", ["doc"])
        assert result is None  # no query text, graceful None

    def test_no_documents(self):
        c = ColBERTReranker()
        result = c.score("query", [])
        assert result is None

    def test_rerank_preserves_ids(self, monkeypatch):
        monkeypatch.setenv("ENGRAM_SKIP_COLBERT", "1")
        c = ColBERTReranker()
        candidates = [
            {"text": "a", "score": 1.0, "id": "m1"},
            {"text": "b", "score": 0.5, "id": "m2"},
        ]
        result = c.rerank("q", candidates)
        ids = {r.get("id") for r in result}
        assert ids == {"m1", "m2"}

    def test_score_returns_none_when_unavailable(self):
        c = ColBERTReranker()
        assert c.score("q", ["d1"]) is None  # no model loaded


class TestColBERTMaxSim:
    def test_maxsim_computation(self):
        """Verify MaxSim formula with synthetic embeddings."""
        import numpy as np

        c = ColBERTReranker()
        # Create fake model that returns known embeddings
        with patch.object(c, "_encode") as mock_encode:
            # query: 2 tokens, doc1: 3 tokens, doc2: 2 tokens
            q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            d1 = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
            d2 = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)

            mock_encode.return_value = np.array([q, d1, d2], dtype=object)

            scores = c._compute_maxsim("q", ["d1", "d2"])

            # For d1: q[0]·d1 = [1.0, 0.5, 0.0] → max=1.0
            #          q[1]·d1 = [0.0, 0.5, 1.0] → max=1.0
            #          maxsim = 2.0
            # For d2: q[0]·d2 = [0.0, -1.0] → max=0.0
            #          q[1]·d2 = [1.0, 0.0] → max=1.0
            #          maxsim = 1.0
            assert len(scores) == 2
            assert abs(scores[0] - 2.0) < 0.01
            assert abs(scores[1] - 1.0) < 0.01


class TestColBERTAutoDevice:
    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ENGRAM_COLBERT_DEVICE", "cuda")
        from engram_router.colbert_reranker import _auto_device
        assert _auto_device() == "cuda"
