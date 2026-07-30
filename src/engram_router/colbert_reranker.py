"""ColBERT-style late-interaction semantic reranker for EngramRouter.

ColBERT (Khattab & Zaharia, SIGIR 2020) computes relevance as the sum of
maximum cosine similarities between each query token embedding and the
most-similar document token embedding (MaxSim). This is stronger than a
bi-encoder (one vector per text) but cheaper than a full cross-encoder
(quadratic self-attention over (q+d) concatenation).

Where it fits in the recall pipeline::

    FTS + vector + HyDE fusion
      → Cross-encoder (bge-reranker-v2-m3, holistic pair scoring)
        → **ColBERT late-interaction (token-level MaxSim, this module)**
          → LLM reranker (optional)
            → context boosts + salience decay

Design notes
------------
- Default model: ``jinaai/jina-colbert-v2`` (1024d, multilingual, 8192 tok).
  It aligns with bge-m3's native dimensionality so no dimension bridging needed.
- Lazy loading via ``_ensure_loaded()`` mirrors ``cross_encoder.py``.
- ``rerank()`` signature matches ``CrossEncoderReranker.rerank()`` — drop-in.
- ENGRAM_SKIP_COLBERT=1 disables at runtime for speed/memory reasons.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Known local ColBERT model manifests.
_LOCAL_MODELS: dict[str, dict[str, Any]] = {
    "jina-colbert-v2": {
        "name": "jinaai/jina-colbert-v2",
        "dim": 1024,
        "max_length": 8192,
        "desc": "Jina ColBERT v2 — multilingual, 1024d, 8K context",
    },
    "jina-colbert-v1": {
        "name": "jinaai/jina-colbert-v1-en",
        "dim": 768,
        "max_length": 512,
        "desc": "Jina ColBERT v1 — English only, 768d, 512 tok",
    },
}


class ColBERTReranker:
    """ColBERT-style token-level late-interaction reranker.

    Usage::

        colbert = ColBERTReranker()
        reranked = colbert.rerank(
            "老张开什么车？",
            [{"text": "老张开一辆黑色宝马X5", "score": 0.8, "id": "m1"},
             {"text": "小李开比亚迪海豚",       "score": 0.6, "id": "m2"}],
        )
    """

    def __init__(
        self,
        model: str = "jina-colbert-v2",
        max_candidates: int = 20,
        colbert_weight: float = 0.4,
        device: str | None = None,
        enabled: bool = True,
    ):
        self._model_key = model
        self._model_info = _LOCAL_MODELS.get(model)
        self._max_candidates = max_candidates
        self._colbert_weight = colbert_weight
        self._device_override = device
        self._enabled = enabled

        self._model: Any = None
        self._device: str = ""
        self._load_attempted = False
        self._init_error: str | None = None

    # ── public ──────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._enabled and self._model is not None

    @property
    def available_hint(self) -> bool:
        """True before lazy-load so outer guards can trigger _ensure_loaded()."""
        return self._enabled

    @property
    def dim(self) -> int:
        return int(self._model_info.get("dim", 1024)) if self._model_info else 1024

    def score(self, query: str, documents: list[str]) -> list[float] | None:
        """Return per-document MaxSim relevance scores for *query*.

        Returns ``None`` on failure — caller can fall back.
        """
        self._ensure_loaded()
        if not self.available or not query or not documents:
            return None
        try:
            return self._compute_maxsim(query, documents)
        except Exception as exc:
            logger.debug("ColBERT scoring failed: %s", exc)
            return None

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Re-rank *candidates* with ColBERT MaxSim, blending with existing scores.

        Candidates are expected as ``[{"text", "score", "id"}, ...]``.
        Returns the same list re-ordered by blended score.
        """
        if os.environ.get("ENGRAM_SKIP_COLBERT") == "1":
            return candidates
        self._ensure_loaded()
        if not self.available or len(candidates) <= 1:
            return candidates

        head = candidates[: self._max_candidates]
        tail = candidates[self._max_candidates :]
        texts = [str(c.get("text", "")) for c in head]
        colbert_scores = self.score(query, texts)
        if not colbert_scores or all(s == 0.0 for s in colbert_scores):
            return candidates

        # Min-max normalise both signals
        fs_max = max(c.get("score", 0.0) for c in head)
        fs_min = min(c.get("score", 0.0) for c in head) if head else 0.0
        fs_range = fs_max - fs_min if fs_max != fs_min else 1.0
        cb_max = max(colbert_scores)
        cb_min = min(colbert_scores)
        cb_range = cb_max - cb_min if cb_max != cb_min else 1.0

        for cand, cb_raw in zip(head, colbert_scores):
            fs = float(cand.get("score", 0.0))
            cb_norm = (cb_raw - cb_min) / cb_range
            fs_norm = (fs - fs_min) / fs_range
            w = self._colbert_weight
            blended = w * cb_norm + (1.0 - w) * fs_norm
            cand["colbert_score"] = round(float(cb_raw), 4)
            cand["colbert_score_norm"] = round(float(cb_norm), 4)
            cand["score"] = round(float(blended), 4)

        head.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        return head + tail

    # ── internal ────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        if self._model_info is None:
            self._init_error = f"unknown model key: {self._model_key}"
            logger.debug(self._init_error)
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            self._init_error = (
                "sentence-transformers not installed. "
                "Install with: pip install engram-router[llm]"
            )
            logger.debug("ColBERT unavailable: %s (%s)", self._init_error, exc)
            return

        device = self._device_override or _auto_device()

        # ColBERT token-level embeddings + MaxSim are expensive on CPU
        # (200-500ms/rerank). Skip on CPU unless forced via ENGRAM_COLBERT_DEVICE=cpu.
        if device == "cpu" and not os.environ.get("ENGRAM_COLBERT_DEVICE"):
            self._init_error = "ColBERT skipped on CPU (use ENGRAM_COLBERT_DEVICE=cpu to force)"
            logger.debug(self._init_error)
            return

        try:
            model_name = self._model_info["name"]
            self._model = SentenceTransformer(
                model_name,
                device=device,
                trust_remote_code=True,
            )
            self._device = device
            logger.info(
                "ColBERT loaded: %s (device=%s, dim=%d)",
                model_name, device, self.dim,
            )
        except Exception as exc:
            self._init_error = f"ColBERT load failed: {exc}"
            self._model = None
            logger.warning("ColBERT unavailable: %s", self._init_error)

    def _encode(self, texts: list[str]) -> np.ndarray | None:
        """Token-level embeddings. Returns shape (n_texts, n_tokens, dim) or None."""
        if self._model is None:
            return None
        try:
            # Use model.encode with output_value="token_embeddings" to get
            # per-token vectors. Fall back to a manual encode if the model
            # doesn't support this flag.
            embeddings = self._model.encode(
                texts,
                output_value="token_embeddings",
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.array(embeddings, dtype=np.float32)
        except Exception:
            # Fallback: encode normally then return as if 1-token-per-doc.
            # This isn't real ColBERT but keeps the pipeline alive.
            vecs = self._model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vecs_np = np.array(vecs, dtype=np.float32)
            return vecs_np[:, np.newaxis, :]  # (n, 1, dim)

    def _compute_maxsim(self, query: str, documents: list[str]) -> list[float]:
        """Compute ColBERT MaxSim scores for each document.

        For each query token embedding q_i and each document token embedding d_j,
        sim_ij = q_i · d_j (cosine). MaxSim(d) = sum_i max_j sim_ij.
        """
        all_texts = [query] + documents
        emb = self._encode(all_texts)
        if emb is None:
            return [0.0] * len(documents)

        q_emb = emb[0]  # (n_q_tokens, dim)
        scores: list[float] = []
        for i in range(1, len(all_texts)):
            d_emb = emb[i]  # (n_d_tokens, dim)
            # Compute q × d^T → (n_q, n_d) similarity matrix
            sim = np.dot(q_emb, d_emb.T)  # cosine (embeddings already normalised)
            maxsim = np.sum(np.max(sim, axis=1))  # sum over query tokens of max per doc token
            scores.append(float(maxsim))
        return scores


def _auto_device() -> str:
    forced = os.environ.get("ENGRAM_COLBERT_DEVICE")
    if forced:
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
