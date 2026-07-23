"""Cross-encoder based semantic reranker for EngramRouter.

A cross-encoder scores (query, candidate) pairs end-to-end, producing a much
stronger relevance signal than the sparse-vector or bi-encoder fusion that
recall() computes today. It slots into the pipeline **after** vector/FTS/graph
fusion and **before** context boosts / salience decay so that:

  1. The candidate pool has already been broadened by three retrieval paths.
  2. Business rules (brand/identity/eval boosts, correction penalty) run last
     and stay authoritative — CE never overrides an explicit business rule.

Default model: BAAI/bge-reranker-v2-m3 (568M, multilingual, 8192 tok context).
On Apple Silicon it runs on Metal (mps) with 60–120 ms per batch of 16 pairs.

Fallbacks are aggressive: any import / load / scoring failure downgrades
``available`` to False and recall() silently continues without CE. The
component never blocks a query, and its absence never changes the shape of
the returned records.

Design notes
------------
- Lazy loading (`_ensure_loaded`) mirrors ``embedding.py:148 _init_backend``:
  the model is not fetched from HuggingFace until the first score() call.
- ``rerank()`` accepts the same ``list[dict[str, Any]]`` shape as
  ``LLMReranker.rerank`` so it's a drop-in replacement inside
  ``MemoryStore.recall`` (see ``store.py:1717``).
- Cloud (Cohere Rerank) is a stub: contracts are declared but the network
  path is left for a follow-up commit once the local path is validated.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# Known local model manifests. Kept small on purpose; extension via config
# rather than an ever-growing catalog inside code.
_LOCAL_MODELS: dict[str, dict[str, Any]] = {
    "bge-v2-m3": {
        "name": "BAAI/bge-reranker-v2-m3",
        "size_mb": 1100,
        "max_length": 8192,
    },
    "bge-base": {
        "name": "BAAI/bge-reranker-base",
        "size_mb": 430,
        "max_length": 512,
    },
    "bge-large": {
        "name": "BAAI/bge-reranker-large",
        "size_mb": 1100,
        "max_length": 512,
    },
}

DEFAULT_MODEL_KEY = "bge-v2-m3"


class CrossEncoderReranker:
    """CPU / Metal / CUDA cross-encoder reranker with lazy load + fallback.

    Args:
        model: key into :data:`_LOCAL_MODELS`, or a raw HuggingFace repo id.
        max_candidates: only the top-N of the incoming candidate list are
            re-scored; remainder is appended below CE-scored items in the
            original relative order. This keeps latency bounded even when
            fusion returns hundreds of items.
        device: ``None`` → auto-pick (mps > cuda > cpu). Pass a string to
            override for tests or benchmarks.
        ce_weight: how much CE contributes to the final blended score. The
            remainder ``(1 - ce_weight)`` retains the fusion score, so a
            broken CE that returns all-zeros only demotes ordering by
            ``ce_weight`` of its magnitude, never fully overrides fusion.
        allow_cloud: reserved for Cohere-rerank support; currently unused
            but plumbed so recall() can pass the privacy gate through.
        remote_provider: ``"cohere"`` reserved. Local-only for now.

    The instance is safe to construct even when ``sentence-transformers`` is
    missing: ``available`` will be ``False`` and every method becomes a no-op
    passthrough. This mirrors the graceful-degradation pattern used by
    :class:`engram_router.embedding.EmbeddingEngine`.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL_KEY,
        max_candidates: int = 20,
        device: str | None = None,
        ce_weight: float = 0.6,
        allow_cloud: bool | None = None,
        remote_provider: str | None = None,
    ) -> None:
        if not (0.0 <= ce_weight <= 1.0):
            raise ValueError(f"ce_weight must be in [0, 1], got {ce_weight}")
        self._model_key = model
        self._model_info = _LOCAL_MODELS.get(model)
        # If caller passed a raw HF repo id (contains '/'), treat it as opaque.
        if self._model_info is None and "/" in model:
            self._model_info = {"name": model, "size_mb": 0, "max_length": 512}
        self._max_candidates = max(1, int(max_candidates))
        self._ce_weight = float(ce_weight)
        self._device_override = device
        self._remote_provider = remote_provider
        # Privacy: gate cloud path exactly as llm_reranker does.
        if allow_cloud is None:
            try:
                from .config import env_allows_cloud

                allow_cloud = env_allows_cloud("reranker")
            except Exception:
                allow_cloud = False
        self._allow_cloud = bool(allow_cloud)

        self._model: Any | None = None
        self._device: str | None = None
        self._init_error: str | None = None
        self._load_attempted = False
        # available becomes True after a successful lazy load OR immediately
        # if the caller wired a remote provider; the first score() call
        # decides for local.
        self._available_hint = self._model_info is not None

    # ── public API ──────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """Whether reranking is expected to succeed.

        Before the first score() call this reflects only the static hints
        (model key valid, sentence-transformers importable). After the first
        call it reflects the actual load outcome.
        """
        if self._load_attempted:
            return self._model is not None
        return self._available_hint

    @property
    def model_name(self) -> str:
        return self._model_info["name"] if self._model_info else self._model_key

    @property
    def device(self) -> str | None:
        return self._device

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return raw CE relevance scores in the same order as ``texts``.

        Falls back to a list of zeros if the model can't be loaded. Never
        raises — reranking is best-effort.
        """
        if not texts:
            return []
        self._ensure_loaded()
        if self._model is None:
            return [0.0] * len(texts)
        try:
            pairs = [(query, t) for t in texts]
            raw = self._model.predict(pairs, show_progress_bar=False)
            return [float(x) for x in raw]
        except Exception as exc:
            logger.debug("Cross-encoder score failed: %s", exc)
            return [0.0] * len(texts)

    def rerank(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Score + blend + reorder candidates.

        Contract mirrors :meth:`LLMReranker.rerank`:
        - Input list contains dicts with ``"text"`` and ``"score"``.
        - Output has the same objects (mutated) sorted by the new blended
          score, most relevant first.
        - Non-scored tail (below ``max_candidates``) is kept in original
          order and appended after the reranked head.
        """
        if not self.available or len(candidates) <= 1:
            return candidates

        head = candidates[: self._max_candidates]
        tail = candidates[self._max_candidates :]
        texts = [str(c.get("text", "")) for c in head]
        ce_scores = self.score(query, texts)
        if not ce_scores or all(s == 0.0 for s in ce_scores):
            # Silent failure inside score(); leave original order.
            return candidates

        # Min-max normalise both signals across this head so blending is
        # scale-invariant. Sigmoid would compress bge-reranker's typical
        # 0.001 vs 0.7 spread into ~0.5 vs ~0.67 and destroy the very
        # discriminative signal we want CE to carry. Min-max keeps the
        # relative gaps intact (a 700x score ratio becomes a 1.0 vs 0.0
        # gap after normalisation).
        fusion_scores = [float(c.get("score", 0.0)) for c in head]
        fs_max = max(fusion_scores) if fusion_scores else 1.0
        fs_min = min(fusion_scores) if fusion_scores else 0.0
        fs_range = fs_max - fs_min if fs_max != fs_min else 1.0
        ce_max = max(ce_scores) if ce_scores else 1.0
        ce_min = min(ce_scores) if ce_scores else 0.0
        ce_range = ce_max - ce_min if ce_max != ce_min else 1.0

        for cand, ce_raw, fs in zip(head, ce_scores, fusion_scores):
            ce_norm = (ce_raw - ce_min) / ce_range
            fs_norm = (fs - fs_min) / fs_range
            blended = self._ce_weight * ce_norm + (1.0 - self._ce_weight) * fs_norm
            cand["ce_score"] = round(float(ce_raw), 4)
            cand["ce_score_norm"] = round(float(ce_norm), 4)
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
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            self._init_error = (
                "sentence-transformers not installed. "
                "Install with: pip install engram-router[llm]"
            )
            logger.debug("CE unavailable: %s (%s)", self._init_error, exc)
            return

        device = self._device_override or self._auto_device()
        try:
            model_name = self._model_info["name"]
            max_length = int(self._model_info.get("max_length", 512))
            self._model = CrossEncoder(
                model_name, device=device, max_length=max_length
            )
            self._device = device
            logger.info(
                "Cross-encoder loaded: %s (device=%s, max_len=%d)",
                model_name,
                device,
                max_length,
            )
        except Exception as exc:
            self._init_error = f"cross-encoder load failed: {exc}"
            self._model = None
            logger.warning("Cross-encoder unavailable: %s", self._init_error)

    @staticmethod
    def _auto_device() -> str:
        # ENGRAM_CE_DEVICE lets tests / benchmarks force a specific device.
        forced = os.environ.get("ENGRAM_CE_DEVICE")
        if forced:
            return forced
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
