"""Embedding engine for EngramRouter Phase 2.

Provides a unified interface for text-to-vector encoding:
  - Local: sentence-transformers (bge-small-zh-v1.5, offline, 24MB)
  - Remote: OpenAI-compatible embeddings API (requires API key)

Graceful degradation: if neither backend is available, returns None
signals to the caller that vector search is disabled.

Usage:
    engine = EmbeddingEngine()  # auto-detect best backend
    vec = engine.encode("张三送了我一把HHKB键盘")
    # vec = np.ndarray of shape (dim,) or None if unavailable
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.request
from typing import Any, Sequence, cast

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Unified text-to-vector encoding with local-first fallback chain.

    Priority: local model > remote API > disabled
    """

    # Well-known lightweight Chinese-English models
    _MODELS = {
        "bge-small": {
            "name": "BAAI/bge-small-zh-v1.5",
            "dim": 512,
            "size_mb": 24,
            "desc": "最佳中文轻量 (24MB, 512d)",
        },
        "text2vec": {
            "name": "shibing624/text2vec-base-chinese",
            "dim": 768,
            "size_mb": 400,
            "desc": "纯中文大模型 (400MB, 768d)",
        },
        "e5-small": {
            "name": "intfloat/multilingual-e5-small",
            "dim": 384,
            "size_mb": 118,
            "desc": "多语言轻量 (118MB, 384d)",
        },
    }

    def __init__(
        self,
        model: str = "bge-small",
        backend: str = "auto",           # "auto" | "local" | "remote"
        api_base: str | None = None,
        api_key: str | None = None,
        api_model: str = "text-embedding-3-small",
        cache_size: int = 1024,
        allow_remote: bool | None = None,
    ):
        self._model_key = model
        self._model_info = self._MODELS.get(model, self._MODELS["bge-small"])
        self._backend = backend
        self._api_base = api_base or os.environ.get("ENGRAM_EMBEDDING_API_BASE")
        self._api_key = api_key or os.environ.get("ENGRAM_EMBEDDING_API_KEY")
        self._api_model = api_model
        # allow_remote resolution:
        #   explicit True/False  → honor caller
        #   None (unspecified)   → default off, but ENGRAM_ALLOW_CLOUD /
        #                          ENGRAM_ALLOW_CLOUD_EMBEDDING can flip on
        if allow_remote is None:
            from .config import env_allows_cloud
            allow_remote = env_allows_cloud("embedding")
        self._allow_remote = allow_remote
        self._local_model = None
        self._lock = threading.Lock()
        self._initialized = False
        self._init_error: str | None = None

        # Try to initialize the best available backend
        self._init_backend()

    # ── public ────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._initialized

    @property
    def dim(self) -> int:
        return cast(int, self._model_info["dim"])

    @property
    def backend_name(self) -> str:
        return self._backend

    @property
    def init_error(self) -> str | None:
        return self._init_error

    def encode(self, texts: str | Sequence[str]) -> Any | None:
        """Encode one or more texts to vector(s).

        Returns:
            np.ndarray shape (dim,) for single text, (N, dim) for batch,
            or None if encoding is unavailable.
        """
        if not self._initialized:
            return None
        single = isinstance(texts, str)
        batch: list[str] = [texts] if single else list(texts)  # type: ignore[list-item]
        try:
            if self._backend in ("local", "auto") and self._local_model:
                vecs = self._encode_local(batch)
            elif self._api_key:
                vecs = self._encode_remote(batch)
            else:
                return None
            return vecs[0] if single else vecs
        except Exception as exc:
            logger.warning("encode failed: %s", exc)
            return None

    def encode_batch(
        self, texts: Sequence[str], batch_size: int = 32
    ) -> Any | None:
        """Encode a batch, chunking large inputs."""
        if not self._initialized:
            return None
        import numpy as np
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            vecs = self.encode(chunk)
            if vecs is None:
                return None
            all_vecs.append(vecs)
        return np.concatenate(all_vecs, axis=0) if all_vecs else None

    # ── internal ──────────────────────────────────────────────

    def _init_backend(self) -> None:
        """Try local, then remote, then give up."""
        if self._backend in ("auto", "local"):
            try:
                from sentence_transformers import SentenceTransformer

                model_name = self._model_info["name"]
                self._local_model = SentenceTransformer(model_name)
                self._backend = "local"
                self._initialized = True
                logger.info(
                    "Embedding engine: local %s (%dd, ~%dMB) loaded",
                    model_name, self.dim, self._model_info["size_mb"],
                )
                return
            except ImportError:
                self._init_error = (
                    "sentence-transformers not installed. "
                    "Install with: pip install engram-router[llm]"
                )
            except Exception as exc:
                self._init_error = f"local model load failed: {exc}"

        if self._backend in ("auto", "remote"):
            if self._allow_remote and self._api_key:
                self._backend = "remote"
                self._initialized = True
                logger.info(
                    "Embedding engine: remote API (model=%s)", self._api_model
                )
                return
            if not self._init_error:
                self._init_error = "No API key configured"

        logger.warning(
            "Embedding engine unavailable: %s. Vector search disabled.",
            self._init_error,
        )

    def _encode_local(self, texts: list[str]) -> Any:
        assert self._local_model is not None
        return self._local_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _encode_remote(self, texts: list[str]) -> Any:
        import numpy as np

        url = (self._api_base or "https://api.openai.com/v1") + "/embeddings"
        body = json.dumps({
            "model": self._api_model,
            "input": texts,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode()
            raise RuntimeError(f"Embedding API error {exc.code}: {err_body[:200]}")

        # Sort by index to maintain input order
        items = sorted(data["data"], key=lambda x: x["index"])
        vecs = np.array([item["embedding"] for item in items], dtype=np.float32)
        return vecs
