"""Naive vector baseline: bge-small-zh + numpy cosine.

Serves as the "just embed everything" straw-man in the evaluation matrix.
No FTS, no graph, no LLM — the point is to isolate how much of engram's
lift comes from its multi-layer retrieval vs plain semantic similarity.

Reuses engram-router's ``EmbeddingEngine`` for a fair comparison (same
model, same normalisation). If the model can't be loaded (``available``
is False), :meth:`open` raises so the matrix driver logs the skip clearly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from engram_router.embedding import EmbeddingEngine

from .base import MemoryProvider, ProviderRecord

logger = logging.getLogger(__name__)


class NaiveVectorProvider(MemoryProvider):
    def __init__(self) -> None:
        self._engine: EmbeddingEngine | None = None
        self._texts: list[str] = []
        self._ids: list[str] = []
        self._vectors: np.ndarray | None = None
        self._workspace: Path | None = None
        self._counter = 0

    @property
    def name(self) -> str:
        return "naive-vector"

    def open(self, workspace: Path) -> None:
        self._workspace = workspace
        self._engine = EmbeddingEngine(allow_remote=False)
        if not self._engine.available:
            raise RuntimeError(
                "NaiveVectorProvider unavailable: EmbeddingEngine failed to load. "
                "Ensure sentence-transformers is installed and bge-small-zh weights "
                "are reachable."
            )
        self._texts = []
        self._ids = []
        self._vectors = None
        self._counter = 0

    def save(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        assert self._engine is not None
        vec = self._engine.encode(text)
        if vec is None:
            raise RuntimeError(f"Embedding failed for text: {text[:40]!r}")
        arr = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
        if self._vectors is None:
            self._vectors = arr
        else:
            self._vectors = np.concatenate([self._vectors, arr], axis=0)
        self._counter += 1
        mid = f"nv_{self._counter}"
        self._ids.append(mid)
        self._texts.append(text)
        return mid

    def recall(self, query: str, top_k: int = 5) -> list[ProviderRecord]:
        assert self._engine is not None
        if self._vectors is None or not self._texts:
            return []
        vec = self._engine.encode(query)
        if vec is None:
            return []
        q = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
        sims = (self._vectors @ q.T).flatten()
        top = np.argsort(sims)[::-1][:top_k]
        return [
            ProviderRecord(
                id=self._ids[i], text=self._texts[i], score=float(sims[i]),
                metadata={"cosine": float(sims[i])},
            )
            for i in top
        ]

    def close(self) -> None:
        self._engine = None
        self._texts = []
        self._ids = []
        self._vectors = None
