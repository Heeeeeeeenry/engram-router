"""EngramRouter provider adapter for the evaluation matrix.

Wraps ``MemoryStore`` so it slots into :class:`MemoryProvider`. Keeps CE on
by default (that's the current production configuration exercised by
eval_v2 with ``ENGRAM_FORCE_CE=1``) and HyDE off — HyDE currently poisons
negative cases (see 2026-07-21 log in OPTIMIZATION_ROADMAP.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from engram_router.store import MemoryStore, RecallWeights

from .base import MemoryProvider, ProviderRecord


class EngramProvider(MemoryProvider):
    def __init__(
        self,
        *,
        ce_enabled: bool = True,
        hyde_enabled: bool = False,
        ce_weight: float | None = None,
        namespace: str = "default",
    ) -> None:
        self._ce_enabled = ce_enabled
        self._hyde_enabled = hyde_enabled
        self._ce_weight = ce_weight
        self._namespace = namespace
        self._store: MemoryStore | None = None

    @property
    def name(self) -> str:
        parts = ["engram"]
        parts.append("ce+" if self._ce_enabled else "ce-")
        parts.append("hyde+" if self._hyde_enabled else "hyde-")
        if self._ce_weight is not None:
            parts.append(f"w{self._ce_weight:g}")
        return "/".join(parts)

    def open(self, workspace: Path) -> None:
        db_path = workspace / "engram.db"
        # ce_weight override lets the matrix driver sweep the blend factor
        # without editing store code.
        weight_kwargs: dict[str, Any] = dict(
            ce_enabled=self._ce_enabled,
            hyde_enabled=self._hyde_enabled,
        )
        if self._ce_weight is not None:
            weight_kwargs["ce_weight"] = self._ce_weight
        weights = RecallWeights(**weight_kwargs)
        # The LLM reranker is a legacy A/B path that competes with CE and
        # noisily 401s against non-OpenAI endpoints. Disable it for the matrix
        # so we're measuring CE (or CE+HyDE) in isolation. Callers that need
        # the LLM reranker can inject their own MemoryStore.
        from engram_router.llm_reranker import LLMReranker

        class _NoopReranker:
            available = False

            def rerank(self, query: str, cands: list[Any]) -> list[Any]:
                return cands

        self._store = MemoryStore(
            path=db_path, weights=weights, reranker=_NoopReranker(),
        )

    def save(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        assert self._store is not None, "open() must be called first"
        mid = self._store.save(text, metadata=metadata, namespace=self._namespace)
        return mid

    def recall(self, query: str, top_k: int = 5) -> list[ProviderRecord]:
        assert self._store is not None, "open() must be called first"
        records = self._store.recall(query, top_k=top_k, namespace=self._namespace)
        return [
            ProviderRecord(
                id=r.id, text=r.raw_text, score=float(r.score),
                metadata={
                    "match_reason": r.match_reason,
                    "created_at": (r.metadata or {}).get("created_at", ""),
                },
            )
            for r in records
        ]

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None
