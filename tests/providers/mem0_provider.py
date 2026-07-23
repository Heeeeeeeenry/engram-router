"""mem0 provider adapter for the evaluation matrix.

Configuration choices (kept minimal, biased toward local + reproducible):

- **LLM**: OpenAI-compatible endpoint, defaults to the same DeepSeek key
  that engram-router uses (``DEEPSEEK_API_KEY`` + ``ENGRA_LLM_BASE_URL``).
  mem0 needs an LLM to run its fact-extraction step; without one it can't
  add memories.
- **Embedder**: Hugging Face ``bge-small-zh-v1.5``, same model as
  engram-router — this isolates *architecture* differences from *encoder*
  differences.
- **Vector store**: Chroma, local persistent path under the workspace.

``open()`` raises with a clear reason if any dependency is missing so the
matrix driver can log the skip instead of crashing the whole run.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .base import MemoryProvider, ProviderRecord

logger = logging.getLogger(__name__)


class Mem0Provider(MemoryProvider):
    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        embedding_dims: int = 512,
        user_id: str = "engram_eval",
        infer: bool = True,
    ) -> None:
        self._llm_model = model or os.environ.get("ENGRA_LLM_MODEL", "deepseek-v4-pro")
        self._base_url = base_url or os.environ.get(
            "ENGRA_LLM_BASE_URL", "https://api.openai.com/v1"
        )
        self._api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get(
            "OPENAI_API_KEY"
        )
        self._embedding_model = embedding_model
        self._embedding_dims = embedding_dims
        self._user_id = user_id
        self._infer = infer
        self._client: Any = None

    @property
    def name(self) -> str:
        return "mem0"

    def open(self, workspace: Path) -> None:
        try:
            from mem0 import Memory
        except ImportError as exc:
            raise RuntimeError(
                "mem0ai not installed. `pip install mem0ai`"
            ) from exc
        if not self._api_key:
            raise RuntimeError(
                "mem0 needs an LLM API key. Set DEEPSEEK_API_KEY or OPENAI_API_KEY."
            )

        # mem0 always initialises OpenAI internally with env vars; injecting
        # them here scopes the change to this provider.
        os.environ["OPENAI_API_KEY"] = self._api_key
        os.environ["OPENAI_BASE_URL"] = self._base_url

        chroma_path = workspace / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)

        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self._llm_model,
                    "api_key": self._api_key,
                    "openai_base_url": self._base_url,
                    "temperature": 0.0,
                    "max_tokens": 2000,
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": self._embedding_model,
                    "embedding_dims": self._embedding_dims,
                },
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "engram_eval",
                    "path": str(chroma_path),
                },
            },
            "history_db_path": str(workspace / "mem0_history.db"),
            "version": "v1.1",
        }
        self._client = Memory.from_config(config)

    def save(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        assert self._client is not None
        # mem0 expects a message list; wrap the raw memory as a single user
        # turn. `infer=self._infer` toggles between "extract structured facts
        # via LLM" (default) and "store verbatim". We keep the default so we
        # exercise mem0's advertised capability.
        result = self._client.add(
            [{"role": "user", "content": text}],
            user_id=self._user_id,
            metadata=metadata,
            infer=self._infer,
        )
        # mem0 returns {"results": [{"id": ..., "event": "ADD", ...}, ...]}
        if isinstance(result, dict) and result.get("results"):
            first = result["results"][0]
            return str(first.get("id", "mem0_?"))
        return "mem0_?"

    def recall(self, query: str, top_k: int = 5) -> list[ProviderRecord]:
        assert self._client is not None
        # mem0 2.x requires user_id under filters={}, not as a top-level kwarg.
        raw = self._client.search(
            query, top_k=top_k, filters={"user_id": self._user_id}
        )
        hits = raw.get("results", []) if isinstance(raw, dict) else raw
        records: list[ProviderRecord] = []
        for h in hits:
            text = h.get("memory") or h.get("text") or h.get("content", "")
            records.append(ProviderRecord(
                id=str(h.get("id", "")),
                text=str(text),
                score=float(h.get("score", 0.0)),
                metadata={k: v for k, v in h.items()
                          if k not in {"id", "memory", "text", "content", "score"}},
            ))
        return records

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.reset()
            except Exception:
                pass
            self._client = None
