"""Long-context baseline: dump every memory into the LLM prompt and ask it
to pick the relevant ones.

This is the "why do we need retrieval at all" straw-man. Modern LLMs claim
huge context windows, so the question is: for the toy scenarios engram is
tuned on (a few dozen memories per case), does a plain "shove everything
into the prompt" beat a structured memory store?

Design choices:
- The LLM sees numbered memories and is asked to return the ids of the k
  most relevant, in rank order, as a JSON array. No summarisation, no
  paraphrasing — we're not comparing writing skill, only retrieval.
- Score is inverse rank so eval_v2's rank-based metrics work unchanged.
- Failure returns an empty list; eval_v2 treats that as no hit.

Uses the same DeepSeek-compatible endpoint engram/mem0 use, so the LLM is
never a confounder.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from .base import MemoryProvider, ProviderRecord

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
你是一个记忆检索助手。用户会给你一组编号的记忆片段和一个查询。
请返回与查询最相关的记忆编号,按相关度从高到低排序,最多 K 条。
只输出一个 JSON 数组,例如 [3, 1, 7]。不要解释、不要额外文字。
如果没有任何一条相关,返回空数组 []。
"""


class LongContextProvider(MemoryProvider):
    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        max_memory_chars: int = 40000,
    ) -> None:
        self._model = model or os.environ.get("ENGRA_LLM_MODEL", "deepseek-v4-pro")
        self._base_url = base_url or os.environ.get(
            "ENGRA_LLM_BASE_URL", "https://api.openai.com/v1"
        )
        self._api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get(
            "OPENAI_API_KEY"
        )
        self._max_chars = max_memory_chars
        self._memories: list[tuple[str, str]] = []  # (id, text)
        self._counter = 0
        self._client: Any = None

    @property
    def name(self) -> str:
        return "long-context"

    def open(self, workspace: Path) -> None:
        if not self._api_key:
            raise RuntimeError(
                "LongContextProvider needs an LLM key. "
                "Set DEEPSEEK_API_KEY or OPENAI_API_KEY."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("`openai` python package required") from exc
        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        self._memories = []
        self._counter = 0

    def save(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        self._counter += 1
        mid = f"lc_{self._counter}"
        self._memories.append((mid, text))
        return mid

    def recall(self, query: str, top_k: int = 5) -> list[ProviderRecord]:
        if not self._memories or self._client is None:
            return []
        # Build the numbered prompt. Numbering starts at 1 so the LLM has less
        # chance to reply with a 0-vs-1-based confusion.
        lines: list[str] = []
        total_chars = 0
        for idx, (_mid, text) in enumerate(self._memories, 1):
            if total_chars + len(text) > self._max_chars:
                # Truncate silently: for eval-set scenarios (few dozen short
                # memories) we never hit this, but be defensive.
                logger.warning("Long-context truncated after %d memories", idx - 1)
                break
            lines.append(f"[{idx}] {text}")
            total_chars += len(text)
        block = "\n".join(lines)
        user_msg = (
            f"记忆列表:\n{block}\n\n"
            f"查询: {query}\n\n"
            f"返回最相关的最多 K={top_k} 条记忆编号(1-based),按相关度降序。"
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.debug("Long-context LLM call failed: %s", exc)
            return []

        indices = self._parse_index_array(raw)
        results: list[ProviderRecord] = []
        for rank, idx in enumerate(indices[:top_k], 1):
            if 1 <= idx <= len(self._memories):
                mid, text = self._memories[idx - 1]
                # Rank-based score: rank 1 → 1.0, rank 2 → 0.5, ...
                results.append(ProviderRecord(
                    id=mid, text=text, score=round(1.0 / rank, 4),
                    metadata={"rank": rank},
                ))
        return results

    def close(self) -> None:
        self._client = None
        self._memories = []

    @staticmethod
    def _parse_index_array(raw: str) -> list[int]:
        if not raw:
            return []
        # Strip markdown fences.
        cleaned = re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"\s*```", "", cleaned).strip()
        # Try direct JSON parse.
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return [int(x) for x in parsed if isinstance(x, (int, float))]
        except (json.JSONDecodeError, ValueError):
            pass
        # Fallback: find first "[...]" and parse.
        m = re.search(r"\[[^\]]*\]", cleaned)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, list):
                    return [int(x) for x in parsed
                            if isinstance(x, (int, float))]
            except (json.JSONDecodeError, ValueError):
                return []
        return []
