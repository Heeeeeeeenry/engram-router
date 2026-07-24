"""HyDE (Hypothetical Document Embeddings) query expander.

Phase 2 of ``docs/design/rerank_and_hyde.md``. The idea:

  Given a query like "我为什么心情不好", where the actual memory reads
  "今天在会议室被老板批评了", there is zero lexical overlap. Bi-encoder
  similarity of a *user question* to a *narrative statement* is unreliable
  because they're written in different registers.

  HyDE bridges this by asking an LLM to *first invent a plausible answer*
  (a "hypothetical document") in the register the memory would use, then
  embedding **that** answer and searching. The synthetic answer sits closer
  to the real answer in embedding space than the question ever will.

This module never invents facts — the prompt constrains the LLM to describe
structure and concept only ("someone did something for a reason"), not
specific names/numbers/dates. Facts come exclusively from the retrieved
memories.

Failure modes are silent:
- Missing LLM key → ``available = False`` → recall() skips HyDE.
- Missing embedding engine → returns ``[]`` from ``expand_and_recall``.
- LLM raises / returns garbage → cached as empty and reused.

Gates (short-circuit before touching the LLM):
- ``len(query.strip()) < min_query_chars`` — too little signal to expand
- Negative-form patterns (e.g. "没…吗", "不…吗") — HyDE would only
  strengthen the wrong direction. Detected by :func:`_looks_negative`.
- Caller-provided ``should_run(query) -> bool`` — lets ``MemoryStore`` plug
  in its ``should_inject`` heuristic without importing it here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, cast

logger = logging.getLogger(__name__)


_HYDE_SYSTEM_PROMPT = """\
你是"记忆假想扩写"助手,任务是为个人记忆检索系统生成假想候选记忆。
用户会给你一句提问(通常关于自己或身边人的经历、状态、原因、时间),
你要生成 3 段简短的"可能记忆",每段一句,只描述结构与关键概念,
用于向量检索的锚点。

严格约束:
1. **禁止编造具体名字、数字、日期、地点、品牌**——用"某人/若干年前/一个地方/某个物品"这样的占位。
2. 每段 15~40 字,写成陈述句(而不是问句),风格像用户日记或聊天记录。
3. 涵盖不同的语义方向:因果、状态、时间、对象。
4. 只输出 JSON 数组,不要 markdown fence、不要解释,例:
   ["...", "...", "..."]

例:
提问: "我为什么心情不好"
输出: ["今天被人批评了心情很低落", "遇到不顺利的事情情绪不好", "工作上出了差错感到沮丧"]
"""

_HYDE_USER_TEMPLATE = "提问: {query}"


# 常见的否定/反问 pattern:命中即禁用 HyDE(否则假想答案会强化错误方向)。
# 例如 "李四买特斯拉了吗"—— HyDE 会生成"某人买了特斯拉"的假想句,把向量拉向
# 张三买特斯拉那条错误记忆。
#
# ⚠️ 只用严格模式:必须是"否定词+疑问"或"是不是/有没有",不能光看到 不/没
# 就 skip。像 "我为什么心情不好" 里的 不 是形容词一部分,不是否定构造。
_NEGATIVE_PATTERNS = (
    re.compile(r"[没不别未]\s*[^吗么]*[吗么]"),  # 有…吗 / 不…么
    re.compile(r"是不是"),
    re.compile(r"有没有"),
)


def _looks_negative(query: str) -> bool:
    """Best-effort negation / polar-question detection.

    Return True when HyDE should be skipped. Conservative: false positives are
    fine (we just fall back to the standard recall path); false negatives
    would waste an LLM call and risk poisoning the ranking.
    """
    q = query.strip()
    if not q:
        return False
    # A polar question that ends in 吗 is always yes/no — skip HyDE.
    # Note: we intentionally do NOT trigger on trailing 么 alone because
    # information-seeking queries like "我要吃什么" also end in 么 and
    # they're exactly what HyDE is designed to help.
    if q.endswith("吗") or q.endswith("嘛"):
        return True
    for pat in _NEGATIVE_PATTERNS:
        if pat.search(q):
            return True
    return False


def _parse_hypotheses(raw: str) -> list[str]:
    """Extract the JSON array of strings from an LLM response.

    Tolerates markdown fences and stray leading/trailing prose. Returns an
    empty list rather than raising when parsing fails — the caller treats
    "no hypotheses" as a soft failure.
    """
    if not raw:
        return []
    cleaned = raw.strip()
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```", "", cleaned).strip()
    # Direct parse if it *is* an array.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        pass
    # Fallback: locate the first '[' … ']' pair with balanced brackets.
    depth = 0
    start = -1
    for i, ch in enumerate(cleaned):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    parsed = json.loads(cleaned[start : i + 1])
                    if isinstance(parsed, list):
                        return [str(x).strip() for x in parsed if str(x).strip()]
                except json.JSONDecodeError:
                    pass
                start = -1
    logger.debug("HyDE parse failed: %.200s", raw)
    return []


@dataclass
class HyDEResult:
    """Bundle returned by :meth:`HyDEExpander.generate` for observability."""

    query: str
    hypotheses: list[str] = field(default_factory=list)
    source: str = "none"  # "none" | "cache" | "fresh" | "skipped"
    latency_ms: float = 0.0
    skip_reason: str = ""


class _HyDECache:
    """LRU cache for hypotheses keyed by sha256 of the normalized query.

    Kept minimal on purpose — the module is fine using
    :class:`query_expansion.ExpansionCache`, but that cache's value type is
    tightly coupled to ``ExpandedQuery`` and mixing HyDE payloads there would
    force layout changes we don't want in this Phase-2 slice.
    """

    def __init__(self, max_size: int = 256) -> None:
        self._max_size = max_size
        self._data: OrderedDict[str, list[str]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(query: str) -> str:
        normalized = " ".join(query.lower().split())
        return "hyde:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, query: str) -> list[str] | None:
        with self._lock:
            key = self._key(query)
            if key in self._data:
                self._data.move_to_end(key)
                return list(self._data[key])
            return None

    def put(self, query: str, hypotheses: list[str]) -> None:
        with self._lock:
            key = self._key(query)
            self._data[key] = list(hypotheses)
            self._data.move_to_end(key)
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class HyDEExpander:
    """Generate hypothetical answers and use them to seed vector recall.

    Design notes:
    - LLM dependency is injected via ``client`` (protocol-typed) so tests can
      swap in a stub without importing ``llm_extractor``.
    - ``expand_and_recall`` returns ``list[tuple[str, float]]`` — exactly the
      shape :func:`fusion.reciprocal_rank_fusion` consumes. This is the
      integration surface with ``store.recall``.
    - HyDE is **vector-channel only** by design (see rerank_and_hyde.md § 3):
      the hypothetical answers describe scenes, not keywords, so putting them
      through FTS would pull in every memory that mentions any of the
      concept words.
    """

    def __init__(
        self,
        client: Any | None = None,
        num_hypotheses: int = 3,
        max_len: int = 80,
        min_query_chars: int = 10,
        cache_size: int = 256,
        allow_cloud: bool | None = None,
        should_run: Callable[[str], bool] | None = None,
    ) -> None:
        if num_hypotheses < 1:
            raise ValueError(f"num_hypotheses must be >= 1, got {num_hypotheses}")
        if max_len < 8:
            raise ValueError(f"max_len must be >= 8, got {max_len}")
        self._num = int(num_hypotheses)
        self._max_len = int(max_len)
        self._min_chars = int(min_query_chars)
        self._cache = _HyDECache(max_size=cache_size)
        self._should_run = should_run

        if client is None:
            try:
                from .llm_extractor import LLMClient

                client = LLMClient()
            except Exception as exc:  # pragma: no cover — defensive
                logger.debug("HyDE: LLMClient init failed: %s", exc)
                client = None
        self._client = client

        # Privacy: same gate as LLMReranker / QueryExpander.
        if allow_cloud is None:
            try:
                from .config import env_allows_cloud

                allow_cloud = env_allows_cloud("llm")
            except Exception:
                allow_cloud = False
        self._allow_cloud = bool(allow_cloud)

    # ── public API ──────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True iff generation *could* succeed for a well-formed query."""
        if self._client is None:
            return False
        if not self._allow_cloud:
            return False
        client_available = getattr(self._client, "available", False)
        return bool(client_available)

    @property
    def cache_size(self) -> int:
        return self._cache.size

    def generate(self, query: str) -> HyDEResult:
        """Return hypothetical answers for ``query``, using cache when possible.

        Skip reasons are exposed in ``HyDEResult.skip_reason`` so the caller
        can log or record them for eval telemetry.
        """
        t0 = time.perf_counter()
        q = (query or "").strip()

        if len(q) < self._min_chars:
            return HyDEResult(query=q, source="skipped",
                              skip_reason=f"query<{self._min_chars} chars",
                              latency_ms=(time.perf_counter() - t0) * 1000)

        if _looks_negative(q):
            return HyDEResult(query=q, source="skipped",
                              skip_reason="negative/polar-question pattern",
                              latency_ms=(time.perf_counter() - t0) * 1000)

        if self._should_run is not None:
            try:
                if not self._should_run(q):
                    return HyDEResult(query=q, source="skipped",
                                      skip_reason="should_run predicate rejected",
                                      latency_ms=(time.perf_counter() - t0) * 1000)
            except Exception as exc:
                logger.debug("HyDE should_run raised, treating as skip: %s", exc)
                return HyDEResult(query=q, source="skipped",
                                  skip_reason="should_run predicate errored",
                                  latency_ms=(time.perf_counter() - t0) * 1000)

        cached = self._cache.get(q)
        if cached is not None:
            return HyDEResult(query=q, hypotheses=list(cached), source="cache",
                              latency_ms=(time.perf_counter() - t0) * 1000)

        if not self.available:
            return HyDEResult(query=q, source="skipped",
                              skip_reason="LLM client unavailable",
                              latency_ms=(time.perf_counter() - t0) * 1000)

        raw = self._call_llm(q)
        hypotheses = _parse_hypotheses(raw)[: self._num]
        # Truncate each so a long-winded LLM doesn't blow through the embed
        # token budget. Keep them roughly the same length as a real memory.
        hypotheses = [h[: self._max_len] for h in hypotheses]

        # Cache even empty responses — an empty list is the correct "we
        # couldn't help" state and reusing it avoids re-hammering the LLM.
        self._cache.put(q, hypotheses)

        return HyDEResult(query=q, hypotheses=hypotheses, source="fresh",
                          latency_ms=(time.perf_counter() - t0) * 1000)

    def expand_and_recall(
        self,
        query: str,
        embedding: Any,
        vector_index: Any,
        k: int = 20,
    ) -> tuple[list[tuple[str, float]], HyDEResult]:
        """Full HyDE step: generate → embed → vector search → dedupe by mem id.

        Returns a merged ``(id, score)`` list ready for RRF and the
        :class:`HyDEResult` for logging. Scores are the raw cosine
        similarities from :meth:`VectorIndex.search`; RRF (in ``store.recall``)
        will disregard the magnitudes and use rank only, so the caller
        doesn't need to renormalise.

        Short-circuits before the LLM call when both embedding AND vector index
        are unavailable, saving 300-600ms of wasted latency.
        """
        if embedding is None and vector_index is None:
            return [], HyDEResult(query=query, source="skipped",
                                  skip_reason="embedding and vector index unavailable")

        result = self.generate(query)
        if not result.hypotheses:
            return [], result

        # De-dupe by memory id, keep highest per-id similarity across the
        # hypotheses to avoid a single memory monopolising the RRF slot.
        best: dict[str, float] = {}
        for hyp in result.hypotheses:
            try:
                vec = embedding.encode(hyp)
                if vec is None:
                    continue
                hits = vector_index.search(vec, k=k)
                for mid, score in hits:
                    if not mid:
                        continue
                    prev = best.get(mid)
                    if prev is None or score > prev:
                        best[mid] = float(score)
            except Exception as exc:
                logger.debug("HyDE search failed for hypothesis %r: %s", hyp, exc)
                continue

        merged = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        return merged, result

    # ── internal ────────────────────────────────────────────────────────

    def _call_llm(self, query: str) -> str:
        if self._client is None:
            return ""
        messages = [
            {"role": "system", "content": _HYDE_SYSTEM_PROMPT},
            {"role": "user", "content": _HYDE_USER_TEMPLATE.format(query=query)},
        ]
        try:
            return cast(str, self._client.chat(messages, temperature=0.0,
                                               max_tokens=400))
        except Exception as exc:
            logger.debug("HyDE LLM call failed: %s", exc)
            return ""
