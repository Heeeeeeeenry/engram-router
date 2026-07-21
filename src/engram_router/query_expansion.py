"""Query Expansion module for EngramRouter.

Provides:
  - SynonymTable: zero-dependency synonym dictionary (<1ms)
  - LLMQueryRewriter: optional LLM query rewriting (colloquial → search variants)
  - ExpansionCache: thread-safe LRU cache with async update support
  - QueryExpander: orchestrator that picks the optimal expansion path

Design principle: zero-dependency core works without LLM; LLM enhancement is
optional and async to keep first-query latency under 200ms.
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
from typing import Any, cast

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ExpandedQuery:
    """Result of query expansion.

    Attributes:
        original: The original query string (unchanged).
        variants: Rewritten search variants, each sent to recall → RRF fused.
        extra_entities: Additional entities extracted from the query.
        synonyms: Token-level synonym mappings for term expansion.
        source: Expansion source ("none" | "synonym-only" | "llm-cached" | "llm-fresh").
        latency_ms: Time spent on expansion in milliseconds.
    """

    original: str
    variants: list[str] = field(default_factory=list)
    extra_entities: list[dict[str, Any]] = field(default_factory=list)
    synonyms: dict[str, list[str]] = field(default_factory=dict)
    source: str = "none"
    latency_ms: float = 0.0


@dataclass
class RewriteResult:
    """Result of a single LLM rewrite call."""

    variants: list[str] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ExpansionStats:
    """Cumulative monitoring stats for QueryExpander."""

    total_calls: int = 0
    cache_hits: int = 0
    synonym_only: int = 0
    llm_cached: int = 0
    llm_fresh: int = 0
    llm_errors: int = 0
    avg_latency_ms: float = 0.0
    cache_size: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# Built-in synonym table
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULT_SYNONYMS: dict[str, list[str]] = {
    # ── 键盘 ──
    "HHKB": ["机械键盘", "键盘", "HHKB键盘", "静电容键盘"],
    "Keychron": ["机械键盘", "键盘", "客制化键盘"],
    "MX": ["机械键盘", "键盘", "Cherry键盘"],
    "机械键盘": ["键盘", "外设"],
    "键盘": ["按键", "外设"],

    # ── 车 ──
    "特斯拉": ["车", "电动车", "Model 3", "Model Y"],
    "Model 3": ["特斯拉", "车", "电动车"],
    "电动车": ["车", "电车"],

    # ── 手机 ──
    "iPhone": ["手机", "苹果手机"],
    "iPad": ["平板", "iPad平板"],

    # ── 食品 ──
    "宫保鸡丁": ["菜", "川菜", "炒菜"],
    "红烧肉": ["菜", "肉菜", "炒菜"],
    "糖醋排骨": ["菜", "排骨"],

    # ── 人物关系 ──
    "同事": ["同僚", "工友"],
    "朋友": ["好友", "哥们", "闺蜜"],
    "老板": ["领导", "上司", "Boss"],

    # ── 属性 ──
    "什么牌子": ["品牌", "型号", "哪个牌子"],
    "多少钱": ["价格", "多少钱", "价位", "费用"],
    "怎么样": ["评价", "体验", "好用吗"],

    # ── 上位词（品类→上位词方向，增强 FTS5 召回） ──
    "交通工具": ["车", "电动车", "汽车", "自行车", "地铁", "火车", "飞机"],
    "外设": ["键盘", "鼠标", "显示器", "耳机", "音响", "摄像头"],
    "电子设备": ["手机", "平板", "电脑", "笔记本"],
    "家具": ["桌子", "椅子", "沙发", "床", "柜子"],
    "宠物": ["猫", "狗", "鱼", "鸟"],
}


# ═══════════════════════════════════════════════════════════════════════════
# SynonymTable
# ═══════════════════════════════════════════════════════════════════════════


class SynonymTable:
    """Zero-dependency synonym mapping table.

    Data sources (merged in order):
      1. Built-in defaults (from object_topic_aliases expansion).
      2. User-provided extra synonyms.
      3. Runtime additions via ``add()``.

    Performance: O(k * n) where k ≈ 100 entries, n ≈ 50 chars → < 0.1ms.
    """

    def __init__(self, extra_synonyms: dict[str, list[str]] | None = None) -> None:
        self._map: dict[str, list[str]] = dict(_DEFAULT_SYNONYMS)
        if extra_synonyms:
            for k, v in extra_synonyms.items():
                self._map[k] = list(v)

    def expand(self, text: str) -> dict[str, list[str]]:
        """Scan text for matching synonym keys.

        Longest-key-first matching ensures "机械键盘" isn't eaten by "键盘".
        """
        result: dict[str, list[str]] = {}
        # Sort by key length descending: long words take priority.
        sorted_keys = sorted(self._map.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if key in text and key not in result:
                result[key] = list(self._map[key])
        return result

    def add(self, term: str, synonyms: list[str]) -> None:
        """Dynamically register a synonym entry."""
        self._map[term] = list(synonyms)

    def remove(self, term: str) -> bool:
        """Remove a synonym entry. Returns True if it existed."""
        if term in self._map:
            del self._map[term]
            return True
        return False

    @property
    def size(self) -> int:
        """Number of synonym entries."""
        return len(self._map)


# ═══════════════════════════════════════════════════════════════════════════
# LLM query rewriting
# ═══════════════════════════════════════════════════════════════════════════

_QUERY_REWRITE_SYSTEM_PROMPT = """\
You are a query expansion engine for a personal memory retrieval system.
Given a colloquial Chinese user query, produce search variants and extra entities to improve recall.

Output ONLY valid JSON. No explanation, no markdown fences.

Schema:
{
  "variants": ["改写变体1", "改写变体2", ...],
  "entities": [
    {"name": "实体名", "kind": "person|object|company|topic|...", "evidence": "原文"}
  ]
}

Rules:
1. variants: 2~4 条搜索变体。
   - 每条变体是完整的搜索短语（不是分词）。
   - 保留关键实体（人名、品牌）不变。
   - 去除口语化噪音（"那个""什么""这个""啊"）。
   - 加入可能的同义词替换（HHKB → 机械键盘）。
   例: "我那个同事送我的键盘什么牌子"
     → ["同事送的键盘品牌", "张三送的HHKB型号", "同事 机械键盘 品牌"]

2. entities: 从查询中提取的关键实体。
   - 只提取查询中明确存在的实体（不推测数据库中有什么）。
   - 不要重复 rule-based 已覆盖的实体（同事/朋友/键盘 等）。
   - 重点：品牌名简称、口语化指代、隐含话题。
   例: "我那个同事送我的键盘什么牌子"
     → [{"name": "键盘", "kind": "object", "evidence": "键盘"}]
     （注：同事/牌子已被 rule-based 覆盖，不再重复）

3. 如果查询很简短（<6个字符）或无需改写，variants 可以为空数组 []。
"""

_QUERY_REWRITE_USER_TEMPLATE = "User query: {query}"


def _parse_json_response(raw: str) -> dict[str, Any]:
    """Parse LLM JSON response, robust to reasoning text and markdown fences."""
    cleaned = raw.strip()

    # Strip markdown fences.
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```", "", cleaned)

    # Direct parse.
    try:
        return cast(dict[str, Any], json.loads(cleaned))
    except json.JSONDecodeError:
        pass

    # Find all top-level JSON objects; pick the one with "variants" or "entities".
    candidates: list[dict[str, Any]] = []
    depth = 0
    start = -1
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(cleaned[start:i + 1])
                    if isinstance(obj, dict):
                        candidates.append(obj)
                except json.JSONDecodeError:
                    pass
                start = -1

    for obj in candidates:
        if "variants" in obj or "entities" in obj:
            return cast(dict[str, Any], obj)

    if candidates:
        return candidates[0]

    logger.warning("Failed to parse LLM rewrite response (len=%d): %.200s", len(raw), raw)
    return {"variants": [], "entities": []}


class LLMQueryRewriter:
    """LLM query rewriter: colloquial queries → multiple search variants.

    Reuses ``LLMClient`` from ``llm_extractor`` (zero extra dependencies).
    """

    def __init__(
        self,
        client: Any | None = None,
        max_variants: int = 4,
    ) -> None:
        from .llm_extractor import LLMClient

        self._client: Any = client if client is not None else LLMClient()
        self._max_variants = max_variants

    @property
    def available(self) -> bool:
        """Whether the LLM API is available (has API key)."""
        return cast(bool, self._client.available)

    def rewrite(self, query: str) -> RewriteResult:
        """Call LLM to rewrite a colloquial query into search variants.

        Returns an empty RewriteResult if LLM is unavailable, query too short,
        or any error occurs.
        """
        if not self.available:
            return RewriteResult()

        # Short queries don't need rewriting.
        if len(query.strip()) < 6:
            return RewriteResult()

        messages = [
            {"role": "system", "content": _QUERY_REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": _QUERY_REWRITE_USER_TEMPLATE.format(query=query)},
        ]

        try:
            raw = self._client.chat(messages, temperature=0.0)
            parsed = _parse_json_response(raw)
            variants = parsed.get("variants", [])[:self._max_variants]
            entities = parsed.get("entities", [])
            return RewriteResult(variants=variants, entities=entities)
        except Exception:
            logger.exception("LLM query rewrite failed for: %s", query)
            return RewriteResult()


# ═══════════════════════════════════════════════════════════════════════════
# ExpansionCache
# ═══════════════════════════════════════════════════════════════════════════


class ExpansionCache:
    """Thread-safe LRU cache for query expansion results.

    Key: sha256 of the normalized query string.
    Value: ExpandedQuery.

    Supports async update: LLM results can be merged into a cache entry
    that was originally populated with synonym-only results.
    """

    def __init__(self, max_size: int = 256) -> None:
        self._max_size = max_size
        self._data: OrderedDict[str, ExpandedQuery] = OrderedDict()
        self._lock = threading.Lock()

    def _make_key(self, query: str) -> str:
        """Normalize and hash query for cache key."""
        normalized = " ".join(query.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, query: str) -> ExpandedQuery | None:
        """Look up a cached expansion result. Returns None on miss."""
        key = self._make_key(query)
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
            return None

    def put(self, query: str, eq: ExpandedQuery) -> None:
        """Insert or update a cache entry."""
        key = self._make_key(query)
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = eq
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)

    def update(self, query: str, variants: list[str], entities: list[dict[str, Any]]) -> bool:
        """Async update: merge LLM results into an existing cache entry.

        Returns:
            True if the entry existed and was updated, False if not found.
        """
        key = self._make_key(query)
        with self._lock:
            existing = self._data.get(key)
            if existing is None:
                return False
            existing.variants = variants
            existing.extra_entities = entities
            existing.source = "llm-cached"
            return True

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._data.clear()

    @property
    def size(self) -> int:
        """Number of entries in cache."""
        with self._lock:
            return len(self._data)

    def __len__(self) -> int:
        return self.size


# ═══════════════════════════════════════════════════════════════════════════
# QueryExpander — main orchestrator
# ═══════════════════════════════════════════════════════════════════════════


class QueryExpander:
    """Orchestrates query expansion across synonym table, cache, and optional LLM.

    Design:
      - Step 1: Check LRU cache (< 0.5ms).
      - Step 2: Apply synonym table expansion (< 1ms).
      - Step 3: Conditionally trigger async LLM rewrite (background thread).
      - LLM results arrive asynchronously and populate the cache for the
        *next* identical query.

    Usage::

        expander = QueryExpander()
        eq = expander.expand("我同事送的键盘什么牌子")       # synonym-only, <2ms
        eq = expander.expand("我同事送的键盘什么牌子")       # cache hit (now with LLM)
    """

    def __init__(
        self,
        synonym_table: SynonymTable | None = None,
        llm_client: Any | None = None,
        cache_size: int = 256,
        enable_llm: bool = True,
        allow_cloud_llm: bool | None = None,
    ) -> None:
        self._synonym_table = synonym_table or SynonymTable()
        self._cache = ExpansionCache(max_size=cache_size)
        # allow_cloud_llm=None → default off, but env var can flip on.
        if allow_cloud_llm is None:
            from .config import env_allows_cloud
            allow_cloud_llm = env_allows_cloud("llm")
        self._enable_llm = enable_llm and allow_cloud_llm
        self._rewriter = LLMQueryRewriter(client=llm_client, max_variants=4)

        # Stats.
        self._stats = ExpansionStats()
        self._stats_lock = threading.Lock()

    # ── Public API ──────────────────────────────────────────────────────────

    def expand(self, query: str, async_llm: bool = False) -> ExpandedQuery:
        """Execute query expansion.

        Args:
            query: The original query string.
            async_llm: If True, trigger a background LLM rewrite (fire-and-forget)
                       and return synonym-only results immediately.

        Returns:
            ExpandedQuery with all available expansions.
            Sync path latency: < 2ms (cache hit < 0.5ms, synonym-only < 1ms).
        """
        t0 = time.perf_counter()

        # Step 1: Check cache.
        cached = self._cache.get(query)
        if cached is not None:
            elapsed = (time.perf_counter() - t0) * 1000
            cached.latency_ms = elapsed
            self._record_call(cached.source, elapsed)
            logger.debug("Cache hit for query %r (source=%s, %.1fms)",
                         query[:60], cached.source, elapsed)
            return cached

        # Step 2: Synonym expansion (zero dependency, always runs).
        synonyms = self._synonym_table.expand(query)

        # Step 3: LLM expansion.
        llm_variants: list[str] = []
        llm_entities: list[dict[str, Any]] = []
        source = "synonym-only"

        if self._enable_llm and self._rewriter.available:
            if async_llm:
                # Fire-and-forget: background LLM, return synonym-only now.
                self._trigger_async_rewrite(query)
            else:
                # Sync LLM (testing / debugging only).
                try:
                    result = self._rewriter.rewrite(query)
                    llm_variants = result.variants
                    llm_entities = result.entities
                    source = "llm-fresh"
                except Exception:
                    logger.exception("Sync LLM rewrite failed for: %s", query)
                    self._increment_llm_errors()

        # Step 4: Assemble result.
        elapsed = (time.perf_counter() - t0) * 1000
        eq = ExpandedQuery(
            original=query,
            variants=llm_variants,
            extra_entities=llm_entities,
            synonyms=synonyms,
            source=source,
            latency_ms=elapsed,
        )

        # Cache the result (even synonym-only; LLM results update it asynchronously).
        self._cache.put(query, eq)
        self._record_call(source, elapsed)

        logger.info("Query expanded: %r → %d variants, %d entities, source=%s, %.1fms",
                     query[:80], len(eq.variants), len(eq.extra_entities),
                     eq.source, eq.latency_ms)

        return eq

    def expand_sync(self, query: str) -> ExpandedQuery:
        """Synchronous expansion with blocking LLM call (for testing/debugging).

        Warning: may take 500ms–2s if LLM is called.
        """
        return self.expand(query, async_llm=False)

    def prewarm(self, queries: list[str]) -> None:
        """Pre-warm the cache by triggering async LLM rewrites for a batch of queries.

        Does not block; results will be available on subsequent calls.
        """
        for q in queries:
            if self._cache.get(q) is None and self._enable_llm and self._rewriter.available:
                self._trigger_async_rewrite(q)

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def synonym_table(self) -> SynonymTable:
        """The SynonymTable in use (for runtime inspection / dynamic add)."""
        return self._synonym_table

    @property
    def stats(self) -> ExpansionStats:
        """Cumulative statistics for monitoring and tuning."""
        with self._stats_lock:
            s = self._stats
            s.cache_size = self._cache.size
            s.avg_latency_ms = self._compute_avg_latency()
            return s

    @property
    def cache_size(self) -> int:
        """Current number of cached expansion results."""
        return self._cache.size

    @property
    def llm_available(self) -> bool:
        """Whether the LLM rewriter is available."""
        return self._enable_llm and self._rewriter.available

    # ── Internal ────────────────────────────────────────────────────────────

    def _trigger_async_rewrite(self, query: str) -> None:
        """Start a background thread to call LLM rewrite and update cache.

        Fire-and-forget: current request returns immediately with synonym-only
        results; the LLM result populates the cache for the next call.
        """

        def _run() -> None:
            try:
                result = self._rewriter.rewrite(query)
                if result.variants or result.entities:
                    updated = self._cache.update(query, result.variants, result.entities)
                    if updated:
                        logger.debug("Async LLM rewrite cached for: %.60s", query)
                    else:
                        logger.debug("Async LLM rewrite: cache entry expired for: %.60s", query)
            except Exception:
                logger.exception("Async query rewrite failed for: %.60s", query)
                self._increment_llm_errors()

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def _record_call(self, source: str, latency_ms: float) -> None:
        """Update stats counters (thread-safe)."""
        with self._stats_lock:
            self._stats.total_calls += 1
            total = self._stats.total_calls
            # Exponential moving average.
            old_avg = self._stats.avg_latency_ms
            self._stats.avg_latency_ms = old_avg + (latency_ms - old_avg) / max(total, 1)

            if source == "synonym-only":
                self._stats.synonym_only += 1
            elif source == "llm-cached":
                self._stats.llm_cached += 1
                self._stats.cache_hits += 1
            elif source == "llm-fresh":
                self._stats.llm_fresh += 1
            # Cache hits that return from cache include source from original entry;
            # we track cache_hits separately in the expand() method path.

            self._stats.cache_size = self._cache.size

    def _increment_llm_errors(self) -> None:
        with self._stats_lock:
            self._stats.llm_errors += 1

    def _compute_avg_latency(self) -> float:
        return self._stats.avg_latency_ms

    def reset_stats(self) -> None:
        """Reset cumulative statistics to zero."""
        with self._stats_lock:
            self._stats = ExpansionStats()
