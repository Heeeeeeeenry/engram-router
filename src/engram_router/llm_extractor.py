"""LLM-powered entity & edge extraction for EngramRouter.

This module provides a pluggable LLM extractor that can replace or augment the
rule-based entity extraction in ``entities.py``.  The LLM is used as a
*structured annotator* — it tags entities, classifies salience, and identifies
typed relations — but it never rewrites or fabricates memory content.

Design principles:
  1. LLM output is ALWAYS tagged ``source="llm"`` with lower confidence.
  2. Raw text is NEVER modified.  The LLM only produces annotations that hang
     off the original evidence.
  3. The rule-based extractor remains the default; LLM is opt-in.
  4. A single LLM call does entities + salience + edges for efficiency.

Enhancements (2026-07-01):
  - LRUResultCache: sha256-based result cache with 4096 capacity, thread-safe.
  - extract_batch(): batch extraction for up to 10 texts in one LLM call.
  - EdgeRelation enum: 8 LLM-extractable + 4 system-generated edge types.
  - _validate_edges(): filters self-loops, illegal types, and non-existent entities.
  - _post_process_salience(): auto-generates DECISION_CAUSED_BY, HAPPENED_AT,
    CONSTRAINS, and HAS_POLARITY edges from salience signals.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from collections import OrderedDict
from enum import Enum
from typing import Any, cast

logger = logging.getLogger(__name__)

# --- Configuration -----------------------------------------------------------

# Default model: DeepSeek V3 (public API, no proxy issues).
# For OneAPI (internal proxy), set ENGRA_LLM_BASE_URL and ENGRA_LLM_MODEL.
DEFAULT_LLM_MODEL = "deepseek-v4-pro"
DEFAULT_API_BASE = "https://oneapi-comate.baidu-int.com/v1"
DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"

# Confidence for LLM-annotated entities / edges.
# Lower than rule-based (1.0) because LLM can hallucinate.
LLM_CONFIDENCE = 0.8

# Maximum input characters sent to the LLM (truncated with suffix if longer).
MAX_INPUT_CHARS = 2000

# Batch extraction: maximum number of texts per LLM call.
MAX_BATCH_SIZE = 10

# LRU cache capacity for single-text extraction results.
CACHE_CAPACITY = 4096


def _get_api_key() -> str | None:
    """Return the API key from environment, or None if unavailable."""
    return os.environ.get(DEFAULT_API_KEY_ENV)


def _get_base_url() -> str:
    return os.environ.get("ENGRA_LLM_BASE_URL", DEFAULT_API_BASE)


def _get_llm_model() -> str:
    return os.environ.get("ENGRA_LLM_MODEL", DEFAULT_LLM_MODEL)


# --- EdgeRelation enum -------------------------------------------------------

class EdgeRelation(str, Enum):
    """Typed edge relations between entities.

    LLM-extractable (8 types):
      These are the relations the LLM prompt can produce directly.
    """
    # ── LLM-extractable ──────────────────────────────────────────────────
    CO_OCCURS_WITH = "CO_OCCURS_WITH"          # Default: shared context
    CAUSED_BY = "CAUSED_BY"                    # A caused by B
    HAS_ATTRIBUTE = "HAS_ATTRIBUTE"            # Entity has an attribute
    OWNS = "OWNS"                              # Ownership (I OWNS HHKB)
    REPLACES = "REPLACES"                      # A replaces B
    PREFERS = "PREFERS"                        # Preference (A prefers B)
    DISLIKES = "DISLIKES"                      # Dislike relation
    RELATED_TO = "RELATED_TO"                  # Catch-all: general relation

    # ── System-generated (4 types) ───────────────────────────────────────
    # These are added during _post_process_salience() based on salience_class
    # signals, NOT extracted by the LLM directly.
    DECISION_CAUSED_BY = "DECISION_CAUSED_BY"  # decision entity -> reason
    CONSTRAINS = "CONSTRAINS"                   # constraint applies to
    HAPPENED_AT = "HAPPENED_AT"                # event -> time entity
    HAS_POLARITY = "HAS_POLARITY"              # sensory -> polarity value

    @classmethod
    def llm_extractable(cls) -> set[str]:
        """Return the set of relation types the LLM is allowed to produce."""
        return {
            cls.CO_OCCURS_WITH,
            cls.CAUSED_BY,
            cls.HAS_ATTRIBUTE,
            cls.OWNS,
            cls.REPLACES,
            cls.PREFERS,
            cls.DISLIKES,
            cls.RELATED_TO,
        }

    @classmethod
    def system_generated(cls) -> set[str]:
        """Return the set of relation types added by post-processing only."""
        return {
            cls.DECISION_CAUSED_BY,
            cls.CONSTRAINS,
            cls.HAPPENED_AT,
            cls.HAS_POLARITY,
        }

    @classmethod
    def is_valid(cls, relation: str) -> bool:
        """Check whether a relation string is a valid EdgeRelation value."""
        return relation in cls.llm_extractable() or relation in cls.system_generated()


# --- LRU Result Cache --------------------------------------------------------

class LRUResultCache:
    """Thread-safe LRU cache for LLM extraction results, keyed by sha256(text).

    Capacity: 4096 entries by default.
    """

    def __init__(self, capacity: int = CACHE_CAPACITY) -> None:
        self._capacity = capacity
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(text: str) -> str:
        """Compute a deterministic cache key from text content."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> dict[str, Any] | None:
        """Return cached result for *text*, or None on miss."""
        key = self._key(text)
        with self._lock:
            if key in self._cache:
                # Move to end (most-recently-used).
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def put(self, text: str, result: dict[str, Any]) -> None:
        """Store *result* under *text*, evicting oldest entry if at capacity."""
        key = self._key(text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = result
            while len(self._cache) > self._capacity:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


# --- Prompt template ---------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """\
You are a precise entity & relationship annotator for a personal memory engine.
Extract structured annotations from the given Chinese (or mixed) conversation text.

Output ONLY valid JSON. No explanation, no markdown fences.

Schema:
{
  "entities": [
    {
      "name": "实体名称",
      "kind": "person|object|company|time|reason|topic|attribute",
      "salience_class": "base_attr|constraint|decision|sensory|event",
      "evidence": "原文中的原文片段"
    }
  ],
  "edges": [
    {
      "src": "源实体名称",
      "dst": "目标实体名称", 
      "relation": "CO_OCCURS_WITH|CAUSED_BY|HAS_ATTRIBUTE|OWNS|REPLACES|PREFERS|DISLIKES|RELATED_TO",
      "confidence": 0.0-1.0
    }
  ]
}

Entity kind guide:
- person: 人名、亲属称谓(妈妈/爸爸/同事/朋友)、角色指代
- object: 具体物品、产品名(含英文品牌)、食物
- company: 公司/组织名
- time: 时间表达(昨天/上周/2024年3月)
- reason: 因果标记(因为/由于/为了)
- topic: 抽象话题(工作/职业/健康/旅行/技术选型)
- attribute: 人的属性(年龄/性别/籍贯/血型/星座)

Salience_class guide:
- base_attr: 恒常不变的属性(性别/年龄/名字/籍贯/血型) —— 只在被直接问到时有价值
- constraint: 硬性约束(不能/必须/最多/不超过)
- decision: 已做出的决定(决定/确定/选择/采用)
- sensory: 感官评价/情感(好吃/温柔/脾气大/喜欢/讨厌)
- event: 一次性事件(去了/做了/买了/吃了) —— 默认

Edge relation guide:
- CO_OCCURS_WITH: 同一上下文中共同出现(默认关系)
- CAUSED_BY: 明确的因果关系(A because of B)
- HAS_ATTRIBUTE: 实体具有某属性(妈妈 HAS_ATTRIBUTE 温柔)
- OWNS: 拥有关系(我 OWNS HHKB)
- REPLACES: 替换关系(Claude REPLACES GPT-4)
- PREFERS: 偏好(A 更喜欢 B)
- DISLIKES: 不喜欢
- RELATED_TO: 其他一般性关联

Rules:
1. Only extract entities that are clearly present in the text. Do NOT invent.
2. Every entity must have a non-empty "evidence" field with the exact source text.
3. Every edge's src/dst must match entity names exactly.
4. Return empty lists if nothing is found.
5. For CJK names, prefer 2-3 character names.
"""

_EXTRACTION_USER_TEMPLATE = "Text to annotate:\n\n{text}"

# ── Batch extraction prompt ──────────────────────────────────────────────

_BATCH_SYSTEM_PROMPT = """\
You are a precise entity & relationship annotator for a personal memory engine.
You will receive multiple text segments, each labeled with an index like [0], [1], etc.
For EACH segment, extract entities and edges independently.

Output ONLY valid JSON. No explanation, no markdown fences.

Schema:
{
  "results": [
    {
      "index": 0,
      "entities": [
        {
          "name": "实体名称",
          "kind": "person|object|company|time|reason|topic|attribute",
          "salience_class": "base_attr|constraint|decision|sensory|event",
          "evidence": "原文中的原文片段"
        }
      ],
      "edges": [
        {
          "src": "源实体名称",
          "dst": "目标实体名称",
          "relation": "CO_OCCURS_WITH|CAUSED_BY|HAS_ATTRIBUTE|OWNS|REPLACES|PREFERS|DISLIKES|RELATED_TO",
          "confidence": 0.0-1.0
        }
      ]
    }
  ]
}

Rules:
1. Only extract entities that are clearly present in the text. Do NOT invent.
2. Every entity must have a non-empty "evidence" field with the exact source text.
3. Every edge's src/dst must match entity names exactly.
4. Return results for EVERY index, even if empty: {"index": N, "entities": [], "edges": []}.
5. For CJK names, prefer 2-3 character names.
6. Use the same entity kind and salience classification as the single-text mode.
"""

_BATCH_USER_TEMPLATE = "Text segments to annotate:\n\n{segments}"


def _build_batch_user_prompt(texts: list[str]) -> str:
    """Build user prompt for batch extraction, labeling each text with its index."""
    segments = "\n\n".join(
        f"[{i}] {text}" for i, text in enumerate(texts)
    )
    return _BATCH_USER_TEMPLATE.format(segments=segments)


# --- Client ------------------------------------------------------------------

class LLMClient:
    """Minimal OpenAI-compatible API client (no external deps required)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        import urllib.request
        import urllib.error

        self.api_key = api_key or _get_api_key()
        self.base_url = (base_url or _get_base_url()).rstrip("/")
        self.model = model or _get_llm_model()
        self._urllib = urllib

        if not self.api_key:
            logger.warning(
                "LLMClient initialized without API key. "
                "Set %s env var or pass api_key explicitly.",
                DEFAULT_API_KEY_ENV,
            )

    @property
    def available(self) -> bool:
        return self.api_key is not None

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.0,
             max_tokens: int = 2048) -> str:
        """Send a chat completion request and return the text response."""
        import ssl
        import urllib.request
        import urllib.error

        url = f"{self.base_url}/chat/completions"
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        # SSL context — default to secure, allow opt-out for internal proxies.
        if os.environ.get("ENGRAM_SSL_VERIFY", "1").lower() in ("0", "false", "no"):
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            logger.warning("SSL verification disabled (ENGRAM_SSL_VERIFY=0). "
                          "API keys are transmitted over unverified connections!")
        else:
            ctx = ssl.create_default_context()

        try:
            with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choice = data["choices"][0]["message"]
            # Use content only; reasoning_content is the model's internal
            # thinking and should never be treated as structured output.
            content = choice.get("content", "")
            if not content:
                logger.warning(
                    "LLM returned empty content (reasoning used all tokens?). "
                    "Try increasing max_tokens."
                )
            return str(content)
        except urllib.error.HTTPError as e:
            logger.error("LLM API HTTP %d: %s", e.code, e.read().decode("utf-8", errors="replace")[:500])
            raise
        except Exception as e:
            logger.error("LLM API error: %s", e)
            raise


# --- Extractor ---------------------------------------------------------------

class LLMExtractor:
    """LLM-based entity, salience, and edge extractor.

    Usage::

        extractor = LLMExtractor()
        # Wrap MemoryStore with LLM-enhanced extraction:
        store = MemoryStore(path="memory.db", llm_extractor=extractor)
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        enabled: bool = True,
        cache_capacity: int = CACHE_CAPACITY,
        allow_cloud: bool = True,
    ) -> None:
        self._client = client or LLMClient()
        self.enabled = enabled and self._client.available and allow_cloud
        self._cache = LRUResultCache(capacity=cache_capacity)

    @property
    def available(self) -> bool:
        return self.enabled

    @property
    def cache(self) -> LRUResultCache:
        """Expose the internal cache for inspection / clearing."""
        return self._cache

    def _truncate(self, text: str) -> str:
        if len(text) <= MAX_INPUT_CHARS:
            return text
        return text[:MAX_INPUT_CHARS] + "\n...(truncated)"

    def _parse_response(self, raw: str, text: str = "") -> dict[str, Any]:
        """Parse LLM JSON response, with robustness for reasoning models.

        DeepSeek V4 Pro and other reasoning models may output:
          1. Reasoning text (Chinese/English explanation) then JSON.
          2. JSON inside markdown fences.
          3. Clean JSON.

        We extract the outermost JSON object that contains an 'entities' key.
        """
        cleaned = raw.strip()

        # Strip markdown fences (```json ... ``` or ``` ... ```).
        cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```", "", cleaned)

        # Try direct parse first.
        try:
            return cast(dict[str, Any], json.loads(cleaned))
        except json.JSONDecodeError:
            pass

        # Find all top-level JSON objects; pick the one with "entities".
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
            if "entities" in obj:
                return cast(dict[str, Any], obj)

        # Fallback: return first dict or empty.
        if candidates:
            return candidates[0]

        logger.warning("Failed to parse LLM response (len=%d): %.200s", len(raw), raw)
        return {"entities": [], "edges": []}

    def _parse_batch_response(self, raw: str, count: int) -> list[dict[str, Any]]:
        """Parse LLM JSON response for batch extraction.

        Returns a list of per-text results in index order.  Missing indices
        are filled with empty results.
        """
        cleaned = raw.strip()
        cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```", "", cleaned)

        try:
            data = json.loads(cleaned)
            if "results" in data:
                results = data["results"]
                # Build index map, fill missing slots.
                by_index: dict[int, dict[str, Any]] = {}
                for item in results:
                    idx = int(item.get("index", -1))
                    if 0 <= idx < count:
                        by_index[idx] = {
                            "entities": item.get("entities", []),
                            "edges": item.get("edges", []),
                        }
                return [
                    by_index.get(i, {"entities": [], "edges": []})
                    for i in range(count)
                ]
        except json.JSONDecodeError:
            pass

        # Fallback: try extracting top-level objects and matching.
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
                        if isinstance(obj, dict) and "results" in obj:
                            candidates.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = -1

        if candidates:
            results = candidates[0].get("results", [])
            by_index_fb: dict[int, dict[str, Any]] = {}
            for item in results:
                idx = int(item.get("index", -1))
                if 0 <= idx < count:
                    by_index_fb[idx] = {
                        "entities": item.get("entities", []),
                        "edges": item.get("edges", []),
                    }
            return [
                by_index_fb.get(i, {"entities": [], "edges": []})
                for i in range(count)
            ]

        logger.warning("Failed to parse batch LLM response (len=%d)", len(raw))
        return [{"entities": [], "edges": []} for _ in range(count)]

    def _normalize_result(self, parsed: dict[str, Any]) -> dict[str, Any]:
        """Normalize and tag a single parsed extraction result."""
        for ent in parsed.get("entities", []):
            ent["source"] = "llm"
            ent.setdefault("kind", "unknown")
            ent.setdefault("salience_class", "event")
            ent.setdefault("evidence", ent.get("name", ""))

        for edge in parsed.get("edges", []):
            raw_conf = float(edge.get("confidence", 0.8))
            edge["confidence"] = min(raw_conf * LLM_CONFIDENCE, 1.0)
            edge.setdefault("relation", "CO_OCCURS_WITH")
            edge["source"] = "llm"

        return parsed

    # ── Edge validation ───────────────────────────────────────────────────

    def _validate_edges(
        self,
        edges: list[dict[str, Any]],
        entity_names: set[str],
    ) -> list[dict[str, Any]]:
        """Filter invalid edges: self-loops, illegal types, and references to
        entities not present in the extraction.

        Args:
            edges: List of edge dicts with ``src``, ``dst``, ``relation``, etc.
            entity_names: Set of valid entity names from this extraction.

        Returns:
            Filtered list of valid edges.
        """
        valid_relations = EdgeRelation.llm_extractable()
        cleaned: list[dict[str, Any]] = []

        for edge in edges:
            src = edge.get("src", "")
            dst = edge.get("dst", "")
            relation = edge.get("relation", "")

            # Filter: self-loops.
            if src == dst:
                logger.debug("_validate_edges: dropped self-loop %s -> %s", src, dst)
                continue

            # Filter: illegal relation types (only LLM-extractable allowed here).
            if relation not in valid_relations:
                logger.debug(
                    "_validate_edges: dropped edge with illegal relation %r (%s -> %s)",
                    relation, src, dst,
                )
                continue

            # Filter: references to non-existent entities.
            if src not in entity_names:
                logger.debug("_validate_edges: dropped edge with unknown src %r", src)
                continue
            if dst not in entity_names:
                logger.debug("_validate_edges: dropped edge with unknown dst %r", dst)
                continue

            cleaned.append(edge)

        return cleaned

    # ── Salience post-processing ──────────────────────────────────────────

    @staticmethod
    def _detect_polarity(entity_name: str, text: str) -> str:
        """Detect sensory polarity (positive/negative/neutral) from text.

        Simple rule-based detection using polarity markers.
        """
        positive_markers = [
            "好吃", "喜欢", "好用", "舒服", "温柔", "开心", "优秀",
            "完美", "棒", "赞", "厉害", "满意", "高兴", "快乐",
            "漂亮", "帅", "好", "爱", "想",
        ]
        negative_markers = [
            "难吃", "讨厌", "难用", "难受", "脾气大", "伤心", "糟糕",
            "差", "烂", "不满", "后悔", "烦", "累", "生气",
            "丑", "恨", "怕",
        ]

        # Look for markers near the entity evidence.
        for marker in positive_markers:
            if marker in text:
                return "positive"
        for marker in negative_markers:
            if marker in text:
                return "negative"
        return "neutral"

    def _post_process_salience(
        self,
        entities: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        text: str,
    ) -> list[dict[str, Any]]:
        """Auto-generate system edges from salience signals.

        Rules:
          - decision entity  → DECISION_CAUSED_BY edge to reason entities
          - constraint entity → CONSTRAINS edge to affected entities
          - event entity     → HAPPENED_AT edge to time entities
          - sensory entity   → HAS_POLARITY edge with polarity value

        These edges supplement (never replace) LLM-extracted edges.
        """
        system_edges: list[dict[str, Any]] = []

        # Index entities by salience_class.
        decisions = [e for e in entities if e.get("salience_class") == "decision"]
        constraints = [e for e in entities if e.get("salience_class") == "constraint"]
        events = [e for e in entities if e.get("salience_class") == "event"]
        sensory_ents = [e for e in entities if e.get("salience_class") == "sensory"]
        reasons = [e for e in entities if e.get("kind") == "reason"]
        times = [e for e in entities if e.get("kind") == "time"]

        # decision → DECISION_CAUSED_BY (link to reason entities).
        for dec in decisions:
            for reason in reasons:
                if dec["name"] != reason["name"]:
                    system_edges.append({
                        "src": dec["name"],
                        "dst": reason["name"],
                        "relation": EdgeRelation.DECISION_CAUSED_BY.value,
                        "confidence": 0.75,
                        "source": "system",
                    })

        # constraint → CONSTRAINS (link to all non-reason, non-time entities).
        for con in constraints:
            for other in entities:
                if other["name"] != con["name"] and other.get("kind") not in ("reason", "time"):
                    system_edges.append({
                        "src": con["name"],
                        "dst": other["name"],
                        "relation": EdgeRelation.CONSTRAINS.value,
                        "confidence": 0.7,
                        "source": "system",
                    })

        # event → HAPPENED_AT (link to time entities).
        for evt in events:
            for time_ent in times:
                if evt["name"] != time_ent["name"]:
                    system_edges.append({
                        "src": evt["name"],
                        "dst": time_ent["name"],
                        "relation": EdgeRelation.HAPPENED_AT.value,
                        "confidence": 0.6,
                        "source": "system",
                    })

        # sensory → HAS_POLARITY.
        for sen in sensory_ents:
            polarity = self._detect_polarity(sen["name"], text)
            system_edges.append({
                "src": sen["name"],
                "dst": polarity,  # "positive", "negative", or "neutral"
                "relation": EdgeRelation.HAS_POLARITY.value,
                "confidence": 0.65,
                "source": "system",
            })

        return edges + system_edges

    # ── Public API ────────────────────────────────────────────────────────

    def extract(self, text: str) -> dict[str, Any]:
        """Extract entities + edges from text via LLM.

        Returns a dict:
            {
              "entities": [{"name", "kind", "salience_class", "evidence", "source"}],
              "edges": [{"src", "dst", "relation", "confidence"}]
            }

        All entities have ``source="llm"`` and edges have confidence multiplied
        by ``LLM_CONFIDENCE``.

        Caching: results are cached by sha256(text); cache hits skip the LLM
        call entirely.
        """
        if not self.available:
            return {"entities": [], "edges": []}

        # Check cache first.
        cached = self._cache.get(text)
        if cached is not None:
            logger.debug("LLMExtractor cache hit for text len=%d", len(text))
            return cached

        truncated = self._truncate(text)
        messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": _EXTRACTION_USER_TEMPLATE.format(text=truncated)},
        ]

        try:
            raw = self._client.chat(messages, temperature=0.0)
            parsed = self._parse_response(raw, text)
        except Exception:
            logger.exception("LLM extraction failed; falling back to rule-based")
            return {"entities": [], "edges": []}

        # Normalize.
        result = self._normalize_result(parsed)

        # Collect entity names for validation.
        entity_names = {e["name"] for e in result.get("entities", [])}

        # Validate edges (filter self-loops, illegal types, unknown entities).
        raw_edges = result.get("edges", [])
        result["edges"] = self._validate_edges(raw_edges, entity_names)

        # Post-process salience: add system-generated edges.
        result["edges"] = self._post_process_salience(
            result.get("entities", []), result["edges"], text
        )

        # Store in cache.
        self._cache.put(text, result)
        return result

    def extract_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Extract entities + edges from multiple texts in one LLM call.

        Groups texts into batches of up to ``MAX_BATCH_SIZE`` (10), checking
        the cache for each text first.  Un-cached texts are sent together in
        one batch LLM call; cached results are merged back in.

        Returns a list of result dicts in the same order as *texts*.
        """
        if not self.available:
            return [{"entities": [], "edges": []} for _ in texts]

        if not texts:
            return []

        n = len(texts)
        results: list[dict[str, Any] | None] = [None] * n
        uncached_indices: list[int] = []

        # Phase 1: check cache.
        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)

        # Phase 2: batch the uncached texts in groups of MAX_BATCH_SIZE.
        for start in range(0, len(uncached_indices), MAX_BATCH_SIZE):
            chunk_indices = uncached_indices[start:start + MAX_BATCH_SIZE]
            chunk_texts = [texts[i] for i in chunk_indices]

            # Build batch prompt.
            messages = [
                {"role": "system", "content": _BATCH_SYSTEM_PROMPT},
                {"role": "user", "content": _build_batch_user_prompt(chunk_texts)},
            ]

            try:
                raw = self._client.chat(messages, temperature=0.0, max_tokens=4096)
                batch_parsed = self._parse_batch_response(raw, len(chunk_texts))
            except Exception:
                logger.exception("LLM batch extraction failed; returning empty for batch")
                batch_parsed = [
                    {"entities": [], "edges": []} for _ in chunk_texts
                ]

            # Normalize each per-text result.
            for local_idx, global_idx in enumerate(chunk_indices):
                parsed = batch_parsed[local_idx]
                result = self._normalize_result(parsed)

                entity_names = {e["name"] for e in result.get("entities", [])}
                result["edges"] = self._validate_edges(
                    result.get("edges", []), entity_names
                )
                result["edges"] = self._post_process_salience(
                    result.get("entities", []), result["edges"], texts[global_idx]
                )

                results[global_idx] = result
                self._cache.put(texts[global_idx], result)

        return [r or {"entities": [], "edges": []} for r in results]


# --- Convenience functions that mirror entities.py API ----------------------

# Module-level singleton (lazy init).
_extractor: LLMExtractor | None = None


def _get_extractor() -> LLMExtractor | None:
    """Return the module-level LLM extractor if configured."""
    global _extractor
    if _extractor is None:
        if os.environ.get("ENGRA_LLM_ENABLED", "").lower() in ("1", "true", "yes"):
            _extractor = LLMExtractor()
        else:
            _extractor = LLMExtractor(enabled=False)
    return _extractor if _extractor.available else None


def extract_entities_llm(text: str) -> list[dict[str, Any]]:
    """Extract entities via LLM, in the same format as entities.extract_entities()."""
    ext = _get_extractor()
    if ext is None:
        return []
    result = ext.extract(text)
    entities = result.get("entities", [])
    # Convert to the format expected by store._index_entities:
    # {"name", "kind", "evidence", "source": "llm"}
    return [
        {
            "name": e["name"],
            "kind": e["kind"],
            "evidence": e.get("evidence", e["name"]),
            "source": "llm",
            "salience_class": e.get("salience_class", "event"),
        }
        for e in entities
    ]


def extract_edges_llm(text: str) -> list[dict[str, Any]]:
    """Extract typed edges via LLM."""
    ext = _get_extractor()
    if ext is None:
        return []
    result = ext.extract(text)
    return cast(list[dict[str, Any]], result.get("edges", []))


def llm_extractor_instance() -> LLMExtractor | None:
    """Return the module-level LLM extractor, or None."""
    return _get_extractor()
