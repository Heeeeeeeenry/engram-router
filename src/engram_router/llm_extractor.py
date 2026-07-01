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
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# --- Configuration -----------------------------------------------------------

# Default model: DeepSeek V3 (public API, no proxy issues).
# For OneAPI (internal proxy), set ENGRA_LLM_BASE_URL and ENGRA_LLM_MODEL.
DEFAULT_LLM_MODEL = "deepseek-chat"
DEFAULT_API_BASE = "https://api.deepseek.com/v1"
DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"

# Confidence for LLM-annotated entities / edges.
# Lower than rule-based (1.0) because LLM can hallucinate.
LLM_CONFIDENCE = 0.8

# Maximum input characters sent to the LLM (truncated with suffix if longer).
MAX_INPUT_CHARS = 2000


def _get_api_key() -> str | None:
    """Return the API key from environment, or None if unavailable."""
    return os.environ.get(DEFAULT_API_KEY_ENV)


def _get_base_url() -> str:
    return os.environ.get("ENGRA_LLM_BASE_URL", DEFAULT_API_BASE)


def _get_llm_model() -> str:
    return os.environ.get("ENGRA_LLM_MODEL", DEFAULT_LLM_MODEL)


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

Rules:
1. Only extract entities that are clearly present in the text. Do NOT invent.
2. Every entity must have a non-empty "evidence" field with the exact source text.
3. Every edge's src/dst must match entity names exactly.
4. Return empty lists if nothing is found.
5. For CJK names, prefer 2-3 character names.
"""

_EXTRACTION_USER_TEMPLATE = "Text to annotate:\n\n{text}"


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

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        """Send a chat completion request and return the text response."""
        import ssl
        import urllib.request
        import urllib.error

        url = f"{self.base_url}/chat/completions"
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2048,
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
            with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
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
            return content
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
    ) -> None:
        self._client = client or LLMClient()
        self.enabled = enabled and self._client.available

    @property
    def available(self) -> bool:
        return self.enabled

    def _truncate(self, text: str) -> str:
        if len(text) <= MAX_INPUT_CHARS:
            return text
        return text[:MAX_INPUT_CHARS] + "\n...(truncated)"

    def _parse_response(self, raw: str, text: str) -> dict[str, Any]:
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
            return json.loads(cleaned)
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
                return obj

        # Fallback: return first dict or empty.
        if candidates:
            return candidates[0]

        logger.warning("Failed to parse LLM response (len=%d): %.200s", len(raw), raw)
        return {"entities": [], "edges": []}

    def extract(self, text: str) -> dict[str, Any]:
        """Extract entities + edges from text via LLM.

        Returns a dict:
            {
              "entities": [{"name", "kind", "salience_class", "evidence", "source"}],
              "edges": [{"src", "dst", "relation", "confidence"}]
            }

        All entities have ``source="llm"`` and edges have confidence multiplied
        by ``LLM_CONFIDENCE``.
        """
        if not self.available:
            return {"entities": [], "edges": []}

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

        # Tag LLM-sourced entities and normalize confidence.
        for ent in parsed.get("entities", []):
            ent["source"] = "llm"
            # Ensure required fields exist.
            ent.setdefault("kind", "unknown")
            ent.setdefault("salience_class", "event")
            ent.setdefault("evidence", ent.get("name", ""))

        for edge in parsed.get("edges", []):
            # Scale LLM confidence.
            raw_conf = float(edge.get("confidence", 0.8))
            edge["confidence"] = min(raw_conf * LLM_CONFIDENCE, 1.0)
            edge.setdefault("relation", "CO_OCCURS_WITH")
            edge["source"] = "llm"

        return parsed


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
    return result.get("edges", [])


def llm_extractor_instance() -> LLMExtractor | None:
    """Return the module-level LLM extractor, or None."""
    return _get_extractor()
