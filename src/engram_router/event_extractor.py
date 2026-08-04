"""Structured event extraction from memory text (L2.1).

Extracts (subject, verb, object, time_expr, confidence) triples from
memory text using rule-based patterns + optional LLM augmentation.

Design principles:
  1. Rule-based first — no LLM dependency for basic SVO extraction.
  2. LLM augmentation is opt-in (``ENGRAM_EVENT_LLM=1``).
  3. Each event carries ``confidence`` and a ``memory_id`` backlink
     so downstream fact-voting and contradiction detection are auditable.
  4. Events are scoped by ``namespace``.

Usage::

    from engram_router.event_extractor import extract_events
    events = extract_events("张三昨天在北京买了一台电脑", memory_id="mem_1")
    # → [Event(subject="张三", verb="买", object="电脑", time_expr="昨天", ...)]
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Event dataclass ──────────────────────────────────────────────────────

@dataclass
class Event:
    """One structured event extracted from a memory."""

    subject: str = ""
    verb: str = ""
    object: str = ""
    time_expr: str = ""
    confidence: float = 0.8
    memory_id: str = ""
    namespace: str = "default"

    def to_sql_row(self, event_id: str) -> tuple[str, ...]:
        return (
            event_id,
            self.memory_id,
            self.subject,
            self.verb,
            self.object,
            self.time_expr,
            str(round(self.confidence, 4)),
            self.namespace,
        )

    @property
    def triple(self) -> tuple[str, str, str]:
        """Return (subject, verb, object) for dedup/matching."""
        return (self.subject, self.verb, self.object)


# ── Rule-based extraction ────────────────────────────────────────────────

# Common Chinese verbs that indicate action events.
# Each maps to a canonical form for dedup.
_VERB_MAP: dict[str, str] = {
    "买了": "买",
    "买了台": "买",
    "买了一台": "买",
    "买了个": "买",
    "买了一个": "买",
    "购入了": "买",
    "买了本": "买",
    "给了": "给",
    "送给了": "给",
    "送了": "送",
    "送给": "送",
    "去了": "去",
    "到了": "到",
    "来了": "来",
    "吃了": "吃",
    "写了": "写",
    "做了": "做",
    "开始": "做",
    "完成了": "完成",
    "学会了": "学会",
    "换了": "换",
    "辞职了": "辞职",
    "面试了": "面试",
    "住在": "住",
    "住了": "住",
    "住": "住",
    "工作了": "工作",
    "工作": "工作",
    "取得了": "取得",
    "获得了": "获得",
    "赢得了": "赢得",
    "投资了": "投资",
    "招聘了": "招聘",
    "开除了": "开除",
    "收购了": "收购",
    "研发了": "研发",
    "开发了": "开发",
    "发布了": "发布",
    "上线了": "上线",
    "部署了": "部署",
    "发表了": "发表",
    "入职了": "入职",
    "离职了": "离职",
    "上班": "工作",
    "在": "在"
}

# Common object modifiers to strip for canonicalization.
_OBJ_STRIP = frozenset({
    "一台", "一个", "一本", "一部", "一辆", "一件",
    "一家", "一次", "一笔", "一批", "一趟", "一番",
})

# Person name patterns — reusable from entity extraction.
# Simplified: check for known subject patterns.
_SUBJECT_MARKERS = frozenset({
    "我", "你", "他", "她", "我们", "你们", "他们",
    "公司", "团队", "老板", "客户",
})

# Time expressions that should be captured.
_TIME_PATTERNS = [
    (r"(昨天|今天|明天)", "day"),
    (r"(上周|本周|下周)", "week"),
    (r"(\d+月\d+日?)", "date"),
    (r"(\d+年\d+月)", "date"),
    (r"(刚才|刚刚|不久前)", "recent"),
    (r"(前\s*[两三四五六七八九十百]?\s*天)", "ago"),
    (r"(前\s*[两三四五六七八九十百]?\s*周)", "ago"),
    (r"(前\s*[两三四五六七八九十百]?\s*个?\s*月)", "ago"),
]

# ── Public API ───────────────────────────────────────────────────────────

def extract_events(
    text: str,
    *,
    memory_id: str = "",
    namespace: str = "default",
) -> list[Event]:
    """Extract structured events from *text*.

    Returns a list of Event objects, possibly empty if no event is detected.
    Events are rule-based by default; LLM augmentation is opt-in via
    ``ENGRAM_EVENT_LLM=1``.
    """
    events = _extract_rule_based(text, memory_id=memory_id, namespace=namespace)

    # LLM augmentation: only if explicitly enabled.
    if os.environ.get("ENGRAM_EVENT_LLM") in ("1", "true", "yes"):
        llm_events = _extract_llm(text, memory_id=memory_id, namespace=namespace)
        events = _merge_events(events, llm_events)

    return events


def _extract_rule_based(
    text: str,
    memory_id: str = "",
    namespace: str = "default",
) -> list[Event]:
    """Rule-based SVO extraction using surface patterns."""
    import re

    events: list[Event] = []

    # --- Find time expressions ---
    time_expr = ""
    for pat, _kind in _TIME_PATTERNS:
        m = re.search(pat, text)
        if m:
            time_expr = m.group(1)
            break

    # --- Find verb patterns ---
    for raw_verb, canonical in _VERB_MAP.items():
        if raw_verb in text:
            parts = text.split(raw_verb, 1)
            subject = _extract_subject(parts[0])
            obj = _extract_object(parts[1]) if len(parts) > 1 else ""

            if subject or obj:
                events.append(Event(
                    subject=subject,
                    verb=canonical,
                    object=obj,
                    time_expr=time_expr,
                    confidence=0.7 if subject and obj else 0.5,
                    memory_id=memory_id,
                    namespace=namespace,
                ))

    return events


def _extract_subject(before_verb: str) -> str:
    """Extract subject from text before the verb."""
    before = before_verb.strip()
    if not before:
        return ""

    import re
    surnames = "张李王赵刘陈杨黄周吴徐孙马朱胡林郭何高罗" \
               "郑梁谢宋唐许邓韩冯曹彭曾萧田董潘袁蔡蒋余" \
               "于杜叶程苏魏吕丁任卢姚沈钟姜崔谭陆范汪" \
               "廖石金韦贾夏付方邹熊孟秦邱江尹薛阎段雷" \
               "侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔汤"

    # Find surname-name pair closest to verb: surname + 1-2 CJK chars
    m = re.search(rf"[{surnames}老小][一-鿿]{{1,2}}$", before)
    if m:
        return m.group(0)

    # Fallback: named entity (company, team, etc.)
    for marker in ("公司", "团队", "部门", "项目组"):
        if marker in before:
            return marker

    return ""


def _extract_object(after_verb: str) -> str:
    """Extract object from text after the verb."""
    import re
    # Strip quantity prefixes
    for prefix in sorted(_OBJ_STRIP, key=len, reverse=True):
        after_verb = after_verb.replace(prefix, "")

    # Find the first noun phrase (CJK + optional ASCII)
    m = re.match(r"\s*([一-鿿A-Za-z0-9]{2,15})", after_verb)
    if m:
        obj = m.group(1)
        # Strip sentence-ending particles
        for suffix in ("了", "，", "。", "!", "！", "的", "还", "也"):
            if obj.endswith(suffix):
                obj = obj[:-1]
        return obj
    return ""


def _extract_llm(
    text: str,
    memory_id: str = "",
    namespace: str = "default",
) -> list[Event]:
    """LLM-based event extraction (opt-in)."""
    try:
        from .llm_extractor import LLMClient
        from .prompt_guard import wrap_user_content as _wrap

        client = LLMClient()
        if not client.available:
            return []

        messages = [
            {"role": "system", "content": _EVENT_LLM_SYSTEM_PROMPT},
            {"role": "user", "content": f"Text: {_wrap(text, max_chars=1000)}"},
        ]
        raw = client.chat(messages, temperature=0.0, max_tokens=512)
        return _parse_llm_events(raw, memory_id=memory_id, namespace=namespace)
    except Exception as exc:
        logger.debug("LLM event extraction failed: %s", exc)
        return []


_EVENT_LLM_SYSTEM_PROMPT = """\
You are a precise event extractor for a personal memory engine.
Extract (subject, verb, object, time) from the given text.
Output ONLY a JSON array, no markdown fences.

Schema:
[
  {
    "subject": "person or entity name",
    "verb": "action (canonical form: 买 not 买了)",
    "object": "thing or target",
    "time": "time expression if present (empty string if none)",
    "confidence": 0.5-1.0
  }
]

Rules:
1. subject MUST be a named person/company, NOT empty.
2. verb MUST be canonical (present tense, no aspect markers).
3. If no clear event, return empty array [].
4. One event per action (a single text may have multiple).
"""


def _parse_llm_events(
    raw: str,
    memory_id: str = "",
    namespace: str = "default",
) -> list[Event]:
    """Parse LLM JSON response into Event objects."""
    import json
    import re

    cleaned = raw.strip()
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```", "", cleaned)

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [
                Event(
                    subject=e.get("subject", ""),
                    verb=e.get("verb", ""),
                    object=e.get("object", ""),
                    time_expr=e.get("time", ""),
                    confidence=min(float(e.get("confidence", 0.8)), 1.0),
                    memory_id=memory_id,
                    namespace=namespace,
                )
                for e in data
                if e.get("subject")
            ]
    except json.JSONDecodeError:
        pass
    return []


def _merge_events(rule: list[Event], llm: list[Event]) -> list[Event]:
    """Merge rule-based and LLM events, deduplicating by triple."""
    seen: set[tuple[str, str, str]] = set()
    merged: list[Event] = []

    # LLM events first (higher confidence when available)
    for e in llm + rule:
        key = e.triple
        if key not in seen and e.subject:
            seen.add(key)
            merged.append(e)
    return merged
