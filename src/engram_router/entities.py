"""Conservative, rule-based entity extraction for EngramRouter.

No LLM. We only emit entities we can defend from surface patterns, so that
extraction never invents facts (consistent with the project's evidence-first
stance). Each entity carries a ``kind`` and the ``evidence`` substring it was
drawn from.

All word lists and patterns are configurable via ~/.engram/config.yaml.
See engram_router.config for the full schema and defaults.
"""

from __future__ import annotations

import re
from typing import Any

from .config import config as cfg


# --- compiled regex (built once at import) -----------------------------------
_ASCII_OBJECT_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,}")


def classify_salience(entity: dict[str, Any], source_text: str) -> str:
    """Classify an entity's salience_class from its name + the source sentence."""
    name = entity.get("name", "")
    kind = entity.get("kind", "")

    if kind == "attribute":
        return "base_attr"

    for pat in cfg.salience.base_attr_name_patterns:
        if re.search(pat, name):
            return "base_attr"

    if any(m in source_text for m in cfg.salience.constraint_markers):
        return "constraint"
    if any(m in source_text for m in cfg.salience.decision_markers):
        return "decision"

    for pat in cfg.salience.sensory_patterns:
        if re.search(pat, source_text):
            return "sensory"

    if kind == "time":
        return "event"
    if any(m in source_text for m in cfg.salience.event_markers):
        return "event"

    if kind == "person" and any(re.search(c, source_text) for c in cfg.salience.base_attr_context):
        return "base_attr"

    return "event"


def _dedup(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for e in entities:
        key = (e["name"], e["kind"])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _extract_persons(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for kin in cfg.entities.kinship_words:
        if kin in text:
            out.append({"name": kin, "kind": "person", "evidence": kin})
    # Named-person-before-role: "张三是我同事" → 张三. But guard against
    # pronoun/filler prefixes ("我那个同事", "他朋友") — there the role word is
    # an abstract reference with NO named individual, so we must not capture a
    # bogus name like "我那个". Such a spurious person poisons the conflicting-
    # person penalty (it can't match any memory's person → everything conflicts).
    _NAME_STOP = ("我", "你", "他", "她", "它", "咱", "那个", "这个", "那位", "这位", "某")
    for m in re.finditer(
        r"([\u4e00-\u9fff]{2,3})(?:是我|是)?(?:" + "|".join(cfg.entities.role_words) + r")",
        text,
    ):
        name = m.group(1)
        # Drop the capture if it is (or ends with) a pronoun/filler — the role
        # word alone (同事/朋友) is then the only person reference.
        if name in _NAME_STOP or any(name.endswith(s) for s in _NAME_STOP):
            continue
        out.append({"name": name, "kind": "person", "evidence": m.group(0)})
    surname = cfg.entities.surname_chars
    # Surname-name: "张三" / "李四". Use a negative lookbehind (prev char is
    # neither another surname char nor 老/小) instead of requiring a leading
    # separator, so a name glued to a role/kinship word is still caught:
    # "同事李四" / "我朋友李四" → 李四 (previously missed because "事"/"友" are
    # not separators). The lookbehind still stops mid-name over-capture inside
    # an already-consumed 老/小 name ("老张" handled below, not re-split here).
    for m in re.finditer(
        rf"(?<![{surname}老小])([{surname}][\u4e00-\u9fff])([\u4e00-\u9fff]?)",
        text,
    ):
        base = m.group(1)
        third = m.group(2)
        name = base
        if third and third not in cfg.entities.name_breakers:
            name = base + third
        out.append({"name": name, "kind": "person", "evidence": name})
    # 老/小 + single char nickname: "老张" / "小李". Drop the leading-separator
    # requirement too (a negative lookbehind guards against 叠字), so "同事老张"
    # yields 老张 rather than nothing.
    for m in re.finditer(rf"(?<![{surname}老小])[老小]([\u4e00-\u9fff])", text):
        out.append({"name": m.group(0).strip(), "kind": "person", "evidence": m.group(0).strip()})
    return out


def _extract_companies(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for comp in cfg.entities.known_companies:
        if comp in text:
            out.append({"name": comp, "kind": "company", "evidence": comp})
    for marker in cfg.entities.company_markers:
        for m in re.finditer(r"([\u4e00-\u9fff]{2,6})" + marker, text):
            out.append({"name": m.group(0), "kind": "company", "evidence": m.group(0)})
    return out


def _extract_objects(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in _ASCII_OBJECT_RE.finditer(text):
        token = m.group(0)
        if token.lower() in cfg.entities.ascii_stop_words:
            continue
        out.append({"name": token, "kind": "object", "evidence": token})
    for obj in cfg.entities.known_objects:
        if obj in text:
            out.append({"name": obj, "kind": "object", "evidence": obj})
    for food in cfg.entities.food_words:
        if food in text:
            out.append({"name": food, "kind": "object", "evidence": food})
    return out


def _extract_attributes(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pat in cfg.entities.attr_patterns:
        for m in re.finditer(pat, text):
            out.append({"name": m.group(0), "kind": "attribute", "evidence": m.group(0)})
    return out


def _extract_time(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pat in cfg.entities.time_patterns:
        for m in re.finditer(pat, text):
            out.append({"name": m.group(0), "kind": "time", "evidence": m.group(0)})
    return out


def _extract_reason(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for marker in cfg.entities.reason_markers:
        if marker in text:
            out.append({"name": marker, "kind": "reason", "evidence": text})
            break
    return out


def _extract_topics(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for topic in cfg.entities.topic_words:
        if topic in text:
            out.append({"name": topic, "kind": "topic", "evidence": topic})
    for product, topic in cfg.entities.object_topic_aliases.items():
        if re.search(rf"(?<![A-Za-z]){re.escape(product)}(?![A-Za-z])", text):
            out.append({"name": topic, "kind": "topic", "evidence": product})
    return out


# ═══════════════════════════════════════════════════════════════════════════
# CJK n-gram 兜底实体提取（覆盖不在 known_objects 中的未知词）
# ═══════════════════════════════════════════════════════════════════════════

_CJK_STOP_BIGRAMS: set[str] = {
    "可以", "不是", "没有", "这个", "那个", "什么", "怎么", "一个", "这样",
    "就是", "还是", "因为", "所以", "但是", "如果", "虽然", "已经", "非常",
    "特别", "比较",
}


def _extract_cjk_bigrams(text: str) -> list[dict[str, Any]]:
    """兜底：真正的滑动窗口提取所有2字CJK连续子串（降权）。
    
    使用步长=1的滑动窗口，提取所有连续2字CJK子串作为候选实体。
    过滤常见停用词，标记为 cjk_ngram 类型，salience=0.3 避免干扰精确实体。
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    i = 0
    n = len(text)
    while i <= n - 2:
        w = text[i:i + 2]
        # 检查是否是纯 CJK (U+4E00-U+9FFF)
        if '\u4e00' <= w[0] <= '\u9fff' and '\u4e00' <= w[1] <= '\u9fff':
            if w not in seen and w not in _CJK_STOP_BIGRAMS:
                seen.add(w)
                out.append({
                    "name": w,
                    "kind": "cjk_ngram",
                    "evidence": w,
                    "salience": 0.3,
                })
            i += 1  # 滑动窗口: 步长=1
        else:
            i += 1
    return out


def extract_entities(text: str) -> list[dict[str, Any]]:
    """Return a deduped list of conservatively-extracted entities."""
    entities: list[dict[str, Any]] = []
    entities.extend(_extract_persons(text))
    entities.extend(_extract_companies(text))
    entities.extend(_extract_objects(text))
    entities.extend(_extract_attributes(text))
    entities.extend(_extract_time(text))
    entities.extend(_extract_reason(text))
    entities.extend(_extract_topics(text))
    # 兜底：CJK n-gram 实体提取，覆盖不在 known_objects 中的未知词
    entities.extend(_extract_cjk_bigrams(text))
    return _dedup(entities)
