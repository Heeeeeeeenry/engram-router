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
    for m in re.finditer(
        r"([\u4e00-\u9fff]{2,3})(?:是我|是)?(?:" + "|".join(cfg.entities.role_words) + r")",
        text,
    ):
        out.append({"name": m.group(1), "kind": "person", "evidence": m.group(0)})
    surname = cfg.entities.surname_chars
    for m in re.finditer(
        rf"(?:^|[，。、！？\s和跟与])([{surname}][\u4e00-\u9fff])([\u4e00-\u9fff]?)",
        text,
    ):
        base = m.group(1)
        third = m.group(2)
        name = base
        if third and third not in cfg.entities.name_breakers:
            name = base + third
        out.append({"name": name, "kind": "person", "evidence": name})
    for m in re.finditer(r"(?:^|[，。、！？\s])[老小]([\u4e00-\u9fff])", text):
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
    return _dedup(entities)
