"""
Persona: cross-session character profile aggregation for EngramRouter.

A Persona distills a person's stable attributes — age, occupation, personality
traits, preferences, and possessions — by aggregating evidence from multiple
tables (memories, entities, edges) across many sessions.  The results are
confidence-weighted (multi-source agreement → high confidence; contradictory
evidence → flagged conflict) and stored in a dedicated ``persona_attrs`` table
so the existing schema is never touched.

Design principles (consistent with the rest of EngramRouter):
  1. Every attribute is backed by at least one evidence reference.
  2. Confidence is a function of source count, source type, and contradiction.
  3. LLM is an optional enhancement — the rule-based aggregator works without it.
  4. Updates are incremental: only changed attributes are persisted.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class AttrEvidence:
    """A single piece of evidence for a persona attribute."""

    value: str
    """The asserted attribute value, e.g. ''32'', ''engineer''."""

    source: str = "rule"
    """Source type: 'rule', 'llm', 'edge', or 'memory'."""

    memory_id: str = ""
    """The memory that supplied this evidence (for auditability)."""

    confidence: float = 1.0
    """Per-source confidence (0.0–1.0)."""


@dataclass
class PersonaAttr:
    """A resolved attribute with aggregated evidence and confidence."""

    key: str
    """Normalised attribute key, e.g. 'age', 'occupation', 'personality'."""

    value: str
    """Best-guess value after evidence resolution."""

    confidence: float = 0.0
    """Aggregated confidence (0.0–1.0).  Multi-source agreement drives this up."""

    evidence: list[AttrEvidence] = field(default_factory=list)
    """All evidence items that contributed to this attribute."""

    conflict: bool = False
    """True when contradictory evidence was found but one value was chosen."""

    conflicting_values: list[str] = field(default_factory=list)
    """When ``conflict`` is True, the alternative values that were rejected."""


@dataclass
class Persona:
    """A cross-session character profile for a named person or entity."""

    name: str
    """Normalised person name (as stored in the entities table)."""

    attrs: dict[str, PersonaAttr] = field(default_factory=dict)
    """key → PersonaAttr.  Keys are normalised to lowercase English slugs:
    'age', 'occupation', 'personality', 'preference', etc."""

    objects: list[dict[str, Any]] = field(default_factory=list)
    """Possessions / owned objects inferred from edges + entity mentions.
    Each entry: {'name': str, 'evidence': [AttrEvidence], 'confidence': float}."""

    last_updated: str = ""
    """ISO-8601 timestamp of the last full aggregation."""

    @property
    def age(self) -> str | None:
        """Convenience accessor for age (if known)."""
        a = self.attrs.get("age")
        return a.value if a else None

    @property
    def occupation(self) -> str | None:
        """Convenience accessor for occupation (if known)."""
        a = self.attrs.get("occupation")
        return a.value if a else None

    @property
    def personality(self) -> str | None:
        """Convenience accessor for personality traits (if known)."""
        a = self.attrs.get("personality")
        return a.value if a else None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (for JSON export / API responses)."""
        return {
            "name": self.name,
            "attrs": {
                k: asdict(v) for k, v in self.attrs.items()
            },
            "objects": self.objects,
            "last_updated": self.last_updated,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Attribute key normalisation
# ──────────────────────────────────────────────────────────────────────────────

# Maps Chinese attribute names and relation labels to normalised English keys.
_ATTR_KEY_MAP: dict[str, str] = {
    # From config.entities.attr_patterns
    "年龄": "age",
    "性别": "gender",
    "生日": "birthday",
    "出生": "birthday",
    "职业": "occupation",
    "工作": "occupation",
    "性格": "personality",
    "个性": "personality",
    "喜好": "preference",
    "喜欢": "preference",
    "偏好": "preference",
    "国籍": "nationality",
    "家乡": "hometown",
    "住在": "residence",
    "住": "residence",
    # From edges relation labels
    "HAS_ATTRIBUTE": "attribute",
    "PREFERS": "preference",
    "KNOWS": "skill",
    "IS_A": "role",
    # Default fallback
}


def _normalise_attr_key(raw: str) -> str:
    """Map a raw attribute key (Chinese / relation label) to a normalised slug."""
    # Direct lookup
    if raw in _ATTR_KEY_MAP:
        return _ATTR_KEY_MAP[raw]
    # Partial match: check if the raw key contains a known key
    for cn_key, en_key in _ATTR_KEY_MAP.items():
        if cn_key in raw:
            return en_key
    # Fallback: lowercase the raw key
    return raw.lower().strip()


# ──────────────────────────────────────────────────────────────────────────────
# Rule-based attribute extraction from raw text
# ──────────────────────────────────────────────────────────────────────────────


def _extract_attrs_from_text(text: str) -> list[dict[str, Any]]:
    """Apply the same attribute patterns as entities.py to extract explicit
    attribute-value pairs from raw memory text.

    Returns a list of dicts with 'key' (normalised) and 'value'.
    """
    attrs: list[dict[str, Any]] = []
    for pat in config.entities.attr_patterns:
        for m in __import__("re").finditer(pat, text):
            raw_val = m.group(0)
            # Determine the key from the pattern match context
            key = _normalise_attr_key(raw_val)
            attrs.append({"key": key, "value": raw_val})
    return attrs


# ──────────────────────────────────────────────────────────────────────────────
# PersonaStore
# ──────────────────────────────────────────────────────────────────────────────


class PersonaStore:
    """Aggregates cross-session personality attributes for named entities.

    ``PersonaStore`` wraps an existing ``MemoryStore``'s database connection
    and adds a single new table (``persona_attrs``) on first use.  It does
    **not** alter the existing schema.

    Args:
        store: An initialised ``MemoryStore`` instance whose ``.conn``
               will be used for all queries.
        llm_extractor: Optional LLM extractor for higher-quality attribute
                       resolution (used only when available and API key set).
    """

    def __init__(
        self,
        store: Any,  # MemoryStore (avoid circular import)
        llm_extractor: Any | None = None,
    ) -> None:
        self._store = store
        self._conn: sqlite3.Connection = store.conn  # type: ignore[assignment]
        self._llm_extractor = llm_extractor
        self._llm_available = (
            llm_extractor is not None and getattr(llm_extractor, "available", False)
        )
        self._init_table()

    # ── schema ────────────────────────────────────────────────────────────

    def _init_table(self) -> None:
        """Create the ``persona_attrs`` table if it does not exist.

        This is the **only** schema addition.  It does not touch any existing
        table.
        """
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS persona_attrs (
                id TEXT PRIMARY KEY,
                person_name TEXT NOT NULL,
                attr_key TEXT NOT NULL,
                attr_value TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.0,
                evidence_json TEXT NOT NULL DEFAULT '[]',
                conflict INTEGER NOT NULL DEFAULT 0,
                conflicting_values TEXT NOT NULL DEFAULT '[]',
                last_updated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_pa_person_name
                ON persona_attrs(person_name);

            CREATE INDEX IF NOT EXISTS idx_pa_key
                ON persona_attrs(person_name, attr_key);
            """
        )
        self._conn.commit()

    # ── id allocation ─────────────────────────────────────────────────────

    def _next_id(self) -> str:
        """Generate a unique id via the existing id_sequences mechanism."""
        # Reuse the store's _next_id if we can; otherwise allocate directly.
        try:
            return self._store._next_id("persona_attrs", "pa")
        except (AttributeError, Exception):
            # Fallback: simple monotic counter via the id_sequences table.
            row = self._conn.execute(
                "UPDATE id_sequences SET next_val = next_val + 1 "
                "WHERE name = 'persona_attrs' RETURNING next_val - 1 AS value"
            ).fetchone()
            if row is None:
                max_row = self._conn.execute(
                    "SELECT MAX(CAST(SUBSTR(id, 4) AS INTEGER)) FROM persona_attrs"
                ).fetchone()
                seed = (int(max_row[0]) + 1) if max_row and max_row[0] else 1
                self._conn.execute(
                    "INSERT OR IGNORE INTO id_sequences (name, next_val) VALUES ('persona_attrs', ?)",
                    (seed,),
                )
                row = self._conn.execute(
                    "UPDATE id_sequences SET next_val = next_val + 1 "
                    "WHERE name = 'persona_attrs' RETURNING next_val - 1 AS value"
                ).fetchone()
            return f"pa_{int(row['value'])}"

    # ── aggregation ───────────────────────────────────────────────────────

    def aggregate(self, person_name: str) -> Persona:
        """Aggregate all known attributes for *person_name* across all sessions.

        The pipeline:
          1. Find the entity entry for *person_name*.
          2. Collect all memories that mention this person (via memory_entities).
          3. Extract explicit attribute-value pairs from the raw text of each memory.
          4. Collect edge-based attributes (HAS_ATTRIBUTE, PREFERS, KNOWS, etc.).
          5. Merge evidence with confidence weighting; detect conflicts.
          6. Optionally invoke an LLM to resolve conflicts (when available).

        Returns a ``Persona`` dataclass.
        """
        persona = Persona(name=person_name)

        # 1. Locate the entity(ies) for this person.
        entity_rows = self._conn.execute(
            "SELECT id, name, kind FROM entities WHERE name = ?",
            (person_name,),
        ).fetchall()
        if not entity_rows:
            return persona  # No entity → empty persona.

        entity_ids = [r["id"] for r in entity_rows]

        # 2. Collect memory ids that mention this person.
        placeholders = ",".join("?" for _ in entity_ids)
        me_rows = self._conn.execute(
            f"SELECT DISTINCT memory_id, evidence FROM memory_entities "
            f"WHERE entity_id IN ({placeholders})",
            entity_ids,
        ).fetchall()
        memory_ids = [r["memory_id"] for r in me_rows]
        if not memory_ids:
            return persona  # No memories → empty persona.

        # 3. Fetch the actual memory text.
        mem_placeholders = ",".join("?" for _ in memory_ids)
        mem_rows = self._conn.execute(
            f"SELECT id, raw_text, summary, confidence FROM memories "
            f"WHERE id IN ({mem_placeholders})",
            memory_ids,
        ).fetchall()
        memory_map: dict[str, sqlite3.Row] = {r["id"]: r for r in mem_rows}

        # 4. Extract attribute evidence from memories.
        evidence_by_key: dict[str, list[AttrEvidence]] = {}

        for me_row in me_rows:
            mid = me_row["memory_id"]
            mem = memory_map.get(mid)
            if mem is None:
                continue
            text = mem["raw_text"]
            mem_conf = float(mem["confidence"])

            for attr in _extract_attrs_from_text(text):
                key = attr["key"]
                ev = AttrEvidence(
                    value=attr["value"],
                    source="rule",
                    memory_id=mid,
                    confidence=mem_conf,
                )
                evidence_by_key.setdefault(key, []).append(ev)

        # 5. Collect edge-based evidence.
        edge_evidence_by_key = self._collect_edge_evidence(entity_ids, memory_map)
        for key, ev_list in edge_evidence_by_key.items():
            evidence_by_key.setdefault(key, []).extend(ev_list)

        # 6. Collect personality traits from descriptive text.
        trait_evidence = self._extract_traits_from_texts(
            person_name, [r["raw_text"] for r in mem_rows]
        )
        if trait_evidence:
            evidence_by_key.setdefault("personality", []).extend(trait_evidence)

        # 7. Collect preference evidence.
        pref_evidence = self._collect_preference_evidence(entity_ids, memory_map)
        if pref_evidence:
            evidence_by_key.setdefault("preference", []).extend(pref_evidence)

        # 8. Resolve evidence into PersonaAttr values.
        for key, evidence_list in evidence_by_key.items():
            attr = self._resolve_evidence(key, evidence_list)
            persona.attrs[key] = attr

        # 9. Collect owned objects.
        persona.objects = self._collect_objects(entity_ids, memory_map)

        # 10. Optional LLM enhancement.
        if self._llm_available:
            persona = self._enhance_with_llm(persona)

        # 11. Timestamp.
        from datetime import datetime, timezone

        persona.last_updated = datetime.now(timezone.utc).isoformat()

        return persona

    # ── edge evidence ─────────────────────────────────────────────────────

    def _collect_edge_evidence(
        self,
        entity_ids: list[str],
        memory_map: dict[str, sqlite3.Row],
    ) -> dict[str, list[AttrEvidence]]:
        """Collect attribute-like evidence from edges where this person is
        the source or destination.

        Relevant relations:
          - HAS_ATTRIBUTE: src PERSON, dst ATTRIBUTE_VALUE
          - PREFERS: src PERSON, dst PREFERRED_ITEM
          - KNOWS: src PERSON, dst SKILL/TOPIC
          - IS_A: src PERSON, dst ROLE
        """
        result: dict[str, list[AttrEvidence]] = {}
        id_placeholders = ",".join("?" for _ in entity_ids)

        edge_rows = self._conn.execute(
            f"SELECT e.id, e.src_id, e.dst_id, e.relation, e.confidence, e.evidence_ref "
            f"FROM edges e "
            f"WHERE (e.src_id IN ({id_placeholders}) OR e.dst_id IN ({id_placeholders})) "
            f"AND e.relation IN ('HAS_ATTRIBUTE', 'PREFERS', 'KNOWS', 'IS_A')",
            entity_ids + entity_ids,
        ).fetchall()

        for edge in edge_rows:
            relation = edge["relation"]
            edge_conf = float(edge["confidence"])
            evidence_ref = edge["evidence_ref"]

            # Determine which endpoint is the value.
            if edge["src_id"] in entity_ids:
                # Person is source; destination carries the value.
                dst_ent = self._conn.execute(
                    "SELECT name, kind FROM entities WHERE id = ?",
                    (edge["dst_id"],),
                ).fetchone()
                if dst_ent:
                    key = _normalise_attr_key(relation)
                    ev = AttrEvidence(
                        value=dst_ent["name"],
                        source="edge",
                        memory_id=evidence_ref.split(":")[0],
                        confidence=edge_conf,
                    )
                    result.setdefault(key, []).append(ev)
            elif edge["dst_id"] in entity_ids:
                # Person is destination; source carries the value.
                src_ent = self._conn.execute(
                    "SELECT name, kind FROM entities WHERE id = ?",
                    (edge["src_id"],),
                ).fetchone()
                if src_ent:
                    key = _normalise_attr_key(relation)
                    ev = AttrEvidence(
                        value=src_ent["name"],
                        source="edge",
                        memory_id=evidence_ref.split(":")[0],
                        confidence=edge_conf,
                    )
                    result.setdefault(key, []).append(ev)

        return result

    # ── trait extraction ──────────────────────────────────────────────────

    #: Chinese personality trait markers — each word is a mild signal.
    TRAIT_PATTERNS: list[str] = [
        # Personality
        r"(性格|个性|脾气|为人)(很|非常|比较|有点|挺|相当)?([\u4e00-\u9fff]{2,6})",
        # Explicit trait description: "他是一个X的人"
        r"(是|算是)(一?个?)(很|非常|比较|有点|挺|相当)?([\u4e00-\u9fff]{2,4})的?人",
    ]

    def _extract_traits_from_texts(
        self, person_name: str, texts: list[str]
    ) -> list[AttrEvidence]:
        """Scan memory texts for personality trait descriptions."""
        import re

        evidence: list[AttrEvidence] = []
        for text in texts:
            if person_name not in text:
                continue
            for pat in self.TRAIT_PATTERNS:
                for m in re.finditer(pat, text):
                    # The last group is the trait value.
                    groups = m.groups()
                    if groups:
                        trait_value = groups[-1]
                        if trait_value:
                            evidence.append(
                                AttrEvidence(
                                    value=trait_value,
                                    source="rule",
                                    confidence=0.7,  # Traits are subjective, lower confidence
                                )
                            )
        return evidence

    # ── preference evidence ───────────────────────────────────────────────

    PREFERENCE_PATTERNS: list[str] = [
        r"(喜欢|讨厌|爱|恨|偏好|青睐)([\u4e00-\u9fffA-Za-z0-9]{1,10})",
        r"(特别|最|非常|很)(喜欢|讨厌|爱)([\u4e00-\u9fffA-Za-z0-9]{1,10})",
    ]

    def _collect_preference_evidence(
        self,
        entity_ids: list[str],
        memory_map: dict[str, sqlite3.Row],
    ) -> list[AttrEvidence]:
        """Extract preference signals from memory texts."""
        import re

        evidence: list[AttrEvidence] = []
        for mem in memory_map.values():
            text = mem["raw_text"]
            for pat in self.PREFERENCE_PATTERNS:
                for m in re.finditer(pat, text):
                    groups = m.groups()
                    if groups:
                        preference_value = groups[-1]
                        if preference_value and len(preference_value) >= 2:
                            evidence.append(
                                AttrEvidence(
                                    value=preference_value,
                                    source="rule",
                                    memory_id=mem["id"],
                                    confidence=float(mem["confidence"]),
                                )
                            )
        return evidence

    # ── evidence resolution ───────────────────────────────────────────────

    def _resolve_evidence(
        self, key: str, evidence_list: list[AttrEvidence]
    ) -> PersonaAttr:
        """Merge multiple evidence items into a single resolved attribute.

        Strategy:
          - Group by value string.
          - The value with the most evidence wins.
          - Confidence = sum of per-source confidences / max possible (capped at 1.0).
          - If multiple values have evidence, mark as conflict and record alternatives.
        """
        if not evidence_list:
            return PersonaAttr(key=key, value="", confidence=0.0)

        # Group evidence by value and sum confidence.
        value_scores: dict[str, float] = {}
        value_evidence: dict[str, list[AttrEvidence]] = {}
        for ev in evidence_list:
            v = ev.value.strip()
            if not v:
                continue
            value_scores[v] = value_scores.get(v, 0.0) + ev.confidence
            value_evidence.setdefault(v, []).append(ev)

        if not value_scores:
            return PersonaAttr(key=key, value="", confidence=0.0)

        # Sort values by aggregated confidence (descending).
        sorted_values = sorted(value_scores.items(), key=lambda x: x[1], reverse=True)
        best_value, best_score = sorted_values[0]

        # Normalise confidence: max possible = len(evidence) * 1.0, but
        # we soften it so a single source doesn't get 1.0.
        max_possible = max(len(evidence_list), 2)
        normalised_conf = min(best_score / max_possible, 1.0)

        # Check for conflict: other values with non-trivial evidence.
        conflict = False
        conflicting = []
        for alt_value, alt_score in sorted_values[1:]:
            if alt_score > 0.0:
                conflict = True
                conflicting.append(alt_value)

        return PersonaAttr(
            key=key,
            value=best_value,
            confidence=round(normalised_conf, 4),
            evidence=value_evidence.get(best_value, []),
            conflict=conflict,
            conflicting_values=conflicting,
        )

    # ── owned objects ─────────────────────────────────────────────────────

    def _collect_objects(
        self,
        entity_ids: list[str],
        memory_map: dict[str, sqlite3.Row],
    ) -> list[dict[str, Any]]:
        """Collect objects that belong to / are associated with this person.

        Strategy:
          - Find object entities that co-occur with this person in memories.
          - Look at CO_OCCURS_WITH edges where one endpoint is the person
            and the other is an object.
          - Look for ownership markers (我的, 他的, 她的, 送了我, 给了我).
        """
        import re

        id_placeholders = ",".join("?" for _ in entity_ids)

        # 1. Objects that co-occur with this person in memories.
        obj_rows = self._conn.execute(
            f"SELECT DISTINCT e.name, e.kind, me.memory_id, me.evidence "
            f"FROM memory_entities me "
            f"JOIN entities e ON e.id = me.entity_id "
            f"WHERE e.kind = 'object' "
            f"AND me.memory_id IN ("
            f"  SELECT DISTINCT memory_id FROM memory_entities "
            f"  WHERE entity_id IN ({id_placeholders})"
            f")",
            entity_ids,
        ).fetchall()

        # 2. Objects connected via edges.
        edge_obj_rows = self._conn.execute(
            f"SELECT e.name AS obj_name, ed.evidence_ref AS memory_id, ed.confidence "
            f"FROM edges ed "
            f"JOIN entities e ON ("
            f"  (ed.src_id IN ({id_placeholders}) AND ed.dst_id = e.id) "
            f"  OR (ed.dst_id IN ({id_placeholders}) AND ed.src_id = e.id)"
            f") "
            f"WHERE e.kind = 'object' "
            f"AND ed.relation IN ('CO_OCCURS_WITH', 'DESCRIBES', 'OWNS')",
            entity_ids + entity_ids,
        ).fetchall()

        # Merge by object name with evidence and confidence.
        obj_evidence: dict[str, dict[str, Any]] = {}

        for row in obj_rows:
            name = row["name"]
            if name not in obj_evidence:
                obj_evidence[name] = {
                    "name": name,
                    "evidence": [],
                    "confidence": 0.0,
                }
            obj_evidence[name]["evidence"].append(
                {
                    "source": "co_occurrence",
                    "memory_id": row["memory_id"],
                    "confidence": 0.4,  # Co-occurrence is weak evidence.
                }
            )
            obj_evidence[name]["confidence"] += 0.4

        for row in edge_obj_rows:
            name = row["obj_name"]
            if name not in obj_evidence:
                obj_evidence[name] = {
                    "name": name,
                    "evidence": [],
                    "confidence": 0.0,
                }
            obj_evidence[name]["evidence"].append(
                {
                    "source": "edge",
                    "memory_id": row["memory_id"].split(":")[0],
                    "confidence": float(row["confidence"]),
                }
            )
            obj_evidence[name]["confidence"] += float(row["confidence"])

        # Boost confidence for objects with ownership markers.
        for name, entry in obj_evidence.items():
            ownership_patterns = [
                re.compile(rf"{re.escape(name)}.*?(?:我的|他的|她的|送了我|给了我|拥有|持有)"),
            ]
            for mem in memory_map.values():
                text = mem["raw_text"]
                for pat in ownership_patterns:
                    if pat.search(text):
                        entry["confidence"] += 0.5

        # Normalise confidence.
        for entry in obj_evidence.values():
            max_conf = max(len(entry["evidence"]), 3)
            entry["confidence"] = min(entry["confidence"] / max_conf, 1.0)
            entry["confidence"] = round(entry["confidence"], 4)

        return sorted(
            obj_evidence.values(),
            key=lambda x: x["confidence"],
            reverse=True,
        )

    # ── LLM enhancement ───────────────────────────────────────────────────

    def _enhance_with_llm(self, persona: Persona) -> Persona:
        """Use an LLM to resolve conflicting attributes and suggest merges.

        This is a best-effort enhancement.  If the LLM call fails, the
        rule-based persona is returned unmodified.
        """
        if not self._llm_extractor:
            return persona

        try:
            # Build a prompt summarising the evidence.
            lines = [f"Person: {persona.name}"]
            for key, attr in persona.attrs.items():
                if attr.conflict:
                    lines.append(
                        f"  {key}: best='{attr.value}' "
                        f"(conf={attr.confidence:.2f}) "
                        f"conflicts={attr.conflicting_values}"
                    )
                else:
                    lines.append(
                        f"  {key}: '{attr.value}' (conf={attr.confidence:.2f})"
                    )

            prompt = (
                "You are a character profile analyst. Given the following "
                "aggregated evidence about a person, resolve any conflicting "
                "attributes and suggest the most likely value for each.\n\n"
                + "\n".join(lines)
                + "\n\nRespond with a JSON object mapping attribute keys to "
                "resolved values. Only include attributes where you can "
                "improve the resolution. Format: "
                '{"age": "32", "occupation": "software engineer"}'
            )

            # Use the LLM extractor's client if available.
            if hasattr(self._llm_extractor, "_call_llm"):
                response = self._llm_extractor._call_llm(prompt)
            elif hasattr(self._llm_extractor, "client"):
                response = self._llm_extractor.client.chat(prompt)
            else:
                return persona

            # Parse JSON from the response.
            resolved = self._parse_llm_json(response)
            for key, value in resolved.items():
                norm_key = _normalise_attr_key(key)
                if norm_key in persona.attrs:
                    # Update the value but preserve evidence.
                    persona.attrs[norm_key].value = str(value)
                    persona.attrs[norm_key].conflict = False
                    persona.attrs[norm_key].conflicting_values = []
                else:
                    persona.attrs[norm_key] = PersonaAttr(
                        key=norm_key,
                        value=str(value),
                        confidence=0.8,  # LLM-derived confidence.
                        evidence=[
                            AttrEvidence(
                                value=str(value),
                                source="llm",
                                confidence=0.8,
                            )
                        ],
                    )
        except Exception as exc:
            logger.debug("LLM persona enhancement failed: %s", exc)

        return persona

    @staticmethod
    def _parse_llm_json(raw: str) -> dict[str, Any]:
        """Best-effort JSON extraction from LLM output."""
        import re

        # Try to find a JSON object anywhere in the response.
        json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        # Try parsing the entire string directly.
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    # ── persistence ────────────────────────────────────────────────────────

    def save(self, persona: Persona) -> None:
        """Persist a Persona to the ``persona_attrs`` table.

        Uses an incremental upsert: existing rows for the same (person_name, attr_key)
        are updated; new rows are inserted.
        """
        now = self._now_iso()
        for key, attr in persona.attrs.items():
            evidence_json = json.dumps(
                [asdict(e) for e in attr.evidence],
                ensure_ascii=False,
            )
            conflicting_json = json.dumps(
                attr.conflicting_values, ensure_ascii=False
            )

            # Check for existing row.
            existing = self._conn.execute(
                "SELECT id FROM persona_attrs WHERE person_name = ? AND attr_key = ?",
                (persona.name, key),
            ).fetchone()

            if existing:
                self._conn.execute(
                    """UPDATE persona_attrs
                       SET attr_value = ?, confidence = ?, evidence_json = ?,
                           conflict = ?, conflicting_values = ?, last_updated = ?
                       WHERE id = ?""",
                    (
                        attr.value,
                        attr.confidence,
                        evidence_json,
                        1 if attr.conflict else 0,
                        conflicting_json,
                        now,
                        existing["id"],
                    ),
                )
            else:
                new_id = self._next_id()
                self._conn.execute(
                    """INSERT INTO persona_attrs
                       (id, person_name, attr_key, attr_value, confidence,
                        evidence_json, conflict, conflicting_values, last_updated)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        new_id,
                        persona.name,
                        key,
                        attr.value,
                        attr.confidence,
                        evidence_json,
                        1 if attr.conflict else 0,
                        conflicting_json,
                        now,
                    ),
                )

        self._conn.commit()

    def load(self, person_name: str) -> Persona | None:
        """Load a previously saved Persona from the database.

        Returns ``None`` if no attributes are stored for the name.
        """
        rows = self._conn.execute(
            "SELECT * FROM persona_attrs WHERE person_name = ? ORDER BY attr_key",
            (person_name,),
        ).fetchall()

        if not rows:
            return None

        persona = Persona(name=person_name)
        for row in rows:
            evidence_raw = row["evidence_json"]
            try:
                evidence_data = json.loads(evidence_raw)
                evidence = [
                    AttrEvidence(**e) for e in evidence_data
                ]
            except (json.JSONDecodeError, TypeError):
                evidence = []

            conflicting_raw = row["conflicting_values"]
            try:
                conflicting = json.loads(conflicting_raw)
            except (json.JSONDecodeError, TypeError):
                conflicting = []

            persona.attrs[row["attr_key"]] = PersonaAttr(
                key=row["attr_key"],
                value=row["attr_value"],
                confidence=float(row["confidence"]),
                evidence=evidence,
                conflict=bool(row["conflict"]),
                conflicting_values=conflicting,
            )

        persona.last_updated = rows[0]["last_updated"] if rows else ""
        return persona

    def update(self, persona: Persona) -> Persona:
        """Incrementally update an existing persona rather than fully rebuilding.

        Strategy:
          1. Load the existing persisted state (if any).
          2. Re-aggregate only from new/changed memories (incremental).
          3. Merge new evidence with existing evidence.
          4. Re-resolve and save.

        For now, this falls back to a full ``aggregate + save`` when no
        existing persisted state is found, which is the simplest correct
        implementation.  Future optimisations can track a ``last_aggregated_id``
        watermark per person.
        """
        existing = self.load(persona.name)
        if existing is None:
            # No prior state; do a full aggregation.
            fresh = self.aggregate(persona.name)
            self.save(fresh)
            return fresh

        # Merge: for each attribute in the fresh persona, merge evidence
        # with existing evidence (avoid duplicate memory_ids).
        fresh = self.aggregate(persona.name)

        for key, fresh_attr in fresh.attrs.items():
            if key in existing.attrs:
                existing_attr = existing.attrs[key]
                existing_mem_ids = {e.memory_id for e in existing_attr.evidence}
                for ev in fresh_attr.evidence:
                    if ev.memory_id not in existing_mem_ids:
                        existing_attr.evidence.append(ev)
                        existing_mem_ids.add(ev.memory_id)
                # Re-resolve with merged evidence.
                resolved = self._resolve_evidence(key, existing_attr.evidence)
                existing.attrs[key] = resolved
            else:
                existing.attrs[key] = fresh_attr

        # Merge objects.
        existing_obj_names = {o["name"] for o in existing.objects}
        for obj in fresh.objects:
            if obj["name"] not in existing_obj_names:
                existing.objects.append(obj)
                existing_obj_names.add(obj["name"])

        from datetime import datetime, timezone

        existing.last_updated = datetime.now(timezone.utc).isoformat()
        self.save(existing)
        return existing

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _now_iso() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()
