"""Causal chain reasoning and temporal event timeline for EngramRouter.

CausalChain:
    trace_causes(event)   → follow CAUSED_BY edges backward (causal chain).
    trace_effects(event)  → follow CAUSED_BY edges forward (impact chain).
    infer_chains()        → infer new causal links from CAUSED_BY +
                              CO_OCCURS_WITH patterns at low confidence (0.3).

Timeline:
    get_timeline(person=None) → time-ordered events, optionally filtered by person.
    get_events_between(start, end) → event range query on timed_events.

Integration:
    The MemoryStore gains a ``timed_events`` table that is auto-populated from
    entities(kind='time') linked via memory_entities.  The Timeline reads from
    this table for O(1) indexed temporal queries.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CausalChain
# ---------------------------------------------------------------------------

@dataclass
class CausalEdge:
    """A single edge in a causal or inferred chain."""
    src_id: str
    dst_id: str
    relation: str
    confidence: float
    evidence_ref: str
    src_name: str = ""
    dst_name: str = ""


@dataclass
class CausalPath:
    """A sequence of CausalEdge objects forming a chain."""
    edges: list[CausalEdge] = field(default_factory=list)
    confidence: float = 1.0  # product of edge confidences

    @property
    def length(self) -> int:
        return len(self.edges)

    @property
    def entities(self) -> list[str]:
        """Ordered entity names along the path."""
        if not self.edges:
            return []
        names = [self.edges[0].src_name]
        for e in self.edges:
            names.append(e.dst_name)
        return names

    def __repr__(self) -> str:
        chain = " → ".join(self.entities) if self.entities else "(empty)"
        return f"CausalPath({chain}, conf={self.confidence:.3f})"


class CausalChain:
    """Trace causal chains through the entity/edge graph.

    This class operates on a SQLite connection that has the standard
    EngramRouter schema (entities, edges, memory_entities, memories).

    Args:
        conn: A ``sqlite3.Connection`` with ``row_factory = sqlite3.Row``.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # -- helpers ---------------------------------------------------------------

    def _entity_name(self, entity_id: str) -> str:
        row = self.conn.execute(
            "SELECT name FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        return row["name"] if row else entity_id

    def _load_edges(self) -> list[CausalEdge]:
        rows = self.conn.execute(
            """SELECT e.id, e.src_id, e.dst_id, e.relation, e.confidence, e.evidence_ref
               FROM edges e"""
        ).fetchall()
        edges: list[CausalEdge] = []
        for r in rows:
            edges.append(
                CausalEdge(
                    src_id=r["src_id"],
                    dst_id=r["dst_id"],
                    relation=r["relation"],
                    confidence=r["confidence"],
                    evidence_ref=r["evidence_ref"],
                    src_name=self._entity_name(r["src_id"]),
                    dst_name=self._entity_name(r["dst_id"]),
                )
            )
        return edges

    # -- trace_causes ----------------------------------------------------------

    def trace_causes(
        self,
        event_entity_id: str,
        max_depth: int = 5,
    ) -> list[CausalPath]:
        """Trace the causal chain backwards from an entity.

        Follows CAUSED_BY edges in reverse: for each edge ``A --CAUSED_BY--> B``,
        B is the cause of A.  Starting from *event_entity_id* (an effect), we
        walk *inbound* CAUSED_BY edges: the entity appears as the *src* of a
        CAUSED_BY edge, and we follow the *dst* as its cause.

        Returns one or more causal paths, each a chain from the original event
        back through its causes.
        """
        edges = self._load_edges()
        caused_by = [e for e in edges if e.relation == "CAUSED_BY"]

        # Build an adjacency map: entity_id -> list of (cause_entity_id, edge)
        # An edge A --CAUSED_BY--> B means B caused A.  So if we're tracing
        # causes for entity X, we look for edges where src_id == X and follow
        # dst_id (the cause).
        cause_map: dict[str, list[tuple[str, CausalEdge]]] = {}
        for e in caused_by:
            cause_map.setdefault(e.src_id, []).append((e.dst_id, e))

        paths: list[CausalPath] = []

        def _dfs(current_id: str, path_edges: list[CausalEdge], visited: set[str]) -> None:
            if len(path_edges) >= max_depth:
                return
            for cause_id, edge in cause_map.get(current_id, []):
                if cause_id in visited:
                    continue
                new_visited = visited | {cause_id}
                new_path = path_edges + [edge]
                paths.append(CausalPath(
                    edges=list(new_path),
                    confidence=self._chain_confidence(new_path),
                ))
                _dfs(cause_id, new_path, new_visited)

        _dfs(event_entity_id, [], {event_entity_id})
        return paths

    # -- trace_effects ---------------------------------------------------------

    def trace_effects(
        self,
        cause_entity_id: str,
        max_depth: int = 5,
    ) -> list[CausalPath]:
        """Trace the effects / impact chain forward from a cause entity.

        Follows CAUSED_BY edges forward: for each edge ``A --CAUSED_BY--> B``,
        A is the effect of B.  Starting from *cause_entity_id* (a cause), we
        walk edges where the entity appears as the *dst* of a CAUSED_BY edge,
        and follow the *src* as its effect.
        """
        edges = self._load_edges()
        caused_by = [e for e in edges if e.relation == "CAUSED_BY"]

        # Build adjacency: for cause X, find all effects (edges where dst_id == X)
        effect_map: dict[str, list[tuple[str, CausalEdge]]] = {}
        for e in caused_by:
            effect_map.setdefault(e.dst_id, []).append((e.src_id, e))

        paths: list[CausalPath] = []

        def _dfs(current_id: str, path_edges: list[CausalEdge], visited: set[str]) -> None:
            if len(path_edges) >= max_depth:
                return
            for effect_id, edge in effect_map.get(current_id, []):
                if effect_id in visited:
                    continue
                new_visited = visited | {effect_id}
                new_path = path_edges + [edge]
                paths.append(CausalPath(
                    edges=list(new_path),
                    confidence=self._chain_confidence(new_path),
                ))
                _dfs(effect_id, new_path, new_visited)

        _dfs(cause_entity_id, [], {cause_entity_id})
        return paths

    # -- infer_chains ----------------------------------------------------------

    def infer_chains(
        self,
        min_confidence: float = 0.0,
    ) -> list[CausalEdge]:
        """Infer new causal chains from existing CAUSED_BY and CO_OCCURS_WITH edges.

        Inference rule (transitive closure):
            If  A --CAUSED_BY--> B   and   B --CO_OCCURS_WITH--> C
            then we *might* infer   A --INFERRED_CAUSED_BY--> C  (conf 0.3).

        The inferred edge confidence is the product of the two source edge
        confidences, capped at 0.3, so these are always clearly marked as
        low-confidence inference — never promoted to fact.

        Returns:
            List of inferred CausalEdge objects (not persisted).
        """
        edges = self._load_edges()

        caused_by = [e for e in edges if e.relation == "CAUSED_BY"]
        co_occurs = [e for e in edges if e.relation == "CO_OCCURS_WITH"]

        # Build lookup: for co-occur edges, we want undirected reachability.
        # CO_OCCURS_WITH is written as a single directed edge per pair (from the
        # earlier-indexed to later-indexed entity), but semantically it's
        # undirected.  We build both directions.
        co_map: dict[str, set[str]] = {}
        for e in co_occurs:
            co_map.setdefault(e.src_id, set()).add(e.dst_id)
            co_map.setdefault(e.dst_id, set()).add(e.src_id)

        inferred: list[CausalEdge] = []
        seen: set[tuple[str, str]] = set()

        for cb in caused_by:
            # cb: effect --CAUSED_BY--> cause
            effect_id = cb.src_id
            cause_id = cb.dst_id

            # If the cause co-occurs with some other entity C, then the effect
            # may also be (indirectly) related to C via the cause.
            for co_entity in co_map.get(cause_id, set()):
                if co_entity == effect_id:
                    continue
                key = (effect_id, co_entity)
                if key in seen:
                    continue
                seen.add(key)

                conf = round(min(cb.confidence * 0.4, 0.3), 3)
                if conf < min_confidence:
                    continue

                inferred.append(
                    CausalEdge(
                        src_id=effect_id,
                        dst_id=co_entity,
                        relation="INFERRED_CAUSED_BY",
                        confidence=conf,
                        evidence_ref=f"inferred:{cb.evidence_ref}",
                        src_name=cb.src_name,
                        dst_name=self._entity_name(co_entity),
                    )
                )

        # Also infer: if two entities share a common cause (both have CAUSED_BY
        # pointing to the same cause), they might be causally related.
        # A1 --CAUSED_BY--> C   and   A2 --CAUSED_BY--> C
        # → A1 --INFERRED_SHARED_CAUSE--> A2  (and vice versa)
        cause_to_effects: dict[str, list[CausalEdge]] = {}
        for cb in caused_by:
            cause_to_effects.setdefault(cb.dst_id, []).append(cb)

        for cause_id, effects in cause_to_effects.items():
            if len(effects) < 2:
                continue
            for i in range(len(effects)):
                for j in range(i + 1, len(effects)):
                    a1, a2 = effects[i], effects[j]
                    for src, dst in [(a1.src_id, a2.src_id), (a2.src_id, a1.src_id)]:
                        key = (src, dst)
                        if key in seen:
                            continue
                        seen.add(key)

                        conf = round(
                            min(a1.confidence * a2.confidence * 0.35, 0.3), 3
                        )
                        if conf < min_confidence:
                            continue

                        inferred.append(
                            CausalEdge(
                                src_id=src,
                                dst_id=dst,
                                relation="INFERRED_SHARED_CAUSE",
                                confidence=conf,
                                evidence_ref=f"inferred:shared_cause:{cause_id}",
                                src_name=self._entity_name(src),
                                dst_name=self._entity_name(dst),
                            )
                        )

        return inferred

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _chain_confidence(edges: list[CausalEdge]) -> float:
        """Product of edge confidences along a chain."""
        if not edges:
            return 1.0
        c = 1.0
        for e in edges:
            c *= e.confidence
        return round(c, 6)


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

@dataclass
class TimedEvent:
    """A single event anchored to a point in time."""
    id: str
    time_entity_id: str
    time_name: str           # e.g. "昨天", "前天", "上周"
    memory_id: str
    raw_text: str
    person_name: str | None   # extracted person entity name, if any
    created_at: str


class Timeline:
    """Query the temporal event timeline.

    The timeline is built from the ``timed_events`` table in the store, which
    is auto-populated from entities(kind='time') linked through memory_entities.

    Args:
        conn: A ``sqlite3.Connection`` with ``row_factory = sqlite3.Row``.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def get_timeline(
        self,
        person: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TimedEvent]:
        """Return time-ordered events, optionally filtered by person.

        Events are ordered by recency (most recent first).  The ``person``
        filter matches against person entities that appear in the same memory.

        Args:
            person: Optional person name filter.
            limit: Max number of events to return.
            offset: Pagination offset.
        """
        if person is not None:
            rows = self.conn.execute(
                """SELECT te.id, te.time_entity_id, te.time_name, te.memory_id,
                          te.raw_text, te.person_name, te.created_at
                   FROM timed_events te
                   WHERE te.person_name = ?
                   ORDER BY te.sort_order ASC, te.created_at DESC
                   LIMIT ? OFFSET ?""",
                (person, limit, offset),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT te.id, te.time_entity_id, te.time_name, te.memory_id,
                          te.raw_text, te.person_name, te.created_at
                   FROM timed_events te
                   ORDER BY te.sort_order ASC, te.created_at DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()

        return [
            TimedEvent(
                id=r["id"],
                time_entity_id=r["time_entity_id"],
                time_name=r["time_name"],
                memory_id=r["memory_id"],
                raw_text=r["raw_text"],
                person_name=r["person_name"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def get_events_between(
        self,
        start: str,
        end: str,
        limit: int = 100,
    ) -> list[TimedEvent]:
        """Return events whose time-name falls between *start* and *end*.

        This uses a simple string-based comparison on sort_order as a proxy
        for temporal ordering.  The sort_order is a numeric rank assigned
        during auto-population (lower = more recent).

        Args:
            start: Start time name (e.g. "前天").
            end: End time name (e.g. "上周").
            limit: Max events to return.
        """
        # Resolve sort_orders for the boundary terms
        start_order = self._time_sort_order(start)
        end_order = self._time_sort_order(end)

        if start_order < 0 or end_order < 0:
            # One or both time terms not in the database; fall back to string
            # pattern matching.
            rows = self.conn.execute(
                """SELECT te.id, te.time_entity_id, te.time_name, te.memory_id,
                          te.raw_text, te.person_name, te.created_at
                   FROM timed_events te
                   WHERE te.time_name >= ? AND te.time_name <= ?
                   ORDER BY te.sort_order ASC, te.created_at DESC
                   LIMIT ?""",
                (start, end, limit),
            ).fetchall()
        else:
            lo = min(start_order, end_order)
            hi = max(start_order, end_order)
            rows = self.conn.execute(
                """SELECT te.id, te.time_entity_id, te.time_name, te.memory_id,
                          te.raw_text, te.person_name, te.created_at
                   FROM timed_events te
                   WHERE te.sort_order BETWEEN ? AND ?
                   ORDER BY te.sort_order ASC, te.created_at DESC
                   LIMIT ?""",
                (lo, hi, limit),
            ).fetchall()

        return [
            TimedEvent(
                id=r["id"],
                time_entity_id=r["time_entity_id"],
                time_name=r["time_name"],
                memory_id=r["memory_id"],
                raw_text=r["raw_text"],
                person_name=r["person_name"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def _time_sort_order(self, time_name: str) -> int:
        """Resolve the numeric sort_order for a given time name.

        Returns -1 if the time name is unknown.
        """
        row = self.conn.execute(
            "SELECT sort_order FROM timed_events WHERE time_name = ? LIMIT 1",
            (time_name,),
        ).fetchone()
        return row["sort_order"] if row else -1


# ---------------------------------------------------------------------------
# timed_events table management (integrated into MemoryStore)
# ---------------------------------------------------------------------------

# Sort-order mapping for common Chinese time expressions.
# Lower = more recent (closer to "now").
_TIME_SORT_ORDER: dict[str, int] = {
    "刚才": 1,
    "今天": 2,
    "昨天": 3,
    "前天": 4,
    "前两天": 5,
    "前几天": 6,
    "这周": 10,
    "上周": 11,
    "上上周": 12,
    "这个月": 20,
    "上个月": 21,
}


def _resolve_sort_order(time_name: str) -> int:
    """Assign a numeric sort order to a time expression.

    Known expressions get fixed positions; numeric patterns (前N天, N天前)
    get derived positions.  Unrecognised expressions default to 50.
    """
    import re

    if time_name in _TIME_SORT_ORDER:
        return _TIME_SORT_ORDER[time_name]

    # "前N天" → N+4 (前天=4, so 前3天=7)
    m = re.match(r"前(\d+)天", time_name)
    if m:
        return 4 + int(m.group(1))

    # "最近" → 1 (very recent)
    if time_name == "最近":
        return 1

    # Date patterns like 2024年, 6月15日 → sort by year then month-day
    m = re.match(r"(\d{4})年", time_name)
    if m:
        year = int(m.group(1))
        return 1000 + year  # far future / past, but ordered

    m = re.match(r"(\d{1,2})月(\d{1,2})日", time_name)
    if m:
        return 30 + int(m.group(1))  # month-day within this year

    return 50  # unknown — put in the middle


def ensure_timed_events_table(conn: sqlite3.Connection) -> None:
    """Create the ``timed_events`` table if it doesn't exist.

    Call this from ``MemoryStore._init_schema()`` or ``_migrate_schema()``.
    """
    conn.execute(
        """CREATE TABLE IF NOT EXISTS timed_events (
            id TEXT PRIMARY KEY,
            time_entity_id TEXT NOT NULL,
            time_name TEXT NOT NULL,
            sort_order INTEGER NOT NULL DEFAULT 50,
            memory_id TEXT NOT NULL,
            raw_text TEXT NOT NULL DEFAULT '',
            person_name TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(time_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
            FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_timed_events_sort "
        "ON timed_events(sort_order, created_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_timed_events_person "
        "ON timed_events(person_name)"
    )


def populate_timed_events(conn: sqlite3.Connection) -> int:
    """Scan memory_entities for time-kind entities and populate timed_events.

    For each (memory_id, time_entity) pair, insert one row into timed_events
    if not already present.  Also attaches the first person entity found in
    the same memory.

    Returns the count of newly inserted rows.
    """
    # Find all (memory_id, entity_id, time_name) for kind='time'
    rows = conn.execute(
        """SELECT me.memory_id, me.entity_id AS time_entity_id,
                  e.name AS time_name, m.raw_text, m.created_at
           FROM memory_entities me
           JOIN entities e ON e.id = me.entity_id
           JOIN memories m ON m.id = me.memory_id
           WHERE e.kind = 'time'
           ORDER BY m.created_at DESC"""
    ).fetchall()

    # Collect existing keys to avoid duplicates
    existing = {
        r["memory_id"] + "|" + r["time_entity_id"]
        for r in conn.execute(
            "SELECT memory_id, time_entity_id FROM timed_events"
        ).fetchall()
    }

    # Look up person entities per memory
    memory_persons: dict[str, str | None] = {}
    person_rows = conn.execute(
        """SELECT me.memory_id, e.name
           FROM memory_entities me
           JOIN entities e ON e.id = me.entity_id
           WHERE e.kind = 'person'"""
    ).fetchall()
    for pr in person_rows:
        # Take the first person per memory
        if pr["memory_id"] not in memory_persons:
            memory_persons[pr["memory_id"]] = pr["name"]

    count = 0
    for r in rows:
        key = r["memory_id"] + "|" + r["time_entity_id"]
        if key in existing:
            continue

        time_name = r["time_name"]
        sort_order = _resolve_sort_order(time_name)
        person_name = memory_persons.get(r["memory_id"])

        # Generate a stable id from the composite key
        event_id = f"tev_{r['memory_id']}_{r['time_entity_id']}"

        conn.execute(
            """INSERT INTO timed_events
               (id, time_entity_id, time_name, sort_order, memory_id,
                raw_text, person_name, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                r["time_entity_id"],
                time_name,
                sort_order,
                r["memory_id"],
                r["raw_text"],
                person_name,
                r["created_at"],
            ),
        )
        count += 1

    return count
