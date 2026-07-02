"""SQLite-backed memory store for EngramRouter.

The store keeps raw evidence as the source of truth. Summaries and distilled
memories are retrieval aids, not replacements for raw evidence.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import config
from .entities import classify_salience, extract_entities
from .fusion import reciprocal_rank_fusion
from .llm_extractor import LLMExtractor, extract_edges_llm, extract_entities_llm
from .query_expansion import QueryExpander, ExpandedQuery

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    raw_text: str
    summary: str
    confidence: float = 1.0
    metadata: dict[str, Any] | None = None
    evidence_refs: list[str] | None = None
    score: float = 0.0
    match_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecallWeights:
    """Centralised recall scoring weights, injectable via MemoryStore.__init__.

    Every number that was previously a hard-coded literal lives here, with its
    current value as the default, so constructing ``RecallWeights()`` is
    behaviourally identical to the old hard-coded world.
    """

    # ── token scoring (_term_weight) ──────────────────────────────────────
    ascii_base: float = 4.0
    ascii_per_char_cap: int = 6
    ascii_per_char: float = 0.5
    cjk_multi_base: float = 2.0
    cjk_multi_per_char: float = 0.5
    stop_char_weight: float = 0.05
    single_cjk_weight: float = 0.4

    # ── _score semantic boosts ────────────────────────────────────────────
    colleague_boost: float = 1.0
    reason_marker_boost: float = 1.5

    # ── recall pipeline (_build_scored_candidates) ────────────────────────
    fts_boost: float = 0.1
    shared_entity_multiplier: float = 1.2
    conflicting_person_penalty: float = 2.2
    person_match_boost: float = 2.0
    entity_tie_break_bonus: float = 0.01

    # ── context boosts (_apply_context_boosts) ────────────────────────────
    brand_boost: float = 2.0
    occupation_boost: float = 1.5
    identity_base_attr_boost: float = 2.0
    eval_sensory_boost: float = 1.5

    # ── correction ────────────────────────────────────────────────────────
    correction_penalty: float = 0.3

    # ── spreading activation (was DEFAULT_MAX_HOPS / DEFAULT_DECAY / _ACT_THRESHOLD)
    max_recall_hops: int = 2
    recall_decay: float = 0.5
    activation_threshold: float = 0.03

    # ── associative reach (_apply_salience_decay) ─────────────────────────
    assoc_reach_base_attr: float = 0.15
    assoc_reach_constraint: float = 0.6
    assoc_reach_decision: float = 0.7
    assoc_reach_sensory: float = 1.0
    assoc_reach_event: float = 1.0

    # ── scale protection ──────────────────────────────────────────────────
    full_scan_limit: int = 2000
    """Cap rows loaded into Python when FTS5 returns no candidates.
    At 10K+ memories the full scan would OOM; 2000 rows is a generous
    budget for the Python ranker to find top_k from."""

    def __post_init__(self) -> None:
        """Validate weights on construction so misconfiguration is caught early."""
        if not (0 < self.recall_decay <= 1):
            raise ValueError(f"recall_decay must be in (0, 1], got {self.recall_decay}")
        if self.activation_threshold <= 0:
            raise ValueError(
                f"activation_threshold must be > 0, got {self.activation_threshold}"
            )
        if self.full_scan_limit < 1:
            raise ValueError(
                f"full_scan_limit must be >= 1, got {self.full_scan_limit}"
            )
        if self.max_recall_hops < 1:
            raise ValueError(
                f"max_recall_hops must be >= 1, got {self.max_recall_hops}"
            )


def _default_weights() -> RecallWeights:
    """Build RecallWeights from config, keeping defaults as fallback."""
    c = config.recall
    return RecallWeights(
        ascii_base=c.ascii_base, ascii_per_char_cap=c.ascii_per_char_cap,
        ascii_per_char=c.ascii_per_char, cjk_multi_base=c.cjk_multi_base,
        cjk_multi_per_char=c.cjk_multi_per_char, stop_char_weight=c.stop_char_weight,
        single_cjk_weight=c.single_cjk_weight,
        colleague_boost=c.colleague_boost, reason_marker_boost=c.reason_marker_boost,
        fts_boost=c.fts_boost, shared_entity_multiplier=c.shared_entity_multiplier,
        conflicting_person_penalty=c.conflicting_person_penalty,
        person_match_boost=c.person_match_boost,
        entity_tie_break_bonus=c.entity_tie_break_bonus,
        brand_boost=c.brand_boost, occupation_boost=c.occupation_boost,
        identity_base_attr_boost=c.identity_base_attr_boost,
        eval_sensory_boost=c.eval_sensory_boost,
        correction_penalty=c.correction_penalty,
        max_recall_hops=c.max_recall_hops, recall_decay=c.recall_decay,
        activation_threshold=c.activation_threshold,
        assoc_reach_base_attr=c.assoc_reach_base_attr,
        assoc_reach_constraint=c.assoc_reach_constraint,
        assoc_reach_decision=c.assoc_reach_decision,
        assoc_reach_sensory=c.assoc_reach_sensory,
        assoc_reach_event=c.assoc_reach_event,
        full_scan_limit=c.full_scan_limit,
    )


class MemoryStore:
    """Local SQLite memory store.

    Args:
        path: SQLite database path. If omitted, uses an in-memory database.
    """

    REASON_MARKERS = ("因为", "原因", "由于", "为了", "生日", "所以")

    def __init__(
        self,
        path: str | Path | None = None,
        max_recall_hops: int | None = None,
        recall_decay: float | None = None,
        weights: RecallWeights | None = None,
        llm_extractor: LLMExtractor | None = None,
        llm_query_extract: bool = False,
        reranker: Any | None = None,
        embedding_engine: Any | None = None,
        vector_index: Any | None = None,
        query_expander: QueryExpander | None = None,
    ) -> None:
        self.weights = weights if weights is not None else _default_weights()
        self.path = str(path) if path is not None else ":memory:"
        self.max_recall_hops = (
            max_recall_hops
            if max_recall_hops is not None
            else self.weights.max_recall_hops
        )
        self.recall_decay = (
            recall_decay
            if recall_decay is not None
            else self.weights.recall_decay
        )
        self.llm_extractor = llm_extractor
        self.llm_query_extract = llm_query_extract
        self.reranker = reranker
        self.embedding_engine = embedding_engine
        self.vector_index = vector_index
        self.query_expander = query_expander if query_expander is not None else QueryExpander()
        self._vector_enabled = (
            embedding_engine is not None
            and embedding_engine.available
            and vector_index is not None
        )
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        if sqlite3.sqlite_version_info < (3, 35, 0):
            raise RuntimeError(
                "EngramRouter requires SQLite >= 3.35.0 for atomic id allocation "
                f"(found {sqlite3.sqlite_version})"
            )
        self.conn = sqlite3.connect(self.path, timeout=5.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA busy_timeout = 5000")
        self.conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            try:
                self.conn.execute("PRAGMA journal_mode = WAL")
            except sqlite3.OperationalError:
                pass
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                raw_text TEXT NOT NULL,
                summary TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'conversation',
                confidence REAL NOT NULL DEFAULT 1.0,
                metadata TEXT NOT NULL DEFAULT '{}',
                namespace TEXT NOT NULL DEFAULT 'default',
                access_count INTEGER NOT NULL DEFAULT 0,
                accessed_at TEXT,
                forgotten INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS evidence (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                quote TEXT NOT NULL,
                source_location TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS raw_logs (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS distilled_memories (
                id TEXT PRIMARY KEY,
                raw_log_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                distilled_text TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(raw_log_id) REFERENCES raw_logs(id) ON DELETE CASCADE,
                FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'unknown',
                salience_class TEXT NOT NULL DEFAULT 'event'
            );

            CREATE TABLE IF NOT EXISTS memory_entities (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                evidence TEXT NOT NULL DEFAULT '',
                salience_class TEXT NOT NULL DEFAULT 'event',
                FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY(entity_id) REFERENCES entities(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                src_id TEXT NOT NULL,
                dst_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                evidence_ref TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS corrections (
                id TEXT PRIMARY KEY,
                target_id TEXT NOT NULL,
                correction_text TEXT NOT NULL,
                evidence_ref TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            -- Monotonic id allocator. One row per logical table; next_val only
            -- ever climbs, so deleting rows can never make _next_id hand back an
            -- id that already exists (the old COUNT(*)+1 scheme did exactly that
            -- and crashed on the PRIMARY KEY after any delete).
            CREATE TABLE IF NOT EXISTS id_sequences (
                name TEXT PRIMARY KEY,
                next_val INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS timed_events (
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
            );
            """
        )
        self._migrate_schema()
        self._init_fts()
        self._init_indices()
        self.conn.commit()

    def _migrate_schema(self) -> None:
        """Add columns introduced after the initial schema, idempotently.

        SQLite has no ``ADD COLUMN IF NOT EXISTS``; we probe ``PRAGMA
        table_info`` and add only what's missing so existing databases upgrade
        in place without data loss.
        """
        ent_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(entities)").fetchall()}
        if "salience_class" not in ent_cols:
            self.conn.execute(
                "ALTER TABLE entities ADD COLUMN salience_class TEXT NOT NULL DEFAULT 'event'"
            )
        me_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(memory_entities)").fetchall()}
        if "salience_class" not in me_cols:
            self.conn.execute(
                "ALTER TABLE memory_entities ADD COLUMN salience_class TEXT NOT NULL DEFAULT 'event'"
            )

        mem_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(memories)").fetchall()}
        if "namespace" not in mem_cols:
            self.conn.execute(
                "ALTER TABLE memories ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'"
            )

        # Phase 3: forgetting / decay columns.
        if "access_count" not in mem_cols:
            self.conn.execute(
                "ALTER TABLE memories ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0"
            )
        if "accessed_at" not in mem_cols:
            self.conn.execute(
                "ALTER TABLE memories ADD COLUMN accessed_at TEXT"
            )
        if "forgotten" not in mem_cols:
            self.conn.execute(
                "ALTER TABLE memories ADD COLUMN forgotten INTEGER NOT NULL DEFAULT 0"
            )

        # Phase 3: timed_events table for causal + timeline features.
        self.conn.execute(
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

    def _init_indices(self) -> None:
        """Indices for the hot recall paths (entity<->memory map, edge hops).

        These only speed up the existing queries; they do not change results.
        """
        self.conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id);
            CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id);
            CREATE INDEX IF NOT EXISTS idx_edges_src_cover
                ON edges(src_id, dst_id, relation, confidence);
            CREATE INDEX IF NOT EXISTS idx_me_entity ON memory_entities(entity_id);
            CREATE INDEX IF NOT EXISTS idx_me_memory ON memory_entities(memory_id);
            CREATE INDEX IF NOT EXISTS idx_entities_name_kind ON entities(name, kind);
            CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
            CREATE INDEX IF NOT EXISTS idx_memories_ns_created
                ON memories(namespace, created_at DESC, id DESC);
            CREATE INDEX IF NOT EXISTS idx_evidence_memory ON evidence(memory_id);
            CREATE INDEX IF NOT EXISTS idx_distilled_memory ON distilled_memories(memory_id);
            CREATE INDEX IF NOT EXISTS idx_timed_events_sort
                ON timed_events(sort_order, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_timed_events_person
                ON timed_events(person_name);
            """
        )

    def _init_fts(self) -> None:
        """Create the FTS5 trigram virtual table used as a candidate source.

        FTS5 is a *candidate* retrieval path, not the ranker. The trigram
        tokenizer matches substrings of ≥3 characters (ASCII brands like HHKB,
        and CJK words of ≥3 chars such as 机械键盘). It cannot match 2-char CJK
        queries (键盘/张三 -> 0 hits, trigram needs ≥3 chars), so recall falls
        back to the weighted ranker over all rows when FTS yields nothing. We
        probe support once; if the build lacks FTS5/trigram we degrade silently
        to the full-scan path (correctness is unchanged, only the candidate
        pre-filter is skipped).
        """
        self._fts_enabled = False
        try:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts "
                "USING fts5(memory_id UNINDEXED, content, tokenize='trigram')"
            )
            self._fts_enabled = True
            # P0-6: FTS5 ghost entries after delete are harmless — the recall
            # pipeline (_fts_candidates, line ~557) re-queries memories to
            # confirm every FTS5 hit exists.  FTS5 content-mode 'delete' is
            # not available without an external content table, so we rely on
            # the pipeline filter instead.  Bulk rebuild on demand if ghost
            # count grows large.  See REVIEW_FINDINGS_2026-06-30_v2.md §P0-6.
        except sqlite3.OperationalError:
            self._fts_enabled = False

    def _fts_remove(self, memory_id: str) -> None:
        """Best-effort FTS5 cleanup stub.

        FTS5 content-mode 'delete' requires an external content table to
        work (the standalone virtual table does not support the 'delete'
        special command).  Instead, the recall pipeline naturally filters
        ghost entries: _fts_candidates() re-queries memories to confirm
        every FTS5 hit exists (line ~559).

        When FTS5 is rebuilt (e.g. after a bulk delete), this method can
        be replaced with a full table rebuild.
        """
        # Currently a no-op: ghost entries are harmless because the recall
        # pipeline cross-checks every FTS5 candidate against the memories
        # table.  See _fts_candidates() lines 557-563.
        pass

    MAX_TEXT_BYTES = 10240  # P0: prevent OOM from unbounded input

    def save(self, text: str, source: str = "conversation", metadata: dict[str, Any] | None = None,
             namespace: str = "default") -> str:
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        if len(text.encode("utf-8")) > self.MAX_TEXT_BYTES:
            raise ValueError(
                f"text exceeds {self.MAX_TEXT_BYTES} bytes "
                f"(got {len(text.encode('utf-8'))})"
            )
        next_id = self._next_id("memories", "mem")
        summary = self._summarize(text)
        self.conn.execute(
            "INSERT INTO memories (id, raw_text, summary, source, confidence, metadata, namespace) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (next_id, text, summary, source, 1.0, self._serialize_metadata(metadata), namespace),
        )
        evidence_id = self._next_id("evidence", "evi")
        self.conn.execute(
            "INSERT INTO evidence (id, memory_id, quote, source_location) VALUES (?, ?, ?, ?)",
            (evidence_id, next_id, text, source),
        )
        self._index_entities(next_id, text)
        self._populate_timed_events_for_memory(next_id)
        if self._fts_enabled:
            self.conn.execute(
                "INSERT INTO memories_fts (memory_id, content) VALUES (?, ?)",
                (next_id, f"{text} {summary}"),
            )
        self.conn.commit()

        # ── Phase 2: Auto-encode for vector search ──
        if self._vector_enabled and self.embedding_engine and self.vector_index:
            try:
                emb = self.embedding_engine.encode(text)
                if emb is not None:
                    self.vector_index.add(next_id, emb)
            except Exception as exc:
                logger.debug("Vector encode skipped for %s: %s", next_id, exc)

        return next_id

    def delete(self, memory_id: str) -> bool:
        """Delete a memory and all associated rows (evidence, entities, edges, FTS5).

        FK ON DELETE CASCADE cleans up evidence, distilled_memories, and
        memory_entities.  FTS5 is a virtual table outside FK scope and must
        be explicitly synced via _fts_remove.
        """
        self._fts_remove(memory_id)

        # ── Phase 2: Remove from vector index ──
        if self._vector_enabled and self.vector_index:
            try:
                self.vector_index.remove(memory_id)
            except Exception:
                pass

        cursor = self.conn.execute(
            "DELETE FROM memories WHERE id = ?", (memory_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    # --- entity indexing -----------------------------------------------------

    def _index_entities(self, memory_id: str, text: str) -> None:
        # 1. Rule-based entities (always runs — the safety net).
        indexed: list[dict[str, Any]] = []
        seen_names: set[tuple[str, str]] = set()

        for ent in extract_entities(text):
            salience = classify_salience(ent, text)
            entity_id = self._get_or_create_entity(ent["name"], ent["kind"], salience)
            link_id = self._next_id("memory_entities", "me")
            self.conn.execute(
                "INSERT INTO memory_entities (id, memory_id, entity_id, evidence, salience_class) VALUES (?, ?, ?, ?, ?)",
                (link_id, memory_id, entity_id, ent.get("evidence", ""), salience),
            )
            indexed.append({"id": entity_id, "name": ent["name"], "kind": ent["kind"], "salience_class": salience})
            seen_names.add((ent["name"], ent["kind"]))

        # 2. LLM entities (supplements rule-based, tagged with source=llm).
        if self.llm_extractor is not None and self.llm_extractor.available:
            for ent in extract_entities_llm(text):
                key = (ent["name"], ent["kind"])
                if key in seen_names:
                    continue  # Rule-based already caught this one.
                salience = ent.get("salience_class", "event")
                entity_id = self._get_or_create_entity(ent["name"], ent["kind"], salience)
                link_id = self._next_id("memory_entities", "me")
                self.conn.execute(
                    "INSERT INTO memory_entities (id, memory_id, entity_id, evidence, salience_class) VALUES (?, ?, ?, ?, ?)",
                    (link_id, memory_id, entity_id, ent.get("evidence", ""), salience),
                )
                indexed.append({
                    "id": entity_id, "name": ent["name"],
                    "kind": ent["kind"], "salience_class": salience
                })
                seen_names.add(key)

        self._index_edges(memory_id, indexed, text)

    def _index_edges(self, memory_id: str, indexed: list[dict[str, Any]],
                     text: str = "") -> None:
        """Write typed relations between the entities co-occurring in one memory.

        Causal-edge hard boundary:
          - Co-occurrence inside a memory is an *inference*, so it is only ever
            written as CO_OCCURS_WITH at low confidence (0.4) and must never be
            promoted to a fact without evidence (≥3 confirmations or an explicit
            user statement).
          - A *user-stated* cause -- surfaced as a ``reason`` entity from a
            causal marker (因为/由于/...) in this very turn -- is the user
            asserting causation. We honour that as a CAUSED_BY edge at high
            confidence (0.95), pointing from each non-reason entity to the
            reason (effect -> cause).
          - When LLM extraction is enabled, LLM-typed edges (HAS_ATTRIBUTE,
            REPLACES, PREFERS, etc.) are merged in with source=llm confidence.

        Every edge's ``evidence_ref`` points back to the memory it was drawn
        from, so the inference is always auditable / revocable.
        """
        # Deduplicate entity ids within this memory while keeping kind/name.
        uniq: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for ent in indexed:
            if ent["id"] in seen_ids:
                continue
            seen_ids.add(ent["id"])
            uniq.append(ent)
        if len(uniq) < 2:
            return

        reasons = [e for e in uniq if e["kind"] == "reason"]
        nonreason = [e for e in uniq if e["kind"] != "reason"]

        written: set[tuple[str, str, str]] = set()

        def _add_edge(src: str, dst: str, relation: str, confidence: float) -> None:
            if src == dst:
                return
            key = (src, dst, relation)
            if key in written:
                return
            written.add(key)
            edge_id = self._next_id("edges", "edge")
            self.conn.execute(
                "INSERT INTO edges (id, src_id, dst_id, relation, confidence, evidence_ref) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (edge_id, src, dst, relation, confidence, memory_id),
            )

        # CO_OCCURS_WITH: every unordered pair, written once as a directed edge
        # from the earlier-indexed entity to the later one (low confidence).
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                _add_edge(uniq[i]["id"], uniq[j]["id"], "CO_OCCURS_WITH", 0.4)

        # CAUSED_BY: user-stated causation (effect -> cause). The reason entity
        # is the cause; connect it only to the nearest non-reason entities
        # (capped at 3 per reason) to avoid the all-to-all noise problem.
        # High confidence because the user said so.
        for cause in reasons:
            nearby = nonreason[:3]
            for effect in nearby:
                _add_edge(effect["id"], cause["id"], "CAUSED_BY", 0.95)

        # DESCRIBES: product→topic (e.g. HHKB → 键盘).  When both a product
        # alias and its topic appear in the same memory, create a high-confidence
        # directed edge so recall can follow the semantic link.
        for ent in uniq:
            topic = config.entities.object_topic_aliases.get(ent["name"])
            if topic is None:
                continue
            topic_ent = next((e for e in uniq if e["name"] == topic), None)
            if topic_ent is not None:
                _add_edge(ent["id"], topic_ent["id"], "DESCRIBES", 0.9)

        # LLM-typed edges: supplement rule-based edges with LLM-annotated
        # relations (HAS_ATTRIBUTE, REPLACES, PREFERS, etc.).  LLM edges use
        # lower confidence (×0.8) and are tagged source=llm in evidence_ref
        # for auditability.
        if self.llm_extractor is not None and self.llm_extractor.available and text:
            name_to_id: dict[str, str] = {e["name"]: e["id"] for e in uniq}
            for edge in extract_edges_llm(text):
                src_id = name_to_id.get(edge.get("src", ""))
                dst_id = name_to_id.get(edge.get("dst", ""))
                if src_id is None or dst_id is None:
                    continue
                relation = edge.get("relation", "CO_OCCURS_WITH")
                confidence = float(edge.get("confidence", 0.8))
                # Don't re-write CO_OCCURS_WITH or CAUSED_BY if already handled.
                if relation in ("CO_OCCURS_WITH", "CAUSED_BY"):
                    continue
                edge_id = self._next_id("edges", "edge")
                self.conn.execute(
                    "INSERT INTO edges (id, src_id, dst_id, relation, confidence, evidence_ref) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (edge_id, src_id, dst_id, relation, confidence,
                     f"{memory_id}:llm"),
                )

    def _get_or_create_entity(self, name: str, kind: str, salience_class: str = "event") -> str:
        # salience is per-memory (stored on memory_entities), NOT a global
        # property of the entity -- the same 妈妈 is a base_attr in one memory
        # and a sensory anchor in another. So entities.salience_class is only a
        # coarse hint and we do not let later memories mutate it.
        row = self.conn.execute(
            "SELECT id FROM entities WHERE name = ? AND kind = ?", (name, kind)
        ).fetchone()
        if row is not None:
            return str(row["id"])
        entity_id = self._next_id("entities", "ent")
        self.conn.execute(
            "INSERT INTO entities (id, name, kind, salience_class) VALUES (?, ?, ?, ?)",
            (entity_id, name, kind, salience_class),
        )
        return entity_id

    def _populate_timed_events_for_memory(self, memory_id: str) -> None:
        """Insert timed_events rows for any time-kind entities in this memory.

        Called automatically after entity indexing on every :meth:`save`.
        Extracts the first person entity (if any) from the same memory to
        attach as ``person_name``.
        """
        from .causal import _resolve_sort_order

        # Find time entities linked to this memory.
        time_rows = self.conn.execute(
            """SELECT e.id AS entity_id, e.name AS time_name, m.raw_text, m.created_at
               FROM memory_entities me
               JOIN entities e ON e.id = me.entity_id
               JOIN memories m ON m.id = me.memory_id
               WHERE me.memory_id = ? AND e.kind = 'time'""",
            (memory_id,),
        ).fetchall()

        if not time_rows:
            return

        # First person entity in the same memory (if any).
        person_row = self.conn.execute(
            """SELECT e.name
               FROM memory_entities me
               JOIN entities e ON e.id = me.entity_id
               WHERE me.memory_id = ? AND e.kind = 'person'
               LIMIT 1""",
            (memory_id,),
        ).fetchone()
        person_name = person_row["name"] if person_row else None

        for tr in time_rows:
            # Skip if already present.
            existing = self.conn.execute(
                "SELECT 1 FROM timed_events WHERE memory_id = ? AND time_entity_id = ?",
                (memory_id, tr["entity_id"]),
            ).fetchone()
            if existing:
                continue

            sort_order = _resolve_sort_order(tr["time_name"])
            event_id = f"tev_{memory_id}_{tr['entity_id']}"

            self.conn.execute(
                """INSERT INTO timed_events
                   (id, time_entity_id, time_name, sort_order, memory_id,
                    raw_text, person_name, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event_id,
                    tr["entity_id"],
                    tr["time_name"],
                    sort_order,
                    memory_id,
                    tr["raw_text"],
                    person_name,
                    tr["created_at"],
                ),
            )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @staticmethod
    def _serialize_metadata(metadata: dict[str, Any] | None) -> str:
        return json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _parse_metadata(raw: str | None) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                logger.warning("Could not parse memory metadata; returning empty metadata")
                return {}
        return parsed if isinstance(parsed, dict) else {}

    def entities_for(self, memory_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT e.id, e.name, e.kind, me.evidence
            FROM memory_entities me JOIN entities e ON e.id = me.entity_id
            WHERE me.memory_id = ?
            """,
            (memory_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def _entity_names_for(self, memory_id: str) -> set[str]:
        rows = self.conn.execute(
            "SELECT e.name FROM memory_entities me JOIN entities e ON e.id = me.entity_id WHERE me.memory_id = ?",
            (memory_id,),
        ).fetchall()
        return {r["name"] for r in rows}

    def _entities_for(self, memory_id: str) -> list[dict[str, Any]]:
        # salience_class is read from memory_entities (per-memory framing), not
        # the global entities row: the same 妈妈 is base_attr framing in one
        # memory and a sensory anchor in another.
        rows = self.conn.execute(
            "SELECT e.name, e.kind, me.salience_class FROM memory_entities me "
            "JOIN entities e ON e.id = me.entity_id WHERE me.memory_id = ?",
            (memory_id,),
        ).fetchall()
        return [{"name": r["name"], "kind": r["kind"], "salience_class": r["salience_class"]} for r in rows]

    # Associative-reach decay per salience class. This is NOT a per-query weight
    # matrix (that would be unauditable). It is a single structural fact: how far
    # a memory should be *carried by association* when it is only reached via the
    # graph (never directly matched). A base attribute (性别/年龄) reached purely
    # by hopping should fade hard; an event/sensory leaf travels well. Directly
    # matched memories are NEVER subjected to this — so "妈妈是谁" still surfaces
    # the base-attr memory because it is a direct hit, while "妈妈昨天干啥" does
    # not drag base attributes in through association.
    # Values now live in RecallWeights.assoc_reach_* fields.

    # --- ranker plug point ---------------------------------------------------
    # The default ranker is the weighted token/entity scorer below. FTS5 is a
    # *candidate source* (trigram substring match), never the ranker: whatever
    # rows survive candidate selection are still ordered by the weighted score.
    # A different ranker can be supplied by setting ``store.ranker`` to a
    # callable ``(query, terms, haystack, store) -> float``; it replaces the
    # base term/entity score while the entity/brand/edge boosts still apply.
    ranker = None

    def _base_score(self, query: str, terms: list[str], haystack: str) -> float:
        if self.ranker is not None:
            return float(self.ranker(query, terms, haystack, self))
        return self._score(query, terms, haystack)

    def _fts_candidates(self, query: str, terms: list[str],
                        namespace: str = "default") -> set[str] | None:
        """Return the set of memory ids the FTS5 trigram index matches.

        Returns ``None`` (meaning "no usable candidate filter, scan everything")
        when FTS is unavailable or the query has no trigram-eligible term. The
        trigram tokenizer needs substrings of ≥3 chars, so 2-char CJK queries
        (键盘, 张三) and bare single chars produce no FTS terms -> ``None`` ->
        the weighted full scan + entity/edge fallback handles them. This is the
        documented Chinese short-query fallback.

        For short mixed tokens (B轮, 20亿) where FTS5 trigram produces no match,
        a LIKE-based fallback scans for individual terms ≥1 char to recover
        candidates that would otherwise be missed entirely.
        """
        if not self._fts_enabled:
            return None
        fts_terms = [t for t in terms if len(t) >= 3]
        raw_ids: set[str] = set()

        # --- FTS5 trigram path ---
        if fts_terms:
            match_expr = " OR ".join('"' + t.replace('"', '""') + '"' for t in fts_terms)
            try:
                rows = self.conn.execute(
                    "SELECT memory_id FROM memories_fts WHERE memories_fts MATCH ?",
                    (match_expr,),
                ).fetchall()
                raw_ids = {r["memory_id"] for r in rows}
            except sqlite3.OperationalError:
                pass

        # --- LIKE fallback for short/mixed tokens that FTS5 trigram misses ---
        # Trigram tokenizer can't match "B轮" (2 chars), "20亿" (split at space),
        # "5000"(in "5000 万") — LIKE recovers these with substring matching.
        # Runs always (not gated on raw_ids empty) so it can ADD to FTS results.
        short_terms = [t for t in terms if 1 <= len(t) < 3]
        if short_terms:
            like_clauses = []
            like_params = []
            for t in short_terms:
                like_clauses.append("raw_text LIKE ?")
                like_params.append(f"%{t}%")
            if like_clauses:
                try:
                    like_rows = self.conn.execute(
                        f"SELECT id FROM memories WHERE namespace = ? AND ({' OR '.join(like_clauses)})",
                        (namespace,) + tuple(like_params),
                    ).fetchall()
                    raw_ids |= {r["id"] for r in like_rows}
                except sqlite3.OperationalError:
                    pass

        # --- Entity-name fallback (second level) ---
        # Topic aliases add entities (e.g. "钱") to memories whose raw_text
        # doesn't contain the entity name. LIKE searches raw_text, so recover
        # these via the entity table. Always merges with existing raw_ids.
        if short_terms:
            try:
                entity_rows = self.conn.execute(
                    f"""SELECT DISTINCT me.memory_id
                        FROM memory_entities me
                        JOIN entities e ON e.id = me.entity_id
                        JOIN memories m ON m.id = me.memory_id
                        WHERE m.namespace = ?
                        AND ({" OR ".join("e.name LIKE ?" for _ in short_terms)})""",
                    (namespace,) + tuple(f"%{t}%" for t in short_terms),
                ).fetchall()
                raw_ids |= {r["memory_id"] for r in entity_rows}
            except sqlite3.OperationalError:
                pass

        if raw_ids:
            placeholders = ",".join("?" for _ in raw_ids)
            ns_rows = self.conn.execute(
                f"SELECT id FROM memories WHERE id IN ({placeholders}) AND namespace = ?",
                tuple(raw_ids) + (namespace,),
            ).fetchall()
            return {r["id"] for r in ns_rows}
        return raw_ids if raw_ids else None

    _SQLITE_IN_BATCH = 900

    def _rows_by_ids(self, ids: list[str], ordered: bool = False,
                     namespace: str | None = None) -> list[sqlite3.Row]:
        if not ids:
            return []
        rows: list[sqlite3.Row] = []
        suffix = " ORDER BY created_at DESC, id DESC" if ordered else ""
        ns_clause = " AND namespace = ?" if namespace else ""
        for i in range(0, len(ids), self._SQLITE_IN_BATCH):
            batch = ids[i : i + self._SQLITE_IN_BATCH]
            placeholders = ",".join("?" for _ in batch)
            params = tuple(batch) + ((namespace,) if namespace else ())
            rows.extend(
                self.conn.execute(
                    f"SELECT * FROM memories WHERE id IN ({placeholders}){ns_clause}" + suffix,
                    params,
                ).fetchall()
            )
        return rows

    def _row_by_id(self, mem_id: str) -> sqlite3.Row | None:
        """Fetch a single memory row by id. Returns None if not found."""
        row = self.conn.execute(
            "SELECT * FROM memories WHERE id = ?", (mem_id,)
        ).fetchone()
        return row

    def _memory_rows(self, fts_ids: set[str] | None,
                     namespace: str = "default") -> list[sqlite3.Row]:
        """Fetch rows to score, using non-empty FTS hits as a candidate filter."""
        if fts_ids:
            return self._rows_by_ids(list(fts_ids), ordered=True)
        # Full-scan fallback: cap at a generous limit to bound scoring cost,
        # then let the Python ranker pick the best top_k.  ORDER BY created_at
        # DESC ensures recent memories win ties; the composite index
        # idx_memories_ns_created covers the WHERE + ORDER BY.
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE namespace = ? "
            "ORDER BY created_at DESC, id DESC "
            "LIMIT ?",
            (namespace, self.weights.full_scan_limit),
        ).fetchall()
        if len(rows) >= self.weights.full_scan_limit:
            logger.warning(
                "Full-scan recall hit limit (%d rows) for namespace=%r — "
                "consider FTS5 or namespace partitioning at scale",
                self.weights.full_scan_limit, namespace,
            )
        return rows

    def _entities_for_memories(self, memory_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        if not memory_ids:
            return {}
        rows: list[sqlite3.Row] = []
        for i in range(0, len(memory_ids), self._SQLITE_IN_BATCH):
            batch = memory_ids[i : i + self._SQLITE_IN_BATCH]
            placeholders = ",".join("?" for _ in batch)
            rows.extend(
                self.conn.execute(
                    "SELECT me.memory_id, e.name, e.kind, me.salience_class "
                    "FROM memory_entities me JOIN entities e ON e.id = me.entity_id "
                    f"WHERE me.memory_id IN ({placeholders})",
                    tuple(batch),
                ).fetchall()
            )
        out: dict[str, list[dict[str, Any]]] = {mid: [] for mid in memory_ids}
        for r in rows:
            out.setdefault(r["memory_id"], []).append(
                {"name": r["name"], "kind": r["kind"], "salience_class": r["salience_class"]}
            )
        return out

    def _record_access(self, memory_ids: list[str]) -> None:
        """Increment access_count and update accessed_at for recalled memories.

        Called at the end of every successful recall so the forgetting engine
        can see which memories have been recently accessed.
        """
        if not memory_ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        for mid in memory_ids:
            self.conn.execute(
                "UPDATE memories SET access_count = access_count + 1, "
                "accessed_at = ? WHERE id = ?",
                (now, mid),
            )
        self.conn.commit()

    def should_inject(self, query: str) -> bool:
        """快速判断查询是否需要记忆注入。

        节省上下文：闲聊/通用知识/数学/编程 不注入。
        仅在查询涉及个人信息/历史/偏好时注入。

        规则（在 recall 之前快速判定，不触发召回）：
        - 含"什么/谁/怎么/哪个/多大/之前/说过/记得" → 需要
        - 纯闲聊("你好/哈哈") → 不需要
        - 通用知识("Python/定义/代码") → 不需要
        - 实时数据("天气/新闻/股票") → 不需要
        """
        import re
        q = query.strip()

        # 纯闲聊 / 纯表情
        if re.match(r'^[你好嗨嗯哦啊哈哈嘿]{1,4}$', q):
            return False

        # 通用知识 / 编程 / 数学 — LLM 自有能力
        _general = ["python", "代码", "定义", "什么是", "如何", "怎么用",
                    "1+1", "等于", "数学", "公式", "算法", "编程",
                    "天气", "新闻", "股票", "汇率", "时间"]
        if any(kw in q.lower() for kw in _general):
            return False

        # 依赖个人记忆的关键词
        _needs_memory = ["什么", "谁", "怎么", "为什么", "哪个", "哪里",
                         "多大", "几岁", "之前", "说过", "记得",
                         "上次", "历史", "聊过", "知道",
                         "最近", "罗列", "列出", "回顾", "总结"]
        if any(kw in q for kw in _needs_memory):
            return True

        # 默认不注入（保守，避免浪费上下文）
        return False

    def recall(self, query: str, top_k: int = 5,
               namespace: str = "default") -> list[MemoryRecord]:
        """Recall top-k memories by a weighted composite score.

        The pipeline:
        1. Tokenise the query and extract entities.
        2. Optionally filter candidates through the FTS5 trigram index.
        3. Expand the candidate set with one-hop edge associations.
        4. Score every candidate through the composable pipeline.
        5. Sort, truncate, and convert to MemoryRecord with batched refs.
        """
        # ── Phase 2: Query Expansion ───────────────────────────────────
        if self.query_expander is not None:
            eq = self.query_expander.expand(query, async_llm=True)

            # 1. Merge synonyms into extra terms for token matching.
            extra_terms: list[str] = []
            for synonyms in eq.synonyms.values():
                extra_terms.extend(synonyms)

            # 2. Extract entities for the base query.
            query_entity_objs = extract_entities(query)

            # 3. Merge LLM extra entities into query entities.
            existing = {(e["name"], e.get("kind", "")) for e in query_entity_objs}
            for ent in eq.extra_entities:
                key = (ent.get("name", ""), ent.get("kind", ""))
                if key[0] and key not in existing:
                    query_entity_objs.append(ent)
                    existing.add(key)

            # 4. If there are variants, run multi-path recall → RRF fusion.
            if eq.variants:
                all_results: list[list[tuple[str, float]]] = []

                # Primary: recall with original query + expanded terms/entities.
                primary_terms = list(dict.fromkeys(self._terms(query) + extra_terms))
                primary = self._recall_single(
                    query, primary_terms, query_entity_objs, namespace,
                )
                all_results.append([(r.id, r.score) for r in primary])

                # Each variant gets its own recall path.
                for variant in eq.variants:
                    v_terms = list(dict.fromkeys(self._terms(variant)))
                    v_entities = extract_entities(variant)
                    # Also merge LLM extra entities into variant entities.
                    v_existing = {(e["name"], e.get("kind", "")) for e in v_entities}
                    for ent in eq.extra_entities:
                        key = (ent.get("name", ""), ent.get("kind", ""))
                        if key[0] and key not in v_existing:
                            v_entities.append(ent)
                            v_existing.add(key)
                    v_results = self._recall_single(
                        variant, v_terms, v_entities, namespace,
                    )
                    all_results.append([(r.id, r.score) for r in v_results])

                # RRF fuse all recall paths.
                merged = reciprocal_rank_fusion(all_results, k=60)

                # Sort merged results by RRF score.
                scored: list[tuple[float, str, Any]] = []
                for mem_id, rrf_score in merged:
                    row = self._row_by_id(mem_id)
                    if row is not None:
                        # Boost RRF scores to comparable range
                        scored.append((rrf_score * 10, "rrf-fused", row))

                scored.sort(key=lambda x: x[0], reverse=True)
                return self._build_recall_response(scored, top_k, query, namespace=namespace)
            else:
                # No variants: use expanded terms/entities with standard pipeline.
                terms = list(dict.fromkeys(self._terms(query) + extra_terms))
                # Fall through to standard pipeline below.
        else:
            terms = self._terms(query)
            query_entity_objs = extract_entities(query)

        # LLM query augmentation: supplement rule-based entities with LLM-
        # extracted ones for better recall (e.g., unlisted brands, topics).
        if self.llm_query_extract and self.llm_extractor is not None and self.llm_extractor.available:
            llm_ents = extract_entities_llm(query)
            existing = {(e["name"], e["kind"]) for e in query_entity_objs}
            for ent in llm_ents:
                if (ent["name"], ent["kind"]) not in existing:
                    query_entity_objs.append(ent)

        query_entities = {e["name"] for e in query_entity_objs}
        query_topics = {e["name"] for e in query_entity_objs if e["kind"] == "topic"}
        query_identity_subjects = self._identity_subjects(query_entity_objs)
        # When an identity question asks about a subject the rule-based
        # extractor missed (pet names like 咪咪, nicknames, rare objects),
        # fall back to scanning the DB for entity names that appear in the
        # query text.  Without this, "咪咪几岁了" would leak 妈妈's age
        # because no query entity is extracted → scope check never fires.
        if self._asks_identity(query) and not query_identity_subjects:
            for r in self.conn.execute(
                "SELECT name, kind FROM entities"
            ).fetchall():
                if r["name"] in query:
                    query_identity_subjects.add(r["name"])
            # If STILL no subjects found AND the query is short (likely a
            # named-entity identity question like "咪咪几岁了"), extract CJK
            # bigrams as fallback subjects.  Skip for long technical queries
            # ("数据库迁移方案是谁做的？") where bigrams would be noise.
            if not query_identity_subjects and len(query) <= 8:
                _STOP_CJK = {"什么", "怎么", "这个", "那个", "哪个", "因为",
                             "所以", "还是", "但是", "虽然", "如果", "可以",
                             "没有", "不是", "时候", "几岁", "多大", "多少"}
                for m in re.finditer(r"[\u4e00-\u9fff]{2,3}", query):
                    w = m.group()
                    if w not in _STOP_CJK:
                        query_identity_subjects.add(w)

        # Corrections: fetch corrected memory ids so we can down-weight them
        # during scoring. We do NOT hard-delete; the original text and correction
        # history stay in the corrections table for audit.
        corrected_ids = self._get_corrected_ids()

        # f3: FTS5 trigram candidate selection.
        fts_ids = self._fts_candidates(query, terms, namespace=namespace)
        rows = self._memory_rows(fts_ids, namespace=namespace)
        entity_map = self._entities_for_memories([r["id"] for r in rows])

        # f2: one-hop edge expansion.
        edge_bonus = self._edge_expansion(query, terms, rows, entity_map=entity_map,
                                          namespace=namespace)
        missing_edge_ids = sorted(set(edge_bonus) - {r["id"] for r in rows})
        if missing_edge_ids:
            rows.extend(self._rows_by_ids(missing_edge_ids))
            entity_map = self._entities_for_memories([r["id"] for r in rows])

        scored = self._build_scored_candidates(
            query=query, terms=terms, rows=rows,
            entity_map=entity_map, edge_bonus=edge_bonus,
            fts_ids=fts_ids, corrected_ids=corrected_ids,
            query_entities=query_entities, query_topics=query_topics,
            query_identity_subjects=query_identity_subjects,
            query_entity_objs=query_entity_objs,
        )
        return self._build_recall_response(scored, top_k, query, namespace=namespace)

    def _build_scored_candidates(
        self,
        query: str,
        terms: list[str],
        rows: list[sqlite3.Row],
        entity_map: dict[str, list[dict[str, Any]]],
        edge_bonus: dict[str, tuple[float, str]],
        fts_ids: set[str] | None = None,
        corrected_ids: set[str] | None = None,
        query_entities: set[str] | None = None,
        query_topics: set[str] | None = None,
        query_identity_subjects: set[str] | None = None,
        query_entity_objs: list[dict[str, Any]] | None = None,
    ) -> list[tuple[float, str, sqlite3.Row]]:
        """Main scoring loop: term matching + entity hop + composed boosts.

        Returns a list of ``(score, reason, row)`` tuples ready for sorting.
        """
        if corrected_ids is None:
            corrected_ids = set()
        if query_entities is None:
            query_entities = set()
        if query_entity_objs is None:
            query_entity_objs = []

        # Pre-compute query context once for the whole batch.
        asks_brand = self._asks_brand(query)
        asks_identity = self._asks_identity(query)
        asks_eval = self._asks_eval(query)

        scored: list[tuple[float, str, sqlite3.Row]] = []
        for row in rows:
            haystack = f"{row['raw_text']} {row['summary']}".lower()
            base = self._base_score(query, terms, haystack)
            reason = self._match_reason(terms, haystack, base)
            score = base
            directly_matched = base > 0  # a real surface-token hit

            # FTS provenance boost.
            if fts_ids is not None and row["id"] in fts_ids:
                score += self.weights.fts_boost
                directly_matched = True
                reason = (reason + "; fts trigram candidate").lstrip("; ")

            # Entity/topic hop: reward memories sharing extracted entities.
            mem_entity_objs = entity_map.get(row["id"], [])
            mem_entities = {e["name"] for e in mem_entity_objs}
            shared = query_entities & mem_entities
            if shared:
                score += self.weights.shared_entity_multiplier * len(shared)
                directly_matched = True
                reason = (reason + "; shared entities: " + ", ".join(sorted(shared))).lstrip("; ")

            # Conflicting-person penalty: when the query asks about a specific
            # person (张三) but the memory is about a different person (妈妈),
            # apply a mild penalty so same-score ties break toward the right person.
            # ALSO: when persons conflict, strip the entity bonus from topic-type
            # entities — "老张开什么车" should NOT boost 小李's car via shared "车".
            # Exception: role words (同事/朋友) are abstract references, not specific
            # individuals — "同事的键盘" should still find 张三's HHKB.
            _ROLE_WORDS = set(config.entities.role_words)
            query_persons = {e["name"] for e in query_entity_objs if e["kind"] == "person"}
            mem_persons = {e["name"] for e in mem_entity_objs if e["kind"] == "person"}
            matching_person = bool(query_persons & mem_persons)
            conflicting_person = (
                query_persons and mem_persons
                and not matching_person
                and not query_persons.issubset(_ROLE_WORDS)
            )
            if matching_person:
                # Query specifically names this person — significant boost
                score += self.weights.person_match_boost
                reason = (reason + "; matching person: " + ", ".join(sorted(query_persons & mem_persons))).lstrip("; ")
            if conflicting_person:
                score -= self.weights.conflicting_person_penalty
                reason = (reason + "; conflicting person").lstrip("; ")
                # Recalculate shared entities WITHOUT topic-kind entities
                non_topic_shared = {
                    e["name"] for e in query_entity_objs if e["kind"] != "topic"
                } & {e["name"] for e in mem_entity_objs if e["kind"] != "topic"}
                # Remove the topic-contributed entity bonus and replace with non-topic only
                if shared and non_topic_shared != shared:
                    score -= self.weights.shared_entity_multiplier * len(shared)
                    score += self.weights.shared_entity_multiplier * len(non_topic_shared)
                    shared = non_topic_shared

            # Tie-break micro-bonus: when two memories would tie, prefer the one
            # that shares MORE entities with the query (not just a "+1.2 per").
            if shared:
                score += self.weights.entity_tie_break_bonus * len(shared)

            # Context-aware boosts (brand / identity / eval).
            result = self._apply_context_boosts(
                query, row, score, reason, mem_entity_objs, directly_matched,
                asks_brand=asks_brand, asks_identity=asks_identity,
                asks_eval=asks_eval, query_topics=query_topics or set(),
                query_identity_subjects=query_identity_subjects or set(),
            )
            if result is None:
                continue  # identity-subject mismatch → skip this row
            score, reason = result

            # Edge-association bonus.
            bonus = edge_bonus.get(row["id"])
            if bonus is not None:
                score += bonus[0]
                reason = (reason + "; " + bonus[1]).lstrip("; ")

            # Correction penalty.
            if row["id"] in corrected_ids:
                score *= self.weights.correction_penalty
                reason = (reason + "; user_corrected").lstrip("; ")

            # Salience decay for association-only memories.
            score, reason = self._apply_salience_decay(
                row, score, reason, mem_entity_objs, directly_matched,
            )

            if score > 0 or not terms:
                scored.append((score, reason, row))

        return scored

    def _apply_context_boosts(
        self,
        query: str,
        row: sqlite3.Row,
        score: float,
        reason: str,
        mem_entity_objs: list[dict[str, Any]],
        directly_matched: bool,
        asks_brand: bool = False,
        asks_identity: bool = False,
        asks_eval: bool = False,
        query_topics: set[str] | None = None,
        query_identity_subjects: set[str] | None = None,
    ) -> tuple[float, str] | None:
        """Apply brand, identity, and evaluation boosts.

        Returns ``(new_score, new_reason)``, or ``None`` when the row should
        be skipped entirely (identity-subject mismatch).
        """
        if query_topics is None:
            query_topics = set()
        if query_identity_subjects is None:
            query_identity_subjects = set()

        # Identity scope check: skip rows that don't share the query's subject.
        mem_subjects = self._identity_subjects(mem_entity_objs) if asks_identity else set()
        # Augment with CJK bigrams from the memory's raw_text that also appear
        # in the query — catches entity-extraction misses like pet names (咪咪).
        if asks_identity and query_identity_subjects and not (query_identity_subjects & mem_subjects):
            for m in re.finditer(r"[\u4e00-\u9fff]{2,3}", row["raw_text"]):
                w = m.group()
                if w in query_identity_subjects:
                    mem_subjects.add(w)
        if asks_identity and query_identity_subjects and not (query_identity_subjects & mem_subjects):
            return None

        # Brand/product boost.
        if asks_brand and query_topics:
            product_ents = [
                e["name"]
                for e in mem_entity_objs
                if e["kind"] == "object" and self._looks_like_product(e["name"])
            ]
            topic_hit = query_topics & {e["name"] for e in mem_entity_objs if e["kind"] == "topic"}
            if product_ents and topic_hit:
                score += self.weights.brand_boost
                reason = (reason + "; brand-bearing product: " + ", ".join(sorted(product_ents))).lstrip("; ")

        # Occupation boost: when query asks about 职业/工作 and memory has an
        # occupation-like topic (教师/医生/...), boost that memory above peers.
        if query_topics & {"职业", "工作"}:
            mem_topics = {e["name"] for e in mem_entity_objs if e["kind"] == "topic"}
            if mem_topics & set(config.recall.occupation_topics):
                score += self.weights.occupation_boost
                reason = (reason + "; occupation topic match").lstrip("; ")

        # Identity base-attribute boost.
        if asks_identity:
            has_base_attr = any(
                e.get("salience_class") == "base_attr" for e in mem_entity_objs
            )
            scoped_identity = bool(query_identity_subjects & mem_subjects)
            if has_base_attr and scoped_identity:
                score += self.weights.identity_base_attr_boost
                reason = (reason + "; identity-question base-attr boost (matched subject)").lstrip("; ")

        # Evaluation sensory boost.
        if asks_eval and directly_matched:
            has_sensory = any(
                e.get("salience_class") == "sensory" for e in mem_entity_objs
            )
            if has_sensory:
                score += self.weights.eval_sensory_boost
                reason = (reason + "; evaluation-question sensory boost").lstrip("; ")

        return (score, reason)

    def _apply_salience_decay(
        self,
        row: sqlite3.Row,
        score: float,
        reason: str,
        mem_entity_objs: list[dict[str, Any]],
        directly_matched: bool,
    ) -> tuple[float, str]:
        """Apply associative-reach salience decay to non-directly-matched rows.

        Directly-matched memories are exempt so identity questions still
        surface base-attribute memories at full strength.
        """
        if directly_matched or score <= 0 or not mem_entity_objs:
            return (score, reason)

        reach = min(
            getattr(self.weights, f"assoc_reach_{e.get('salience_class', 'event')}", 1.0)
            for e in mem_entity_objs
        )
        if reach < 1.0:
            score *= reach
            reason = (reason + f"; assoc-reach×{reach:.2f}").lstrip("; ")

        return (score, reason)

    def _recall_single(
        self,
        query: str,
        terms: list[str],
        query_entity_objs: list[dict[str, Any]],
        namespace: str = "default",
    ) -> list[MemoryRecord]:
        """Run the standard recall pipeline for a single query variant.

        This is a self-contained recall path that extracts entities,
        performs FTS + edge expansion + scoring, and returns scored records.
        Used by the multi-variant RRF fusion path in ``recall()``.
        """
        query_entities = {e["name"] for e in query_entity_objs}
        query_topics = {e["name"] for e in query_entity_objs if e["kind"] == "topic"}
        query_identity_subjects = self._identity_subjects(query_entity_objs)

        if self._asks_identity(query) and not query_identity_subjects:
            for r in self.conn.execute(
                "SELECT name, kind FROM entities"
            ).fetchall():
                if r["name"] in query:
                    query_identity_subjects.add(r["name"])
            if not query_identity_subjects and len(query) <= 8:
                _STOP_CJK = {"什么", "怎么", "这个", "那个", "哪个", "因为",
                             "所以", "还是", "但是", "虽然", "如果", "可以",
                             "没有", "不是", "时候", "几岁", "多大", "多少"}
                for m in re.finditer(r"[\u4e00-\u9fff]{2,3}", query):
                    w = m.group()
                    if w not in _STOP_CJK:
                        query_identity_subjects.add(w)

        corrected_ids = self._get_corrected_ids()
        fts_ids = self._fts_candidates(query, terms, namespace=namespace)
        rows = self._memory_rows(fts_ids, namespace=namespace)
        entity_map = self._entities_for_memories([r["id"] for r in rows])
        edge_bonus = self._edge_expansion(query, terms, rows,
                                          entity_map=entity_map, namespace=namespace)
        missing_edge_ids = sorted(set(edge_bonus) - {r["id"] for r in rows})
        if missing_edge_ids:
            rows.extend(self._rows_by_ids(missing_edge_ids))
            entity_map = self._entities_for_memories([r["id"] for r in rows])

        scored = self._build_scored_candidates(
            query=query, terms=terms, rows=rows,
            entity_map=entity_map, edge_bonus=edge_bonus,
            fts_ids=fts_ids, corrected_ids=corrected_ids,
            query_entities=query_entities, query_topics=query_topics,
            query_identity_subjects=query_identity_subjects,
            query_entity_objs=query_entity_objs,
        )
        # Return a larger top-k for RRF to fuse; the final truncation happens
        # after fusion in the caller.
        return self._build_recall_response(scored, max(50, len(scored)), query, namespace=namespace)

    def _build_recall_response(
        self,
        scored: list[tuple[float, str, sqlite3.Row]],
        top_k: int,
        query: str = "",
        namespace: str = "default",
    ) -> list[MemoryRecord]:
        """Sort scored candidates, take top-k, and convert to MemoryRecords.

        Evidence refs are batch-fetched (one query each for evidence and
        raw_logs) to fix the N+1 query problem in ``_row_to_record``.

        When keyword results are below top_k, supplements with recent items
        (sorted by created_at DESC) so queries like "罗列对话"/"历史记录"
        don't return empty.
        """
        scored.sort(
            key=lambda item: (item[0], item[2]["created_at"], item[2]["id"]),
            reverse=True,
        )
        top = scored[:top_k]
        if not top:
            # No keyword/vector hits at all — fall back to recent items.
            recent_rows = self.conn.execute(
                "SELECT * FROM memories WHERE namespace = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (namespace, top_k),
            ).fetchall()
            fb_records: list[MemoryRecord] = []
            for row in recent_rows:
                fb_records.append(self._row_to_record(row, score=0.5,
                    match_reason="recent fallback (no keyword match)"))
            return fb_records

        mem_ids = [item[2]["id"] for item in top]
        evidence_map = self._batch_evidence_refs(mem_ids)
        raw_refs_map = self._batch_raw_refs(mem_ids)

        records: list[MemoryRecord] = []
        for score, reason, row in top:
            mid = row["id"]
            records.append(
                self._row_to_record(
                    row, score=score, match_reason=reason,
                    evidence_refs=evidence_map.get(mid, []),
                    raw_refs=raw_refs_map.get(mid, []),
                )
            )

        # ── Phase 2: Vector-search fusion ──
        if query and self._vector_enabled and self.embedding_engine and self.vector_index:
            try:
                vec = self.embedding_engine.encode(query)
                if vec is not None:
                    vector_results = self.vector_index.search(vec, k=top_k * 4)
                    vector_list = [(mid, score) for mid, score in vector_results]
                    keyword_list = [(r.id, r.score) for r in records if r.score > 0]
                    merged = reciprocal_rank_fusion(
                        [keyword_list, vector_list],
                        k=60,
                        weights=[0.4, 0.6],
                    )
                    rrf_scores = dict(merged)
                    # Build new list (MemoryRecord is frozen)
                    new_records: list[MemoryRecord] = []
                    existing_ids = {r.id for r in records}
                    for r in records:
                        # Blend: keep keyword score as base, boost if also found by vector
                        rrf_s = rrf_scores.get(r.id, 0)
                        if rrf_s:
                            new_score = r.score + rrf_s * 10  # keyword base + vector boost
                        else:
                            new_score = r.score
                        new_records.append(MemoryRecord(
                            id=r.id, raw_text=r.raw_text, summary=r.summary,
                            confidence=r.confidence, metadata=r.metadata,
                            evidence_refs=r.evidence_refs,
                            score=new_score, match_reason=r.match_reason,
                        ))
                    for mid, rrf_score in merged:
                        if mid not in existing_ids:
                            row = self._rows_by_ids([mid])
                            if row:
                                rw = row[0]
                                new_records.append(MemoryRecord(
                                    id=mid, raw_text=rw["raw_text"],
                                    summary=rw.get("summary", rw["raw_text"][:160]),
                                    confidence=rw.get("confidence", 1.0),
                                    metadata=self._parse_metadata(rw.get("metadata")),
                                    evidence_refs=[], score=rrf_score,
                                    match_reason="vector search (RRF fusion)",
                                ))
                    new_records.sort(key=lambda x: x.score, reverse=True)
                    records = new_records
            except Exception as exc:
                logger.debug("Vector search skipped: %s", exc)

        # ── Phase 2.5: LLM reranker (semantic re-rank, optional) ──
        if self.reranker and self.reranker.available and records and len(records) > 1:
            try:
                candidates = [{"text": r.raw_text, "score": r.score} for r in records]
                reranked = self.reranker.rerank(query, candidates)
                # Rebuild MemoryRecord with new scores
                rrmap = {rr.get("text",""): rr.get("score",0) for rr in reranked}
                records = [MemoryRecord(
                    id=r.id, raw_text=r.raw_text, summary=r.summary,
                    confidence=r.confidence, metadata=r.metadata,
                    evidence_refs=r.evidence_refs,
                    score=rrmap.get(r.raw_text, r.score),
                    match_reason=r.match_reason,
                ) for r in records]
                records.sort(key=lambda x: x.score, reverse=True)
                logger.debug("Reranker applied to %d records", len(records))
            except Exception as exc:
                logger.debug("Reranker skipped: %s", exc)

        # ── Phase 3: Record access for forgetting engine ──
        self._record_access([r.id for r in records])

        # ── Phase 4: Recent fallback ──
        # When keyword/vector recall returns fewer than top_k results,
        # supplement with recently-saved items. This handles meta-queries
        # like "罗列最近对话" where no content keyword matches exist.
        if len(records) < top_k:
            existing_ids = {r.id for r in records}
            recent_limit = top_k - len(records)
            recent_rows = self.conn.execute(
                "SELECT * FROM memories WHERE namespace = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (namespace, recent_limit + top_k),  # extra buffer for dedup
            ).fetchall()
            for row in recent_rows:
                rid = row["id"]
                if rid not in existing_ids and len(records) < top_k:
                    records.append(
                        self._row_to_record(row, score=0.5,
                            match_reason="recent fallback"))
                    existing_ids.add(rid)

        return records

    def _batch_evidence_refs(self, memory_ids: list[str]) -> dict[str, list[str]]:
        """Single-batch fetch of evidence ids for multiple memory rows."""
        if not memory_ids:
            return {}
        out: dict[str, list[str]] = {mid: [] for mid in memory_ids}
        for i in range(0, len(memory_ids), self._SQLITE_IN_BATCH):
            batch = memory_ids[i : i + self._SQLITE_IN_BATCH]
            placeholders = ",".join("?" for _ in batch)
            rows = self.conn.execute(
                f"SELECT id, memory_id FROM evidence WHERE memory_id IN ({placeholders}) ORDER BY id",
                tuple(batch),
            ).fetchall()
            for r in rows:
                out.setdefault(r["memory_id"], []).append(r["id"])
        return out

    def _batch_raw_refs(self, memory_ids: list[str]) -> dict[str, list[str]]:
        """Single-batch fetch of raw-log refs for multiple memory rows."""
        if not memory_ids:
            return {}
        out: dict[str, list[str]] = {mid: [] for mid in memory_ids}
        for i in range(0, len(memory_ids), self._SQLITE_IN_BATCH):
            batch = memory_ids[i : i + self._SQLITE_IN_BATCH]
            placeholders = ",".join("?" for _ in batch)
            rows = self.conn.execute(
                f"SELECT raw_log_id, memory_id FROM distilled_memories WHERE memory_id IN ({placeholders}) ORDER BY id",
                tuple(batch),
            ).fetchall()
            for r in rows:
                out.setdefault(r["memory_id"], []).append(r["raw_log_id"])
        return out

    def _edge_expansion(
        self,
        query: str,
        terms: list[str],
        rows: list[sqlite3.Row],
        max_hops: int | None = None,
        decay: float | None = None,
        entity_map: dict[str, list[dict[str, Any]]] | None = None,
        namespace: str = "default",
    ) -> dict[str, tuple[float, str]]:
        """N-hop spreading activation over the typed ``edges`` table.

        Returns ``{memory_id: (bonus, reason)}`` for memories reached by
        following edges out of a *seed* memory (one the direct pass already
        matches on tokens/entities). Edges let recall reach a memory that
        shares no token and no entity with the query but sits a few hops away
        in the association graph (妈妈 → 做饭好吃 → 红烧肉 → 那顿饭).

        Activation model (bounded BFS, re-propagating):
          - seed entities start at activation 1.0;
          - crossing an edge multiplies activation by ``decay × confidence``;
          - a node reached again by a STRONGER path is updated AND re-queued so
            the stronger activation propagates onward (this is the bug the
            0002 draft had — it never re-queued, so multi-path gains died);
          - activation below ``activation_threshold`` stops spreading;
          - ``max_hops`` caps depth.

        Defaults: ``max_hops=2``, ``decay=0.5``. Two hops covers the
        三段联想 the user described while keeping the dense low-confidence
        CO_OCCURS_WITH graph from lighting up the whole store on small data.
        The bonus is the neighbour entity's final activation, so a strong
        CAUSED_BY (0.95) hop pulls far harder than a weak CO_OCCURS_WITH (0.4).
        """
        max_hops = self.max_recall_hops if max_hops is None else max_hops
        decay = self.recall_decay if decay is None else decay

        # 1. Find seed memories: those with a positive direct token/entity match.
        query_entity_objs = extract_entities(query)
        query_entities = {e["name"] for e in query_entity_objs}
        seed_ids: list[str] = []
        for row in rows:
            haystack = f"{row['raw_text']} {row['summary']}".lower()
            if self._base_score(query, terms, haystack) > 0:
                seed_ids.append(row["id"])
                continue
            if entity_map is not None:
                mem_entities = {e["name"] for e in entity_map.get(row["id"], [])}
            else:
                mem_entities = self._entity_names_for(row["id"])
            if query_entities & mem_entities:
                seed_ids.append(row["id"])
        if not seed_ids:
            return {}

        # 2. Collect the entity ids each seed memory carries (the BFS frontier).
        seed_entity_ids: set[str] = set()
        if seed_ids:
            placeholders = ",".join("?" for _ in seed_ids)
            for r in self.conn.execute(
                f"SELECT DISTINCT entity_id FROM memory_entities "
                f"WHERE memory_id IN ({placeholders})",
                tuple(seed_ids),
            ).fetchall():
                seed_entity_ids.add(r["entity_id"])
        if not seed_entity_ids:
            return {}

        # 3. Bounded, re-propagating BFS over the edge graph.
        #    activated[entity_id] = (activation, relation, hop, path_names)
        from collections import deque

        # Preload entity names so _name() never hits the DB inside the BFS loop.
        entity_names: dict[str, str] = {}
        if seed_entity_ids:
            e_ph = ",".join("?" for _ in seed_entity_ids)
            for r in self.conn.execute(
                f"SELECT id, name FROM entities WHERE id IN ({e_ph})",
                tuple(seed_entity_ids),
            ).fetchall():
                entity_names[r["id"]] = r["name"]

        def _name(eid: str) -> str:
            if eid in entity_names:
                return entity_names[eid]
            r = self.conn.execute("SELECT name FROM entities WHERE id = ?", (eid,)).fetchone()
            if r:
                entity_names[eid] = r["name"]
                return str(r["name"])
            return eid

        activated: dict[str, tuple[float, str, int, list[str]]] = {}
        queue: deque[tuple[str, float, int]] = deque()
        for eid in seed_entity_ids:
            queue.append((eid, 1.0, 0))

        while queue:
            src_eid, src_act, hop = queue.popleft()
            if hop >= max_hops:
                continue
            next_hop = hop + 1
            for src_col, dst_col in (("src_id", "dst_id"), ("dst_id", "src_id")):
                for e in self.conn.execute(
                    f"SELECT {dst_col} AS nbr, relation, confidence FROM edges WHERE {src_col} = ?",
                    (src_eid,),
                ).fetchall():
                    nbr = e["nbr"]
                    if nbr in seed_entity_ids:
                        continue  # never re-activate a seed entity
                    new_act = src_act * decay * float(e["confidence"])
                    if new_act < self.weights.activation_threshold:
                        continue
                    prev = activated.get(nbr)
                    if prev is None or new_act > prev[0]:
                        src_path = activated.get(src_eid, (0.0, "", 0, [_name(src_eid)]))[3]
                        activated[nbr] = (new_act, e["relation"], next_hop, src_path + [_name(nbr)])
                        # Re-queue so the improved activation propagates onward.
                        queue.append((nbr, new_act, next_hop))
        if not activated:
            return {}

        # 4. Map activated neighbour entities -> the memories carrying them
        #    (excluding seeds), keep the strongest activation per memory.
        #    Filter by namespace so edge associations never cross tenants.
        result: dict[str, tuple[float, str]] = {}
        seed_set = set(seed_ids)
        activated_ids = list(activated.keys())
        if activated_ids:
            a_ph = ",".join("?" for _ in activated_ids)
            for r in self.conn.execute(
                "SELECT me.memory_id, me.entity_id FROM memory_entities me "
                "JOIN memories m ON m.id = me.memory_id "
                f"WHERE me.entity_id IN ({a_ph}) AND m.namespace = ?",
                tuple(activated_ids) + (namespace,),
            ).fetchall():
                mid = r["memory_id"]
                if mid in seed_set:
                    continue
                entity_id = r["entity_id"]
                act, relation, hop, path = activated[entity_id]
                reason = f"edge assoc hop={hop} act={act:.3f} [{' → '.join(path)}] ({relation})"
                existing = result.get(mid)
                if existing is None or act > existing[0]:
                    result[mid] = (act, reason)
        return result

    def _get_corrected_ids(self) -> set[str]:
        """Return the set of memory ids that have been user-corrected.

        Corrections are traced through the ``corrections`` table; the original
        memory stays in ``memories`` so the evidence chain is never severed.
        """
        rows = self.conn.execute("SELECT DISTINCT target_id FROM corrections").fetchall()
        return {r["target_id"] for r in rows}

    @staticmethod
    def _looks_like_product(name: str) -> bool:
        """A product-like object name is one carrying specific identity, i.e.
        an ASCII/alnum token (HHKB, MX-3) rather than a generic CJK noun (键盘)."""
        return bool(re.search(r"[A-Za-z0-9]", name))

    @staticmethod
    def _identity_subjects(entities: list[dict[str, Any]]) -> set[str]:
        """Anchors an identity attribute can belong to.

        Identity boosts must be scoped: a memory with 妈妈's 55岁 should not
        answer 张三/室友/咪咪 age questions. Person entities are the primary
        anchors; named objects are included so pet/device identity questions can
        still bind to specific non-person subjects (咪咪 / HHKB / iPhone).
        """
        subjects: set[str] = set()
        for ent in entities:
            kind = ent.get("kind")
            name = ent.get("name", "")
            if kind in ("person", "object", "topic"):
                subjects.add(name)
        return subjects

    @staticmethod
    def _asks_brand(query: str) -> bool:
        return any(k in query for k in ("牌子", "品牌", "什么型号", "型号", "哪个牌", "什么款"))

    @staticmethod
    def _asks_identity(query: str) -> bool:
        """Identity questions ('是谁/叫什么/多大/几岁/什么星座') ask for the
        constant base attributes (name/age/gender) of a subject."""
        return any(k in query for k in (
            "是谁", "叫什么", "叫啥", "名字", "多大", "几岁", "多少岁", "年龄",
            "什么星座", "属相", "什么血型", "哪里人", "哪儿人", "是男是女", "性别",
        ))

    @staticmethod
    def _asks_eval(query: str) -> bool:
        """Evaluation/quality questions ('怎么样/好不好/如何/好吃吗/手艺') ask for
        a judgement, which lives in sensory/evaluative content."""
        return any(k in query for k in (
            "怎么样", "好不好", "好吃吗", "好吃不", "如何", "厉害吗", "手艺",
            "性格", "脾气", "好看吗", "漂亮吗",
        ))

    def gap_check(self, query: str, memories: list[MemoryRecord] | None = None,
                  namespace: str = "default", scan_all: bool = False) -> dict[str, Any]:
        if memories is None:
            if scan_all:
                # Scan all memories in namespace, bypassing recall scoring.
                rows = self.conn.execute(
                    "SELECT raw_text FROM memories WHERE namespace = ?", (namespace,)
                ).fetchall()
                combined = "\n".join(r["raw_text"] for r in rows)
            else:
                memories = self.recall(query, namespace=namespace)
                combined = "\n".join(m.raw_text for m in memories)
        else:
            combined = "\n".join(m.raw_text for m in memories)
        missing: list[str] = []
        # Five gap types: reason, person, time, location, object.
        if self._asks_reason(query) and not self._has_reason(combined):
            missing.append("reason")
        if self._asks_person(query) and not self._has_person_like(combined):
            missing.append("person")
        if self._asks_time(query) and not self._has_time(combined):
            missing.append("time")
        if self._asks_location(query) and not self._has_location(combined):
            missing.append("location")
        if self._asks_object(query) and not self._has_object(combined):
            missing.append("object")
        sufficient = not missing
        return {
            "sufficient": sufficient,
            "missing": missing,
            "suggested_question": "" if sufficient else self._suggest_question(missing),
        }

    def save_raw_log(self, text: str, kind: str = "conversation") -> str:
        raw_id = self._next_id("raw_logs", "raw")
        self.conn.execute("INSERT INTO raw_logs (id, kind, text) VALUES (?, ?, ?)", (raw_id, kind, text))
        self.conn.commit()
        return raw_id

    def get_raw_log(self, raw_id: str) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM raw_logs WHERE id = ?", (raw_id,)).fetchone()
        if row is None:
            raise KeyError(raw_id)
        return dict(row)

    def compact(self, raw_log_id: str, distilled_text: str,
                namespace: str = "default") -> str:
        # Raw log remains untouched. Distilled memory points back to raw evidence.
        self.get_raw_log(raw_log_id)
        memory_id = self.save(distilled_text, source="compaction", metadata={"raw_log_id": raw_log_id},
                              namespace=namespace)
        distilled_id = self._next_id("distilled_memories", "dst")
        self.conn.execute(
            "INSERT INTO distilled_memories (id, raw_log_id, memory_id, distilled_text) VALUES (?, ?, ?, ?)",
            (distilled_id, raw_log_id, memory_id, distilled_text),
        )
        evidence_id = self._next_id("evidence", "evi")
        self.conn.execute(
            "INSERT INTO evidence (id, memory_id, quote, source_location) VALUES (?, ?, ?, ?)",
            (evidence_id, memory_id, distilled_text, raw_log_id),
        )
        self.conn.commit()
        return distilled_id

    def _row_to_record(
        self,
        row: sqlite3.Row,
        score: float,
        match_reason: str,
        evidence_refs: list[str] | None = None,
        raw_refs: list[str] | None = None,
    ) -> MemoryRecord:
        """Convert a database row to a MemoryRecord.

        When ``evidence_refs`` / ``raw_refs`` are provided (pre-fetched by
        ``_build_recall_response``), the N+1 queries are skipped.
        """
        if evidence_refs is None:
            evidence_refs = [
                r["id"]
                for r in self.conn.execute(
                    "SELECT id FROM evidence WHERE memory_id = ? ORDER BY id",
                    (row["id"],),
                ).fetchall()
            ]
        if raw_refs is None:
            raw_refs = [
                r["raw_log_id"]
                for r in self.conn.execute(
                    "SELECT raw_log_id FROM distilled_memories WHERE memory_id = ? ORDER BY id",
                    (row["id"],),
                ).fetchall()
            ]
        metadata = self._parse_metadata(row["metadata"])
        metadata.update({"source": row["source"], "created_at": row["created_at"]})
        # Phase 3: surface access/forgetting columns for the forgetting engine.
        if row["accessed_at"] is not None:
            metadata["accessed_at"] = row["accessed_at"]
        metadata["access_count"] = int(row["access_count"]) if row["access_count"] is not None else 0
        metadata["forgotten"] = bool(row["forgotten"])
        return MemoryRecord(
            id=row["id"],
            raw_text=row["raw_text"],
            summary=row["summary"],
            confidence=float(row["confidence"]),
            metadata=metadata,
            evidence_refs=evidence_refs + raw_refs,
            score=score,
            match_reason=match_reason,
        )

    # Tables whose ids carry an "<prefix>_<n>" numeric tail, so we can recover
    # the high-water mark from existing rows when seeding a legacy database.
    _ID_TABLES = {
        "memories": "mem",
        "evidence": "evi",
        "memory_entities": "me",
        "entities": "ent",
        "edges": "edge",
        "raw_logs": "raw",
        "distilled_memories": "dst",
    }

    def _next_id(self, table: str, prefix: str) -> str:
        """Hand back a never-before-used id via the monotonic id_sequences table.

        The old scheme returned ``f"{prefix}_{COUNT(*)+1}"``; after any DELETE
        the count fell and a live id was reissued, crashing the next INSERT on
        the PRIMARY KEY (Phase 5 consolidation deletes/down-weights memories, so
        this was a guaranteed crash). Instead we keep a per-table counter that
        only ever climbs.

        Legacy databases written under the old scheme have rows but no
        id_sequences state; on first allocation we seed the counter past the
        largest existing ``<prefix>_<n>`` so we never collide with rows already
        on disk.

        Allocation is a single-row atomic UPDATE so independent connections do
        not read the same value and hand back duplicate ids. We deliberately do
        not open an explicit transaction here: callers already run inside
        sqlite3's implicit transaction and commit at the end of save/compact.
        """
        row = self.conn.execute(
            "UPDATE id_sequences SET next_val = next_val + 1 WHERE name = ? RETURNING next_val - 1 AS value",
            (table,),
        ).fetchone()
        if row is None:
            seed = self._seed_sequence(table, prefix)
            self.conn.execute(
                "INSERT OR IGNORE INTO id_sequences (name, next_val) VALUES (?, ?)",
                (table, seed),
            )
            row = self.conn.execute(
                "UPDATE id_sequences SET next_val = next_val + 1 WHERE name = ? RETURNING next_val - 1 AS value",
                (table,),
            ).fetchone()
        return f"{prefix}_{int(row['value'])}"

    def _seed_sequence(self, table: str, prefix: str) -> int:
        """Return the first counter value for ``table``: 1 for an empty table, or
        max(existing ``<prefix>_<n>`` tail)+1 for a legacy table with rows."""
        assert table in self._ID_TABLES, f"unknown table: {table}"
        try:
            row = self.conn.execute(
                f"SELECT MAX(CAST(SUBSTR(id, ?) AS INTEGER)) FROM {table} "
                f"WHERE id LIKE ?",
                (len(prefix) + 2, f"{prefix}_%"),
            ).fetchone()
        except sqlite3.OperationalError:
            return 1
        val = row[0]
        return int(val) + 1 if val is not None else 1

    # ---- filler-word cleaning -------------------------------------------------
    FILLER_WORDS: list[str] = [
        "就是说", "就是说呢", "就是说啊", "就是说呀",
        "那个什么", "那个啥", "那什么", "这个那个",
        "对吧", "对不对", "是不是", "是吧", "你懂吧", "你知道吧",
        "怎么说呢", "怎么说",
        "那个", "这个", "内个", "然后呢", "反正",
    ]
    FILLER_CHARS: frozenset[str] = frozenset(
        "嗯呃啊嘛呢哈哦噢哎呀哇嘞咧呐呗嘿呵嘻嗷咳呸哟"
    )

    @staticmethod
    def _clean_sentence(text: str) -> str:
        """Remove Chinese filler words / discourse markers.

        Longer phrases are removed first so partial matches don't leave
        orphan fragments behind.  Single-char fillers are stripped from
        leading / trailing positions and between punctuation boundaries.
        """
        result = text
        for filler in sorted(MemoryStore.FILLER_WORDS, key=len, reverse=True):
            result = result.replace(filler, "")

        # Strip leading/trailing filler chars (they are noise at edges).
        filler_set: frozenset[str] = MemoryStore.FILLER_CHARS
        while result and result[0] in filler_set:
            result = result[1:]
        while result and result[-1] in filler_set:
            result = result[:-1]

        # Remove filler chars that sit between whitespace / punctuation.
        # We rebuild character-by-character to avoid regex engine issues
        # (Python's re forbids variable-width look-behind).
        boundary: set[str] = {" ", "\t", "，", ",", "。", ".", "！",
                               "!", "？", "?", "\n"}
        chars: list[str] = []
        n = len(result)
        for i, ch in enumerate(result):
            if ch in filler_set:
                left_ok = i == 0 or result[i - 1] in boundary
                right_ok = i == n - 1 or result[i + 1] in boundary
                if left_ok and right_ok:
                    continue  # skip isolated filler
            chars.append(ch)
        result = "".join(chars)

        # Collapse repeated punctuation / whitespace created by removal.
        result = re.sub(r"[，,]{2,}", "，", result)
        result = re.sub(r"[。.!！?？]{2,}", "。", result)
        result = re.sub(r"\s{2,}", " ", result)
        return result.strip()

    @staticmethod
    def _truncate_cjk(text: str, max_chars: int = 120) -> str:
        """Truncate *text* to at most *max_chars*, never cutting inside a
        CJK code-point.

        Python 3 ``str`` slicing already operates on code-points so CJK
        characters are inherently safe.  This method strips trailing
        whitespace / punctuation so the result reads cleanly.
        """
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip()

    @staticmethod
    def _summarize(text: str) -> str:
        """Distil the first sentence into a lossless-leaning summary.

        Strategy:
        1. Isolate the first sentence (best-effort, CJK + ASCII end marks).
        2. Remove filler / discourse-marker noise.
        3. Truncate to ≤ 120 characters on a CJK-safe boundary.

        This keeps entities, numbers and proper nouns intact — pure
        character-count truncation would chop them, violating the
        project's anti-lossy-summary philosophy.
        """
        m = re.search(r"[。！？!?\n]", text)
        first = text[: m.start() + 1] if m else text

        cleaned = MemoryStore._clean_sentence(first)
        if not cleaned.strip():
            cleaned = first  # fallback: cleaning stripped everything

        return MemoryStore._truncate_cjk(cleaned, 120)

    @staticmethod
    def _terms(query: str) -> list[str]:
        ascii_terms = re.findall(r"[A-Za-z0-9_]+", query.lower())
        cjk_terms = re.findall(r"[\u4e00-\u9fff]{2,}", query)
        chars = [ch for ch in query if "\u4e00" <= ch <= "\u9fff"]
        return list(dict.fromkeys(ascii_terms + cjk_terms + chars))

    # Common interrogative / filler CJK chars carry almost no retrieval signal.
    # Without down-weighting them, a noise turn that happens to share 我/的/是
    # outscores the turn holding the actual answer.
    STOP_CHARS = frozenset(config.recall.stop_chars)

    def _term_weight(self, term: str) -> float:
        """Weight a term by how much retrieval signal it carries.

        - ASCII / alphanumeric tokens (brands like HHKB, dates, numbers) are
          high-signal and rare -> heaviest weight.
        - Multi-char CJK words (机械键盘, 同事, 腾讯) are meaningful -> medium.
        - Single common CJK chars (我, 的, 是) are noise -> near zero.
        """
        if re.fullmatch(r"[A-Za-z0-9_]+", term):
            # Rare proper nouns / brands / IDs: scale a little with length.
            return self.weights.ascii_base + min(len(term), self.weights.ascii_per_char_cap) * self.weights.ascii_per_char
        if len(term) >= 2:
            return self.weights.cjk_multi_base + (len(term) - 2) * self.weights.cjk_multi_per_char
        # Single CJK char.
        if term in MemoryStore.STOP_CHARS:
            return self.weights.stop_char_weight
        return self.weights.single_cjk_weight

    # --- consolidate ----------------------------------------------------------

    def consolidate(self) -> dict[str, Any]:
        """合并重复实体名、清理孤立边，返回清理统计。

        合并规则：同名实体的大小写/空白变体会被合并到保留实体。
        孤立边：src_id 或 dst_id 不存在的边，以及 evidence_ref 指向不存在
        memory 的边均被删除。

        返回 ``{"merged_entities": N, "removed_edges": N, "removed_self_loops": N,
        "removed_duplicate_edges": N}``。
        """
        stats: dict[str, int] = {
            "merged_entities": 0,
            "removed_edges": 0,
            "removed_self_loops": 0,
            "removed_duplicate_edges": 0,
        }

        # ── 1. 合并重复实体名 (大小写/空白变体) ────────────────────────
        rows = self.conn.execute(
            "SELECT id, name, kind FROM entities ORDER BY id"
        ).fetchall()

        # 按规范化键分组: norm_key -> [(id, original_name, kind)]
        groups: dict[str, list[tuple[str, str, str]]] = {}
        for r in rows:
            norm = r["name"].strip().lower()
            groups.setdefault(norm, []).append((r["id"], r["name"], r["kind"]))

        for norm, entries in groups.items():
            if len(entries) <= 1:
                continue
            entries.sort(key=lambda x: x[0])  # 保留 id 最小的
            canonical_id = entries[0][0]
            for dup_id, dup_name, _ in entries[1:]:
                # 将 memory_entities 指向 dup 的全部改向 canonical；
                # 若有 UNIQUE(memory_id, entity_id) 冲突则删除重复行
                self.conn.execute(
                    "UPDATE OR REPLACE memory_entities SET entity_id = ? "
                    "WHERE entity_id = ?",
                    (canonical_id, dup_id),
                )
                # 将 edges 中引用 dup 的改为 canonical
                self.conn.execute(
                    "UPDATE edges SET src_id = ? WHERE src_id = ?",
                    (canonical_id, dup_id),
                )
                self.conn.execute(
                    "UPDATE edges SET dst_id = ? WHERE dst_id = ?",
                    (canonical_id, dup_id),
                )
                # 删除 dup 实体 (FK CASCADE 已清理 memory_entities)
                self.conn.execute("DELETE FROM entities WHERE id = ?", (dup_id,))
                stats["merged_entities"] += 1

        # ── 清理合并后产生的自环边 ────────────────────────────────────
        cur = self.conn.execute("DELETE FROM edges WHERE src_id = dst_id")
        stats["removed_self_loops"] = cur.rowcount

        # ── 清理重复边 (同 src,dst,relation 只保留一条) ──────────────
        cur = self.conn.execute(
            "DELETE FROM edges WHERE id NOT IN ("
            "SELECT MIN(id) FROM edges GROUP BY src_id, dst_id, relation"
            ")"
        )
        stats["removed_duplicate_edges"] = cur.rowcount

        # ── 2. 清理孤立边 ─────────────────────────────────────────────
        # 2a. 源/目标实体已不存在的边
        cur = self.conn.execute(
            "DELETE FROM edges WHERE src_id NOT IN (SELECT id FROM entities)"
        )
        removed = cur.rowcount
        cur = self.conn.execute(
            "DELETE FROM edges WHERE dst_id NOT IN (SELECT id FROM entities)"
        )
        removed += cur.rowcount
        # 2b. evidence_ref 指向不存在的 memory
        cur = self.conn.execute(
            "DELETE FROM edges WHERE evidence_ref != '' "
            "AND evidence_ref NOT IN (SELECT id FROM memories)"
        )
        removed += cur.rowcount
        stats["removed_edges"] = removed

        self.conn.commit()
        self._fts_rebuild()
        return stats

    def _fts_rebuild(self) -> None:
        """Rebuild FTS5 index to purge ghost entries (deleted memories still in FTS)."""
        try:
            self.conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        except sqlite3.OperationalError:
            pass  # FTS5 table may not exist

    def _score(self, query: str, terms: list[str], haystack: str) -> float:
        score = 0.0
        for term in terms:
            if term.lower() in haystack:
                score += self._term_weight(term)
        # Very small semantic-ish boosts for common Chinese memory questions.
        if "同事" in query and "同事" in haystack:
            score += self.weights.colleague_boost
        if "为什么" in query and any(marker in haystack for marker in MemoryStore.REASON_MARKERS):
            score += self.weights.reason_marker_boost
        return score

    @staticmethod
    def _match_reason(terms: list[str], haystack: str, score: float) -> str:
        matched = [term for term in terms if term.lower() in haystack]
        if matched:
            return "matched terms: " + ", ".join(matched[:8])
        return f"fallback score={score:.2f}"

    @staticmethod
    def _asks_reason(query: str) -> bool:
        return any(marker in query for marker in ("为什么", "原因", "为啥", "为何"))

    @staticmethod
    def _asks_person(query: str) -> bool:
        return any(marker in query for marker in ("谁", "哪位", "哪个人"))

    @classmethod
    def _has_reason(cls, text: str) -> bool:
        return any(marker in text for marker in cls.REASON_MARKERS)

    @staticmethod
    def _has_person_like(text: str) -> bool:
        """Check for CJK bigram/trigram that looks like a person, excluding
        known time/topic/object words that would produce false positives."""
        _NON_PERSON_CJK = {
            "礼物", "键盘", "鼠标", "手机", "电脑", "前天", "昨天",
            "今天", "明天", "后天", "上午", "下午", "晚上", "中午",
            "什么", "怎么", "这个", "那个", "哪个", "因为", "所以",
            "好吃", "好看", "厉害", "红烧", "觉得", "喜欢", "可以",
            "什么", "没有", "不是", "还是", "但是", "虽然", "如果",
        }
        for m in re.finditer(r"[\u4e00-\u9fff]{2,3}", text):
            if m.group() not in _NON_PERSON_CJK:
                return True
        return False

    @staticmethod
    def _asks_time(query: str) -> bool:
        return any(marker in query for marker in ("什么时候", "何时", "几点", "哪天", "哪一天", "几号", "什么时间", "多久", "几点钟"))

    @staticmethod
    def _has_time(text: str) -> bool:
        return bool(re.search(
            r"前[两三四五六七八九十0-9]*天|昨天|今天|明天|前天|后天|"
            r"上[周月]|这[周月]|下[周月]|最近|\d{4}年|\d{1,2}月\d{1,2}日|上周|下周|这周|上午|下午|晚上|早上|中午",
            text,
        ))

    @staticmethod
    def _asks_location(query: str) -> bool:
        return any(marker in query for marker in ("哪里", "哪儿", "在哪", "什么地方", "地点", "位置", "哪个城市", "哪个省"))

    @staticmethod
    def _has_location(text: str) -> bool:
        return bool(re.search(
            r"[\u4e00-\u9fff]{2,}(?:市|省|路|街|区|楼|层|室|房间|家附近|公司|办公室|学校|医院|商场|餐厅|公园)",
            text,
        ))

    @staticmethod
    def _asks_object(query: str) -> bool:
        """Detect object-focused questions.  Strip reason/time phrases first
        so "为什么" / "什么时候" don't suppress a genuine "什么东西" in the
        same query."""
        cleaned = query
        for phrase in ("为什么", "什么时候", "何时", "几点", "哪天", "多久"):
            cleaned = cleaned.replace(phrase, "")
        return any(marker in cleaned for marker in ("什么东西", "什么", "啥", "哪个", "哪种"))

    @staticmethod
    def _has_object(text: str) -> bool:
        return bool(re.search(r"[A-Za-z0-9\-]{2,}", text)) or any(
            obj in text for obj in ("键盘", "鼠标", "耳机", "礼物", "书", "手机", "电脑", "猫", "狗")
        )

    @staticmethod
    def _suggest_question(missing: list[str]) -> str:
        questions: dict[str, str] = {
            "reason": "你之前有说过为什么/出于什么原因吗？",
            "person": "你说的是哪一位？",
            "time": "这大概是什么时候的事？",
            "location": "这发生在哪里？",
            "object": "具体是什么东西？",
        }
        parts = [questions[m] for m in missing if m in questions]
        return " ".join(parts) if parts else "这部分记忆不够完整，你能补充一下吗？"
