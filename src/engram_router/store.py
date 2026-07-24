"""SQLite-backed memory store for EngramRouter.

The store keeps raw evidence as the source of truth. Summaries and distilled
memories are retrieval aids, not replacements for raw evidence.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import config
from .scoring import RecallWeights, _default_weights
from . import candidates
from . import graph
from . import scoring
from .cross_encoder import CrossEncoderReranker
from .embedding import EmbeddingEngine
from .entities import classify_salience, extract_entities
from .fusion import reciprocal_rank_fusion
from .hyde import HyDEExpander
from .llm_extractor import LLMExtractor, extract_edges_llm, extract_entities_llm
from .llm_reranker import LLMReranker
from .query_expansion import QueryExpander
from . import query_intent

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
        cross_encoder: Any | None = None,
        hyde: Any | None = None,
        embedding_engine: Any | None = None,
        vector_index: Any | None = None,
        query_expander: QueryExpander | None = None,
        enable_vector: bool = True,  # set False to skip embedding model loading (useful for tests)
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
        self.llm_query_extract = llm_query_extract
        self.vector_index = vector_index
        self.query_expander = query_expander if query_expander is not None else QueryExpander(
            allow_cloud_llm=config.privacy.allow_cloud_llm,
        )

        # ── Cloud privacy gating: create default instances when caller doesn't
        #    provide them, gated by config.privacy so data never leaves the
        #    machine without explicit consent.
        if llm_extractor is not None:
            self.llm_extractor = llm_extractor
        else:
            self.llm_extractor = LLMExtractor(
                allow_cloud=config.privacy.allow_cloud_llm,
            )

        if reranker is not None:
            self.reranker = reranker
        else:
            self.reranker = LLMReranker(
                allow_cloud=config.privacy.allow_cloud_reranker,
            )

        # ── Cross-encoder reranker (Phase 1 rerank_and_hyde.md) ──
        _skip_ce = (
            os.environ.get("ENGRAM_SKIP_CE") == "1"
        ) and os.environ.get("ENGRAM_FORCE_CE") != "1"
        if cross_encoder is not None:
            self.cross_encoder = cross_encoder
        elif self.weights.ce_enabled and not _skip_ce:
            try:
                self.cross_encoder = CrossEncoderReranker(
                    model=self.weights.ce_model,
                    max_candidates=self.weights.ce_max_candidates,
                    ce_weight=self.weights.ce_weight,
                    allow_cloud=config.privacy.allow_cloud_reranker,
                )
            except Exception as exc:
                logger.debug("CrossEncoderReranker init failed: %s", exc)
                self.cross_encoder = None
        else:
            self.cross_encoder = None

        # ── HyDE (Phase 2 rerank_and_hyde.md) ──
        if hyde is not None:
            self.hyde = hyde
        elif self.weights.hyde_enabled:
            try:
                self.hyde = HyDEExpander(
                    num_hypotheses=self.weights.hyde_num_hypotheses,
                    min_query_chars=self.weights.hyde_min_query_chars,
                    allow_cloud=config.privacy.allow_cloud_llm,
                    should_run=self.should_inject,
                )
            except Exception as exc:
                logger.debug("HyDEExpander init failed: %s", exc)
                self.hyde = None
        else:
            self.hyde = None

        self.vector_index = vector_index
        # Allow skipping vector via env var for fast test runs
        _skip_vector = not enable_vector or os.environ.get("ENGRAM_SKIP_VECTOR") == "1"
        if _skip_vector:
            self.embedding_engine = None
            self._vector_enabled = False
        else:
            if embedding_engine is not None:
                self.embedding_engine = embedding_engine
            else:
                self.embedding_engine = EmbeddingEngine(
                    allow_remote=config.privacy.allow_cloud_embedding,
                )
            # Auto-create vector index if not provided
            if self.vector_index is None and self.embedding_engine.available:
                from .vector_index import VectorIndex
                vec_path = None
                if path is not None:
                    p = Path(path)
                    vec_path = p.parent / f"{p.stem}.faiss"
                self.vector_index = VectorIndex(
                    dim=self.embedding_engine.dim, path=vec_path,
                )
            self._vector_enabled = (
                self.embedding_engine.available
                and self.vector_index is not None
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

        # ── Phase 3: Persona / Causal / Forgetting (lazy-init) ──
        self._persona: Any = None
        self._causal: Any = None
        self._timeline: Any = None
        self._forgetting: Any = None

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
        """Delegates to candidates.init_indices."""
        candidates.init_indices(self.conn)

    def _init_fts(self) -> None:
        """Delegates to candidates.init_fts; stores _fts_enabled on self."""
        self._fts_enabled = candidates.init_fts(self.conn)

    def _fts_remove(self, memory_id: str) -> None:
        """Delegates to candidates.fts_remove."""
        candidates.fts_remove(memory_id)

    MAX_TEXT_BYTES = 49152  # 48KB — raised from 10KB for LongMemEval _s split (0.09% of memories exceed 10KB, mostly tokenization artifacts)

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

        # ── Phase 3: auto-update persona from saved text ──
        try:
            self._update_persona(text)
        except Exception as exc:
            logger.debug("Persona update skipped: %s", exc)

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
        """Delegate to graph.index_edges."""
        graph.index_edges(
            self.conn, self._next_id,
            memory_id=memory_id, indexed=indexed, text=text,
            llm_extractor=self.llm_extractor,
        )


    # ── Phase 3: Persona / Causal / Timeline lazy-init ──

    @property
    def persona(self):
        if self._persona is None:
            from .persona import PersonaStore
            self._persona = PersonaStore(self, self.llm_extractor)
        return self._persona

    @property
    def causal(self):
        if self._causal is None:
            from .causal import CausalChain
            self._causal = CausalChain(self.conn)
        return self._causal

    @property
    def timeline(self):
        if self._timeline is None:
            from .causal import Timeline
            self._timeline = Timeline(self.conn)
        return self._timeline

    @property
    def forgetting(self):
        if self._forgetting is None:
            from .forgetting import ForgettingEngine, ForgettingConfig
            self._forgetting = ForgettingEngine(self, ForgettingConfig())
        return self._forgetting
    def _apply_decay(self, records):
        try:
            engine = self.forgetting
            for r in records:
                if engine.should_forget(r):
                    engine.forget(r.id)
        except Exception:
            pass
    def _update_persona(self, text):
        try:
            from .entities import extract_entities
            for ent in extract_entities(text):
                name = ent.get("name", "")
                if ent.get("kind") == "person" and name not in ("我", "你", "他", "她", "它"):
                    persona = self.persona.aggregate(name)
                    if persona is not None:
                        self.persona.update(persona)
        except Exception:
            pass


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
        return scoring.base_score(
            query, terms, haystack,
            self.weights, MemoryStore.STOP_CHARS,
            MemoryStore.REASON_MARKERS,
            ranker=self.ranker, store=self,
        )

    def _fts_candidates(self, query: str, terms: list[str],
                        namespace: str = "default") -> set[str] | None:
        """Delegates to candidates.fts_candidates."""
        return candidates.fts_candidates(
            self.conn, self._fts_enabled, query, terms, namespace,
        )

    # _SQLITE_IN_BATCH moved to candidates.py; keep as alias for internal callers
    # that still reference self._SQLITE_IN_BATCH (e.g. _batch_evidence_refs).
    _SQLITE_IN_BATCH = candidates.SQLITE_IN_BATCH

    def _rows_by_ids(self, ids: list[str], ordered: bool = False,
                     namespace: str | None = None) -> list[sqlite3.Row]:
        """Delegates to candidates.rows_by_ids."""
        return candidates.rows_by_ids(self.conn, ids, ordered=ordered, namespace=namespace)

    def _row_by_id(self, mem_id: str, namespace: str | None = None) -> sqlite3.Row | None:
        """Delegates to candidates.row_by_id."""
        return candidates.row_by_id(self.conn, mem_id, namespace=namespace)

    def _memory_rows(self, fts_ids: set[str] | None,
                     namespace: str = "default") -> list[sqlite3.Row]:
        """Delegates to candidates.memory_rows."""
        return candidates.memory_rows(self.conn, self.weights, fts_ids, namespace)

    def _entities_for_memories(self, memory_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Delegates to candidates.entities_for_memories."""
        return candidates.entities_for_memories(self.conn, memory_ids)

    def _record_access(self, memory_ids: list[str]) -> None:
        """Delegates to candidates.record_access."""
        candidates.record_access(self.conn, memory_ids)

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
            eq = self.query_expander.expand(query, async_llm=False)

            # 1. Merge synonyms into extra terms for token matching.
            extra_terms: list[str] = []
            for synonyms in eq.synonyms.values():
                extra_terms.extend(synonyms)

            # 2. Extract entities for the base query.
            query_entity_objs = [
                e for e in extract_entities(query)
                if e.get("kind") != "cjk_ngram"
            ]

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
                    v_entities = [
                        e for e in extract_entities(variant)
                        if e.get("kind") != "cjk_ngram"
                    ]
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
                    row = self._row_by_id(mem_id, namespace=namespace)
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
            query_entity_objs = [
                e for e in extract_entities(query)
                if e.get("kind") != "cjk_ngram"
            ]

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
            # Namespace must be threaded here — edge_bonus may include memory
            # ids whose entities are cross-namespace duplicates; without the
            # filter, hydration would leak rows from other namespaces.
            rows.extend(self._rows_by_ids(missing_edge_ids, namespace=namespace))
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
            # cjk_ngram entities are excluded — they're for FTS5 LIKE fallback only.
            mem_entity_objs = entity_map.get(row["id"], [])
            mem_entities = {
                e["name"] for e in mem_entity_objs
                if e.get("kind") != "cjk_ngram"
            }
            query_ents_filtered = {
                e["name"] for e in query_entity_objs
                if e.get("kind") != "cjk_ngram"
            }
            shared = query_ents_filtered & mem_entities
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

            # Edge-association bonus (scaled to compete with direct term matches).
            bonus = edge_bonus.get(row["id"])
            if bonus is not None:
                score += bonus[0] * self.weights.edge_assoc_boost
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
            rows.extend(self._rows_by_ids(missing_edge_ids, namespace=namespace))
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
            # No keyword/vector hits at all — fall back to vector + recent items.
            fb_records: list[MemoryRecord] = []
            existing_ids: set[str] = set()

            # Phase 2 (inline): vector search fallback for keywordless queries
            if query and self._vector_enabled and self.embedding_engine and self.vector_index:
                try:
                    vec = self.embedding_engine.encode(query)
                    if vec is not None:
                        vector_results = self.vector_index.search(vec, k=top_k * 2)
                        for mid, sim in vector_results:
                            if len(fb_records) >= top_k:
                                break
                            row = self._rows_by_ids([mid], namespace=namespace)
                            if row:
                                rw = row[0]
                                summary = rw["summary"] if rw["summary"] else rw["raw_text"][:160]
                                confidence = rw["confidence"] if rw["confidence"] is not None else 1.0
                                fb_records.append(MemoryRecord(
                                    id=mid, raw_text=rw["raw_text"], summary=summary,
                                    confidence=confidence,
                                    metadata=self._parse_metadata(rw["metadata"]),
                                    evidence_refs=[],
                                    score=self.weights.vector_fallback_base + sim * self.weights.vector_fallback_sim_scale,
                                    match_reason="vector search (semantic)",
                                ))
                                existing_ids.add(mid)
                except Exception as exc:
                    logger.debug("Vector fallback skipped: %s", exc)

            # Fill remaining slots with recent items
            if len(fb_records) < top_k:
                recent_rows = self.conn.execute(
                    "SELECT * FROM memories WHERE namespace = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (namespace, top_k * 2),
                ).fetchall()
                for row in recent_rows:
                    rid = row["id"]
                    if rid not in existing_ids and len(fb_records) < top_k:
                        fb_records.append(self._row_to_record(row, score=self.weights.recent_fallback_score,
                            match_reason="recent fallback (no keyword match)"))
                        existing_ids.add(rid)
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
        # Pre-compute the query vector once — used by RRF fusion (Phase 2)
        # and post-CE fallback (Phase 3.5) below.
        _query_vec: Any = None
        if query and self._vector_enabled and self.embedding_engine and self.vector_index:
            try:
                _query_vec = self.embedding_engine.encode(query)
            except Exception:
                _query_vec = None

        # Track hyde/vector ids for hyde-only de-weighting before CE.
        _hyde_ids: set[str] = set()
        _vector_ids: set[str] = set()
        if _query_vec is not None:
            try:
                vec = _query_vec
                if vec is not None:
                    vector_results = self.vector_index.search(vec, k=top_k * 4)
                    vector_list = [(mid, score) for mid, score in vector_results]
                    keyword_list = [(r.id, r.score) for r in records if r.score > 0]

                    # ── HyDE result list (Phase 2 of rerank_and_hyde.md).
                    hyde_list: list[tuple[str, float]] = []
                    hyde_result = None
                    hyde_skipped = False
                    if self.hyde and getattr(self.hyde, "available", False) \
                            and self.weights.hyde_rrf_weight > 0:
                        try:
                            hyde_list, hyde_result = self.hyde.expand_and_recall(
                                query, self.embedding_engine, self.vector_index,
                                k=self.weights.hyde_top_k,
                            )
                            if hyde_result:
                                if hyde_result.source == "skipped":
                                    hyde_skipped = True
                                if hyde_result.hypotheses:
                                    logger.debug("HyDE: %d hypotheses, %d candidates, source=%s",
                                                 len(hyde_result.hypotheses), len(hyde_list),
                                                 hyde_result.source)
                        except Exception as exc:
                            logger.debug("HyDE skipped: %s", exc)
                            hyde_skipped = True

                    hyde_ids: set[str] = {mid for mid, _ in hyde_list}
                    vector_ids: set[str] = {mid for mid, _ in vector_list}
                    _hyde_ids = hyde_ids
                    _vector_ids = vector_ids

                    result_lists: list[list[tuple[str, float]]] = [keyword_list, vector_list]
                    rrf_weights: list[float] = [
                        self.weights.rrf_keyword_weight,
                        self.weights.rrf_vector_weight,
                    ]
                    if hyde_list:
                        result_lists.append(hyde_list)
                        rrf_weights.append(float(self.weights.hyde_rrf_weight))

                    merged = reciprocal_rank_fusion(
                        result_lists, k=60, weights=rrf_weights)
                    rrf_scores = dict(merged)
                    # Build new list (MemoryRecord is frozen)
                    new_records: list[MemoryRecord] = []
                    existing_ids = {r.id for r in records}
                    for r in records:
                        # Blend: keep keyword score as base, boost if also found by vector
                        rrf_s = rrf_scores.get(r.id, 0)
                        if rrf_s:
                            new_score = r.score + rrf_s * self.weights.rrf_score_boost
                        else:
                            new_score = r.score
                        reason = r.match_reason
                        if r.id in hyde_ids and "HyDE" not in reason:
                            reason = (reason + " · HyDE" if reason else "HyDE").strip(" ·")
                        new_records.append(MemoryRecord(
                            id=r.id, raw_text=r.raw_text, summary=r.summary,
                            confidence=r.confidence, metadata=r.metadata,
                            evidence_refs=r.evidence_refs,
                            score=new_score, match_reason=reason,
                        ))
                    for mid, rrf_score in merged:
                        if mid not in existing_ids and len(new_records) < top_k:
                            row = self._rows_by_ids([mid], namespace=namespace)
                            if row:
                                rw = row[0]
                                summary = rw["summary"] if rw["summary"] else rw["raw_text"][:160]
                                reason = "HyDE (hypothetical)" if mid in hyde_ids else "vector search (semantic)"
                                # Reduce confidence for vector-only hits when HyDE
                                # was skipped (negative query → wrong direction).
                                base_score = max(self.weights.vector_fallback_base,
                                                 rrf_score * self.weights.rrf_new_insert_score_scale)
                                if hyde_skipped and mid not in hyde_ids:
                                    base_score *= self.weights.hyde_skip_vector_penalty
                                new_records.append(MemoryRecord(
                                    id=mid, raw_text=rw["raw_text"], summary=summary,
                                    confidence=rw["confidence"] if rw["confidence"] is not None else 1.0,
                                    metadata=self._parse_metadata(rw["metadata"]),
                                    evidence_refs=[], score=base_score,
                                    match_reason=reason,
                                ))
                    new_records.sort(key=lambda x: x.score, reverse=True)
                    records = new_records
            except Exception as exc:
                logger.debug("Vector search skipped: %s", exc)

        # ── Phase 2.35: HyDE-only candidate de-weight ──
        # Candidates found ONLY by HyDE (not in keyword or vector) are lower
        # confidence.  De-weight so CE doesn't accidentally promote a
        # hyde-only match into top-1 and poison rejection accuracy.
        if _hyde_ids and _vector_ids and records:
            keyword_ids_before = {r.id for r in records
                                  if "HyDE" in (r.match_reason or "")
                                  and "vector" not in (r.match_reason or "").lower()}
            hyde_only = _hyde_ids - _vector_ids - keyword_ids_before
            if hyde_only:
                records = [
                    MemoryRecord(
                        id=r.id, raw_text=r.raw_text, summary=r.summary,
                        confidence=r.confidence, metadata=r.metadata,
                        evidence_refs=r.evidence_refs,
                        score=r.score * self.weights.hyde_only_penalty,
                        match_reason=r.match_reason + " · hyde-only deweight",
                    ) if r.id in hyde_only else r
                    for r in records
                ]

        # ── Phase 2.4: Cross-encoder rerank (Phase 1 rerank_and_hyde.md) ──
        # Save pre-CE signal so Phase 3.5 can decide whether to fall back to
        # vector search.  After CE normalisation all scores fall into [0,1],
        # making threshold-based checks (score <= 0.1) semantically broken.
        _had_meaningful_hits = bool(records) and not all(
            r.score <= 0 for r in records)
        if self.cross_encoder and getattr(self.cross_encoder, "available", False) \
                and query and len(records) > 1:
            try:
                cands = [{"text": r.raw_text, "score": r.score, "id": r.id} for r in records]
                reranked = self.cross_encoder.rerank(query, cands)
                new_order: list[MemoryRecord] = []
                for c in reranked:
                    src = next((r for r in records if r.id == c["id"]), None)
                    if src is None:
                        continue
                    reason = src.match_reason
                    if "ce_score" in c and "cross-encoder" not in reason:
                        reason = (reason + " · cross-encoder" if reason else "cross-encoder").strip(" ·")
                    new_order.append(MemoryRecord(
                        id=src.id, raw_text=src.raw_text, summary=src.summary,
                        confidence=src.confidence, metadata=src.metadata,
                        evidence_refs=src.evidence_refs,
                        score=float(c.get("score", src.score)),
                        match_reason=reason,
                    ))
                seen = {r.id for r in new_order}
                for r in records:
                    if r.id not in seen:
                        new_order.append(r)
                records = new_order
                logger.debug("Cross-encoder reranked %d records", len(cands))
            except Exception as exc:
                logger.debug("Cross-encoder skipped: %s", exc)

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
        self._apply_decay(records)

        # ── Phase 3.5: Vector fallback for queries with no keyword matches ──
        # Uses pre-CE flag (_had_meaningful_hits) so CE/LLM normalisation
        # doesn't cause the threshold to misfire (all scores become [0,1]).
        if (not records or not _had_meaningful_hits) \
                and _query_vec is not None:
            try:
                vec = _query_vec
                if vec is not None:
                    vector_results = self.vector_index.search(vec, k=top_k * 2)
                    existing_ids = {r.id for r in records}
                    for mid, sim in vector_results:
                        if mid not in existing_ids and len(records) < top_k:
                            row = self._rows_by_ids([mid], namespace=namespace)
                            if row:
                                rw = row[0]
                                summary = rw["summary"] if rw["summary"] else rw["raw_text"][:160]
                                confidence = rw["confidence"] if rw["confidence"] is not None else 1.0
                                records.append(MemoryRecord(
                                    id=mid, raw_text=rw["raw_text"], summary=summary,
                                    confidence=confidence,
                                    metadata=self._parse_metadata(rw["metadata"]),
                                    evidence_refs=[],
                                    score=self.weights.vector_fallback_base
                                          + sim * self.weights.vector_fallback_sim_scale,
                                    match_reason="vector search (semantic)",
                                ))
                                existing_ids.add(mid)
            except Exception as exc:
                logger.debug("Vector fallback skipped: %s", exc)

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
                        self._row_to_record(row, score=self.weights.recent_fallback_score,
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

    def _entity_query_relevance(
        self,
        entity_name: str,
        entity_kind: str,
        query_terms: list[str],
        query_entities: set[str],
        query_entity_objs: list[dict[str, Any]],
    ) -> float:
        """Delegate to graph.entity_query_relevance."""
        return graph.entity_query_relevance(
            entity_name, entity_kind,
            query_terms, query_entities, query_entity_objs,
            weights=self.weights,
        )

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
        """Delegate to graph.edge_expansion."""
        return graph.edge_expansion(
            self.conn, weights=self.weights,
            query=query, terms=terms, rows=rows,
            max_hops=max_hops, decay=decay,
            entity_map=entity_map, namespace=namespace,
            base_score_fn=self._base_score,
        )

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

    _asks_brand = staticmethod(query_intent.asks_brand)
    _asks_identity = staticmethod(query_intent.asks_identity)
    _asks_eval = staticmethod(query_intent.asks_eval)
    _asks_reason = staticmethod(query_intent.asks_reason)
    _asks_person = staticmethod(query_intent.asks_person)
    _has_reason = staticmethod(query_intent.has_reason)
    _asks_time = staticmethod(query_intent.asks_time)
    _asks_location = staticmethod(query_intent.asks_location)
    _asks_object = staticmethod(query_intent.asks_object)
    _has_time = staticmethod(query_intent.has_time)
    _has_location = staticmethod(query_intent.has_location)
    _has_object = staticmethod(query_intent.has_object)
    _has_person_like = staticmethod(query_intent.has_person_like)
    _suggest_question = staticmethod(query_intent.suggest_question)

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
        return scoring.terms(query)

    # Common interrogative / filler CJK chars carry almost no retrieval signal.
    # Without down-weighting them, a noise turn that happens to share 我/的/是
    # outscores the turn holding the actual answer.
    STOP_CHARS = frozenset(config.recall.stop_chars)

    def _term_weight(self, term: str) -> float:
        """Weight a term by how much retrieval signal it carries.

        Delegates to scoring.term_weight.
        """
        return scoring.term_weight(term, self.weights, MemoryStore.STOP_CHARS)

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
        """Delegates to candidates.fts_rebuild."""
        candidates.fts_rebuild(self.conn)

    def _score(self, query: str, terms: list[str], haystack: str) -> float:
        return scoring.score(
            query, terms, haystack,
            self.weights, MemoryStore.STOP_CHARS,
            MemoryStore.REASON_MARKERS,
        )

    @staticmethod
    def _match_reason(terms: list[str], haystack: str, score: float) -> str:
        return scoring.match_reason(terms, haystack, score)

