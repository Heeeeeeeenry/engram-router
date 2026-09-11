"""SQLite-backed memory store for EngramRouter.

The store keeps raw evidence as the source of truth. Summaries and distilled
memories are retrieval aids, not replacements for raw evidence.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from ..config import config
from .scoring import RecallWeights, _default_weights
from . import candidates
from . import graph
from . import records
from . import scoring
from ..cross_encoder import CrossEncoderReranker
from ..colbert_reranker import ColBERTReranker
from ..coref import CorefTracker
from ..embedding import EmbeddingEngine
from ..entity_canon import canonicalize_entity_name, _migrate_alias_table
from ..entities import classify_salience, extract_entities
from ..fusion import reciprocal_rank_fusion
from ..hyde import HyDEExpander
from ..llm_extractor import LLMExtractor, extract_edges_llm, extract_entities_llm
from ..llm_reranker import LLMReranker
from ..query_expansion import QueryExpander
from . import query_intent
from . import recall as _recall
from .records import MemoryRecord, serialize_metadata, parse_metadata

logger = logging.getLogger(__name__)


class MemoryStore:
    """Local SQLite memory store.

    Args:
        path: SQLite database path. If omitted, uses an in-memory database.
    """

    REASON_MARKERS = ("因为", "原因", "所以", "导致", "出于", "由于", "为了", "生日")

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
        colbert_reranker: Any | None = None,
        hyde: Any | None = None,
        embedding_engine: Any | None = None,
        vector_index: Any | None = None,
        query_expander: QueryExpander | None = None,
        enable_vector: bool = True,  # set False to skip embedding model loading (useful for tests)
        _adapter: Any = None,  # StorageAdapter (internal — use create_adapter() to construct)
    ) -> None:
        # ── Postgres/Turso adapter layer ──
        if _adapter is not None:
            self.adapter = _adapter
        elif path is not None and str(path).startswith(("postgresql://", "postgres://", "libsql://")):
            from .adapters import create_adapter
            self.adapter = create_adapter(str(path))
        else:
            import sqlite3
            from .adapters import SqliteAdapter
            db_path = str(path) if path is not None else ":memory:"
            self.adapter = SqliteAdapter(sqlite3.connect(db_path))
        self.conn = self.adapter.conn
        # Detect backend dialect (used to gate sqlite-specific features)
        try:
            self._dialect = getattr(self.adapter, 'dialect', 'sqlite')
        except Exception:
            self._dialect = 'sqlite'
        self._is_sqlite = self._dialect == 'sqlite'
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

        # ── ColBERT late-interaction reranker ──
        _skip_colbert = os.environ.get("ENGRAM_SKIP_COLBERT") == "1"
        if colbert_reranker is not None:
            self.colbert_reranker = colbert_reranker
        elif not _skip_colbert and self.weights.ce_enabled:
            try:
                self.colbert_reranker = ColBERTReranker(
                    max_candidates=self.weights.ce_max_candidates,
                )
            except Exception as exc:
                logger.debug("ColBERTReranker init failed: %s", exc)
                self.colbert_reranker = None
        else:
            self.colbert_reranker = None

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
                from ..vector_index import VectorIndex
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
        # ── Postgres/Turso adapter layer (N9) ──
        if path is not None and Path(path).parent != Path('.'):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not self._is_sqlite:
            pass  # Postgres/Turso — no PRAGMA needed
        else:
            if sqlite3.sqlite_version_info < (3, 35, 0):
                raise RuntimeError(
                    "EngramRouter requires SQLite >= 3.35.0 for atomic id allocation "
                    f"(found {sqlite3.sqlite_version})"
                )
            conn = self.conn  # already set by adapter above
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 5000")
            conn.execute("PRAGMA foreign_keys = ON")
            if self.path != ":memory:":
                try:
                    conn.execute("PRAGMA journal_mode = WAL")
                except sqlite3.OperationalError:
                    pass
        self._init_schema()

        # ── Phase 3: Persona / Causal / Forgetting (lazy-init) ──
        self._persona: Any = None
        self._causal: Any = None
        self._coref = CorefTracker()
        self._timeline: Any = None
        self._forgetting: Any = None
        self._intent_classifier: Any = None

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

            -- L2.1: Structured events extracted from memory text.
            -- subject/verb/object is a simplified triple; time_expr is a
            -- natural-language time reference (昨天/上周/2024年3月); the
            -- memory_id FK ties each event back to its evidence source.
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                subject TEXT NOT NULL DEFAULT '',
                verb TEXT NOT NULL DEFAULT '',
                object TEXT NOT NULL DEFAULT '',
                time_expr TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.8,
                namespace TEXT NOT NULL DEFAULT 'default',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_events_memory ON events(memory_id);
            CREATE INDEX IF NOT EXISTS idx_events_namespace ON events(namespace);

            -- L2: Aggregated fact votes from events.
            -- Each row is one (subject, verb, object, namespace) fact with a
            -- vote count.  When the same (subject, verb) has >1 object, the
            -- votes reflect how many times each was mentioned (most-recent
            -- object gets a recency boost).  Downstream consumers query this
            -- table directly for the canonical answer to a fact question
            -- without re-scanning raw events.
            CREATE TABLE IF NOT EXISTS fact_votes (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                verb TEXT NOT NULL,
                object TEXT NOT NULL,
                vote_count INTEGER NOT NULL DEFAULT 1,
                last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                namespace TEXT NOT NULL DEFAULT 'default',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_fact_votes_sv
                ON fact_votes(subject, verb, namespace);

            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'unknown',
                salience_class TEXT NOT NULL DEFAULT 'event',
                namespace TEXT NOT NULL DEFAULT 'default'
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
                evidence_ref TEXT NOT NULL DEFAULT '',
                namespace TEXT NOT NULL DEFAULT 'default'
            );

            CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id);
            CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id);
            CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation);

            CREATE TABLE IF NOT EXISTS corrections (
                id TEXT PRIMARY KEY,
                target_id TEXT NOT NULL,
                correction_text TEXT NOT NULL,
                evidence_ref TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            -- L2: GDPR compliance audit log.  Every mutating operation
            -- (save/delete/consolidate/forget) appends one row.  The audit
            -- trail is append-only — rows are never updated or deleted.
            CREATE TABLE IF NOT EXISTS audit_log (
                id TEXT PRIMARY KEY,
                operation TEXT NOT NULL,
                memory_id TEXT NOT NULL DEFAULT '',
                namespace TEXT NOT NULL DEFAULT 'default',
                details TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_audit_log_memory ON audit_log(memory_id);
            CREATE INDEX IF NOT EXISTS idx_audit_log_ns ON audit_log(namespace);

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
                namespace TEXT NOT NULL DEFAULT 'default',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(time_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
            );

            -- Sampled recall traces for offline replay and debugging (L4.3).
            -- Written probabilistically by recall(); sampling_rate controls
            -- how often a call is snapshotted to avoid storage bloat.
            CREATE TABLE IF NOT EXISTS recall_snapshots (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                namespace TEXT NOT NULL DEFAULT 'default',
                top_k INTEGER NOT NULL DEFAULT 5,
                result_ids TEXT NOT NULL DEFAULT '[]',
                trace_json TEXT NOT NULL DEFAULT '{}',
                latency_ms REAL NOT NULL DEFAULT 0.0,
                memory_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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

        # N3: namespace-scoped entities / edges / timed_events.
        ent_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(entities)").fetchall()}
        if "namespace" not in ent_cols:
            self.conn.execute(
                "ALTER TABLE entities ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'"
            )
        edge_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(edges)").fetchall()}
        if "namespace" not in edge_cols:
            self.conn.execute(
                "ALTER TABLE edges ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'"
            )
        te_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(timed_events)").fetchall()}
        if "namespace" not in te_cols:
            self.conn.execute(
                "ALTER TABLE timed_events ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'"
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

        # Phase 3: timed_events table already created in _init_schema()
        # above (line ~295). This block previously duplicated it — kept as
        # a no-op reference for the migration comment location.

        # Phase 5: edges time range columns (valid_from / valid_to).
        edge_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(edges)").fetchall()}
        if "valid_from" not in edge_cols:
            try:
                self.conn.execute("ALTER TABLE edges ADD COLUMN valid_from TEXT")
            except sqlite3.OperationalError:
                pass
        if "valid_to" not in edge_cols:
            try:
                self.conn.execute("ALTER TABLE edges ADD COLUMN valid_to TEXT")
            except sqlite3.OperationalError:
                pass

        # Phase 6: recall_snapshots table (L4.3).
        # Defined in _init_schema() for new DBs; migration path added here
        # so existing DBs don't crash on first snapshot INSERT.
        snap_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(recall_snapshots)").fetchall()}
        if not snap_cols:
            self.conn.execute(
                """CREATE TABLE recall_snapshots (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    result_ids TEXT NOT NULL DEFAULT '[]',
                    trace_json TEXT NOT NULL DEFAULT '{}',
                    latency_ms REAL NOT NULL DEFAULT 0,
                    memory_count INTEGER NOT NULL DEFAULT 0,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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
        """Save a single memory turn."""
        return self.save_batch([(text, source, metadata)], namespace=namespace)[0]

    def save_batch(
        self,
        items: list[tuple[str, str, dict[str, Any] | None]],
        namespace: str = "default",
    ) -> list[str]:
        """Save multiple memory turns in a single batch.

        Batch-encodes vectors and commits once, avoiding HanLP NER +
        FAISS serialization overhead on per-item calls. For 2500+ item
        scenarios this is a ~5-10x speedup over calling ``save()`` 2500x.
        """
        if not items:
            return []

        texts = [t for t, _, _ in items]
        summaries = [self._summarize(t) for t in texts]

        # ── L3: Save dedup ──
        # If vector is enabled, pre-encode all texts and check whether any
        # are near-duplicates (cosine > 0.95) of existing memories in the
        # same namespace.  Duplicates get an extra evidence record rather
        # than a new memory row.
        duplicate_for: dict[int, str] = {}  # idx → existing memory_id
        pre_embs = None
        if self._vector_enabled and self.embedding_engine and self.vector_index:
            try:
                pre_embs = self.embedding_engine.encode_batch(texts)
                if pre_embs is not None:
                    import numpy as np
                    for idx, vec in enumerate(pre_embs):
                        if idx in duplicate_for:
                            continue
                        hits = self.vector_index.search(vec, k=3)
                        for mid, sim in hits:
                            if sim > 0.95:
                                # Verify same namespace
                                ns_row = self.conn.execute(
                                    "SELECT namespace FROM memories WHERE id = ?",
                                    (mid,),
                                ).fetchone()
                                if ns_row and ns_row["namespace"] == namespace:
                                    duplicate_for[idx] = mid
                                    break
            except Exception as exc:
                logger.debug("Save dedup check skipped: %s", exc)

        # Pre-encode all vectors in one batch (L3: reused for both dedup and indexing)
        # NOTE: use `is not None` — pre_embs is a numpy (N, dim) array whose
        # boolean truth value is ambiguous and would raise ValueError.
        batch_embs = pre_embs if pre_embs is not None else None
        if batch_embs is None and self._vector_enabled and self.embedding_engine and self.vector_index:
            try:
                batch_embs = self.embedding_engine.encode_batch(texts)
            except Exception as exc:
                logger.debug("Batch vector encode skipped: %s", exc)

        ids: list[str] = []
        dedup_count = 0
        for idx, (text, source, metadata) in enumerate(items):
            if not text or not text.strip():
                raise ValueError("text must not be empty")
            if len(text.encode("utf-8")) > self.MAX_TEXT_BYTES:
                raise ValueError(f"text exceeds {self.MAX_TEXT_BYTES} bytes")

            # ── L3: Save dedup — merge near-duplicates into existing memory ──
            existing_id = duplicate_for.get(idx)
            if existing_id:
                # Append evidence only; the existing memory row stays.
                evidence_id = self._next_id("evidence", "evi")
                self.conn.execute(
                    "INSERT INTO evidence (id, memory_id, quote, source_location) VALUES (?, ?, ?, ?)",
                    (evidence_id, existing_id, text, source),
                )
                ids.append(existing_id)
                dedup_count += 1
                continue

            next_id = self._next_id("memories", "mem")
            self.conn.execute(
                "INSERT INTO memories (id, raw_text, summary, source, confidence, metadata, namespace) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (next_id, text, summaries[idx], source, 1.0, self._serialize_metadata(metadata), namespace),
            )
            evidence_id = self._next_id("evidence", "evi")
            self.conn.execute(
                "INSERT INTO evidence (id, memory_id, quote, source_location) VALUES (?, ?, ?, ?)",
                (evidence_id, next_id, text, source),
            )
            self._index_entities(next_id, text, namespace=namespace)
            self._populate_timed_events_for_memory(next_id, namespace=namespace)
            self._index_events(next_id, text, namespace=namespace)
            if self._fts_enabled:
                self.conn.execute(
                    "INSERT INTO memories_fts (memory_id, content) VALUES (?, ?)",
                    (next_id, f"{text} {summaries[idx]}"),
                )
            if batch_embs is not None and idx < len(batch_embs):
                try:
                    self.vector_index.add(next_id, batch_embs[idx])
                except ValueError as _dim_err:
                    # N2: fail-loud on dim mismatch — silent-data-loss was
                    # audit finding SRE-blocker. Detect via error message
                    # containing "dim" and increment counter instead of
                    # swallowing silently.
                    from . import metrics as _m
                    _err_str = str(_dim_err).lower()
                    if "dim" in _err_str:
                        try:
                            _m.dropped_vector_count.labels(reason="dim_mismatch").inc()
                        except Exception:
                            pass
                        logger.warning(
                            "Vector dim mismatch for %s (embedding vs index): %s. "
                            "Consider rebuilding the FAISS index after a model swap.",
                            next_id, _dim_err,
                        )
                    else:
                        try:
                            _m.dropped_vector_count.labels(reason="value_error").inc()
                        except Exception:
                            pass
                        logger.debug("Vector add ValueError for %s: %s", next_id, _dim_err)
                except (ImportError, AttributeError):
                    logger.debug("Vector add skipped for %s", next_id)
            ids.append(next_id)

        self.conn.commit()

        # N6: Persist vector index exactly once after all vectors are added.
        # Previously add() auto-saved on every call — O(n) writes that could
        # leave a partially-written file on mid-batch crash.  Now we commit
        # the SQL rows first, then persist the index once (atomic _save).
        if batch_embs is not None and self._vector_enabled and self.vector_index:
            try:
                self.vector_index.save()
            except Exception as exc:
                logger.debug("Vector index save after batch skipped: %s", exc)

        # N1: save_dedup_ratio (L3: now reflects actual dedup, no longer placeholder 1.0)
        total_items = len(items)
        unique_items = total_items - dedup_count
        dedup_ratio = unique_items / total_items if total_items > 0 else 1.0
        try:
            from . import metrics as _m
            _m.save_dedup_ratio.set(dedup_ratio)
        except Exception:
            pass

        # Audit: log save batch
        self._audit("save", "", namespace, f"batch_count={len(ids)}")

        # Batch persona update
        for text in texts:
            try:
                self._update_persona(text, namespace=namespace)
            except (ImportError, ValueError, AttributeError, KeyError):
                logger.debug("_update_persona skipped", exc_info=True)

        return ids

    def delete(self, memory_id: str, namespace: str = "default") -> bool:
        """Delete a memory and all associated rows (evidence, entities, edges, FTS5).

        FK ON DELETE CASCADE cleans up evidence, distilled_memories, and
        memory_entities.  FTS5 is a virtual table outside FK scope and must
        be explicitly synced via _fts_remove.

        N3: namespace-scoped — verifies ownership before touching FTS/vector
        indexes so a caller in tenant A cannot mutate tenant B's index entries.
        """
        # Verify ownership FIRST — before touching FTS or vector index.
        row = self.conn.execute(
            "SELECT namespace FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None or row["namespace"] != namespace:
            return False

        self._fts_remove(memory_id)

        # ── Phase 2: Remove from vector index ──
        if self._vector_enabled and self.vector_index:
            try:
                self.vector_index.remove(memory_id)
            except (ImportError, ValueError, KeyError):
                logger.debug("vector_index.remove skipped for %s", memory_id)

        cursor = self.conn.execute(
            "DELETE FROM memories WHERE id = ? AND namespace = ?",
            (memory_id, namespace),
        )
        deleted = cursor.rowcount > 0
        if deleted:
            self._audit("delete", memory_id, namespace)
        self.conn.commit()
        return deleted

    # --- entity indexing -----------------------------------------------------

    def _index_entities(self, memory_id: str, text: str, namespace: str = "default") -> None:
        # 0. Pronoun coreference: backfill person entities for pronoun-starting
        #    texts ("她特别喜欢猫" → inject 小李 as person entity).
        raw_entities = extract_entities(text)
        raw_entities = self._coref.resolve(text, raw_entities)

        # 1. Rule-based entities (always runs — the safety net).
        indexed: list[dict[str, Any]] = []
        seen_names: set[tuple[str, str]] = set()

        for ent in raw_entities:
            salience = classify_salience(ent, text)
            entity_id = self._get_or_create_entity(ent["name"], ent["kind"], salience, namespace=namespace)
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
                entity_id = self._get_or_create_entity(ent["name"], ent["kind"], salience, namespace=namespace)
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

        self._index_edges(memory_id, indexed, text, namespace=namespace)

    def _index_edges(self, memory_id: str, indexed: list[dict[str, Any]],
                     text: str = "", namespace: str = "default") -> None:
        """Delegate to graph.index_edges."""
        graph.index_edges(
            self.conn, self._next_id,
            memory_id=memory_id, indexed=indexed, text=text,
            llm_extractor=self.llm_extractor,
            namespace=namespace,
        )


    # ── Phase 3: Persona / Causal / Timeline lazy-init ──

    @property
    def persona(self):
        if self._persona is None:
            from ..persona import PersonaStore
            self._persona = PersonaStore(self, self.llm_extractor)
        return self._persona

    @property
    def causal(self):
        if self._causal is None:
            from ..causal import CausalChain
            self._causal = CausalChain(self.conn)
        return self._causal

    @property
    def timeline(self):
        if self._timeline is None:
            from ..causal import Timeline
            self._timeline = Timeline(self.conn)
        return self._timeline

    @property
    def forgetting(self):
        if self._forgetting is None:
            from ..forgetting import ForgettingEngine, ForgettingConfig
            self._forgetting = ForgettingEngine(self, ForgettingConfig())
        return self._forgetting

    @property
    def intent_classifier(self):
        """Lazy-init LLM query intent classifier (falls back to regex internally).

        Wired so recall.build_scored_candidates can honour it via
        ``getattr(store, 'intent_classifier', None)``.  The classifier degrades
        to regex when the LLM/cloud is unavailable, so it is always safe to
        instantiate.
        """
        if self._intent_classifier is None:
            from ..intent_classifier import IntentClassifier
            # Reuse the shared LLM client so the classifier honours the same
            # cloud opt-in + API key as the extractor, without a second client.
            _client = self.llm_extractor._client if self.llm_extractor is not None else None
            self._intent_classifier = IntentClassifier(client=_client)
        return self._intent_classifier

    @intent_classifier.setter
    def intent_classifier(self, classifier):
        """Allow callers/tests to inject a custom intent classifier."""
        self._intent_classifier = classifier

    def _apply_decay(self, records):
        try:
            engine = self.forgetting
            for r in records:
                if engine.should_forget(r):
                    engine.forget(r.id)
        except (ImportError, ValueError, AttributeError):
            logger.debug("_apply_decay skipped", exc_info=True)
    def _update_persona(self, text, namespace: str = "default"):
        try:
            from ..entities import extract_entities
            for ent in extract_entities(text):
                name = ent.get("name", "")
                if ent.get("kind") == "person" and name not in ("我", "你", "他", "她", "它"):
                    persona = self.persona.aggregate(name, namespace)
                    if persona is not None:
                        self.persona.update(persona)
        except (ImportError, ValueError, AttributeError, KeyError):
            logger.debug("_update_persona skipped", exc_info=True)


    def _get_or_create_entity(self, name: str, kind: str, salience_class: str = "event",
                             namespace: str = "default") -> str:
        # Canonicalize: map aliases (老张 → 张伟) to the canonical entity.
        # Conservatively only does direct lookup; fuzzy suggestions are logged.
        # N3: namespace-scoped — (name, kind, namespace) is the identity key.
        canonical = canonicalize_entity_name(self, name, kind, namespace)

        row = self.conn.execute(
            "SELECT id FROM entities WHERE name = ? AND kind = ? AND namespace = ?",
            (canonical, kind, namespace),
        ).fetchone()
        if row is not None:
            return str(row["id"])
        entity_id = self._next_id("entities", "ent")
        self.conn.execute(
            "INSERT INTO entities (id, name, kind, salience_class, namespace) VALUES (?, ?, ?, ?, ?)",
            (entity_id, canonical, kind, salience_class, namespace),
        )
        return entity_id

    def _populate_timed_events_for_memory(self, memory_id: str, namespace: str = "default") -> None:
        """Insert timed_events rows for any time-kind entities in this memory.

        Called automatically after entity indexing on every :meth:`save`.
        Extracts the first person entity (if any) from the same memory to
        attach as ``person_name``.

        N3: namespace-scoped — stores namespace on each row.
        """
        from ..causal import _resolve_sort_order

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
                    raw_text, person_name, namespace, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event_id,
                    tr["entity_id"],
                    tr["time_name"],
                    sort_order,
                    memory_id,
                    tr["raw_text"],
                    person_name,
                    namespace,
                    tr["created_at"],
                ),
            )

    def _index_events(self, memory_id: str, text: str, namespace: str = "default") -> None:
        """L2.1: Extract structured events and write to events table.
        L2 fact_vote: also upsert into fact_votes for majority-vote aggregation.
        """
        try:
            from ..event_extractor import extract_events
            events = extract_events(text, memory_id=memory_id, namespace=namespace)
            for event in events:
                event_id = self._next_id("events", "evt")
                self.conn.execute(
                    """INSERT INTO events
                       (id, memory_id, subject, verb, object, time_expr, confidence, namespace)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    event.to_sql_row(event_id),
                )
                # Upsert into fact_votes: increment vote count on conflict
                if event.subject and event.verb and event.object:
                    # Deterministic id = namespace:subject:verb:object
                    fv_id = f"fv_{event.subject}_{event.verb}_{event.object}"
                    existing = self.conn.execute(
                        "SELECT id, vote_count FROM fact_votes WHERE id = ? AND namespace = ?",
                        (fv_id, namespace),
                    ).fetchone()
                    if existing:
                        self.conn.execute(
                            """UPDATE fact_votes SET vote_count = ? + 1, last_seen_at = CURRENT_TIMESTAMP
                               WHERE id = ?""",
                            (existing["vote_count"], existing["id"]),
                        )
                    else:
                        self.conn.execute(
                            """INSERT INTO fact_votes (id, subject, verb, object, vote_count, namespace)
                               VALUES (?, ?, ?, ?, 1, ?)""",
                            (fv_id, event.subject, event.verb, event.object, namespace),
                        )

                # L3: write-time conflict check for each event
                if os.environ.get("ENGRAM_CONFLICT_CHECK") in ("1", "true", "yes"):
                    try:
                        from ..contradiction import check_new_event
                        conflicts = check_new_event(self, {
                            "id": event_id,
                            "subject": event.subject,
                            "verb": event.verb,
                            "object": event.object,
                            "memory_id": memory_id,
                            "namespace": namespace,
                        })
                        if conflicts:
                            logger.info(
                                "Conflict detected for event %s: %d conflicts",
                                event_id, len(conflicts),
                            )
                            self._audit("conflict_detected", memory_id, namespace,
                                       f"event_id={event_id} conflicts={len(conflicts)}")
                    except ImportError:
                        pass
        except (ImportError, ValueError, AttributeError) as exc:
            logger.debug("_index_events skipped: %s", exc)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @staticmethod
    def _serialize_metadata(metadata: dict[str, Any] | None) -> str:
        """Delegate to records.serialize_metadata."""
        return serialize_metadata(metadata)

    @staticmethod
    def _parse_metadata(raw: str | None) -> dict[str, Any]:
        """Delegate to records.parse_metadata."""
        return parse_metadata(raw)

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
        """Delegate to graph._entity_names_for."""
        return graph._entity_names_for(self.conn, memory_id)

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
        """Delegate to recall.should_inject."""
        return _recall.should_inject(query)

    def recall(self, query: str, top_k: int = 5,
               namespace: str = "default",
               include_forgotten: bool = False) -> list[MemoryRecord]:
        """Delegate to recall.recall, with optional snapshot sampling (L4.3).

        L3: include_forgotten=True surfaces cold-stored (forgotten=1) memories
        alongside active ones.  Default False to filter them out.
        """
        import time as _time

        t0 = _time.perf_counter()
        results = _recall.recall(self, query, top_k=top_k, namespace=namespace)
        # L3 cold storage: filter or keep forgotten based on include_forgotten
        if not include_forgotten:
            results = [r for r in results if not (r.metadata or {}).get('forgotten')]
        elapsed = (_time.perf_counter() - t0) * 1000
        # Capture any trace from the pipeline tracer for snapshot retention.
        trace = getattr(self, '_tracer', None)
        self._write_snapshot(query, results, trace=trace, latency_ms=elapsed,
                             namespace=namespace, top_k=top_k)

        # L4: A/B shadow recall — run a second recall with different weights
        # and log deltas without affecting the primary result.
        if self._shadow_weights is not None:
            self._run_shadow_recall(query, top_k, namespace, include_forgotten)

        return results

    # ── L4: A/B shadow recall ─────────────────────────────────────────

    _shadow_weights: object | None = None  # RecallWeights for shadow profile

    def set_shadow_profile(self, weights: object) -> None:
        """Enable an A/B shadow recall profile.

        On every subsequent recall(), the primary result is returned unchanged,
        but a second recall() runs in the background with *weights* (e.g. from
        profiles.EXPERIMENT_A).  Differences are logged via debug at the end.

        Set to None to disable.
        """
        self._shadow_weights = weights

    def _run_shadow_recall(
        self, query: str, top_k: int, namespace: str, include_forgotten: bool,
    ) -> None:
        """Run shadow recall with different weights; log deltas to debug."""
        try:
            import time as _time2
            t_s = _time2.perf_counter()

            # Save primary weights, install shadow weights
            primary_weights = self.weights
            object.__setattr__(self, 'weights', self._shadow_weights)
            try:
                shadow_results = _recall.recall(self, query, top_k=top_k, namespace=namespace)
                if not include_forgotten:
                    shadow_results = [
                        r for r in shadow_results
                        if not (r.metadata or {}).get('forgotten')
                    ]
            finally:
                object.__setattr__(self, 'weights', primary_weights)
            shadow_elapsed = (_time2.perf_counter() - t_s) * 1000

            shadow_ids = [r.id for r in shadow_results]
            shadow_top1 = shadow_ids[0] if shadow_ids else None

            logger.info(
                "Shadow recall: %d results, top1=%s, %dms (ns=%s)",
                len(shadow_results), shadow_top1, round(shadow_elapsed, 1), namespace,
            )
        except Exception as exc:
            logger.debug("Shadow recall failed: %s", exc)

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
        """Delegate to recall.build_scored_candidates."""
        return _recall.build_scored_candidates(
            self, query=query, terms=terms, rows=rows,
            entity_map=entity_map, edge_bonus=edge_bonus,
            fts_ids=fts_ids, corrected_ids=corrected_ids,
            query_entities=query_entities, query_topics=query_topics,
            query_identity_subjects=query_identity_subjects,
            query_entity_objs=query_entity_objs,
        )

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
        """Delegate to recall.apply_context_boosts."""
        return _recall.apply_context_boosts(
            self, query, row, score, reason, mem_entity_objs, directly_matched,
            asks_brand=asks_brand, asks_identity=asks_identity,
            asks_eval=asks_eval, query_topics=query_topics,
            query_identity_subjects=query_identity_subjects,
        )

    def _apply_salience_decay(
        self,
        row: sqlite3.Row,
        score: float,
        reason: str,
        mem_entity_objs: list[dict[str, Any]],
        directly_matched: bool,
    ) -> tuple[float, str]:
        """Delegate to recall.apply_salience_decay."""
        return _recall.apply_salience_decay(
            self, row, score, reason, mem_entity_objs, directly_matched,
        )

    def _recall_single(
        self,
        query: str,
        terms: list[str],
        query_entity_objs: list[dict[str, Any]],
        namespace: str = "default",
    ) -> list[MemoryRecord]:
        """Delegate to recall.recall_single."""
        return _recall.recall_single(
            self, query, terms, query_entity_objs, namespace,
        )

    def _build_recall_response(
        self,
        scored: list[tuple[float, str, sqlite3.Row]],
        top_k: int,
        query: str = "",
        namespace: str = "default",
    ) -> list[MemoryRecord]:
        """Delegate to recall.build_recall_response."""
        return _recall.build_recall_response(
            self, scored, top_k=top_k, query=query, namespace=namespace,
        )

    def _batch_evidence_refs(self, memory_ids: list[str]) -> dict[str, list[str]]:
        """Delegate to recall.batch_evidence_refs."""
        return _recall.batch_evidence_refs(self.conn, memory_ids)

    def _batch_raw_refs(self, memory_ids: list[str]) -> dict[str, list[str]]:
        """Delegate to recall.batch_raw_refs."""
        return _recall.batch_raw_refs(self.conn, memory_ids)

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

    def _asks_identity(self, query: str) -> bool:
        """Identity check that uses the LLM classifier when available.

        Delegates to recall._asks_identity(self, query) which honours
        intent_classifier + regex fallback. Fixes M4 shadow bug: the
        original staticmethod pointer to query_intent.asks_identity
        silently bypassed the classifier.
        """
        return _recall._asks_identity(self, query)

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
        """Delegate to records.row_to_record."""
        return records.row_to_record(
            self.conn, row, score, match_reason,
            evidence_refs=evidence_refs, raw_refs=raw_refs,
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
        "recall_snapshots": "snap",
        "events": "evt",
        "fact_votes": "fv",
        "audit_log": "audit",
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
        """Delegate to records.clean_sentence."""
        return records.clean_sentence(text)

    @staticmethod
    def _truncate_cjk(text: str, max_chars: int = 120) -> str:
        """Delegate to records.truncate_cjk."""
        return records.truncate_cjk(text, max_chars)

    @staticmethod
    def _summarize(text: str) -> str:
        """Delegate to records.summarize."""
        return records.summarize(text)

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

    def consolidate(self, namespace: str = "default") -> dict[str, Any]:
        """合并重复实体名、清理孤立边，返回清理统计。

        合并规则：同名实体的大小写/空白变体会被合并到保留实体。
        孤立边：src_id 或 dst_id 不存在的边，以及 evidence_ref 指向不存在
        memory 的边均被删除。

        N3: namespace-scoped — all operations constrained to one namespace.

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
            "SELECT id, name, kind FROM entities WHERE namespace = ? ORDER BY id",
            (namespace,),
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
                # 将 edges 中引用 dup 的改为 canonical (namespace-scoped)
                self.conn.execute(
                    "UPDATE edges SET src_id = ? WHERE src_id = ? AND namespace = ?",
                    (canonical_id, dup_id, namespace),
                )
                self.conn.execute(
                    "UPDATE edges SET dst_id = ? WHERE dst_id = ? AND namespace = ?",
                    (canonical_id, dup_id, namespace),
                )
                # 删除 dup 实体 (FK CASCADE 已清理 memory_entities)
                self.conn.execute(
                    "DELETE FROM entities WHERE id = ? AND namespace = ?",
                    (dup_id, namespace),
                )
                stats["merged_entities"] += 1

        # ── 清理合并后产生的自环边 ────────────────────────────────────
        cur = self.conn.execute(
            "DELETE FROM edges WHERE src_id = dst_id AND namespace = ?",
            (namespace,),
        )
        stats["removed_self_loops"] = cur.rowcount

        # ── 清理重复边 (同 src,dst,relation 只保留一条) ──────────────
        cur = self.conn.execute(
            "DELETE FROM edges WHERE id NOT IN ("
            "SELECT MIN(id) FROM edges WHERE namespace = ? "
            "GROUP BY src_id, dst_id, relation"
            ") AND namespace = ?",
            (namespace, namespace),
        )
        stats["removed_duplicate_edges"] = cur.rowcount

        # ── 2. 清理孤立边 ─────────────────────────────────────────────
        # 2a. 源/目标实体已不存在的边
        cur = self.conn.execute(
            "DELETE FROM edges WHERE namespace = ? "
            "AND src_id NOT IN (SELECT id FROM entities WHERE namespace = ?)",
            (namespace, namespace),
        )
        removed = cur.rowcount
        cur = self.conn.execute(
            "DELETE FROM edges WHERE namespace = ? "
            "AND dst_id NOT IN (SELECT id FROM entities WHERE namespace = ?)",
            (namespace, namespace),
        )
        removed += cur.rowcount
        # 2b. evidence_ref 指向不存在的 memory
        cur = self.conn.execute(
            "DELETE FROM edges WHERE evidence_ref != '' AND namespace = ? "
            "AND evidence_ref NOT IN (SELECT id FROM memories WHERE namespace = ?)",
            (namespace, namespace),
        )
        removed += cur.rowcount
        stats["removed_edges"] = removed

        self.conn.commit()
        self._audit("consolidate", "", namespace, str(stats))
        self._fts_rebuild()
        return stats

    def _audit(self, operation: str, memory_id: str, namespace: str, details: str = "") -> None:
        """Append one row to the audit_log (fire-and-forget, never fails)."""
        try:
            audit_id = self._next_id("audit_log", "audit")
            self.conn.execute(
                """INSERT INTO audit_log (id, operation, memory_id, namespace, details)
                   VALUES (?, ?, ?, ?, ?)""",
                (audit_id, operation, memory_id, namespace, details[:2000]),
            )
        except Exception as exc:
            logger.debug("_audit write skipped: %s", exc)

    # ── L2: fact_vote query API ────────────────────────────────────────

    def get_fact(self, subject: str, verb: str, namespace: str = "default") -> dict[str, Any] | None:
        """Return the majority-voted fact for a (subject, verb) pair.

        Returns the fact with the highest vote_count, or None if no facts exist.
        In case of a tie, the most recently seen fact wins (recency bias).
        """
        row = self.conn.execute(
            """SELECT subject, verb, object, vote_count, last_seen_at
               FROM fact_votes
               WHERE subject = ? AND verb = ? AND namespace = ?
               ORDER BY vote_count DESC, last_seen_at DESC
               LIMIT 1""",
            (subject, verb, namespace),
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        # L2: tag fact vs observation based on vote count
        result["is_fact"] = row["vote_count"] >= 2
        result["is_observation"] = row["vote_count"] == 1
        return result

    def get_confirmed_facts(self, namespace: str = "default") -> list[dict[str, Any]]:
        """Return only confirmed facts (vote_count >= 2), excluding observations."""
        rows = self.conn.execute(
            """SELECT subject, verb, object, vote_count, last_seen_at
               FROM fact_votes
               WHERE namespace = ? AND vote_count >= 2
               ORDER BY vote_count DESC""",
            (namespace,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_facts(self, subject: str, namespace: str = "default") -> list[dict[str, Any]]:
        """Return all voted facts for a subject, ranked by confidence."""
        rows = self.conn.execute(
            """SELECT subject, verb, object, vote_count, last_seen_at
               FROM fact_votes
               WHERE subject = ? AND namespace = ?
               ORDER BY vote_count DESC""",
            (subject, namespace),
        ).fetchall()
        return [dict(r) for r in rows]

    def rebuild_fact_votes(self, namespace: str = "default") -> int:
        """Recompute fact_votes from the events table.

        Useful after bulk imports or when the fact_votes table drifts.
        Returns the number of fact rows written.
        """
        # Clear existing votes for this namespace
        self.conn.execute(
            "DELETE FROM fact_votes WHERE namespace = ?", (namespace,)
        )
        # Re-aggregate from events
        self.conn.execute(
            """INSERT INTO fact_votes
               (id, subject, verb, object, vote_count, namespace)
               SELECT
                   'fv_' || subject || '_' || verb || '_' || object || '_' || CAST(COUNT(*) AS TEXT),
                   subject, verb, object, COUNT(*) as vote_count, namespace
               FROM events
               WHERE namespace = ? AND subject != '' AND verb != '' AND object != ''
               GROUP BY subject, verb, object, namespace""",
            (namespace,),
        )
        count = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM fact_votes WHERE namespace = ?",
            (namespace,),
        ).fetchone()
        self.conn.commit()
        return count["cnt"] if count else 0

    # ── L3: maintenance job ───────────────────────────────────────────

    def run_maintenance(self, namespace: str = "default") -> dict[str, int]:
        """Run all housekeeping operations: consolidate + contradiction scan +
        fact_vote rebuild + persona update.  Returns per-step counts.

        Designed to be called periodically (cron / scheduler) rather than
        on every user interaction.
        """
        stats: dict[str, int] = {}
        from datetime import datetime, timezone

        # 1. Forgetting: apply decay to all active memories
        forgetting = getattr(self, 'forgetting', None)
        if forgetting:
            rows = self.conn.execute(
                "SELECT id, raw_text, summary, confidence, access_count, "
                "accessed_at, forgotten FROM memories "
                "WHERE namespace = ? AND forgotten = 0",
                (namespace,),
            ).fetchall()
            forgotten_count = 0
            for row in rows:
                # Create a minimal record object for should_forget
                class _Row:
                    pass
                r = _Row()
                r.id = row["id"]
                r.raw_text = row["raw_text"]
                r.summary = row["summary"]
                r.confidence = row["confidence"]
                r.metadata = {"accessed_at": row["accessed_at"], "access_count": row["access_count"]}
                r.forgotten = row["forgotten"]

                if forgetting.should_forget(r):
                    forgetting.forget(r.id)
                    forgotten_count += 1
            stats["forgotten"] = forgotten_count

        # 2. Consolidate entities
        cons_result = self.consolidate(namespace=namespace)
        stats.update(cons_result)

        # 3. Contradiction scan
        try:
            from ..contradiction import detect_contradictions
            contra_result = detect_contradictions(self, namespace=namespace)
            stats["contradictions_found"] = contra_result["contradictions"]
            stats["contradiction_edges"] = contra_result["edges_written"]
        except ImportError:
            pass

        # 4. Rebuild fact votes
        stats["facts_rebuilt"] = self.rebuild_fact_votes(namespace=namespace)

        # 5. Record maintenance timestamp
        try:
            audit_id = self._next_id("audit_log", "audit")
            self.conn.execute(
                "INSERT INTO audit_log (id, operation, memory_id, namespace, details) "
                "VALUES (?, 'maintenance', '', ?, '{}')",
                (audit_id, namespace),
            )
        except Exception:
            pass

        self.conn.commit()
        return stats

    # ── L3: reflection ────────────────────────────────────────────────

    def reflect(self, namespace: str = "default", num_recent: int = 50) -> list[str]:
        """Run reflection on recent memories; return focus areas.

        Calls the LLM (if available) to summarise recent memories into
        3-5 broad themes/directions.  Safe to call periodically.
        """
        from ..reflection import run_reflection
        return run_reflection(self, namespace=namespace, num_recent=num_recent)

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

    # --- recall snapshots (L4.3) ---

    def _write_snapshot(
        self,
        query: str,
        results: list[MemoryRecord],
        trace: "RecallTracer | None" = None,
        latency_ms: float = 0.0,
        namespace: str = "default",
        top_k: int = 5,
    ) -> None:
        """Probabilistically write a recall snapshot for offline replay.

        Sampling is gated by ``self.weights.sampling_rate``.  Samples capture
        the query text, result ids, trace JSON (from tracer.to_dict()), wall-clock
        latency, and the store's total memory count at call time.
        """
        import json as _json
        import random as _random

        if self.weights.sampling_rate <= 0.0:
            return
        if _random.random() > self.weights.sampling_rate:
            return

        snap_id = self._next_id("recall_snapshots", "snap")
        result_ids = _json.dumps([r.id for r in results], ensure_ascii=False)
        trace_json = _json.dumps(
            trace.to_dict() if trace else {}, ensure_ascii=False,
        )
        memory_count = self.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE namespace = ?",
            (namespace,),
        ).fetchone()[0]

        self.conn.execute(
            """INSERT INTO recall_snapshots
               (id, query, namespace, top_k, result_ids, trace_json, latency_ms, memory_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (snap_id, query, namespace, top_k, result_ids, trace_json, latency_ms, memory_count),
        )
        self.conn.commit()

    def replay_snapshots(
        self, limit: int = 20, namespace: str = "default",
    ) -> list[dict[str, Any]]:
        """Return recent recall snapshots for offline replay analysis.

        Each snapshot is returned as a dict with keys: id, query, namespace,
        top_k, result_ids (parsed list), trace (parsed dict), latency_ms,
        memory_count, created_at.  Results are ordered by recency.
        """
        import json as _json

        rows = self.conn.execute(
            """SELECT id, query, namespace, top_k, result_ids, trace_json,
                      latency_ms, memory_count, created_at
               FROM recall_snapshots
               WHERE namespace = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (namespace, limit),
        ).fetchall()

        return [
            {
                "id": r["id"],
                "query": r["query"],
                "namespace": r["namespace"],
                "top_k": r["top_k"],
                "result_ids": _json.loads(r["result_ids"]),
                "trace": _json.loads(r["trace_json"]),
                "latency_ms": r["latency_ms"],
                "memory_count": r["memory_count"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

