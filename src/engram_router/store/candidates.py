"""FTS5 candidate retrieval, row access, and index helpers for EngramRouter.

Extracted from store.py (Step 4): FTS5 initialization, candidate retrieval
via trigram + LIKE fallback, batch row fetch, entity mapping, and access
tracking.  These are pure database operations — they receive ``conn`` (an
sqlite3.Connection) and ``weights`` (RecallWeights) from the caller so they
don't couple to MemoryStore's lifecycle.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, cast

from .scoring import RecallWeights

logger = logging.getLogger(__name__)

# ── Batch size ---------------------------------------------------------------
# SQLite has a hard limit of SQLITE_MAX_VARIABLE_NUMBER (default 32766).
# We batch at 900 to stay well under it while still being efficient for
# large IN(...) lists across all call sites.
_SQLITE_IN_BATCH = 900


# ── FTS5 initialisation ------------------------------------------------------

def init_fts(conn: sqlite3.Connection) -> bool:
    """Create the FTS5 trigram virtual table used as a candidate source.

    FTS5 is a *candidate* retrieval path, not the ranker. The trigram
    tokenizer matches substrings of ≥3 characters (ASCII brands like HHKB,
    and CJK words of ≥3 chars such as 机械键盘). It cannot match 2-char CJK
    queries (键盘/张三 -> 0 hits, trigram needs ≥3 chars), so recall falls
    back to the weighted ranker over all rows when FTS yields nothing. We
    probe support once; if the build lacks FTS5/trigram we degrade silently
    to the full-scan path (correctness is unchanged, only the candidate
    pre-filter is skipped).

    Returns ``True`` if FTS5 is available and enabled, ``False`` otherwise.
    """
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts "
            "USING fts5(memory_id UNINDEXED, content, tokenize='trigram')"
        )
        return True
    except sqlite3.OperationalError:
        return False


def fts_remove(_memory_id: str) -> None:
    """Best-effort FTS5 cleanup stub.

    FTS5 content-mode 'delete' requires an external content table to
    work (the standalone virtual table does not support the 'delete'
    special command).  Instead, the recall pipeline naturally filters
    ghost entries: ``fts_candidates()`` re-queries memories to confirm
    every FTS5 hit exists.

    When FTS5 is rebuilt (e.g. after a bulk delete), this stub can be
    replaced with a full table rebuild.
    """
    # Currently a no-op: ghost entries are harmless because the recall
    # pipeline cross-checks every FTS5 candidate against the memories
    # table.  See fts_candidates() below for the confirmation step.
    pass


def fts_rebuild(conn: sqlite3.Connection) -> None:
    """Rebuild FTS5 index to purge ghost entries (deleted memories still in FTS)."""
    try:
        conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
    except sqlite3.OperationalError:
        pass  # FTS5 table may not exist


# ── Indices ------------------------------------------------------------------

def init_indices(conn: sqlite3.Connection) -> None:
    """Indices for the hot recall paths (entity<->memory map, edge hops).

    These only speed up the existing queries; they do not change results.
    """
    conn.executescript(
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


# ── FTS5 candidate retrieval -------------------------------------------------

def fts_candidates(
    conn: sqlite3.Connection,
    fts_enabled: bool,
    query: str,
    terms: list[str],
    namespace: str = "default",
) -> set[str] | None:
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
    _ = query  # reserved for future query-based heuristics
    if not fts_enabled:
        return None
    fts_terms = [t for t in terms if len(t) >= 3]
    raw_ids: set[str] = set()

    # --- FTS5 trigram path ---
    if fts_terms:
        match_expr = " OR ".join('"' + t.replace('"', '""') + '"' for t in fts_terms)
        try:
            rows = conn.execute(
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
                like_rows = conn.execute(
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
            entity_rows = conn.execute(
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
        ns_rows = conn.execute(
            f"SELECT id FROM memories WHERE id IN ({placeholders}) AND namespace = ?",
            tuple(raw_ids) + (namespace,),
        ).fetchall()
        return {r["id"] for r in ns_rows}
    return raw_ids if raw_ids else None


# ── Row-level access ---------------------------------------------------------

def rows_by_ids(
    conn: sqlite3.Connection,
    ids: list[str],
    ordered: bool = False,
    namespace: str | None = None,
) -> list[sqlite3.Row]:
    """Fetch memory rows by id list, batched for SQLite variable limits."""
    if not ids:
        return []
    rows: list[sqlite3.Row] = []
    suffix = " ORDER BY created_at DESC, id DESC" if ordered else ""
    ns_clause = " AND namespace = ?" if namespace else ""
    for i in range(0, len(ids), _SQLITE_IN_BATCH):
        batch = ids[i : i + _SQLITE_IN_BATCH]
        placeholders = ",".join("?" for _ in batch)
        params = tuple(batch) + ((namespace,) if namespace else ())
        rows.extend(
            conn.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders}){ns_clause}" + suffix,
                params,
            ).fetchall()
        )
    return rows


def row_by_id(
    conn: sqlite3.Connection,
    mem_id: str,
    namespace: str | None = None,
) -> sqlite3.Row | None:
    """Fetch a single memory row by id. Returns None if not found."""
    if namespace is None:
        row = conn.execute(
            "SELECT * FROM memories WHERE id = ?", (mem_id,)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM memories WHERE id = ? AND namespace = ?",
            (mem_id, namespace),
        ).fetchone()
    return cast("sqlite3.Row | None", row)


def memory_rows(
    conn: sqlite3.Connection,
    weights: RecallWeights,
    fts_ids: set[str] | None,
    namespace: str = "default",
) -> list[sqlite3.Row]:
    """Fetch rows to score, using non-empty FTS hits as a candidate filter."""
    if fts_ids:
        return rows_by_ids(conn, list(fts_ids), ordered=True, namespace=namespace)
    # Full-scan fallback: cap at a generous limit to bound scoring cost,
    # then let the Python ranker pick the best top_k.  ORDER BY created_at
    # DESC ensures recent memories win ties; the composite index
    # idx_memories_ns_created covers the WHERE + ORDER BY.
    rows = conn.execute(
        "SELECT * FROM memories WHERE namespace = ? "
        "ORDER BY created_at DESC, id DESC "
        "LIMIT ?",
        (namespace, weights.full_scan_limit),
    ).fetchall()
    if len(rows) >= weights.full_scan_limit:
        logger.warning(
            "Full-scan recall hit limit (%d rows) for namespace=%r — "
            "consider FTS5 or namespace partitioning at scale",
            weights.full_scan_limit, namespace,
        )
    return rows


# ── Entity lookups -----------------------------------------------------------

def entities_for_memories(
    conn: sqlite3.Connection,
    memory_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Return entity names/kinds/salience for a batch of memory ids."""
    if not memory_ids:
        return {}
    rows: list[sqlite3.Row] = []
    for i in range(0, len(memory_ids), _SQLITE_IN_BATCH):
        batch = memory_ids[i : i + _SQLITE_IN_BATCH]
        placeholders = ",".join("?" for _ in batch)
        rows.extend(
            conn.execute(
                "SELECT me.memory_id, e.name, e.kind, me.salience_class "
                "FROM memory_entities me JOIN entities e ON e.id = me.entity_id "
                f"WHERE me.memory_id IN ({placeholders})",
                tuple(batch),
            ).fetchall()
        )
    out: dict[str, list[dict[str, Any]]] = {mid: [] for mid in memory_ids}
    for r in rows:
        # Exclude cjk_ngram entities from the entity map — they are
        # for FTS5 LIKE fallback only, not for scoring or edge expansion.
        if r["kind"] == "cjk_ngram":
            continue
        out.setdefault(r["memory_id"], []).append(
            {"name": r["name"], "kind": r["kind"], "salience_class": r["salience_class"]}
        )
    return out


# ── Access tracking ----------------------------------------------------------

def record_access(conn: sqlite3.Connection, memory_ids: list[str]) -> None:
    """Increment access_count and update accessed_at for recalled memories.

    Called at the end of every successful recall so the forgetting engine
    can see which memories have been recently accessed.
    """
    if not memory_ids:
        return
    now = datetime.now(timezone.utc).isoformat()
    for mid in memory_ids:
        conn.execute(
            "UPDATE memories SET access_count = access_count + 1, "
            "accessed_at = ? WHERE id = ?",
            (now, mid),
        )
    conn.commit()


# ── Re-export _SQLITE_IN_BATCH for store.py callers that need it ─────────────

SQLITE_IN_BATCH = _SQLITE_IN_BATCH
