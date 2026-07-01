"""Tests for the forgetting and decay module (Phase 3)."""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone

import pytest

from engram_router.forgetting import (
    ACCESS_BOOST_FRACTION,
    DECAY_HALF_LIFE_DAYS,
    FORGET_THRESHOLD,
    ForgettingConfig,
    ForgettingEngine,
)
from engram_router.store import MemoryStore, MemoryRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine(tmp_path) -> ForgettingEngine:
    """ForgettingEngine backed by a fresh temp DB with schema migrated."""
    store = MemoryStore(path=tmp_path / "mem.db")
    return ForgettingEngine(store)


@pytest.fixture
def populated(engine: ForgettingEngine) -> ForgettingEngine:
    """Engine with a few memories saved and a correction applied."""
    store = engine.store
    store.save("张三说我26岁。")
    mid = store.save("张三说他28岁。")
    store.save("李四喜欢红烧肉。")
    # Mark mid as corrected
    store.conn.execute(
        "INSERT INTO corrections (id, target_id, correction_text) VALUES (?, ?, ?)",
        ("corr_x", mid, "年龄更正为28岁"),
    )
    store.conn.commit()
    return engine


def _make_memory(
    store: MemoryStore,
    raw_text: str,
    confidence: float = 1.0,
    access_count: int = 0,
    accessed_at: str | None = None,
    forgotten: bool = False,
    created_at: str | None = None,
) -> MemoryRecord:
    """Insert a memory row directly for precise control and return a MemoryRecord."""
    mid = store.save(raw_text)
    if confidence != 1.0:
        store.conn.execute(
            "UPDATE memories SET confidence = ? WHERE id = ?", (confidence, mid)
        )
    if access_count:
        store.conn.execute(
            "UPDATE memories SET access_count = ? WHERE id = ?", (access_count, mid)
        )
    if accessed_at:
        store.conn.execute(
            "UPDATE memories SET accessed_at = ? WHERE id = ?", (accessed_at, mid)
        )
    if forgotten:
        store.conn.execute(
            "UPDATE memories SET forgotten = 1, confidence = 1e-6 WHERE id = ?", (mid,)
        )
    if created_at:
        store.conn.execute(
            "UPDATE memories SET created_at = ? WHERE id = ?", (created_at, mid)
        )
    store.conn.commit()

    row = store.conn.execute(
        "SELECT * FROM memories WHERE id = ?", (mid,)
    ).fetchone()
    return MemoryRecord(
        id=mid,
        raw_text=row["raw_text"],
        summary=row["summary"],
        confidence=float(row["confidence"]),
        metadata={
            "source": row["source"],
            "created_at": row["created_at"],
            "accessed_at": row["accessed_at"],
            "access_count": int(row["access_count"]) if row["access_count"] else 0,
            "forgotten": bool(row["forgotten"]),
        },
    )


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


def test_new_columns_exist(tmp_path):
    """Memories table must have access_count, accessed_at, forgotten after init."""
    store = MemoryStore(path=tmp_path / "db.sqlite")
    cols = {
        r["name"]
        for r in store.conn.execute("PRAGMA table_info(memories)").fetchall()
    }
    for col in ("access_count", "accessed_at", "forgotten"):
        assert col in cols, f"Column {col} missing after migration"


def test_legacy_db_migration_adds_columns(tmp_path):
    """Old schema databases get the new columns on upgrade."""
    # Create old-style DB without the phase-3 columns
    conn = sqlite3.connect(str(tmp_path / "legacy.sqlite"))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            raw_text TEXT NOT NULL,
            summary TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'conversation',
            confidence REAL NOT NULL DEFAULT 1.0,
            metadata TEXT NOT NULL DEFAULT '{}',
            namespace TEXT NOT NULL DEFAULT 'default',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO memories (id, raw_text, summary) VALUES ('mem_1', 'old', 'old');
    """)
    conn.commit()
    conn.close()

    # Open — migration should add columns.
    store = MemoryStore(path=tmp_path / "legacy.sqlite")
    cols = {
        r["name"]
        for r in store.conn.execute("PRAGMA table_info(memories)").fetchall()
    }
    assert "access_count" in cols
    assert "accessed_at" in cols
    assert "forgotten" in cols

    # Existing row gets default values.
    row = store.conn.execute(
        "SELECT access_count, accessed_at, forgotten FROM memories WHERE id = ?",
        ("mem_1",),
    ).fetchone()
    assert row["access_count"] == 0
    assert row["accessed_at"] is None
    assert row["forgotten"] == 0


# ---------------------------------------------------------------------------
# decay_score
# ---------------------------------------------------------------------------


def test_decay_score_fresh_memory_is_one(engine):
    """A just-created memory with no elapsed time should have score ≈ 1."""
    mem = _make_memory(engine.store, "fresh memory")
    score = engine.decay_score(mem)
    assert 0.95 <= score <= 1.0


def test_decay_score_halves_after_half_life(engine):
    """After exactly half_life_days, score should be conf * decay_fraction."""
    from datetime import timedelta

    old_dt = (datetime.now(timezone.utc) - timedelta(days=DECAY_HALF_LIFE_DAYS)).isoformat()
    mem = _make_memory(engine.store, "old", created_at=old_dt)
    score = engine.decay_score(mem)
    # Expected: conf * decay_fraction ^ (days / half_life_days)
    # = 1.0 * 0.5 ^ 1 = 0.5
    assert 0.45 <= score <= 0.55, f"score={score}"


def test_decay_score_increases_with_access(engine):
    """Access boost should increase the decay score."""
    from datetime import timedelta

    old_dt = (datetime.now(timezone.utc) - timedelta(days=DECAY_HALF_LIFE_DAYS)).isoformat()
    mem_no_access = _make_memory(engine.store, "old", created_at=old_dt, access_count=0)
    mem_with_access = _make_memory(engine.store, "old2", created_at=old_dt, access_count=5)

    score_no = engine.decay_score(mem_no_access)
    score_yes = engine.decay_score(mem_with_access)
    # With 5 accesses, boost = 1 + 5*0.30 = 2.5x
    assert score_yes > score_no, f"{score_yes} > {score_no}"


def test_decay_score_clamped_to_range(engine):
    """decay_score must always return in [0, 1]."""
    # Very old memory with no access.
    old_dt = "2000-01-01T00:00:00+00:00"
    mem = _make_memory(engine.store, "very old", created_at=old_dt, access_count=0)
    score = engine.decay_score(mem)
    assert 0.0 <= score <= 1.0

    # Memory with very high access_count shouldn't exceed 1.
    mem_boost = _make_memory(
        engine.store, "boosted", access_count=100, accessed_at=datetime.now(timezone.utc).isoformat()
    )
    score_boost = engine.decay_score(mem_boost)
    assert 0.0 <= score_boost <= 1.0


# ---------------------------------------------------------------------------
# should_forget
# ---------------------------------------------------------------------------


def test_should_forget_returns_false_for_fresh_memory(engine):
    """Fresh memories are not forgotten."""
    mem = _make_memory(engine.store, "fresh")
    assert not engine.should_forget(mem)


def test_should_forget_returns_true_for_old_unaccessed_memory(engine):
    """Very old, never-accessed memory should be candidate."""
    from datetime import timedelta

    old_dt = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    mem = _make_memory(engine.store, "ancient", created_at=old_dt, access_count=0)
    assert engine.should_forget(mem)


def test_should_forget_returns_false_for_corrected_memory(populated):
    """User-corrected memories are immune to forgetting."""
    # Find the corrected memory
    row = populated.store.conn.execute(
        "SELECT m.id, m.raw_text, m.summary, m.source, m.confidence, m.metadata, m.created_at "
        "FROM memories m JOIN corrections c ON c.target_id = m.id LIMIT 1"
    ).fetchone()
    assert row is not None
    mem = MemoryRecord(
        id=row["id"],
        raw_text=row["raw_text"],
        summary=row["summary"],
        confidence=float(row["confidence"]),
        metadata={"source": row["source"], "created_at": "2000-01-01T00:00:00+00:00"},
    )
    assert not populated.should_forget(mem)


def test_should_forget_returns_false_for_already_forgotten(engine):
    """Already forgotten memories should not be re-evaluated."""
    mem = _make_memory(engine.store, "forgotten", forgotten=True)
    assert not engine.should_forget(mem)


def test_should_forget_recently_accessed_not_candidate(engine):
    """Memory accessed recently (< half_life) should not be forgotten."""
    from datetime import timedelta

    old_dt = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    recent_dt = datetime.now(timezone.utc).isoformat()
    mem = _make_memory(
        engine.store, "old but accessed", created_at=old_dt,
        access_count=1, accessed_at=recent_dt,
    )
    assert not engine.should_forget(mem)


# ---------------------------------------------------------------------------
# forget / unmark_forgotten
# ---------------------------------------------------------------------------


def test_forget_marks_as_forgotten(engine):
    """forget() sets forgotten=1 and confidence to near-zero."""
    mem = _make_memory(engine.store, "to forget")
    assert engine.forget(mem.id)

    row = engine.store.conn.execute(
        "SELECT forgotten, confidence FROM memories WHERE id = ?",
        (mem.id,),
    ).fetchone()
    assert row["forgotten"] == 1
    assert row["confidence"] < 0.01


def test_forget_idempotent(engine):
    """Calling forget on an already-forgotten memory returns False."""
    mem = _make_memory(engine.store, "idem", forgotten=True)
    assert not engine.forget(mem.id)


def test_forget_nonexistent(engine):
    """Calling forget on non-existent id returns False."""
    assert not engine.forget("nonexistent")


def test_unmark_forgotten_restores(engine):
    """unmark_forgotten restores forgotten flag and confidence."""
    mem = _make_memory(engine.store, "restore me", forgotten=True)
    assert engine.unmark_forgotten(mem.id, confidence=0.8)

    row = engine.store.conn.execute(
        "SELECT forgotten, confidence FROM memories WHERE id = ?",
        (mem.id,),
    ).fetchone()
    assert row["forgotten"] == 0
    assert row["confidence"] == 0.8


def test_unmark_forgotten_nonexistent(engine):
    """unmark_forgotten on non-existent id returns False."""
    assert not engine.unmark_forgotten("no_such")


# ---------------------------------------------------------------------------
# consolidate
# ---------------------------------------------------------------------------


def test_consolidate_merges_near_duplicates(engine):
    """Two nearly identical memories should be merged."""
    engine.store.save("张三前两天送我一把HHKB键盘。")
    engine.store.save("张三前两天送我一把HHKB键盘，手感很好。")
    stats = engine.consolidate()
    assert stats["merged"] >= 1
    assert stats["pairs_found"] >= 1


def test_consolidate_keeps_distinct_memories(engine):
    """Very different memories should not be merged."""
    engine.store.save("张三送我一把HHKB键盘。")
    engine.store.save("李四喜欢吃红烧肉。")
    stats = engine.consolidate()
    assert stats["merged"] == 0


def test_consolidate_no_memories(engine):
    """Empty store should return zeros."""
    stats = engine.consolidate()
    assert stats["merged"] == 0
    assert stats["pairs_found"] == 0


def test_consolidate_respects_namespace(engine):
    """Consolidation should be namespace-scoped."""
    engine.store.save("张三送我一把键盘。", namespace="ns1")
    engine.store.save("张三送我一把键盘手感好。", namespace="ns2")
    stats = engine.consolidate(namespace="ns1")
    assert stats["merged"] == 0  # only one in ns1


# ---------------------------------------------------------------------------
# salience protection
# ---------------------------------------------------------------------------


def test_constraint_salience_immune(engine):
    """Memories linked to constraint/decisional salience entities are immune."""
    store = engine.store
    # Create entity with constraint salience
    store.conn.execute(
        "INSERT INTO entities (id, name, kind, salience_class) VALUES (?, ?, ?, ?)",
        ("ent_prot", "deadline", "topic", "constraint"),
    )
    mid = store.save("项目必须在周五前完成。")
    store.conn.execute(
        "INSERT INTO memory_entities (id, memory_id, entity_id, evidence, salience_class) VALUES (?, ?, ?, ?, ?)",
        ("me_prot", mid, "ent_prot", "周五前", "constraint"),
    )
    store.conn.commit()

    mem = _make_memory(store, "already exists", created_at="2000-01-01T00:00:00+00:00")
    # Use the actual memory that has constraint linkage:
    row = store.conn.execute(
        "SELECT * FROM memories WHERE id = ?", (mid,)
    ).fetchone()
    mem2 = MemoryRecord(
        id=mid,
        raw_text=row["raw_text"],
        summary=row["summary"],
        confidence=float(row["confidence"]),
        metadata={
            "source": row["source"],
            "created_at": "2000-01-01T00:00:00+00:00",
        },
    )
    assert not engine.should_forget(mem2)


# ---------------------------------------------------------------------------
# recall access tracking
# ---------------------------------------------------------------------------


def test_recall_increments_access_count(tmp_path):
    """Each recall hit increments access_count and sets accessed_at."""
    store = MemoryStore(path=tmp_path / "acc.db")
    mid = store.save("HHKB 是机械键盘")

    before = store.conn.execute(
        "SELECT access_count, accessed_at FROM memories WHERE id = ?", (mid,)
    ).fetchone()
    assert before["access_count"] == 0
    assert before["accessed_at"] is None

    records = store.recall("HHKB")
    assert any(r.id == mid for r in records)

    after = store.conn.execute(
        "SELECT access_count, accessed_at FROM memories WHERE id = ?", (mid,)
    ).fetchone()
    assert after["access_count"] >= 1
    assert after["accessed_at"] is not None


def test_recall_multi_access_accumulates(tmp_path):
    """Multiple recalls increment access_count cumulatively."""
    store = MemoryStore(path=tmp_path / "multi.db")
    mid = store.save("HHKB 是机械键盘")

    for _ in range(3):
        store.recall("HHKB")

    row = store.conn.execute(
        "SELECT access_count FROM memories WHERE id = ?", (mid,)
    ).fetchone()
    assert row["access_count"] == 3


# ---------------------------------------------------------------------------
# run_maintenance
# ---------------------------------------------------------------------------


def test_run_maintenance_dry_run(tmp_path):
    """Dry run should identify candidates without marking them."""
    store = MemoryStore(path=tmp_path / "maint.db")
    engine = ForgettingEngine(store)

    # Create a very old memory
    mid = store.save("ancient knowledge")
    store.conn.execute(
        "UPDATE memories SET created_at = ? WHERE id = ?",
        ("2000-01-01T00:00:00", mid),
    )
    store.conn.commit()

    result = engine.run_maintenance(dry_run=True)
    assert len(result["candidates_forgotten"]) >= 1

    # Verify not actually forgotten
    row = store.conn.execute(
        "SELECT forgotten FROM memories WHERE id = ?", (mid,)
    ).fetchone()
    assert row["forgotten"] == 0


def test_run_maintenance_wet_run(tmp_path):
    """Wet run actually marks candidates as forgotten."""
    store = MemoryStore(path=tmp_path / "maint2.db")
    engine = ForgettingEngine(store)

    mid = store.save("ancient knowledge")
    store.conn.execute(
        "UPDATE memories SET created_at = ? WHERE id = ?",
        ("2000-01-01T00:00:00", mid),
    )
    store.conn.commit()

    result = engine.run_maintenance(dry_run=False)
    assert result["forgotten_count"] >= 1

    row = store.conn.execute(
        "SELECT forgotten FROM memories WHERE id = ?", (mid,)
    ).fetchone()
    assert row["forgotten"] == 1


# ---------------------------------------------------------------------------
# ForgettingConfig
# ---------------------------------------------------------------------------


def test_config_defaults():
    c = ForgettingConfig()
    assert c.decay_half_life_days == DECAY_HALF_LIFE_DAYS
    assert c.decay_fraction == 0.5
    assert c.access_boost_fraction == ACCESS_BOOST_FRACTION
    assert c.forget_threshold == FORGET_THRESHOLD


def test_custom_config_alters_behavior(tmp_path):
    """A more aggressive decay config should forget faster."""
    store = MemoryStore(path=tmp_path / "aggro.db")
    config = ForgettingConfig(
        decay_half_life_days=0.1,  # very short half-life
        forget_threshold=0.9,      # very high threshold
    )
    engine = ForgettingEngine(store, config=config)

    from datetime import timedelta
    old_dt = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    mem = _make_memory(store, "quick decay", created_at=old_dt)
    assert engine.should_forget(mem)


# ---------------------------------------------------------------------------
# verify soft-delete not hard-delete
# ---------------------------------------------------------------------------


def test_forgotten_memory_still_exists(engine):
    """Forgotten memories are soft-deleted — row still exists."""
    mem = _make_memory(engine.store, "soft delete test")
    engine.forget(mem.id)

    row = engine.store.conn.execute(
        "SELECT id, raw_text, forgotten, confidence FROM memories WHERE id = ?",
        (mem.id,),
    ).fetchone()
    assert row is not None
    assert row["raw_text"] == "soft delete test"
    assert row["forgotten"] == 1
    assert row["confidence"] < 0.01
