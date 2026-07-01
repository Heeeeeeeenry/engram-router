"""Regression tests for the monotonic id allocator (MemoryStore._next_id).

Background: the original allocator computed the next id as COUNT(*)+1. After
any DELETE the count dropped, so a previously-issued id was handed back and the
next INSERT crashed on the PRIMARY KEY. Phase 5 consolidation (which deletes /
down-weights memories) would hit this immediately.

The fix uses an id_sequences allocator whose next_val only ever climbs. These
tests pin that contract: ids increase, deletes never cause collisions, and a
legacy database (rows present, no id_sequences state) is seeded past its
existing ids on first allocation.
"""
from engram_router.store import MemoryStore


def test_ids_increase_monotonically():
    store = MemoryStore()
    ids = [store.save("张三是我前同事"), store.save("他在腾讯工作"), store.save("HHKB键盘")]
    assert ids == ["mem_1", "mem_2", "mem_3"]


def test_delete_middle_then_save_does_not_collide():
    """Core regression: deleting a row must not let _next_id reissue a live id."""
    store = MemoryStore()
    store.save("第一条")
    mid2 = store.save("第二条")
    store.save("第三条")
    store.conn.execute("DELETE FROM memories WHERE id = ?", (mid2,))
    store.conn.commit()
    new_id = store.save("第四条")  # old COUNT(*)+1 -> mem_3 (exists) -> IntegrityError
    assert new_id == "mem_4"
    ids = {r["id"] for r in store.conn.execute("SELECT id FROM memories")}
    assert ids == {"mem_1", "mem_3", "mem_4"}


def test_delete_all_then_save_keeps_climbing():
    store = MemoryStore()
    store.save("a")
    store.save("b")
    store.conn.execute("DELETE FROM memories")
    store.conn.commit()
    assert store.save("c") == "mem_3"


def test_legacy_db_seeds_past_existing_ids():
    """Rows present but no id_sequences state (a db written under COUNT(*)+1):
    first allocation must seed past the max existing id, not restart at 1."""
    store = MemoryStore()
    store.save("one")
    store.save("two")
    store.save("three")
    store.conn.execute("DELETE FROM id_sequences")  # wipe allocator -> legacy state
    store.conn.commit()
    assert store.save("four") == "mem_4"
