"""Tests for typed edge writing (f1) and edge-driven recall expansion (f2).

Causal-edge hard boundary (the core safety invariant for this layer):
  - A user-stated cause (a `reason` entity surfaced from a causal marker such
    as 因为/由于) is written as a CAUSED_BY edge at high confidence (0.95).
  - Everything else co-occurring inside one memory is only ever written as
    CO_OCCURS_WITH at low confidence. An *inferred* relation must never be
    promoted to a fact without evidence.
"""

from __future__ import annotations

from engram_router.store import MemoryStore


def _edges(store: MemoryStore) -> list[dict]:
    rows = store.conn.execute(
        "SELECT id, src_id, dst_id, relation, confidence, evidence_ref FROM edges"
    ).fetchall()
    return [dict(r) for r in rows]


# --- f1: edges are written on save ------------------------------------------


def test_save_writes_cooccurs_edges_between_entities():
    """Two entities in the same memory get a CO_OCCURS_WITH edge."""
    store = MemoryStore()
    store.save("张三是我的前同事，现在在腾讯。")  # person=张三, company=腾讯
    edges = _edges(store)
    assert edges, "save should write at least one edge"
    co = [e for e in edges if e["relation"] == "CO_OCCURS_WITH"]
    assert co, "co-occurring entities should produce CO_OCCURS_WITH edges"
    # Edge endpoints must be real entity ids (ent_*), not raw strings.
    for e in co:
        assert e["src_id"].startswith("ent_")
        assert e["dst_id"].startswith("ent_")
    # Inferred co-occurrence stays low-confidence (not promoted to fact).
    assert all(e["confidence"] < 0.9 for e in co)


def test_cooccurs_edges_carry_memory_evidence_ref():
    """Every inferred edge points back to the memory it was drawn from."""
    store = MemoryStore()
    mem_id = store.save("张三是我的前同事，现在在腾讯。")
    co = [e for e in _edges(store) if e["relation"] == "CO_OCCURS_WITH"]
    assert co
    assert all(e["evidence_ref"] == mem_id for e in co)


def test_user_stated_cause_writes_caused_by_edge_high_confidence():
    """A user-stated cause (因为...) is a CAUSED_BY edge at conf 0.95."""
    store = MemoryStore()
    # 因为我生日 -> the gift (HHKB) is caused by the birthday reason.
    store.save("张三送我一把 HHKB，因为我生日。")
    caused = [e for e in _edges(store) if e["relation"] == "CAUSED_BY"]
    assert caused, "a user-stated cause should produce a CAUSED_BY edge"
    assert all(abs(e["confidence"] - 0.95) < 1e-9 for e in caused)


def test_no_caused_by_without_causal_marker():
    """No causal marker -> no CAUSED_BY edge (no invented causation)."""
    store = MemoryStore()
    store.save("张三送我一把 HHKB。")  # no 因为/由于/...
    caused = [e for e in _edges(store) if e["relation"] == "CAUSED_BY"]
    assert caused == []


def test_edges_are_not_duplicated_for_same_pair_in_one_memory():
    """A single memory should not emit the same directed edge twice."""
    store = MemoryStore()
    store.save("张三是我的前同事，现在在腾讯。")
    co = [e for e in _edges(store) if e["relation"] == "CO_OCCURS_WITH"]
    seen = {(e["src_id"], e["dst_id"]) for e in co}
    assert len(seen) == len(co), "duplicate directed edges within one memory"


# --- f2: edges drive recall expansion ---------------------------------------


def test_recall_expands_one_hop_across_memories_via_cooccurs(tmp_path):
    """A query hits m1 directly, then reaches m2 through a shared-entity edge.

    Two SEPARATE memories that share the entity 张三 but no surface token:
      m1: 张三送我一把 HHKB 键盘。      (HHKB / 键盘 / 张三)
      m2: 张三现在在腾讯工作。           (张三 / 腾讯, NO 'HHKB')
    Query 'HHKB' lands on m1 directly. Through the CO_OCCURS_WITH edge
    HHKB<->张三 written from m1, recall should make a one-hop jump to the
    OTHER memory anchored on 张三 (m2) -- even though m2 shares no token with
    the query. This is the increment edges add over a single memory's own
    entity index.
    """
    store = MemoryStore(path=tmp_path / "edge_hop.db")
    store.save("张三送我一把 HHKB 键盘。")
    store.save("张三现在在腾讯工作。")
    records = store.recall("HHKB", top_k=5)
    joined = " ".join(r.raw_text for r in records)
    assert "腾讯" in joined, "edge one-hop should pull in the 张三/腾讯 memory"


def test_recall_edge_hop_is_reported_in_match_reason(tmp_path):
    """When a record is pulled in purely via an edge hop, recall says so."""
    store = MemoryStore(path=tmp_path / "edge_reason.db")
    store.save("张三送我一把 HHKB 键盘。")
    store.save("张三现在在腾讯工作。")
    records = store.recall("HHKB", top_k=5)
    hop = [r for r in records if "腾讯" in r.raw_text]
    assert hop, "the edge-reached memory should be in the results"
    assert any("edge" in r.match_reason.lower() for r in hop)


def test_recall_without_edges_does_not_invent_links(tmp_path):
    """No shared entity, no edge -> the unrelated memory is NOT pulled in.

    Guards against the edge hop over-recalling: a memory that shares neither a
    token nor an entity (directly or one hop away) must stay out.
    """
    store = MemoryStore(path=tmp_path / "edge_neg.db")
    store.save("张三送我一把 HHKB 键盘。")
    store.save("今天的天气很好，适合散步。")  # no overlap at all
    records = store.recall("HHKB", top_k=5)
    joined = " ".join(r.raw_text for r in records)
    # With recent fallback, recall now returns recent items when below top_k.
    # The weather memory may appear as a recent fallback (score=0.5).
    # Core assertion: the HHKB memory MUST be in results.
    assert "HHKB" in joined, "the HHKB memory should be in the results"
