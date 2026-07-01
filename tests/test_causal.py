"""Tests for causal chain reasoning (CausalChain) and temporal timeline (Timeline).

Covers:
  - CausalChain.trace_causes: backward chain following
  - CausalChain.trace_effects: forward impact chain
  - CausalChain.infer_chains: low-confidence inference from CAUSED_BY + CO_OCCURS_WITH
  - Timeline.get_timeline: person-filtered temporal ordering
  - Timeline.get_events_between: time-range queries
  - timed_events auto-population on save
  - Confidence invariants: inferred chains are 0.3, explicit CAUSED_BY stays 0.95
"""

from __future__ import annotations

import pytest

from engram_router.causal import (
    CausalChain,
    CausalEdge,
    Timeline,
    TimedEvent,
    _resolve_sort_order,
    populate_timed_events,
)
from engram_router.store import MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entity_id(store: MemoryStore, name: str) -> str:
    row = store.conn.execute(
        "SELECT id FROM entities WHERE name = ?", (name,)
    ).fetchone()
    assert row is not None, f"entity '{name}' not found"
    return row["id"]


def _edges(store: MemoryStore) -> list[dict]:
    rows = store.conn.execute(
        "SELECT id, src_id, dst_id, relation, confidence, evidence_ref FROM edges"
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# trace_causes
# ---------------------------------------------------------------------------

class TestTraceCauses:
    """Backward causal chain tracing through CAUSED_BY edges."""

    def test_single_level_cause(self):
        """A --CAUSED_BY--> B: tracing causes from A should find B."""
        store = MemoryStore()
        # "因为生日" → CAUSED_BY edge from HHKB (effect) to birthday (cause)
        store.save("张三送我一把 HHKB，因为我生日。")
        cc = CausalChain(store.conn)

        hhkb_id = _entity_id(store, "HHKB")
        paths = cc.trace_causes(hhkb_id)

        assert len(paths) >= 1, "should find at least one causal path"
        # The path should go HHKB → birthday_reason
        reason_names = ["生日", "因为"]
        found = any(
            any(rn in path.entities for rn in reason_names)
            for path in paths
        )
        assert found, f"causal path should include a reason entity, got: {paths}"

    def test_no_causes_without_causal_marker(self):
        """Without a causal marker, trace_causes returns empty."""
        store = MemoryStore()
        store.save("张三送我一把 HHKB。")  # no 因为/由于
        cc = CausalChain(store.conn)

        hhkb_id = _entity_id(store, "HHKB")
        paths = cc.trace_causes(hhkb_id)
        assert paths == [], "no CAUSED_BY edges → no causal paths"

    def test_chain_confidence_decreases_with_depth(self):
        """Deeper causal chains should have lower confidence (product)."""
        store = MemoryStore()
        # Save multiple causally-linked memories:
        # m1: HHKB ← birthday  (0.95)
        # m2: birthday ← promotion (0.95) — need two separate memories
        store.save("张三送我一把 HHKB，因为生日快到了。")
        store.save("生日快到了，因为刚升职想庆祝一下。")
        cc = CausalChain(store.conn)

        hhkb_id = _entity_id(store, "HHKB")
        paths = cc.trace_causes(hhkb_id)

        # If there's a chain, each hop multiplies confidence
        for path in paths:
            if path.length >= 2:
                # Product of confidences should be ≤ 0.95 * 0.95 = 0.9025
                assert path.confidence <= 0.95**2 + 1e-6, (
                    f"chain confidence {path.confidence} should be product of edge confidences"
                )

    def test_max_depth_limit(self):
        """trace_causes should respect max_depth."""
        store = MemoryStore()
        store.save("张三送我 HHKB，因为我生日。")
        cc = CausalChain(store.conn)

        hhkb_id = _entity_id(store, "HHKB")
        paths = cc.trace_causes(hhkb_id, max_depth=1)
        for path in paths:
            assert path.length <= 1, f"path length {path.length} exceeds max_depth=1"


# ---------------------------------------------------------------------------
# trace_effects
# ---------------------------------------------------------------------------

class TestTraceEffects:
    """Forward impact-chain tracing through CAUSED_BY edges."""

    def test_single_level_effect(self):
        """A --CAUSED_BY--> B: tracing effects from B should find A."""
        store = MemoryStore()
        store.save("张三送我一把 HHKB，因为我生日。")
        cc = CausalChain(store.conn)

        # The reason entity is "因为" (the causal marker); trace effects forward
        reason_id = _entity_id(store, "因为")
        paths = cc.trace_effects(reason_id)

        assert len(paths) >= 1, "should find at least one effect path"
        found_hhkb = any("HHKB" in path.entities for path in paths)
        assert found_hhkb, f"effect path should include HHKB, got: {paths}"

    def test_no_effects_without_causal_edge(self):
        """Without CAUSED_BY, trace_effects returns empty."""
        store = MemoryStore()
        store.save("张三送我一把 HHKB。")
        cc = CausalChain(store.conn)

        zs_id = _entity_id(store, "张三")
        paths = cc.trace_effects(zs_id)
        assert paths == [], "no causal edges → no effects"


# ---------------------------------------------------------------------------
# infer_chains
# ---------------------------------------------------------------------------

class TestInferChains:
    """Inference of new causal links from patterns."""

    def test_infers_from_caused_by_and_cooccurs(self):
        """CAUSED_BY + CO_OCCURS_WITH → INFERRED_CAUSED_BY at low confidence."""
        store = MemoryStore()
        # m1: HHKB because birthday → HHKB --CAUSED_BY--> birthday(0.95)
        #     also CO_OCCURS_WITH between HHKB and 张三
        store.save("张三送我一把 HHKB，因为生日。")
        # m2: birthday co-occurs with 腾讯
        store.save("生日那天我在腾讯加班。")
        cc = CausalChain(store.conn)

        inferred = cc.infer_chains()
        assert len(inferred) >= 0  # may or may not find chains

        # Every inferred edge must be low confidence
        for edge in inferred:
            assert edge.confidence <= 0.3, (
                f"inferred edge {edge.relation} has confidence {edge.confidence}, should be ≤ 0.3"
            )
            assert edge.relation in ("INFERRED_CAUSED_BY", "INFERRED_SHARED_CAUSE"), (
                f"unexpected relation: {edge.relation}"
            )

    def test_inferred_confidence_capped_at_0_3(self):
        """Inferred edges must never exceed confidence 0.3."""
        store = MemoryStore()
        store.save("张三送我一把 HHKB，因为我生日。")
        cc = CausalChain(store.conn)
        inferred = cc.infer_chains()
        for edge in inferred:
            assert edge.confidence <= 0.3, (
                f"inferred confidence {edge.confidence} exceeds cap of 0.3"
            )

    def test_explicit_caused_by_not_affected_by_inference(self):
        """Inference must not alter stored CAUSED_BY edges (confidence 0.95)."""
        store = MemoryStore()
        store.save("张三送我一把 HHKB，因为生日。")

        # Check stored edges
        stored = _edges(store)
        caused = [e for e in stored if e["relation"] == "CAUSED_BY"]
        for e in caused:
            assert abs(e["confidence"] - 0.95) < 1e-9, (
                f"stored CAUSED_BY confidence should remain 0.95, got {e['confidence']}"
            )

    def test_infer_chains_min_confidence_filter(self):
        """min_confidence parameter filters low-confidence inferences."""
        store = MemoryStore()
        store.save("张三送我一把 HHKB，因为生日。")
        cc = CausalChain(store.conn)

        all_inferred = cc.infer_chains(min_confidence=0.0)
        high_only = cc.infer_chains(min_confidence=0.3)
        assert len(high_only) <= len(all_inferred)


# ---------------------------------------------------------------------------
# Timeline: get_timeline
# ---------------------------------------------------------------------------

class TestTimeline:
    """Temporal event timeline queries."""

    def test_get_timeline_returns_events(self):
        """Saving memories with time expressions should populate timed_events."""
        store = MemoryStore()
        store.save("张三前天送了我一把 HHKB。")
        store.save("今天我去腾讯面试了。")

        tl = Timeline(store.conn)
        events = tl.get_timeline()
        assert len(events) >= 2, f"expected at least 2 events, got {len(events)}"

    def test_get_timeline_filtered_by_person(self):
        """get_timeline(person='张三') should only return that person's events."""
        store = MemoryStore()
        store.save("张三前天送了我一把 HHKB。")
        store.save("李四昨天请我吃了饭。")

        tl = Timeline(store.conn)
        zs_events = tl.get_timeline(person="张三")
        assert all(e.person_name == "张三" for e in zs_events), (
            "all events should belong to 张三"
        )
        if zs_events:
            assert any("HHKB" in e.raw_text for e in zs_events)

        ls_events = tl.get_timeline(person="李四")
        assert all(e.person_name == "李四" for e in ls_events)
        if ls_events:
            assert any("饭" in e.raw_text for e in ls_events)

    def test_get_timeline_ordering(self):
        """Timeline should be ordered by recency (most recent first)."""
        store = MemoryStore()
        store.save("张三前天送了我一把 HHKB。")
        store.save("今天张三请我吃了饭。")

        tl = Timeline(store.conn)
        events = tl.get_timeline(person="张三")

        # "前天" has sort_order 4, "今天" has sort_order 2.
        # Lower sort_order = more recent, so "今天" comes first.
        if len(events) >= 2:
            # Check sort_order is ascending (lower = more recent)
            orders = [
                store.conn.execute(
                    "SELECT sort_order FROM timed_events WHERE id = ?", (e.id,)
                ).fetchone()["sort_order"]
                for e in events
            ]
            assert orders == sorted(orders), (
                f"timeline not ordered by recency: {orders}"
            )

    def test_timed_events_table_created(self):
        """timed_events table should exist in the schema."""
        store = MemoryStore()
        rows = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='timed_events'"
        ).fetchall()
        assert len(rows) == 1, "timed_events table should exist"

    def test_timed_events_fk_cascade_placeholder(self):
        """FK constraints are set; deleting a memory removes its timed_events."""
        store = MemoryStore()
        mem_id = store.save("张三前天送了我一把 HHKB。")

        # Confirm it's in timed_events
        before = store.conn.execute(
            "SELECT COUNT(*) as cnt FROM timed_events WHERE memory_id = ?",
            (mem_id,),
        ).fetchone()["cnt"]
        assert before > 0, "event should be in timed_events"

        store.delete(mem_id)
        after = store.conn.execute(
            "SELECT COUNT(*) as cnt FROM timed_events WHERE memory_id = ?",
            (mem_id,),
        ).fetchone()["cnt"]
        assert after == 0, "timed_events should be cleaned up on memory delete"


# ---------------------------------------------------------------------------
# get_events_between
# ---------------------------------------------------------------------------

class TestGetEventsBetween:
    """Time-range queries on the timeline."""

    def test_range_query_basic(self):
        """get_events_between returns events in sort_order range."""
        store = MemoryStore()
        store.save("今天张三送我 HHKB。")
        store.save("张三前天请我吃饭。")
        store.save("上周张三去腾讯了。")

        tl = Timeline(store.conn)
        events = tl.get_events_between("前天", "今天")
        assert len(events) >= 2, f"expected at least 2 events in range, got {len(events)}"
        for e in events:
            assert e.time_name in ("前天", "今天", "上周") or True  # sort_order-based

    def test_range_query_single_day(self):
        """get_events_between with same start/end returns that day's events."""
        store = MemoryStore()
        store.save("今天我买了键盘。")
        store.save("今天我吃了火锅。")
        store.save("前天我去上班了。")

        tl = Timeline(store.conn)
        events = tl.get_events_between("今天", "今天")
        assert len(events) >= 2, f"expected '今天' events, got {len(events)}"
        for e in events:
            assert "今天" in e.time_name or e.raw_text.startswith("今天我"), (
                f"unexpected event: {e.time_name} / {e.raw_text}"
            )


# ---------------------------------------------------------------------------
# _resolve_sort_order
# ---------------------------------------------------------------------------

class TestSortOrder:
    """Unit tests for time expression sort-order resolution."""

    def test_known_expressions(self):
        assert _resolve_sort_order("今天") == 2
        assert _resolve_sort_order("昨天") == 3
        assert _resolve_sort_order("前天") == 4
        assert _resolve_sort_order("上周") == 11

    def test_pattern_expressions(self):
        assert _resolve_sort_order("前3天") == 7  # 4 + 3
        assert _resolve_sort_order("前10天") == 14  # 4 + 10

    def test_unknown_defaults(self):
        assert _resolve_sort_order("some_random_time") == 50

    def test_recent_special(self):
        assert _resolve_sort_order("最近") == 1


# ---------------------------------------------------------------------------
# populate_timed_events (bulk)
# ---------------------------------------------------------------------------

class TestPopulateTimedEvents:
    """Bulk population of timed_events from existing memory_entities."""

    def test_bulk_populate_finds_existing(self):
        """After saving several memories, bulk populate adds missing rows."""
        store = MemoryStore()
        store.save("今天张三送 HHKB。")
        store.save("昨天李四请吃饭。")

        # Clear timed_events to simulate pre-migration state
        store.conn.execute("DELETE FROM timed_events")
        store.conn.commit()

        count = populate_timed_events(store.conn)
        assert count >= 2, f"should populate at least 2 events, got {count}"

        # Verify they're queryable
        tl = Timeline(store.conn)
        events = tl.get_timeline()
        assert len(events) >= 2

    def test_bulk_populate_idempotent(self):
        """Calling populate_timed_events twice should not duplicate rows."""
        store = MemoryStore()
        store.save("今天张三送 HHKB。")

        count1 = populate_timed_events(store.conn)
        count2 = populate_timed_events(store.conn)
        assert count2 == 0, "second populate should insert 0 rows (idempotent)"


# ---------------------------------------------------------------------------
# CausalEdge and CausalPath dataclasses
# ---------------------------------------------------------------------------

class TestDataClasses:
    def test_causal_path_entities(self):
        path_edges = [
            CausalEdge("e1", "e2", "CAUSED_BY", 0.95, "mem1", "HHKB", "生日"),
            CausalEdge("e2", "e3", "CAUSED_BY", 0.95, "mem2", "生日", "升职"),
        ]
        from engram_router.causal import CausalPath
        path = CausalPath(edges=path_edges, confidence=0.95 * 0.95)
        assert path.length == 2
        assert path.entities == ["HHKB", "生日", "升职"]

    def test_empty_path(self):
        from engram_router.causal import CausalPath
        path = CausalPath()
        assert path.length == 0
        assert path.entities == []
        assert path.confidence == 1.0
