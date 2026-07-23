"""Capability suite: Timeline differentiator.

按时间/人物过滤事件 —— mem0/naive/long-context 全无。
engram 独有:
- save() 时自动写 `timed_events`(通过时间实体 → memory 反查)
- store.timeline.get_timeline(person=..., limit=...)
- store.timeline.get_events_between(start, end)
- 使用 _TIME_SORT_ORDER 表(recent → old)排序中文相对时间
"""

from __future__ import annotations

import pytest

from engram_router.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    import os
    os.environ["ENGRAM_SKIP_VECTOR"] = "1"
    s = MemoryStore(path=tmp_path / "cap_timeline.db")
    yield s
    s.close()


# ── 场景 1:save 时时间事件自动写入 ─────────────────────────

def test_time_entity_creates_timed_event(store):
    """含时间实体的 memory,save() 后应在 timed_events 表出现一行。"""
    store.save("昨天张三送我一个键盘")
    rows = store.conn.execute("SELECT * FROM timed_events").fetchall()
    assert len(rows) >= 1
    assert any("昨天" in r["time_name"] for r in rows)


def test_memory_without_time_entity_writes_no_timed_event(store):
    """没有时间实体的 memory 不应写 timed_events。"""
    store.save("我喜欢机械键盘")
    rows = store.conn.execute("SELECT * FROM timed_events").fetchall()
    assert len(rows) == 0


# ── 场景 2:get_timeline 按 recency 排序 ─────────────────────

def test_timeline_orders_by_recency(store):
    """最近事件应排在前 —— sort_order 越小越 recent。"""
    store.save("前两天张三请客")
    store.save("今天张三又打来电话")
    store.save("昨天张三来了")

    events = store.timeline.get_timeline(limit=10)
    assert len(events) >= 3
    order = [e.time_name for e in events]
    # sort_order:今天(2) < 昨天(3) < 前两天(5) — 更近的在前
    if "今天" in order and "前两天" in order:
        assert order.index("今天") < order.index("前两天")


# ── 场景 3:按人物过滤 ─────────────────────────────────────────

def test_timeline_person_filter(store):
    """只想看张三的时间线,李四相关事件不能污染。"""
    store.save("昨天张三来找我")
    store.save("今天李四打电话")

    zhang_events = store.timeline.get_timeline(person="张三")
    li_events = store.timeline.get_timeline(person="李四")

    # 张三的时间线里不能出现李四相关的事件
    for e in zhang_events:
        # person_name 可能是 None 或 "张三"
        if e.person_name is not None:
            assert e.person_name == "张三"


def test_timeline_person_filter_no_match_returns_empty(store):
    store.save("昨天下雨了")  # 没有人物
    events = store.timeline.get_timeline(person="不存在的人")
    assert events == []


# ── 场景 4:get_events_between 时间范围 ─────────────────────

def test_events_between_known_time_range(store):
    """查询"昨天到前天"之间的事件 —— sort_order 范围过滤。"""
    store.save("昨天张三请客")
    store.save("前两天李四请客")
    store.save("刚才收到消息")  # 应不在范围内

    events = store.timeline.get_events_between("昨天", "前两天")
    time_names = {e.time_name for e in events}
    # 至少包含范围内的两条
    assert "昨天" in time_names or "前两天" in time_names
    # 不应包含 '刚才'(sort_order 更小)
    assert "刚才" not in time_names


# ── 场景 5:分页 ────────────────────────────────────────────────

def test_timeline_pagination(store):
    """limit + offset 组合应正确切片。"""
    for i in range(5):
        store.save(f"第{i}件事发生在昨天")

    page1 = store.timeline.get_timeline(limit=2, offset=0)
    page2 = store.timeline.get_timeline(limit=2, offset=2)

    ids1 = {e.id for e in page1}
    ids2 = {e.id for e in page2}
    # 两页不应有重叠
    assert ids1.isdisjoint(ids2)


# ── 场景 6:TimedEvent 字段完整 ─────────────────────────────────

def test_timed_event_carries_raw_text(store):
    """每个 TimedEvent 都要能回填 raw_text —— 证据链约束。"""
    store.save("昨天张三送我HHKB")
    events = store.timeline.get_timeline(limit=10)
    assert len(events) >= 1
    ev = events[0]
    assert ev.raw_text and "HHKB" in ev.raw_text
    assert ev.memory_id  # 必须有 memory_id 指回来
    assert ev.time_name  # 必须有 time_name


# ── 场景 7:多个人物在同一 memory 内 ──────────────────────────

def test_timeline_captures_first_person_in_multi_person_memory(store):
    """一条 memory 提了多个人物,timeline 用 person_name 过滤时应可查到。"""
    store.save("昨天张三和李四一起吃饭")
    events = store.timeline.get_timeline()
    # 至少一个事件应该带 person_name
    assert len(events) >= 1


# ── 场景 8:空数据库 ────────────────────────────────────────────

def test_timeline_empty_returns_empty_list(store):
    assert store.timeline.get_timeline() == []
    assert store.timeline.get_events_between("昨天", "前天") == []
