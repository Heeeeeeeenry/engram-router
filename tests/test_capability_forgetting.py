"""Capability suite: ForgettingEngine differentiator.

Ebbinghaus-inspired forgetting —— mem0/naive/long-context 全无。
engram 独有:
- `store.forgetting.decay_score(memory)` 计算当前衰减分
- `store.forgetting.should_forget(memory)` 判断是否触发遗忘
- `store.forgetting.forget(memory_id)` 打 forgotten 标记(软删除)
- `store.forgetting.unmark_forgotten(memory_id)` 恢复
- `store.forgetting.consolidate()` 批量维护

关键性质:
- forgotten 的 memory 不参与召回(降权,不硬删)
- 原文完整保留(evidence 优先原则)
- corrections 表与 forgetting 是**协同**关系
"""

from __future__ import annotations

import pytest

from engram_router.store import MemoryStore, MemoryRecord


def _get_record(store, mid):
    """Fetch a MemoryRecord for the given id — forgetting engine expects the
    dataclass, not a raw sqlite3.Row."""
    row = store.conn.execute("SELECT * FROM memories WHERE id = ?",
                             (mid,)).fetchone()
    assert row is not None
    return store._row_to_record(row, score=1.0, match_reason="")


@pytest.fixture
def store(tmp_path):
    import os
    os.environ["ENGRAM_SKIP_VECTOR"] = "1"
    s = MemoryStore(path=tmp_path / "cap_forgetting.db")
    yield s
    s.close()


# ── 场景 1:forget 标记不硬删 ──────────────────────────────────

def test_forget_marks_but_does_not_delete_row(store):
    mid = store.save("张三送我一个键盘")
    result = store.forgetting.forget(mid)
    assert result is True

    # memories 表里 memory 还在
    row = store.conn.execute(
        "SELECT * FROM memories WHERE id = ?", (mid,)
    ).fetchone()
    assert row is not None
    assert row["raw_text"] == "张三送我一个键盘"


def test_forget_unknown_id_returns_false(store):
    assert store.forgetting.forget("mem_999") is False


# ── 场景 2:forgotten memory 从 recall 中降权 ─────────────────

def test_forgotten_memory_downweighted_in_recall(store):
    mid1 = store.save("张三送我一个键盘")
    mid2 = store.save("张三送我一本书")

    # 忘掉第一条
    store.forgetting.forget(mid1)

    records = store.recall("张三送我什么", top_k=5)
    ids = [r.id for r in records]
    if mid1 in ids and mid2 in ids:
        r1 = next(r for r in records if r.id == mid1)
        r2 = next(r for r in records if r.id == mid2)
        # 被 forget 的应该分数更低
        assert r1.score <= r2.score


# ── 场景 3:unmark_forgotten 恢复 ─────────────────────────────

def test_unmark_forgotten_restores_visibility(store):
    mid = store.save("张三喜欢钓鱼")
    store.forgetting.forget(mid)
    ok = store.forgetting.unmark_forgotten(mid, confidence=1.0)
    assert ok is True

    records = store.recall("张三喜欢什么", top_k=5)
    assert any(r.id == mid for r in records)


def test_unmark_unknown_id_returns_false(store):
    assert store.forgetting.unmark_forgotten("mem_999") is False


# ── 场景 4:decay_score 计算 ─────────────────────────────────

def test_decay_score_returns_float_between_zero_and_one(store):
    mid = store.save("测试记忆")
    rec = _get_record(store, mid)

    score = store.forgetting.decay_score(rec)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_fresh_memory_has_high_decay_score(store):
    """刚存的 memory 应有较高的 retention 分。"""
    mid = store.save("刚刚发生的事")
    rec = _get_record(store, mid)
    score = store.forgetting.decay_score(rec)
    # 应该 >= 0.5(不会立即降到 0)
    assert score >= 0.5


# ── 场景 5:should_forget 判断 ─────────────────────────────────

def test_should_forget_returns_bool(store):
    mid = store.save("测试记忆")
    rec = _get_record(store, mid)
    result = store.forgetting.should_forget(rec)
    assert isinstance(result, bool)


# ── 场景 6:consolidate 批量维护 ───────────────────────────────

def test_consolidate_returns_report(store):
    for i in range(3):
        store.save(f"记忆 {i}")

    report = store.forgetting.consolidate()
    assert isinstance(report, dict)
    # 报告至少含"处理数"或类似字段
    assert len(report) > 0


# ── 场景 7:forget + evidence 保留 ────────────────────────────

def test_forget_does_not_delete_evidence(store):
    """项目核心主张:软遗忘不能破坏证据链。"""
    mid = store.save("张三送我HHKB")
    ev_rows_before = store.conn.execute(
        "SELECT COUNT(*) c FROM evidence WHERE memory_id = ?", (mid,)
    ).fetchone()["c"]

    store.forgetting.forget(mid)

    ev_rows_after = store.conn.execute(
        "SELECT COUNT(*) c FROM evidence WHERE memory_id = ?", (mid,)
    ).fetchone()["c"]

    assert ev_rows_after == ev_rows_before
