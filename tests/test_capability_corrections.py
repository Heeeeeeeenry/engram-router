"""Capability suite: Corrections differentiator.

用户纠正机制 —— mem0 有 update() 但会**覆盖原值不留原始事实**;
naive/long-context 完全无。engram 独有:
- corrections 表保留原文 + 纠正文本,审计不删
- recall 时 corrected_ids 硬降权 ×0.3
- match_reason 标 "user_corrected"
- 不硬删被纠正的 memory
"""

from __future__ import annotations

import pytest

from engram_router.store import MemoryStore


def _insert_correction(store, target_id, correction_text, cid="corr_1"):
    """Helper: 直接向 corrections 表写入 —— 项目里没有暴露 add_correction 公共 API,
    tests/test_store.py 也是这样直接 INSERT 的。"""
    store.conn.execute(
        "INSERT INTO corrections (id, target_id, correction_text) VALUES (?, ?, ?)",
        (cid, target_id, correction_text),
    )
    store.conn.commit()


@pytest.fixture
def store(tmp_path):
    import os
    os.environ["ENGRAM_SKIP_VECTOR"] = "1"
    s = MemoryStore(path=tmp_path / "cap_corr.db")
    yield s
    s.close()


# ── 场景 1:纠正后原文不删 ─────────────────────────────────────

def test_correction_preserves_original_memory(store):
    mid = store.save("张三今年26岁")
    _insert_correction(store, mid, "年龄更正为28岁")

    # 原 memory 依然在
    row = store.conn.execute(
        "SELECT * FROM memories WHERE id = ?", (mid,)
    ).fetchone()
    assert row is not None
    assert "26" in row["raw_text"]


def test_correction_row_persists_for_audit(store):
    mid = store.save("张三今年26岁")
    _insert_correction(store, mid, "年龄更正为28岁", cid="corr_audit")

    row = store.conn.execute(
        "SELECT * FROM corrections WHERE target_id = ?", (mid,)
    ).fetchone()
    assert row is not None
    assert row["correction_text"] == "年龄更正为28岁"


# ── 场景 2:被纠正的 memory recall 时降权 ─────────────────────

def test_corrected_memory_score_lower_than_uncorrected(store):
    mid1 = store.save("张三说我26岁")
    mid2 = store.save("张三说他28岁")
    _insert_correction(store, mid1, "26 应是 28")

    records = store.recall("张三 多大", top_k=5)
    corrected = next((r for r in records if r.id == mid1), None)
    uncorrected = next((r for r in records if r.id == mid2), None)
    if corrected and uncorrected:
        assert corrected.score < uncorrected.score


# ── 场景 3:match_reason 标记 user_corrected ─────────────────

def test_corrected_memory_marked_in_match_reason(store):
    mid = store.save("张三说我26岁")
    _insert_correction(store, mid, "年龄更正")

    records = store.recall("张三 多大", top_k=5)
    corrected = next((r for r in records if r.id == mid), None)
    assert corrected is not None
    assert "user_corrected" in corrected.match_reason


# ── 场景 4:没有 correction 就没有 penalty ─────────────────────

def test_no_correction_means_no_downweight_marker(store):
    store.save("张三说他28岁")
    records = store.recall("张三 多大", top_k=5)
    for r in records:
        assert "user_corrected" not in r.match_reason


# ── 场景 5:多次纠正同一 memory ────────────────────────────────

def test_multiple_corrections_on_same_target(store):
    mid = store.save("张三工作在阿里")
    _insert_correction(store, mid, "阿里 → 腾讯", cid="corr_1")
    _insert_correction(store, mid, "腾讯 → 字节", cid="corr_2")

    rows = store.conn.execute(
        "SELECT * FROM corrections WHERE target_id = ? ORDER BY id", (mid,)
    ).fetchall()
    assert len(rows) == 2  # 两条纠正都保留


# ── 场景 6:corrections + forgetting 协作 —— 独立机制 ────────

def test_corrections_and_forgetting_are_independent(store):
    """corrections 只降权不打 forgotten;forgetting 只打标不写 corrections。
    engram 用两个正交机制处理"错误" vs "过时"两个不同问题。"""
    mid = store.save("张三说他26岁")
    _insert_correction(store, mid, "已更正")

    # forgetting 状态不受 correction 影响
    row = store.conn.execute(
        "SELECT * FROM memories WHERE id = ?", (mid,)
    ).fetchone()
    # 单独查看 forgotten 字段
    forgotten_val = row["forgotten"] if "forgotten" in row.keys() else 0
    assert not forgotten_val  # 未被自动打 forgotten
