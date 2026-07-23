"""Capability suite: CausalChain differentiator.

沿 CAUSED_BY 边追溯因果链 —— mem0/naive/long-context 全无。

engram 独有:
- `save()` 时自动写 CAUSED_BY 边(reason marker 触发,confidence 0.95)
- `store.causal.trace_causes(entity_id)` 反向追溯
- `store.causal.trace_effects(entity_id)` 正向传播
"""

from __future__ import annotations

import pytest

from engram_router.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    import os
    os.environ["ENGRAM_SKIP_VECTOR"] = "1"
    s = MemoryStore(path=tmp_path / "cap_causal.db")
    yield s
    s.close()


def _entity_id(store, name):
    """Find an entity id by name — small helper for these tests.

    多次匹配时优先返回 non-cjk_ngram 类别(cjk_ngram 是 fallback,不是主要图节点)。
    """
    rows = store.conn.execute(
        "SELECT id, kind FROM entities WHERE name = ?", (name,)
    ).fetchall()
    if not rows:
        return None
    for r in rows:
        if r["kind"] != "cjk_ngram":
            return r["id"]
    return rows[0]["id"]


# ── 场景 1:save 时自动写 CAUSED_BY 边 ───────────────────────────

def test_causal_edge_written_on_save_with_reason_marker(store):
    """含 '因为' 标记的记忆,save() 后应产生 CAUSED_BY 边。"""
    store.save("张三送我HHKB,因为是生日礼物")

    edges = store.conn.execute(
        "SELECT * FROM edges WHERE relation = 'CAUSED_BY'"
    ).fetchall()
    assert len(edges) > 0, "expected at least one CAUSED_BY edge after save"


def test_no_causal_edge_when_no_marker(store):
    """没有因果标记的普通陈述,不应写 CAUSED_BY 边(项目主张:不推断因果)。"""
    store.save("张三送了我一个键盘")
    store.save("我喜欢机械键盘")

    caused_by = store.conn.execute(
        "SELECT * FROM edges WHERE relation = 'CAUSED_BY'"
    ).fetchall()
    assert len(caused_by) == 0, \
        "CAUSED_BY should not be inferred without an explicit reason marker"


# ── 场景 2:trace_causes 反向追溯 ────────────────────────────────

def test_trace_causes_finds_reason_marker_as_cause(store):
    """键盘(结果) → "因为"(原因标记),trace_causes 应能沿 CAUSED_BY 追溯到 reason 节点。

    engram 当前的实现:reason marker(例如"因为")作为独立实体节点,与被解释的对象
    之间挂 CAUSED_BY 边。这不是"生日 → HHKB" 的因果对,而是"HHKB --CAUSED_BY-->
    因为(某事)"。测试聚焦"追到 reason 节点"这个契约,不假设更精细的语义抽取。
    """
    store.save("张三送我HHKB键盘,因为是生日礼物")

    hhkb_id = _entity_id(store, "HHKB")
    assert hhkb_id is not None, "HHKB entity should exist"

    paths = store.causal.trace_causes(hhkb_id, max_depth=3)
    assert len(paths) >= 1, "expected at least one causal path from HHKB"
    all_names: set[str] = set()
    for path in paths:
        for edge in path.edges:
            all_names.add(edge.src_name)
            all_names.add(edge.dst_name)
    assert any("因为" in n or "生日" in n for n in all_names), \
        f"expected reason marker in cause chain, got {all_names}"


# ── 场景 3:多跳因果链 ────────────────────────────────────────────

def test_trace_causes_multi_hop_when_available(store):
    """连续两条因果句 → CAUSED_BY 边应形成图。

    engram 的 reason marker 是共享节点("因为" 所有句子都指向它),所以真正的多跳链
    需要一个专注 causal 的场景。这里检查:至少能追到 1 段路径。多跳能力的完整测试
    需要更结构化的因果抽取器(未来的 Phase 3+),现在断言最小契约。
    """
    store.save("系统变慢,因为内存不足")
    store.save("内存不足,因为部署了新版本")

    # 找任意 CAUSED_BY 边的 src 作为出发点
    row = store.conn.execute(
        "SELECT src_id FROM edges WHERE relation = 'CAUSED_BY' LIMIT 1"
    ).fetchone()
    if row is None:
        pytest.skip("no CAUSED_BY edges written")

    paths = store.causal.trace_causes(row["src_id"], max_depth=5)
    assert len(paths) >= 1, "expected at least one path from a CAUSED_BY src"


# ── 场景 4:trace_effects 正向传播 ────────────────────────────────

def test_trace_effects_from_reason_marker(store):
    """"因为" 节点是所有原因的共享入口,从它出发应能看到被解释的实体。"""
    store.save("张三送我HHKB键盘,因为是生日礼物")

    reason_id = _entity_id(store, "因为")
    if reason_id is None:
        pytest.skip("reason marker not extracted")

    paths = store.causal.trace_effects(reason_id, max_depth=3)
    # 至少一条路径存在
    assert len(paths) >= 1, "expected at least one downstream path from reason marker"


# ── 场景 5:trace_causes 空图不崩溃 ────────────────────────────────

def test_trace_causes_empty_graph_returns_empty_list(store):
    """没有任何 CAUSED_BY 边时,trace_causes 应返回 [] 而不是崩溃。"""
    store.save("普通的一句话没有因果")
    paths = store.causal.trace_causes("ent_999", max_depth=5)
    assert paths == []


# ── 场景 6:CausalPath.length 属性 ────────────────────────────────

def test_causal_path_length_matches_edge_count(store):
    store.save("张三送我HHKB,因为是生日礼物")
    kb_id = _entity_id(store, "HHKB") or _entity_id(store, "键盘")
    if kb_id is None:
        pytest.skip("keyboard entity not extracted")

    paths = store.causal.trace_causes(kb_id, max_depth=3)
    for p in paths:
        assert p.length == len(p.edges)


# ── 场景 7:trace_causes 的 max_depth 遵守 ────────────────────────

def test_trace_causes_respects_max_depth(store):
    """构造一条 3 段链,max_depth=1 时不应返回 3 段路径。"""
    store.save("系统变慢,因为CPU忙")
    store.save("CPU忙,因为进程多")
    store.save("进程多,因为部署了新版本")

    system_id = _entity_id(store, "系统") or _entity_id(store, "CPU")
    if system_id is None:
        pytest.skip("chain root entity not extracted")

    shallow = store.causal.trace_causes(system_id, max_depth=1)
    for p in shallow:
        assert p.length <= 1


# ── 场景 8:confidence 是 CAUSED_BY 的高 confidence ──────────────

def test_causal_edge_has_high_confidence(store):
    """CAUSED_BY 边应有 confidence 接近 0.95(项目 SCHEMA.md 约定)。"""
    store.save("张三送我HHKB,因为是生日礼物")

    row = store.conn.execute(
        "SELECT confidence FROM edges WHERE relation = 'CAUSED_BY' LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row["confidence"] >= 0.9
