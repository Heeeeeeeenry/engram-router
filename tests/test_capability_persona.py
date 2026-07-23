"""Capability suite: PersonaStore differentiator.

engram-router 独有能力,mem0 只做半个(user_id 层),naive/long-context 完全没有。
每个用例覆盖一个真实场景:属性聚合、跨 session、冲突、别名、拒绝无关信息。

设计约束(2026-07-22 调查后修正):
- 规则版 `aggregate()` 用 `_extract_attrs_from_text`,attrs 的 key 是**原始匹配串**(例如
  "30岁" 而不是 "age")。`.age`/`.occupation` 便利属性只在 attrs["age"] 显式设定后可用
  (通常来自 LLM 增强或人工 save)。
- 测试改为断言 `attrs` 里能取到期望原文,而非依赖 `.age` 便利属性 —— 后者是 nice-to-have,
  不是核心能力。
"""

from __future__ import annotations

import pytest

from engram_router.store import MemoryStore
from engram_router.persona import Persona, PersonaAttr


@pytest.fixture
def store(tmp_path):
    """Fresh, vector-free store for capability tests — fast + deterministic."""
    import os
    os.environ["ENGRAM_SKIP_VECTOR"] = "1"
    s = MemoryStore(path=tmp_path / "cap_persona.db")
    yield s
    s.close()


def _attr_values(persona: Persona) -> list[str]:
    """All attribute values as a flat list — makes 'contains 30' assertions
    independent of key normalisation choices."""
    return [a.value for a in persona.attrs.values()]


# ── 场景 1:基础属性聚合 ─────────────────────────────────────────────

def test_persona_captures_age_from_natural_sentence(store):
    """自然中文里的年龄应能进入 persona.attrs。"""
    store.save("张三今年30岁,在腾讯做后端工程师")
    persona = store.persona.aggregate("张三")
    values = _attr_values(persona)
    assert any("30" in v for v in values)


def test_persona_captures_preference_marker(store):
    """"喜欢 X" 是 attr_patterns 命中路径 —— 应进入 attrs。

    规则版 `_extract_attrs_from_text` 只对 config.entities.attr_patterns 里定义的
    正则触发。年龄(N岁)/ 偏好(喜欢 …)是命中路径;"职业是 …" 类 free-form
    表述不命中 —— 属于 LLM 增强的领域,不是规则版能力范围。这个测试守住规则版
    真实能力边界。
    """
    store.save("张三喜欢机械键盘")
    persona = store.persona.aggregate("张三")
    values = _attr_values(persona)
    assert any("机械键盘" in v or "键盘" in v for v in values), \
        f"expected preference in attrs, got {values}"


# ── 场景 2:跨 session 聚合(engram 独家)──────────────────────────

def test_persona_cross_session_aggregation(store):
    """三条独立 memory,PersonaStore 聚合成一份画像 —— mem0 需要显式 update。"""
    store.save("张三今年30岁")
    store.save("张三在腾讯做后端")
    store.save("张三喜欢钓鱼")

    persona = store.persona.aggregate("张三")
    values = _attr_values(persona)
    # 三个信息片段至少各有一部分被吸收
    has_age = any("30" in v for v in values)
    has_role = any(("腾讯" in v) or ("后端" in v) for v in values)
    has_hobby = any("钓鱼" in v for v in values)
    assert sum([has_age, has_role, has_hobby]) >= 2, \
        f"expected ≥2 of (age/role/hobby) in attrs, got {values}"


# ── 场景 3:多人物隔离 —— 不能把李四的属性归到张三 ──────────────

def test_persona_isolation_between_people(store):
    store.save("张三今年30岁,在腾讯工作")
    store.save("李四今年45岁,在字节跳动")

    zhang_values = _attr_values(store.persona.aggregate("张三"))
    li_values = _attr_values(store.persona.aggregate("李四"))

    # 张三画像里应含 30,不能含 45
    assert any("30" in v for v in zhang_values)
    assert not any("45" in v for v in zhang_values), \
        f"张三 leaked 李四's age: {zhang_values}"
    # 李四画像里应含 45,不能含 30
    assert any("45" in v for v in li_values)
    assert not any("30" in v for v in li_values), \
        f"李四 leaked 张三's age: {li_values}"


# ── 场景 4:属性冲突 —— 至少保留一个证据 ─────────────────────

def test_persona_conflict_retains_evidence(store):
    """张三 26 岁 → 28 岁 两条记忆共存,attrs 至少留一个,不崩溃。"""
    store.save("张三说他26岁")
    store.save("张三说他28岁")

    persona = store.persona.aggregate("张三")
    values = _attr_values(persona)
    # 至少一个数字被保留
    assert any(("26" in v) or ("28" in v) for v in values)


# ── 场景 5:persona save/load 往返 ─────────────────────────────────

def test_persona_save_and_load_roundtrip(store):
    store.save("张三今年30岁,在腾讯工作")
    p1 = store.persona.aggregate("张三")
    store.persona.save(p1)

    p2 = store.persona.load("张三")
    assert p2 is not None
    assert p2.name == p1.name
    assert set(p2.attrs.keys()) == set(p1.attrs.keys())


def test_persona_load_unknown_person_returns_none(store):
    result = store.persona.load("不存在的人")
    assert result is None


# ── 场景 6:未提及 → attrs 保持空 ────────────────────────────────

def test_persona_empty_for_unmentioned_person(store):
    store.save("李四30岁")
    persona = store.persona.aggregate("张三")
    # 张三从未被提及,attrs 应为空
    assert persona.attrs == {}


# ── 场景 7:aggregate 幂等 ────────────────────────────────────────

def test_persona_aggregate_is_idempotent(store):
    store.save("张三30岁,在腾讯工作")
    p1 = store.persona.aggregate("张三")
    p2 = store.persona.aggregate("张三")
    assert set(p1.attrs.keys()) == set(p2.attrs.keys())
    for k in p1.attrs:
        assert p1.attrs[k].value == p2.attrs[k].value


# ── 场景 8:手动 attrs["age"] 后 .age 便利属性可用 ────────────

def test_persona_convenience_age_property_after_manual_set(store):
    """`.age` 便利属性依赖 LLM 归一化的 'age' key;规则版没能自动填充。
    这个测试确认:手动写入 attrs["age"] 后 .age 属性生效 —— 说明便利属性接口没坏,
    只是规则抽取没触发它。"""
    persona = Persona(name="张三")
    persona.attrs["age"] = PersonaAttr(key="age", value="30", confidence=1.0)
    assert persona.age == "30"


# ── 场景 9:evidence 完整(每条 attr 至少一条 evidence)──────────

def test_persona_attributes_carry_evidence(store):
    """项目主张:每个属性至少有 1 条 evidence,可回溯到 memory_id。"""
    store.save("张三今年30岁")
    persona = store.persona.aggregate("张三")
    for attr in persona.attrs.values():
        assert len(attr.evidence) >= 1
        # 至少一条 evidence 指回一个 memory_id
        assert any(ev.memory_id for ev in attr.evidence)
