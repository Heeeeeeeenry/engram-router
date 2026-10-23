"""Tests for consolidate_observation / reflect (借鉴 Hindsight 的记忆分层)."""

from __future__ import annotations

from engram_router.consolidate import consolidate_observation, extract_lesson, reflect
from engram_router.store import MemoryStore


def _store(tmp_path, name: str = "c.db") -> MemoryStore:
    return MemoryStore(path=tmp_path / name, enable_vector=False)


# ── opt-in gate ────────────────────────────────────────────────────────────


def test_consolidate_skipped_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("ENGRAM_CONSOLIDATE_LLM", raising=False)
    s = _store(tmp_path)
    s.save("张三养了一只猫")
    s.save("张三喜欢喝咖啡")
    assert consolidate_observation(s, "张三") is None


def test_reflect_skipped_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("ENGRAM_CONSOLIDATE_LLM", raising=False)
    s = _store(tmp_path)
    s.save("张三养了一只猫")
    assert reflect(s, "张三喜欢什么") is None


# ── consolidate needs ≥2 experiences ───────────────────────────────────────


def test_consolidate_needs_two_experiences(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGRAM_CONSOLIDATE_LLM", "1")
    s = _store(tmp_path)
    s.save("张三养了一只猫")
    # 只有 1 条 experience，sources < 2，在 LLM 调用前就返回 None
    assert consolidate_observation(s, "张三", top_k=5) is None


def test_consolidate_skips_observation_records(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGRAM_CONSOLIDATE_LLM", "1")
    s = _store(tmp_path)
    # 只存 observation（非 experience），sources 过滤后 < 2
    s.save(
        "张三喜欢猫",
        source="observation",
        metadata={"kind": "observation", "evidence_refs": ["mem_1"]},
    )
    assert consolidate_observation(s, "张三", top_k=5) is None


# ── full flow with a mocked LLM ────────────────────────────────────────────


class _FakeLLM:
    available = True

    def chat(self, messages, **kwargs):
        # 返回一条固定的观察文本
        return "张三是一个喜欢猫和咖啡的人"


def test_consolidate_stores_observation_with_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGRAM_CONSOLIDATE_LLM", "1")
    monkeypatch.setattr(
        "engram_router.llm_extractor.LLMClient", lambda *a, **k: _FakeLLM()
    )
    s = _store(tmp_path)
    s.save("张三养了一只猫")
    s.save("张三每天喝咖啡")

    mid = consolidate_observation(s, "张三的喜好", top_k=5)

    assert mid is not None
    row = s.conn.execute(
        "SELECT source, metadata FROM memories WHERE id = ?", (mid,)
    ).fetchone()
    assert row["source"] == "observation"

    import json

    meta = json.loads(row["metadata"])
    assert meta["kind"] == "observation"
    assert len(meta["evidence_refs"]) >= 2  # 指向源记忆


def test_consolidate_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGRAM_CONSOLIDATE_LLM", "1")
    monkeypatch.setattr(
        "engram_router.llm_extractor.LLMClient", lambda *a, **k: _FakeLLM()
    )
    s = _store(tmp_path)
    s.save("张三养了一只猫")
    s.save("张三每天喝咖啡")

    mid1 = consolidate_observation(s, "张三的喜好", top_k=5)
    assert mid1 is not None

    # 第二次整合同一批 experience，应该跳过（幂等，不产生重复 observation）
    mid2 = consolidate_observation(s, "张三的喜好", top_k=5)
    assert mid2 is None


def test_reflect_synthesizes_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGRAM_CONSOLIDATE_LLM", "1")
    monkeypatch.setattr(
        "engram_router.llm_extractor.LLMClient", lambda *a, **k: _FakeLLM()
    )
    s = _store(tmp_path)
    s.save("张三养了一只猫")
    s.save("张三每天喝咖啡")

    answer = reflect(s, "张三喜欢什么", top_k=5)
    assert answer == "张三是一个喜欢猫和咖啡的人"


# ── extract_lesson (ECC 持续学习 / Hindsight mental models) ────────────────


class _LessonLLM:
    available = True

    def chat(self, messages, **kwargs):
        return "给张三买礼物别买皮沙发，他的猫会抓坏"


class _NoneLLM:
    available = True

    def chat(self, messages, **kwargs):
        return "NONE"


def test_extract_lesson_skipped_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("ENGRAM_CONSOLIDATE_LLM", raising=False)
    s = _store(tmp_path)
    s.save("张三的猫抓坏了沙发")
    assert extract_lesson(s, "张三的猫") is None


def test_extract_lesson_stores_lesson(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGRAM_CONSOLIDATE_LLM", "1")
    monkeypatch.setattr(
        "engram_router.llm_extractor.LLMClient", lambda *a, **k: _LessonLLM()
    )
    s = _store(tmp_path)
    s.save("张三的猫抓坏了皮沙发")

    mid = extract_lesson(s, "张三的猫抓沙发", top_k=5)
    assert mid is not None

    row = s.conn.execute(
        "SELECT source, metadata FROM memories WHERE id = ?", (mid,)
    ).fetchone()
    assert row["source"] == "lesson"

    import json

    meta = json.loads(row["metadata"])
    assert meta["kind"] == "lesson"
    assert len(meta["evidence_refs"]) >= 1


def test_extract_lesson_returns_none_when_no_lesson(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGRAM_CONSOLIDATE_LLM", "1")
    monkeypatch.setattr(
        "engram_router.llm_extractor.LLMClient", lambda *a, **k: _NoneLLM()
    )
    s = _store(tmp_path)
    s.save("张三养了一只猫")

    assert extract_lesson(s, "张三的猫", top_k=5) is None


def test_extract_lesson_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGRAM_CONSOLIDATE_LLM", "1")
    monkeypatch.setattr(
        "engram_router.llm_extractor.LLMClient", lambda *a, **k: _LessonLLM()
    )
    s = _store(tmp_path)
    s.save("张三的猫抓坏了皮沙发")

    mid1 = extract_lesson(s, "张三的猫抓沙发", top_k=5)
    assert mid1 is not None

    # 第二次对同一批 experience 提取，应该跳过（幂等，不产生重复 lesson）
    mid2 = extract_lesson(s, "张三的猫抓沙发", top_k=5)
    assert mid2 is None
