"""Tests for CorefTracker pronoun coreference resolution."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from engram_router.coref import (
    CorefTracker,
    _starts_with_pronoun,
    resolve_pronouns_in_context,
)


# ── Helper fixtures ───────────────────────────────────────────────────────

def _person(name: str) -> dict:
    return {"name": name, "kind": "person", "evidence": name}


class TestPronounDetection:
    def test_starts_with_ta_male(self):
        assert _starts_with_pronoun("他之前在华为做过") is True

    def test_starts_with_ta_female(self):
        assert _starts_with_pronoun("她特别喜欢猫") is True

    def test_starts_with_ta_neutral(self):
        assert _starts_with_pronoun("它是一只小狗") is True

    def test_starts_with_plural(self):
        assert _starts_with_pronoun("们几个来了") is True  # 们 is a valid CJK pronoun char

    def test_no_pronoun_at_start(self):
        assert _starts_with_pronoun("老张是我们部门的") is False

    def test_pronoun_in_middle_not_matched(self):
        assert _starts_with_pronoun("这是他的书") is False

    def test_empty_text(self):
        assert _starts_with_pronoun("") is False

    def test_leading_whitespace_stripped(self):
        assert _starts_with_pronoun("   她说过") is True


class TestCorefTrackerBasic:
    def test_explicit_person_updates_tracker(self):
        tracker = CorefTracker()
        entities = [_person("张三")]
        result = tracker.resolve("张三是我的同事", entities)
        assert tracker._last_person == "张三"
        assert result == entities  # no augmentation needed

    def test_pronoun_resolves_to_last_person(self):
        tracker = CorefTracker()
        tracker.resolve("张三是我的同事", [_person("张三")])
        result = tracker.resolve("他之前在华为做过", [])
        persons = [e for e in result if e.get("kind") == "person"]
        assert any(p["name"] == "张三" for p in persons)

    def test_distance_gate_no_augmentation_beyond_max(self):
        tracker = CorefTracker(max_distance=2)
        tracker.resolve("张三是同事", [_person("张三")])
        tracker.resolve("我今天吃了川菜", [])  # distance 1
        tracker.resolve("天气真好", [])  # distance 2
        tracker.resolve("这是重要的日子", [])  # distance 3 — exceeds max
        result = tracker.resolve("他很喜欢音乐", [])  # distance 4
        persons = [e for e in result if e.get("kind") == "person"]
        # Beyond max distance, no augmentation
        assert not any(p.get("source") == "coref" for p in persons)

    def test_reset_clears_state(self):
        tracker = CorefTracker()
        tracker.resolve("张三是同事", [_person("张三")])
        tracker.reset()
        assert tracker._last_person is None
        assert tracker._distance == 0

    def test_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("ENGRAM_SKIP_COREF", "1")
        tracker = CorefTracker()
        # Even with pronoun, no augmentation when disabled
        tracker.resolve("张三是同事", [_person("张三")])
        result = tracker.resolve("他很好", [])
        assert not any(e.get("source") == "coref" for e in result)

    def test_multi_speaker_switching(self):
        """When a new person is named, tracker follows them."""
        tracker = CorefTracker()
        tracker.resolve("张三来了", [_person("张三")])
        result_a = tracker.resolve("他喜欢咖啡", [])
        assert any(p["name"] == "张三" for p in result_a if p.get("kind") == "person")

        tracker.resolve("李四也来了", [_person("李四")])
        result_b = tracker.resolve("他喜欢茶", [])
        assert any(p["name"] == "李四" for p in result_b if p.get("kind") == "person")

    def test_no_augment_when_explicit_person(self):
        """Text with explicit person doesn't need coref backfill."""
        tracker = CorefTracker()
        tracker.resolve("张三来了", [_person("张三")])
        result = tracker.resolve("他找到了李四", [_person("李四")])
        # Only the explicit person, no coref-source added
        coref_persons = [e for e in result if e.get("source") == "coref"]
        assert len(coref_persons) == 0


class TestResolvePronounsInContext:
    def test_full_sequence_processing(self):
        memories = [
            "张三是我的同事",
            "他今年30岁",
            "李四也是同事",
            "她非常热情",  # she = ... need female — will resolve to 李四 (last named)
        ]
        results = resolve_pronouns_in_context(memories)
        assert len(results) == 4
        # Result[1] should have coref-injected 张三
        r1_persons = [e for e in results[1] if e.get("source") == "coref"]
        assert len(r1_persons) == 1
        assert r1_persons[0]["name"] == "张三"
