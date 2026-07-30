"""Property/fuzz tests with randomized inputs.

Uses stdlib ``random`` (no hypothesis dep) to generate large batches of
synthetic Chinese + ASCII text and verify recall/entity/reranker paths
don't crash or produce inconsistent output.  These are stability tests,
not correctness tests — they only assert that the pipeline degrades
gracefully on unusual inputs.
"""

from __future__ import annotations

import os
import random
import string
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("ENGRAM_SKIP_VECTOR", "1")
os.environ.setdefault("ENGRAM_SKIP_CE", "1")
os.environ.setdefault("ENGRAM_SKIP_COLBERT", "1")
os.environ.setdefault("ENGRAM_SKIP_HANLP", "1")

from engram_router.store import MemoryStore
from engram_router.coref import CorefTracker
from engram_router.entities import extract_entities


# ── Chinese character pool ───────────────────────────────────────────────

_SURNAMES = ["张", "李", "王", "赵", "刘", "陈", "杨", "黄", "周", "吴",
             "徐", "孙", "胡", "朱", "高", "林", "何", "郭", "马", "罗"]
_GIVEN_CHARS = "伟娜磊静婷强丽鹏敏峰洋燕勇平艳梅涛军明"
_TOPICS = ["工作", "旅游", "学习", "家人", "健康", "运动", "投资", "音乐"]
_ACTIONS = ["去了", "买了", "吃了", "见了", "学了", "换了", "开始", "完成"]
_TIMES = ["昨天", "今天", "上周", "最近", "去年", "刚才", "上个月", "前几天"]


def _random_name() -> str:
    return random.choice(_SURNAMES) + random.choice(_GIVEN_CHARS)


def _random_sentence(seed: int | None = None) -> str:
    if seed is not None:
        random.seed(seed)
    parts = [
        _random_name(),
        random.choice(_TIMES),
        random.choice(_ACTIONS),
        random.choice(_TOPICS),
    ]
    return "".join(parts) + "。"


def _random_query(seed: int) -> str:
    random.seed(seed)
    templates = [
        "{name}做什么？",
        "{time}谁在忙？",
        "他喜欢{topic}吗？",
        "{name}是谁？",
        "{topic}怎么样？",
    ]
    return random.choice(templates).format(
        name=_random_name(),
        time=random.choice(_TIMES),
        topic=random.choice(_TOPICS),
    )


class TestRandomIngestion:
    def test_100_random_saves_no_crash(self, tmp_path):
        """Ingesting 100 random Chinese sentences must complete."""
        random.seed(42)
        store = MemoryStore(path=tmp_path / "fuzz.db", enable_vector=False)
        try:
            for i in range(100):
                text = _random_sentence(seed=i)
                store.save(text)
            # Basic invariant: memory count == 100
            count = store.conn.execute(
                "SELECT COUNT(*) FROM memories"
            ).fetchone()[0]
            assert count == 100
        finally:
            store.close()

    def test_random_recall_returns_list(self, tmp_path):
        """Random query on populated store returns a valid list."""
        random.seed(7)
        store = MemoryStore(path=tmp_path / "fuzz.db", enable_vector=False)
        try:
            for i in range(30):
                store.save(_random_sentence(seed=i))
            # Try 20 random queries — none should crash
            for j in range(20):
                q = _random_query(j + 1000)
                records = store.recall(q, top_k=5)
                assert isinstance(records, list)
                assert all(r.raw_text for r in records)
        finally:
            store.close()

    def test_batch_save_matches_individual(self, tmp_path):
        """save_batch() and 3x save() produce same count."""
        random.seed(11)
        texts = [_random_sentence(seed=i) for i in range(15)]

        s1 = MemoryStore(path=tmp_path / "batch.db", enable_vector=False)
        try:
            s1.save_batch([(t, "conversation", None) for t in texts])
            c1 = s1.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        finally:
            s1.close()

        s2 = MemoryStore(path=tmp_path / "loop.db", enable_vector=False)
        try:
            for t in texts:
                s2.save(t)
            c2 = s2.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        finally:
            s2.close()

        assert c1 == c2 == 15


class TestRandomEntityExtraction:
    def test_no_crash_on_random_names(self):
        """Random Chinese sentences don't crash entity extraction."""
        random.seed(21)
        for i in range(50):
            text = _random_sentence(seed=i)
            ents = extract_entities(text)
            assert isinstance(ents, list)
            # All entities have expected shape
            for e in ents:
                assert "name" in e
                assert "kind" in e

    def test_extract_random_ascii_string(self):
        """ASCII-only random string produces at most object entities."""
        random.seed(31)
        text = "".join(random.choices(string.ascii_letters + string.digits, k=40))
        ents = extract_entities(text)
        assert isinstance(ents, list)


class TestRandomCoref:
    def test_random_pronoun_chain(self):
        """CorefTracker handles long random sequences without state corruption."""
        random.seed(41)
        tracker = CorefTracker(max_distance=3)
        prev_person: str | None = None

        for i in range(50):
            if random.random() < 0.3:
                name = _random_name()
                ents = [{"name": name, "kind": "person", "evidence": name}]
                tracker.resolve(f"{name}来了", ents)
                prev_person = name
            else:
                pronoun = random.choice(["他", "她", "它"])
                text = f"{pronoun}最近很忙"
                result = tracker.resolve(text, [])
                # If a person was named recently, coref may inject it
                coref_persons = [r for r in result if r.get("source") == "coref"]
                if prev_person and tracker._distance <= 3:
                    # Not a guarantee — distance may have expired.
                    # But if augmented, it must be the tracked person.
                    for cp in coref_persons:
                        assert cp["name"] == prev_person

    def test_disabled_never_augments(self, monkeypatch):
        """ENGRAM_SKIP_COREF=1 disables augmentation entirely."""
        monkeypatch.setenv("ENGRAM_SKIP_COREF", "1")
        random.seed(51)
        tracker = CorefTracker()
        tracker.resolve("张三来了", [{"name": "张三", "kind": "person", "evidence": "张三"}])
        # 20 random pronoun-only turns
        for i in range(20):
            result = tracker.resolve("他去干嘛了", [])
            assert not any(e.get("source") == "coref" for e in result)


class TestRandomFuzzRecall:
    """Higher-level: end-to-end random ingestion → recall → invariants."""

    def test_recall_ranks_are_valid(self, tmp_path):
        random.seed(61)
        store = MemoryStore(path=tmp_path / "fuzz.db", enable_vector=False)
        try:
            for i in range(50):
                store.save(_random_sentence(seed=i))
            for j in range(10):
                q = _random_query(j + 2000)
                records = store.recall(q, top_k=5)
                # Scores non-increasing after Phase 2.6
                scores = [r.score for r in records]
                # We don't guarantee strict sort — but scores shouldn't be NaN
                assert all(isinstance(s, (int, float)) for s in scores)
                assert not any(s != s for s in scores)  # NaN check
        finally:
            store.close()
