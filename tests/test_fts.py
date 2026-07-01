"""Tests for the FTS5 trigram candidate path and pluggable ranker (f3).

Design note — avoiding the f2-style false green:
  The weighted token/entity scorer already finds most things. To prove FTS5 is
  a REAL increment (not the existing scan sneaking past), these tests assert on
  artefacts that ONLY the FTS path produces:
    - the `memories_fts` virtual table is actually populated on save;
    - a trigram-eligible query is tagged "fts trigram candidate" in
      match_reason (the weighted scan never emits that string);
    - the documented 2-char CJK fallback (键盘 -> 0 trigram hits) still recalls
      via the entity/topic path, i.e. FTS never SUPPRESSES the fallback;
    - a pluggable ranker replaces the base score yet FTS candidate selection
      still narrows the row set independently of the ranker.

These tests are skipped (not failed) on a SQLite build without FTS5/trigram,
since that is an environment capability, not a code regression.
"""

from __future__ import annotations

import sqlite3

import pytest

from engram_router.store import MemoryStore


def _fts_available() -> bool:
    c = sqlite3.connect(":memory:")
    try:
        c.execute("CREATE VIRTUAL TABLE t USING fts5(x, tokenize='trigram')")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        c.close()


pytestmark = pytest.mark.skipif(
    not _fts_available(), reason="SQLite build lacks FTS5/trigram"
)


def test_save_populates_fts_virtual_table(tmp_path):
    """Every saved memory is mirrored into the FTS5 trigram table."""
    store = MemoryStore(path=tmp_path / "fts_pop.db")
    assert store._fts_enabled, "FTS should be enabled when the build supports it"
    mem_id = store.save("张三送我一把 HHKB 键盘。")
    rows = store.conn.execute(
        "SELECT memory_id, content FROM memories_fts WHERE memory_id = ?", (mem_id,)
    ).fetchall()
    assert len(rows) == 1, "save should mirror the memory into memories_fts"
    assert "HHKB" in rows[0]["content"]


def test_trigram_query_is_tagged_as_fts_candidate(tmp_path):
    """An ASCII (≥3 char) query is selected via the FTS trigram index.

    'HHKB' is trigram-eligible; the matched memory must carry the
    'fts trigram candidate' provenance that ONLY the FTS branch writes.
    """
    store = MemoryStore(path=tmp_path / "fts_tag.db")
    store.save("张三送我一把 HHKB 键盘。")
    records = store.recall("HHKB", top_k=5)
    assert records
    top = records[0]
    assert "HHKB" in top.raw_text
    assert "fts trigram candidate" in top.match_reason.lower()


def test_fts_finds_substring_no_whitespace_token(tmp_path):
    """Trigram matches a substring inside a longer run with no word boundary.

    A plain whitespace tokenizer would miss 'HHKB' glued inside an ASCII run;
    the trigram index matches it as a substring. This isolates the FTS path:
    the answer memory has the brand only as an embedded substring.
    """
    store = MemoryStore(path=tmp_path / "fts_substr.db")
    store.save("settings:keyboardHHKBpro=on")  # HHKB embedded, no boundary
    store.save("今天天气很好。")  # unrelated, no HHKB substring
    records = store.recall("HHKB", top_k=5)
    assert records
    joined = " ".join(r.raw_text for r in records)
    assert "keyboardHHKBpro" in joined
    top = records[0]
    assert "fts trigram candidate" in top.match_reason.lower()


def test_two_char_cjk_query_recalls_via_like_fallback(tmp_path):
    """2-char CJK (键盘) now gets LIKE-based candidates (not just full scan).

    The LIKE fallback was added to recover short mixed tokens (B轮, 20亿)
    that FTS5 trigram misses. Short CJK also benefits — recall still works
    and may carry the FTS candidate tag from the LIKE path.
    """
    store = MemoryStore(path=tmp_path / "fts_fallback.db")
    store.save("张三送我一把 HHKB 键盘。")
    records = store.recall("键盘", top_k=5)
    assert records, "2-char CJK query must still recall"
    joined = " ".join(r.raw_text for r in records)
    assert "HHKB" in joined


def test_fts_candidates_like_fallback_for_short_terms(tmp_path):
    """LIKE fallback returns candidates for short terms that FTS5 trigram misses."""
    store = MemoryStore(path=tmp_path / "fts_like.db")
    store.save("张三送我一把 HHKB 键盘。")
    cand = store._fts_candidates("键盘", store._terms("键盘"))
    assert cand is not None and len(cand) >= 1, "LIKE fallback should find candidates"
    cand2 = store._fts_candidates("HHKB", store._terms("HHKB"))
    assert cand2 is not None and len(cand2) >= 1


def test_pluggable_ranker_overrides_base_score_independently_of_fts(tmp_path):
    """A custom ranker replaces the base score; FTS still narrows candidates.

    We install a ranker that scores every row 0 EXCEPT one specific memory,
    proving the ranker is the scoring authority. Separately, the FTS candidate
    set still gets the provenance tag, proving the two concerns are decoupled.
    """
    store = MemoryStore(path=tmp_path / "fts_ranker.db")
    store.save("张三送我一把 HHKB 键盘。")
    store.save("李四提到 HHKB 也不错。")

    # Ranker prefers any memory mentioning 李四, regardless of the query.
    def ranker(query, terms, haystack, _store):
        return 99.0 if "李四" in haystack else 0.0

    store.ranker = ranker
    records = store.recall("HHKB", top_k=5)
    assert records
    assert "李四" in records[0].raw_text, "custom ranker should decide the top result"
    # FTS candidate tagging is independent of the ranker: both HHKB-bearing
    # memories are trigram candidates.
    tagged = [r for r in records if "fts trigram candidate" in r.match_reason.lower()]
    assert len(tagged) >= 2
