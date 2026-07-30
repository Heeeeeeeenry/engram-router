"""Tests for entity canonicalization + alias table."""

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from engram_router.entity_canon import (
    _migrate_alias_table,
    canonicalize,
    register_alias,
    suggest_aliases,
)


@pytest.fixture
def conn():
    """Fresh in-memory SQLite connection with migrated schema."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    _migrate_alias_table(c)
    return c


class TestAliasTable:
    def test_migrate_creates_table(self, conn):
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='entity_aliases'"
        ).fetchall()
        assert len(rows) == 1

    def test_migrate_idempotent(self, conn):
        _migrate_alias_table(conn)  # second call
        _migrate_alias_table(conn)  # third call
        # Should not raise, table remains
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='entity_aliases'"
        ).fetchall()
        assert len(rows) == 1

    def test_register_and_lookup(self, conn):
        assert register_alias(conn, "老张", "张伟") is True
        assert canonicalize(conn, "老张") == "张伟"
        assert canonicalize(conn, "张伟") == "张伟"  # no alias, returns as-is

    def test_register_replaces_existing(self, conn):
        register_alias(conn, "老张", "张伟")
        register_alias(conn, "老张", "张三")  # update
        assert canonicalize(conn, "老张") == "张三"

    def test_canonicalize_unknown_returns_name(self, conn):
        assert canonicalize(conn, "李四") == "李四"


class TestSuggestAliases:
    def test_bigram_overlap_50_percent(self):
        # 王芳华 bigrams = {王芳, 芳华}, 王芳丽 bigrams = {王芳, 芳丽}
        # intersection = {王芳}, union = {王芳, 芳华, 芳丽} → jaccard = 1/3 ~ 0.33
        # For 50% we need higher overlap. 芳芳 vs 芳华: {芳芳} vs {芳华}, no overlap.
        # Test that heavy-overlap names (3-char alias vs 3-char same base) trigger it.
        existing = ["张三丰"]  # bigrams: {张三, 三丰}
        suggestions = suggest_aliases(existing, "张三")  # bigrams: {张三}
        # intersection = {张三}, union = {张三, 三丰} → jaccard = 1/2 = 0.5
        names = [s[0] for s in suggestions]
        assert "张三丰" in names

    def test_no_overlap_for_dissimilar_names(self):
        existing = ["张三"]
        suggestions = suggest_aliases(existing, "李四")
        # Zero bigram overlap
        assert not any(s[0] == "张三" for s in suggestions)

    def test_lao_prefix_matches_surname(self):
        existing = ["张伟"]
        suggestions = suggest_aliases(existing, "老张")
        # 老张 → surname 张 → matches 张伟
        names = [s[0] for s in suggestions]
        assert "张伟" in names

    def test_xiao_prefix_matches_surname(self):
        existing = ["李娜"]
        suggestions = suggest_aliases(existing, "小李")
        names = [s[0] for s in suggestions]
        assert "李娜" in names

    def test_empty_existing(self):
        assert suggest_aliases([], "老张") == []

    def test_self_excluded(self):
        # New name equals existing — no self-alias
        suggestions = suggest_aliases(["张伟"], "张伟")
        assert suggestions == []
