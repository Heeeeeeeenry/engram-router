"""Tests for tree-sitter code symbol extraction (L5)."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _ts_available():
    try:
        import tree_sitter
        import tree_sitter_python
        return True
    except ImportError:
        return False


ts_required = pytest.mark.skipif(not _ts_available(), reason="tree-sitter not installed")


class TestHeuristicGate:
    def test_python_def_triggers(self):
        from engram_router.code_symbols import is_likely_code
        assert is_likely_code("def hello():\n    pass") is True
        assert is_likely_code("async def fetch():\n    return 1") is True

    def test_python_class_triggers(self):
        from engram_router.code_symbols import is_likely_code
        assert is_likely_code("class Foo(Base):\n    pass") is True

    def test_python_import_triggers(self):
        from engram_router.code_symbols import is_likely_code
        assert is_likely_code("import os\nimport sys") is True
        assert is_likely_code("from dataclasses import dataclass") is True

    def test_markdown_fence_triggers(self):
        from engram_router.code_symbols import is_likely_code
        assert is_likely_code("```python\ndef f(): pass\n```") is True

    def test_chinese_prose_does_not_trigger(self):
        from engram_router.code_symbols import is_likely_code
        assert is_likely_code("张三今天吃了饺子，说很好吃") is False
        assert is_likely_code("我写了一个def函数来测试") is False  # no parens, not code

    def test_english_prose_does_not_trigger(self):
        from engram_router.code_symbols import is_likely_code
        assert is_likely_code("I love the function you built") is False

    def test_indented_prose_triggers(self):
        from engram_router.code_symbols import is_likely_code
        # 3+ lines, 2+ indented → likely code
        text = "header\n    line one\n    line two"
        assert is_likely_code(text) is True


@ts_required
class TestExtractSymbols:
    def test_function_def(self):
        from engram_router.code_symbols import extract_symbols
        code = "def hello():\n    return 'world'"
        result = extract_symbols(code)
        names = [s["name"] for s in result]
        assert "hello" in names

    def test_class_def(self):
        from engram_router.code_symbols import extract_symbols
        code = "class MyClass:\n    def method(self):\n        pass"
        result = extract_symbols(code)
        names = [s["name"] for s in result]
        assert "MyClass" in names
        assert "method" in names  # nested method

    def test_multiple_functions(self):
        from engram_router.code_symbols import extract_symbols
        code = "def a():\n    pass\n\ndef b():\n    pass"
        result = extract_symbols(code)
        names = {s["name"] for s in result}
        assert names >= {"a", "b"}

    def test_all_entities_have_correct_kind(self):
        from engram_router.code_symbols import extract_symbols
        code = "def f(): pass\nclass C: pass"
        result = extract_symbols(code)
        assert len(result) > 0
        for ent in result:
            assert ent["kind"] == "code_symbol"
            assert "name" in ent
            assert "evidence" in ent

    def test_plain_text_returns_empty(self):
        from engram_router.code_symbols import extract_symbols
        assert extract_symbols("你好世界") == []
        assert extract_symbols("今天天气不错") == []


class TestEnrichEntitiesHook:
    def test_hook_skips_when_env_disabled(self):
        from engram_router.code_symbols import enrich_entities
        os.environ["ENGRAM_SKIP_CODE_SYMBOLS"] = "1"
        try:
            existing = [{"name": "x", "kind": "topic"}]
            result = enrich_entities("def f(): pass", existing)
            assert result is existing
            assert len(result) == 1  # no new entities added
        finally:
            del os.environ["ENGRAM_SKIP_CODE_SYMBOLS"]

    def test_hook_preserves_existing(self):
        from engram_router.code_symbols import enrich_entities
        existing = [{"name": "张三", "kind": "person", "evidence": "张三"}]
        result = enrich_entities("def hello(): pass", existing)
        # existing entity must still be present
        assert any(e["name"] == "张三" for e in result)

    def test_hook_does_not_dedup_across_kinds(self):
        """A code_symbol and a person with the same name can coexist."""
        from engram_router.code_symbols import enrich_entities
        existing = [{"name": "main", "kind": "person", "evidence": "main"}]
        result = enrich_entities("def main(): pass", existing)
        kinds = {e["kind"] for e in result}
        assert "person" in kinds


@ts_required
class TestEndToEndStoreSave:
    """Verify code_symbol entities go through save + recall correctly."""

    def test_save_code_memory_creates_code_symbol_entity(self):
        from engram_router.store.core import MemoryStore

        store = MemoryStore()
        code = "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
        mem_id = store.save(code, namespace="test")

        # Check entities table
        rows = store.conn.execute(
            "SELECT name, kind FROM entities WHERE namespace = ?", ("test",)
        ).fetchall()
        names = {r["name"] for r in rows}
        assert "factorial" in names

    def test_code_symbol_recall_by_name(self):
        from engram_router.store.core import MemoryStore

        store = MemoryStore()
        store.save("import os.path as path_utils", namespace="test")

        # Recall should find the code memory by symbol name
        results = store.recall("path_utils", top_k=3, namespace="test")
        assert len(results) > 0
