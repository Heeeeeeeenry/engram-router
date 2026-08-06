"""Code symbol extraction via tree-sitter (L5)."""

from typing import Any

import logging
import os
import sys

logger = logging.getLogger(__name__)

# tree-sitter is an optional dependency — import fails gracefully if absent.
_TS_AVAILABLE = False
_try_code_block_heuristic = None

def _import_tree_sitter():
    global _TS_AVAILABLE, _try_code_block_heuristic
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser

        PY_LANG = Language(tspython.language())
        _parser = Parser(PY_LANG)
        del Language, Parser, tspython  # keep internals hidden
        _TS_AVAILABLE = True
        return _parser

    except ImportError:
        logger.debug("tree-sitter/tree-sitter-python not available — code symbol extraction disabled")
        _TS_AVAILABLE = False
        return None

# -- Heuristic gate ------------------------------------------------------------

def is_likely_code(text: str) -> bool:
    """Return True when *text* looks like it contains code, not prose.

    The gate is intentionally conservative: it only fires on strong signals
    (def / class / import block, markdown code fences, or substantial
    indentation), avoiding false positives on Chinese text that happens to
    contain English words.
    """
    # Markdown code fence
    if "```python" in text or "```py" in text:
        return True
    # Python keyword starters
    import re
    if re.search(r'\bdef\s+\w{2,}\s*\(', text):
        return True
    if re.search(r'\bclass\s+\w{2,}\s*[:\(]', text):
        return True
    if re.search(r'\bimport\s+\w{2,}', text):
        return True
    if re.search(r'\bfrom\s+\w{2,}\s+import', text):
        return True
    # Multi-line with substantial indentation (Python convention)
    lines = text.split('\n')
    if len(lines) >= 3:
        indented = [L for L in lines if L.startswith('    ') or L.startswith('\t')]
        if len(indented) >= 2:
            return True
    return False

# -- Extraction ----------------------------------------------------------------

def extract_symbols(text: str) -> list[dict[str, Any]]:
    """Parse code symbols (function / class definitions) from *text*.

    Uses tree-sitter for Python (only language currently supported).
    Returns entities in the ``engram_router`` entity dict contract:
    ``{"name": ..., "kind": "code_symbol", "evidence": ...}``.

    Gracefully returns an empty list when tree-sitter is not installed
    or *text* contains no code.
    """
    if not _TS_AVAILABLE:
        return []

    if not is_likely_code(text):
        return []

    symbols: list[dict[str, Any]] = []

    # Module-level function / class defs, plus nested defs inside classes
    # All treesitter-query-based extraction happens here.
    try:
        _parser = _import_tree_sitter()
        if _parser is None:
            return []

        # Use tree-sitter Node API directly (avoid requiring tree-sitter Query).
        # Walk the tree: for function_definition / class_definition nodes,
        # extract the name + accompanying line.
        tree = _parser.parse(bytes(text, "utf-8"))
        root_node = tree.root_node

        def _walk(node):
            if node.type == "function_definition":
                _extract_func(node, text, symbols)
                # Also recurse into body for nested functions (methods)
            elif node.type == "class_definition":
                _extract_class(node, text, symbols)
            for child in node.children:
                _walk(child)

        _walk(root_node)
    except Exception:
        logger.debug("code symbol extraction failed", exc_info=True)
        return []

    return symbols


def _extract_func(node, text: str, symbols: list[dict[str, Any]]) -> None:
    """Extract a function_definition node's name + first-line evidence."""
    for child in node.children:
        if child.type == "identifier":
            name = child.text.decode("utf-8")
            # Find the first line of the function definition as evidence
            start_line = node.start_point[0]
            evidence = text.split("\n")[start_line].strip() if text else name
            symbols.append({
                "name": name,
                "kind": "code_symbol",
                "evidence": evidence,
            })
            return


def _extract_class(node, text: str, symbols: list[dict[str, Any]]) -> None:
    """Extract a class_definition node's name + first-line evidence."""
    for child in node.children:
        if child.type == "identifier":
            name = child.text.decode("utf-8")
            start_line = node.start_point[0]
            evidence = text.split("\n")[start_line].strip() if text else name
            symbols.append({
                "name": name,
                "kind": "code_symbol",
                "evidence": evidence,
            })
            return


# -- Entities pipeline hook ------------------------------------------------------

def enrich_entities(
    text: str,
    existing_entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Hook for entities.py: append code symbols to entity list.

    Follows the same append-not-overwrite contract as
    ``hanlp_ner.enrich_entities()``.  Use it in extract_entities() after
    all other extractors have run.

    Usage::

        from .code_symbols import enrich_entities as _enrich_code
        entities = _enrich_code(text, entities)
    """
    if os.environ.get("ENGRAM_SKIP_CODE_SYMBOLS") == "1":
        return existing_entities

    symbols = extract_symbols(text)
    if not symbols:
        return existing_entities

    # Dedup against existing: don't add a symbol if (name, kind) already
    # exists (unlikely for code_symbol in normal text, but safe).
    existing_keys = {(e["name"], e["kind"]) for e in existing_entities}
    for sym in symbols:
        key = (sym["name"], sym["kind"])
        if key not in existing_keys:
            existing_entities.append(sym)

    return existing_entities
