"""Entity canonicalization and alias management for EngramRouter.

Chinese entity extraction often produces multiple surface forms for the same
real-world entity: "王芳" / "芳姐" / "老王" / "王大夫" → same person.
Without canonicalization these are 4 separate entity nodes, fragmenting the
knowledge graph and diluting person-isolation signals.

This module provides a lightweight alias table backed by the SQLite schema:

    entity_aliases(alias TEXT PRIMARY KEY, canonical TEXT NOT NULL)

When ``canonicalize(entity_name)`` is called, it looks up the alias table.
If no match is found, the name itself is returned as canonical.  New aliases
can be registered explicitly (e.g., by user correction or LLM reasoning) or
via automatic fuzzy matching.

Design notes
------------
- Schema migration: ``_migrate_alias_table()`` adds the table if missing.
- Fuzzy matching: for now, CJK bigram overlap ≥ 50% triggers an alias
  suggestion (logged but NOT auto-applied — manual verification required).
  Full LLM-based canonicalization is a future upgrade.
- ENGRAM_SKIP_CANONICAL=1 disables at runtime.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

# Common Chinese nickname/kinship patterns for heuristic alias detection.
# 老X / 小X / X姐 / X哥 / X总 / X工 / X老师 / X先生 / X女士 / 阿X
# These produce candidate aliases with the same surname.
_SURNAME_ALIAS_SUFFIXES = (
    "姐", "哥", "总", "工", "老师", "先生", "女士",
)

_ALIAS_PREFIXES = ("老", "小", "阿")


def _migrate_alias_table(conn: sqlite3.Connection) -> None:
    """Ensure the entity_aliases table exists."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity_aliases (
            alias TEXT PRIMARY KEY,
            canonical TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'heuristic',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)


def register_alias(
    conn: sqlite3.Connection,
    alias: str,
    canonical: str,
    source: str = "heuristic",
) -> bool:
    """Register *alias* → *canonical* in the alias table."""
    try:
        conn.execute(
            "INSERT OR REPLACE INTO entity_aliases (alias, canonical, source) VALUES (?, ?, ?)",
            (alias, canonical, source),
        )
        return True
    except Exception as exc:
        logger.debug("register_alias failed: %s", exc)
        return False


def canonicalize(conn: sqlite3.Connection, name: str) -> str:
    """Return the canonical form of *name*, or *name* itself if no alias exists."""
    row = conn.execute(
        "SELECT canonical FROM entity_aliases WHERE alias = ?",
        (name,),
    ).fetchone()
    return row["canonical"] if row else name


def suggest_aliases(
    existing_names: list[str],
    new_name: str,
) -> list[tuple[str, str]]:
    """Suggest alias candidates for *new_name* against *existing_names*.

    Returns a list of ``(existing_name, reason)`` tuples where a candidate
    alias match was detected.  Callers can auto-apply (aggressive) or log
    for manual review (conservative).

    Currently uses:
    1. CJK bigram overlap ≥ 50% (e.g., "王芳" vs "芳姐" share "芳")
    2. Surname + kinship pattern (e.g., "张工" → "张伟" if both share "张")
    """
    candidates: list[tuple[str, str]] = []

    # Extract CJK bigrams from new name
    new_bigrams = {new_name[i:i+2] for i in range(len(new_name)-1)}

    for existing in existing_names:
        if existing == new_name:
            continue
        existing_bigrams = {existing[i:i+2] for i in range(len(existing)-1)}

        # Bigram overlap heuristic
        if new_bigrams and existing_bigrams:
            intersection = new_bigrams & existing_bigrams
            union = new_bigrams | existing_bigrams
            jaccard = len(intersection) / len(union) if union else 0
            if jaccard >= 0.5:
                candidates.append((existing, f"bigram overlap {jaccard:.2f}"))
                continue

        # Surname pattern: 老张 → 张伟 (shared leading char)
        for prefix in _ALIAS_PREFIXES:
            if new_name.startswith(prefix) and len(new_name) == 2:
                surname = new_name[1]
                if existing.startswith(surname) and existing != new_name:
                    candidates.append((existing, f"{prefix}X → surname {surname}"))
                    break

        # Kinship suffix: 芳姐 → 王芳 (shared trailing char)
        for suffix in _SURNAME_ALIAS_SUFFIXES:
            if new_name.endswith(suffix) and len(new_name) > len(suffix):
                base = new_name[:-len(suffix)]
                if base in existing:
                    candidates.append((existing, f"X{suffix} → base {base}"))
                    break

    return candidates


# ── MemoryStore integration helper ───────────────────────────────────────


def canonicalize_entity_name(
    store: Any,  # MemoryStore instance
    name: str,
    kind: str,
) -> str:
    """Canonicalize *name* within *store*'s DB + suggest new aliases.

    Returns the canonical name (may be *name* itself).
    """
    if os.environ.get("ENGRAM_SKIP_CANONICAL") == "1":
        return name

    _migrate_alias_table(store.conn)

    # 1. Direct lookup
    canonical = canonicalize(store.conn, name)

    # 2. If no alias found, suggest candidates against existing entities
    if canonical == name and kind == "person":
        existing = [
            row[0] for row in
            store.conn.execute(
                "SELECT DISTINCT name FROM entities WHERE kind = ? ORDER BY name",
                (kind,),
            ).fetchall()
        ]
        if len(existing) >= 2:  # Need ≥2 entities to compare for aliases
            suggestions = suggest_aliases(existing, name)
            for target, reason in suggestions:
                logger.info(
                    "Entity alias suggestion: %r → %r (%s)",
                    name, target, reason,
                )
                # Conservative: log only, don't auto-apply.
                # To auto-apply, uncomment:
                # register_alias(store.conn, name, target, "heuristic:" + reason)
                # canonical = target
                # break

    return canonical
