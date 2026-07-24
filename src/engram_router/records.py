"""Memory record data model and row-level helpers.

The module holds the MemoryRecord dataclass plus the pure functions that
convert database rows into records, clean/summarise/truncate text, and
serialise metadata.  These were extracted from store.py to keep that module
focused on the MemoryStore class.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import sqlite3
from dataclasses import asdict, dataclass
from typing import Any, cast

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryRecord:
    id: str
    raw_text: str
    summary: str
    confidence: float = 1.0
    metadata: dict[str, Any] | None = None
    evidence_refs: list[str] | None = None
    score: float = 0.0
    match_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Metadata serialisation (was MemoryStore._serialize_metadata / _parse_metadata)
# ---------------------------------------------------------------------------

def serialize_metadata(metadata: dict[str, Any] | None) -> str:
    return json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)


def parse_metadata(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            logger.warning("Could not parse memory metadata; returning empty metadata")
            return {}
    return parsed if isinstance(parsed, dict) else {}


# ---------------------------------------------------------------------------
# Row → MemoryRecord (was MemoryStore._row_to_record)
# ---------------------------------------------------------------------------

def row_to_record(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    score: float,
    match_reason: str,
    evidence_refs: list[str] | None = None,
    raw_refs: list[str] | None = None,
) -> MemoryRecord:
    """Convert a database row to a MemoryRecord.

    When ``evidence_refs`` / ``raw_refs`` are provided (pre-fetched by
    ``_build_recall_response``), the N+1 queries are skipped.
    """
    if evidence_refs is None:
        evidence_refs = [
            r["id"]
            for r in conn.execute(
                "SELECT id FROM evidence WHERE memory_id = ? ORDER BY id",
                (row["id"],),
            ).fetchall()
        ]
    if raw_refs is None:
        raw_refs = [
            r["raw_log_id"]
            for r in conn.execute(
                "SELECT raw_log_id FROM distilled_memories WHERE memory_id = ? ORDER BY id",
                (row["id"],),
            ).fetchall()
        ]
    metadata = parse_metadata(row["metadata"])
    metadata.update({"source": row["source"], "created_at": row["created_at"]})
    # Phase 3: surface access/forgetting columns for the forgetting engine.
    if row["accessed_at"] is not None:
        metadata["accessed_at"] = row["accessed_at"]
    metadata["access_count"] = int(row["access_count"]) if row["access_count"] is not None else 0
    metadata["forgotten"] = bool(row["forgotten"])
    return MemoryRecord(
        id=row["id"],
        raw_text=row["raw_text"],
        summary=row["summary"],
        confidence=float(row["confidence"]),
        metadata=metadata,
        evidence_refs=evidence_refs + raw_refs,
        score=score,
        match_reason=match_reason,
    )


# ---------------------------------------------------------------------------
# Filler-word cleaning (was MemoryStore.FILLER_WORDS / FILLER_CHARS / _clean_sentence)
# ---------------------------------------------------------------------------

FILLER_WORDS: list[str] = [
    "就是说", "就是说呢", "就是说啊", "就是说呀",
    "那个什么", "那个啥", "那什么", "这个那个",
    "对吧", "对不对", "是不是", "是吧", "你懂吧", "你知道吧",
    "怎么说呢", "怎么说",
    "那个", "这个", "内个", "然后呢", "反正",
]

FILLER_CHARS: frozenset[str] = frozenset(
    "嗯呃啊嘛呢哈哦噢哎呀哇嘞咧呐呗嘿呵嘻嗷咳呸哟"
)


def clean_sentence(text: str) -> str:
    """Remove Chinese filler words / discourse markers.

    Longer phrases are removed first so partial matches don't leave
    orphan fragments behind.  Single-char fillers are stripped from
    leading / trailing positions and between punctuation boundaries.
    """
    result = text
    for filler in sorted(FILLER_WORDS, key=len, reverse=True):
        result = result.replace(filler, "")

    # Strip leading/trailing filler chars (they are noise at edges).
    while result and result[0] in FILLER_CHARS:
        result = result[1:]
    while result and result[-1] in FILLER_CHARS:
        result = result[:-1]

    # Remove filler chars that sit between whitespace / punctuation.
    # We rebuild character-by-character to avoid regex engine issues
    # (Python's re forbids variable-width look-behind).
    boundary: set[str] = {" ", "\t", "，", ",", "。", ".", "！",
                           "!", "？", "?", "\n"}
    chars: list[str] = []
    n = len(result)
    for i, ch in enumerate(result):
        if ch in FILLER_CHARS:
            left_ok = i == 0 or result[i - 1] in boundary
            right_ok = i == n - 1 or result[i + 1] in boundary
            if left_ok and right_ok:
                continue  # skip isolated filler
        chars.append(ch)
    result = "".join(chars)

    # Collapse repeated punctuation / whitespace created by removal.
    result = re.sub(r"[，,]{2,}", "，", result)
    result = re.sub(r"[。.!！?？]{2,}", "。", result)
    result = re.sub(r"\s{2,}", " ", result)
    return result.strip()


# ---------------------------------------------------------------------------
# Truncate (was MemoryStore._truncate_cjk)
# ---------------------------------------------------------------------------

def truncate_cjk(text: str, max_chars: int = 120) -> str:
    """Truncate *text* to at most *max_chars*, never cutting inside a
    CJK code-point.

    Python 3 ``str`` slicing already operates on code-points so CJK
    characters are inherently safe.  This method strips trailing
    whitespace / punctuation so the result reads cleanly.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


# ---------------------------------------------------------------------------
# Summarise (was MemoryStore._summarize)
# ---------------------------------------------------------------------------

def summarize(text: str) -> str:
    """Distil the first sentence into a lossless-leaning summary.

    Strategy:
    1. Isolate the first sentence (best-effort, CJK + ASCII end marks).
    2. Remove filler / discourse-marker noise.
    3. Truncate to ≤ 120 characters on a CJK-safe boundary.

    This keeps entities, numbers and proper nouns intact — pure
    character-count truncation would chop them, violating the
    project's anti-lossy-summary philosophy.
    """
    m = re.search(r"[。！？!?\n]", text)
    first = text[: m.start() + 1] if m else text

    cleaned = clean_sentence(first)
    if not cleaned.strip():
        cleaned = first  # fallback: cleaning stripped everything

    return truncate_cjk(cleaned, 120)
