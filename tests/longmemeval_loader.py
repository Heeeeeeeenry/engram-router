"""LongMemEval adapter for engram-router's evaluation harness.

LongMemEval (Wu et al., ICLR 2025 — arXiv:2410.10813) is the first
non-toy corpus this project runs against. Unlike ``semantic_audit.py`` /
``multi_angle_eval.py`` (15-turn hand-written Chinese scenarios), each
LongMemEval item ships:

  - a haystack of real ChatGPT-style multi-session conversations
    (oracle split: 1-6 sessions, ~22 turns average; the full ``_s``/``_m``
    splits go up to 500 sessions and include distractor sessions with NO
    answer-bearing content — that's the part that actually stresses
    retrieval instead of just recall-everything).
  - one natural-language question + a reference answer.
  - a ``question_type`` tag: single-session-user / single-session-assistant
    / single-session-preference / multi-session / temporal-reasoning /
    knowledge-update.

Data source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
(MIT licensed, commercially usable). Files are fetched once and cached under
``tests/data/longmemeval/`` — not committed to the repo (see .gitignore note
in that directory's README companion, added alongside this file).

Key adaptation decisions
-------------------------
- Each **user turn** becomes one ``store.save()`` call — matching how
  engram-router's other scenarios work (assistant turns are conversational
  filler, not facts to remember, and saving them would let long assistant
  replies dominate FTS/vector candidate pools with no signal).
- Answer matching is **substring-based**, same rule as ``eval_v2.py``'s
  ``EvalCase.expected_ids_or_texts``: a record “contains” the answer if the
  answer string (or a token of it) appears in ``raw_text``. LongMemEval's own
  paper uses a GPT-4 judge for free-form correctness — we don't have that
  judge wired up here, so this loader intentionally reports a **stricter,
  lower-bound** P@1/MRR than the paper's headline numbers. That's a known,
  documented gap (see the module docstring in ``eval_v2_longmemeval.py``).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "tests" / "data" / "longmemeval"
HF_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"

_SPLIT_FILES = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}


@dataclass
class LMECase:
    """One LongMemEval question, decoupled from the eval_v2 EvalCase shape
    so this loader has zero import-time dependency on eval_v2 internals."""

    question_id: str
    question: str
    answer: str
    question_type: str
    # Flattened user-turn texts across every haystack session (order preserved).
    memories: list[str] = field(default_factory=list)
    num_sessions: int = 0
    num_turns: int = 0


def _download(split: str, dest: Path) -> None:
    filename = _SPLIT_FILES[split]
    url = f"{HF_BASE}/{filename}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading LongMemEval split %r from %s", split, url)
    tmp = dest.with_suffix(".tmp")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 — fixed HF URL, not user input
    tmp.rename(dest)


def _ensure_split_file(split: str) -> Path:
    if split not in _SPLIT_FILES:
        raise ValueError(f"unknown split {split!r}; choose from {list(_SPLIT_FILES)}")
    dest = DATA_DIR / _SPLIT_FILES[split]
    if not dest.exists():
        _download(split, dest)
    return dest


def load_longmemeval(
    split: str = "oracle",
    limit: int | None = None,
    question_types: set[str] | None = None,
) -> list[LMECase]:
    """Load and flatten a LongMemEval split into ``LMECase`` objects.

    Args:
        split: "oracle" (only answer-bearing sessions, fast, ~500 items),
            "s" (~40 sessions/item incl. distractors), or "m" (~500
            sessions/item — very large, hours to ingest per system).
        limit: cap the number of items (dev speed-up).
        question_types: only keep these question_type values.
    """
    path = _ensure_split_file(split)
    with path.open() as f:
        raw = json.load(f)

    cases: list[LMECase] = []
    for item in raw:
        qt = item.get("question_type", "unknown")
        if question_types is not None and qt not in question_types:
            continue

        memories: list[str] = []
        sessions = item.get("haystack_sessions", [])
        for session in sessions:
            for turn in session:
                if turn.get("role") == "user":
                    content = (turn.get("content") or "").strip()
                    if content:
                        memories.append(content)

        cases.append(LMECase(
            question_id=item["question_id"],
            question=item["question"],
            # 32 items in the oracle split store the answer as an int (e.g. 3).
            # Coerce to string so downstream substring matching in eval_v2's
            # ``_matches`` doesn't blow up on `int in str` type errors.
            answer=str(item.get("answer", "")),
            question_type=qt,
            memories=memories,
            num_sessions=len(sessions),
            num_turns=sum(len(s) for s in sessions),
        ))
        if limit is not None and len(cases) >= limit:
            break

    return cases


def dataset_stats(cases: list[LMECase]) -> dict[str, Any]:
    """Summary stats — used for sanity-checking a freshly loaded split."""
    if not cases:
        return {}
    by_type: dict[str, int] = {}
    for c in cases:
        by_type[c.question_type] = by_type.get(c.question_type, 0) + 1
    turns = [len(c.memories) for c in cases]
    return {
        "total_cases": len(cases),
        "by_question_type": by_type,
        "memories_per_case_min": min(turns),
        "memories_per_case_max": max(turns),
        "memories_per_case_avg": round(sum(turns) / len(turns), 1),
    }


if __name__ == "__main__":
    # Quick manual smoke: `python tests/longmemeval_loader.py`
    logging.basicConfig(level=logging.INFO)
    cs = load_longmemeval(split="oracle")
    print(json.dumps(dataset_stats(cs), indent=2))
