"""Periodic reflection job (L3).

Reads recent memories and produces a structured summary of "directions
the user currently cares about" — these operate as soft retrieval priors,
guiding recall toward recent topics without artificial penalty on old ones.

Design:
  1. Fetch the N most recent memories from a namespace.
  2. Ask the LLM to summarise them into ≤5 "current focus areas".
  3. Store the result as metadata on a special `reflection` memory row
     so downstream recall can optionally boost matches to focus areas.

The LLM call is opt-in (ENGRAM_REFLECTION_LLM=1) and never blocks — failure
just logs and returns an empty result.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

_REFLECTION_SYSTEM_PROMPT = """\
You are a memory reflection analyst.  Given the user's most recent conversation
memories, identify 3-5 broad themes or directions the user is currently
thinking about, working on, or dealing with.  Output ONLY a JSON array of
short (≤10 word) focus descriptions.

Rules:
1. Focus on **direction** (goal, concern, project), not specific facts.
2. Each entry ≤10 words, written as a noun phrase.
3. If the memories don't suggest clear themes, return an empty array.
4. No markdown fences, no explanation — JSON array only.

Example:
  Input: memories about switching from React to Vue, a new hire onboarding,
         and deployment pipeline issues.
  Output: ["Frontend framework migration", "Team growth and onboarding",
           "CI/CD pipeline reliability"]
"""


def run_reflection(
    store: Any,
    namespace: str = "default",
    num_recent: int = 50,
) -> list[str]:
    """Run reflection on recent memories and return focus areas.

    Args:
        store: MemoryStore instance.
        namespace: namespace to reflect on.
        num_recent: how many recent memories to read.

    Returns:
        List of focus area strings, or empty list on failure.
    """
    if os.environ.get("ENGRAM_REFLECTION_LLM") not in ("1", "true", "yes"):
        logger.debug("Reflection skipped: ENGRAM_REFLECTION_LLM not set")
        return []

    # Fetch recent memories
    rows = store.conn.execute(
        """SELECT id, raw_text, summary FROM memories
           WHERE namespace = ? AND forgotten = 0
           ORDER BY created_at DESC LIMIT ?""",
        (namespace, num_recent),
    ).fetchall()

    if not rows:
        return []

    # Build a compact summary for the LLM
    lines: list[str] = []
    for r in rows:
        text = r["summary"] if r["summary"] and len(r["summary"]) > 10 else r["raw_text"]
        lines.append(text[:200])

    memories_text = "\n".join(f"- {line}" for line in lines)

    # Call LLM
    try:
        from .llm_extractor import LLMClient
        from .prompt_guard import wrap_user_content as _wrap

        client = LLMClient()
        if not client.available:
            return []

        t0 = time.perf_counter()
        raw = client.chat(
            [
                {"role": "system", "content": _REFLECTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Recent memories:\n{_wrap(memories_text)}"},
            ],
            temperature=0.3,
            max_tokens=256,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        # Parse JSON array from LLM response
        import re
        cleaned = raw.strip()
        cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```", "", cleaned)

        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                focus_areas = [str(x).strip() for x in data if str(x).strip()]
                logger.info(
                    "Reflection: %d focus areas from %d memories in %.0fms",
                    len(focus_areas), len(rows), elapsed,
                )
                return focus_areas[:5]
        except json.JSONDecodeError:
            pass

    except Exception as exc:
        logger.debug("Reflection LLM call failed: %s", exc)

    return []


def store_reflection(
    store: Any,
    namespace: str = "default",
    num_recent: int = 50,
) -> str | None:
    """Run reflection and store the result as a memory.

    The reflection memory has source='reflection' and appears in recall()
    results when queried, but is excluded by default from standard recall
    (only surfaced when explicitly asked about "what am I working on").

    Returns the memory_id of the stored reflection, or None if skipped.
    """
    focus_areas = run_reflection(store, namespace=namespace, num_recent=num_recent)
    if not focus_areas:
        return None

    # Store as a special memory row
    text = "当前关注方向:\n" + "\n".join(f"- {f}" for f in focus_areas)
    try:
        memory_id = store.save(
            text,
            source="reflection",
            metadata={
                "reflection": True,
                "focus_areas": focus_areas,
                "reflected_at": _now_iso(),
            },
            namespace=namespace,
        )
        return memory_id
    except Exception as exc:
        logger.debug("store_reflection failed: %s", exc)
        return None


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
