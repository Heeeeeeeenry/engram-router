"""Lightweight pronoun coreference resolution for Chinese text.

Instead of loading a heavy neural model (fastcoref requires PyTorch + 500MB+),
this module uses a simple but effective heuristic:

1. **Last-named-person tracking**: when processing a sequence of memories in
   order, remember the most recent person entity mentioned.  Pronoun-starting
   texts ("他/她/它/他们/她们") inherit that person.

2. **Distance gating**: only resolves pronouns within ``max_distance`` texts
   of the antecedent (default 5).  Beyond that, the text is likely about a
   different topic.

3. **Rule-based CJK pronoun detection**: 他, 她, 它, 他们, 她们, 它们 are
   recognized as pronouns.  When a memory has NO explicit person entity and
   starts with a pronoun, the tracked person entity is injected.

This directly addresses the 2 isolation failures where pronoun-only memories
("她特别喜欢猫" / "他送我的东西") lacked person entities and escaped the
person-isolation filter.

Usage::

    from engram_router.coref import CorefTracker
    tracker = CorefTracker()
    for text in memories:
        entities = extract_entities(text)
        entities = tracker.resolve(text, entities)
        # entities now has person entity backfilled for pronoun texts

Design notes
------------
- ENGRAM_SKIP_COREF=1 disables at runtime.
- NOT a full NLP coref system — intentionally simple, verifiable, fast.
  For production-grade coref, fastcoref or a HanLP coref model can be
  swapped in later via the same interface.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# Chinese third-person pronouns that start a sentence (or clause).
_PRONOUN_PATTERN = re.compile(r"^([他她它们])")


class CorefTracker:
    """Tracks the most recent named person for pronoun resolution."""

    def __init__(self, max_distance: int = 5):
        self._max_distance = max_distance
        self._last_person: str | None = None
        self._distance: int = 0
        self._enabled = os.environ.get("ENGRAM_SKIP_COREF") != "1"

    def resolve(
        self,
        text: str,
        entities: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Resolve pronoun references in *text* and update tracker state.

        Returns *entities* unchanged if coref is disabled, or augmented
        with back-filled person entities for pronoun-starting texts.
        """
        if not self._enabled:
            return entities

        # Check if this text explicitly names a person
        persons = [e for e in entities if e.get("kind") == "person"]
        has_explicit_person = any(
            p["name"] not in ("他", "她", "它", "他们", "她们", "它们", "他之", "她之")
            for p in persons
        )

        if has_explicit_person:
            # Update the tracker with the last named person
            last = persons[-1]
            self._last_person = last["name"]
            self._distance = 0
            return entities

        # Increment distance since last named person
        self._distance += 1

        # If we have a tracked person and the text starts with a pronoun
        if (
            self._last_person is not None
            and self._distance <= self._max_distance
            and _starts_with_pronoun(text)
        ):
            # Inject the tracked person entity
            entities = list(entities)  # don't mutate input
            entities.append({
                "name": self._last_person,
                "kind": "person",
                "evidence": f"coref → {self._last_person}",
                "source": "coref",
            })
            logger.debug(
                "Coref: resolved pronoun in %r → %s (distance=%d)",
                text[:40], self._last_person, self._distance,
            )

        return entities

    def reset(self) -> None:
        """Reset tracker state (e.g., between independent scenarios)."""
        self._last_person = None
        self._distance = 0


def _starts_with_pronoun(text: str) -> bool:
    """Check if text begins with a third-person Chinese pronoun."""
    m = _PRONOUN_PATTERN.match(text.strip())
    return m is not None


# ── High-level integration helper ────────────────────────────────────────


def resolve_pronouns_in_context(
    memories: list[str],
    entity_extractor: Any = None,
) -> list[list[dict[str, Any]]]:
    """Process a sequence of memories through coreference resolution.

    Args:
        memories: List of raw text memories in ingestion order.
        entity_extractor: ``extract_entities`` callable. If None, imports
            ``engram_router.entities.extract_entities``.

    Returns:
        List of entity lists, one per memory, with pronoun backfills applied.
    """
    if entity_extractor is None:
        from engram_router.entities import extract_entities as _ee
        entity_extractor = _ee

    tracker = CorefTracker()
    result: list[list[dict[str, Any]]] = []
    for text in memories:
        entities = entity_extractor(text)
        entities = tracker.resolve(text, entities)
        result.append(entities)
    return result
