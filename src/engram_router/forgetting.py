"""Automatic forgetting and decay module for EngramRouter.

Implements the Ebbinghaus forgetting curve (simplified) with access boosts,
correction immunity, and salience-based protection against forgetting.

Design principle: never hard-delete — only down-weight via the ``forgotten``
flag so the evidence chain remains auditable.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

from .store import MemoryStore, MemoryRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable constants (kept as module-level for easy configuration)
# ---------------------------------------------------------------------------

# Ebbinghaus base decay: 50% reduction per DECAY_PERIOD_DAYS
DECAY_HALF_LIFE_DAYS: float = 7.0
DECAY_FRACTION: float = 0.5  # fraction remaining after one half-life

# Access boost: each recall hit adds this fraction to the current score
ACCESS_BOOST_FRACTION: float = 0.30

# Decay threshold: memories below this score are candidates for forgetting
FORGET_THRESHOLD: float = 0.05

# Consolidation similarity threshold (SequenceMatcher ratio)
CONSOLIDATE_SIMILARITY_THRESHOLD: float = 0.80

# Salience classes that are immune to forgetting
PROTECTED_SALIENCE: frozenset[str] = frozenset({"constraint", "decision"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class ForgettingConfig:
    """Centralised configuration for the forgetting engine.

    All fields have sensible defaults; override any at construction time.
    """

    decay_half_life_days: float = DECAY_HALF_LIFE_DAYS
    decay_fraction: float = DECAY_FRACTION
    access_boost_fraction: float = ACCESS_BOOST_FRACTION
    forget_threshold: float = FORGET_THRESHOLD
    consolidate_similarity_threshold: float = CONSOLIDATE_SIMILARITY_THRESHOLD
    protected_salience: frozenset[str] = PROTECTED_SALIENCE


class ForgettingEngine:
    """Drive automatic memory decay, forgetting, and consolidation.

    Designed to be attached to a ``MemoryStore`` instance.  It reads (and
    potentially writes) the ``memories`` table columns added by the Phase-3
    schema migration:

    * ``access_count``  – INT, incremented on every ``recall()`` hit
    * ``accessed_at``   – TEXT, ISO-8601 timestamp of last recall hit
    * ``forgotten``     – INT, 0=active, 1=forgotten (down-weighted, not deleted)

    Usage::

        store = MemoryStore("data.db")
        engine = ForgettingEngine(store)
        engine.forget("mem_3")               # mark forgotten
        stale = engine.should_forget(row)    # check decay status
        stats = engine.consolidate()         # merge near-duplicates
    """

    def __init__(
        self,
        store: MemoryStore,
        config: ForgettingConfig | None = None,
    ) -> None:
        self.store = store
        self.config = config or ForgettingConfig()

    # -- decay ----------------------------------------------------------------

    def decay_score(self, memory: MemoryRecord) -> float:
        """Return the time-decayed score for a memory record.

        Implements a simplified Ebbinghaus forgetting curve:

            score = confidence * decay^(days_since_half_life_ago / half_life_days)

        where ``decay`` defaults to 0.5 and ``half_life_days`` to 7 (so every
        7 days the retained fraction halves).

        Each prior access adds a boost::

            score = max(0, 1 - exp(-boosted_decay))
            boosted_decay = time_decay * (1 + access_count * access_boost)

        Returns a float in [0, 1].
        """
        conf = float(memory.confidence)
        # --- time decay ---
        days = self._days_since_accessed(memory)
        half_life = self.config.decay_half_life_days
        if half_life <= 0:
            time_factor = 1.0
        else:
            # decay ^ (days/half_life)
            time_factor = self.config.decay_fraction ** (days / half_life)

        # --- access boost ---
        access_count = self._get_access_count(memory)
        boost = 1.0 + access_count * self.config.access_boost_fraction

        score = conf * time_factor * boost
        return max(0.0, min(1.0, score))

    # -- should_forget --------------------------------------------------------

    def should_forget(self, memory: MemoryRecord) -> bool:
        """Return ``True`` when a memory is a candidate for forgetting.

        A memory is eligible when:
        1. Its decayed score is **below** ``forget_threshold``.
        2. It has NOT been recently accessed (within the last half-life period).
        3. It is NOT protected (user-corrected or high-salience).

        Even when this returns ``True`` the caller still has to call
        ``forget()`` to actually mark the record.
        """
        # Already forgotten → skip.
        if self._is_forgotten(memory):
            return False

        # Protected by salience (constraint / decision).
        if self._has_protected_salience(memory):
            return False

        # Protected by user correction.
        if self._is_corrected(memory):
            return False

        # Decay score below threshold?
        score = self.decay_score(memory)
        if score >= self.config.forget_threshold:
            return False

        # Recently accessed? (within half-life)
        days = self._days_since_accessed(memory)
        if days < self.config.decay_half_life_days:
            return False

        return True

    # -- forget ---------------------------------------------------------------

    def forget(self, memory_id: str) -> bool:
        """Mark a memory as forgotten (soft delete — confidence dropped near 0).

        The memory row is **never** hard-deleted.  Instead:
        * ``forgotten`` is set to 1.
        * ``confidence`` is reduced to a near-zero value (1e-6).

        Returns ``True`` if the memory was actually updated, ``False`` if it
        was already forgotten or doesn't exist.
        """
        row = self.store.conn.execute(
            "SELECT id, forgotten FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()

        if row is None:
            logger.debug("forget(%r): memory not found", memory_id)
            return False

        if row["forgotten"]:
            logger.debug("forget(%r): already forgotten", memory_id)
            return False

        self.store.conn.execute(
            "UPDATE memories SET forgotten = 1, confidence = ? WHERE id = ?",
            (1e-6, memory_id),
        )
        self.store.conn.commit()
        logger.info("forget(%r): marked as forgotten", memory_id)
        return True

    def unmark_forgotten(self, memory_id: str, confidence: float = 1.0) -> bool:
        """Reverse a soft forget, restoring the original confidence."""
        row = self.store.conn.execute(
            "SELECT id, forgotten FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()

        if row is None:
            return False

        self.store.conn.execute(
            "UPDATE memories SET forgotten = 0, confidence = ? WHERE id = ?",
            (confidence, memory_id),
        )
        self.store.conn.commit()
        return True

    # -- consolidate ----------------------------------------------------------

    def consolidate(self, namespace: str = "default") -> dict[str, Any]:
        """Merge near-duplicate memories (similarity > threshold).

        For every pair of active (non-forgotten) memories in the same
        namespace whose ``raw_text`` similarity exceeds
        ``consolidate_similarity_threshold`` (default 0.80), the older
        memory is kept and the duplicate is soft-forgotten.

        Returns a stats dict: ``{"pairs_found": N, "merged": N}``.
        """
        rows = self.store.conn.execute(
            "SELECT id, raw_text, created_at FROM memories "
            "WHERE namespace = ? AND forgotten = 0 "
            "ORDER BY created_at ASC, id ASC",
            (namespace,),
        ).fetchall()

        if len(rows) < 2:
            return {"pairs_found": 0, "merged": 0}

        pairs_found = 0
        merged = 0

        # Compare each pair once (O(n²) — acceptable for the expected scale
        # of targeted consolidation calls, not a background loop).
        n = len(rows)
        for i in range(n):
            if rows[i]["id"] is None:
                continue  # already merged away
            for j in range(i + 1, n):
                if rows[j]["id"] is None:
                    continue
                similarity = SequenceMatcher(
                    None, rows[i]["raw_text"], rows[j]["raw_text"]
                ).ratio()
                if similarity >= self.config.consolidate_similarity_threshold:
                    pairs_found += 1
                    # Keep the earlier one (i), forget the later (j).
                    self.forget(str(rows[j]["id"]))
                    merged += 1
                    rows[j] = {**rows[j], "id": None}  # skip further

        return {"pairs_found": pairs_found, "merged": merged}

    # -- helpers --------------------------------------------------------------

    def _days_since_accessed(self, memory: MemoryRecord) -> float:
        """Days since the last access, or since creation if never accessed."""
        ts = None

        # Prefer the metadata field (set by recall() hit).
        if memory.metadata:
            ts = memory.metadata.get("accessed_at")

        # Fall back to created_at.
        if not ts and memory.metadata:
            ts = memory.metadata.get("created_at")

        if not ts:
            return 0.0

        try:
            accessed_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return 0.0

        now = datetime.now(timezone.utc)
        return (now - accessed_dt).total_seconds() / 86400.0

    def _get_access_count(self, memory: MemoryRecord) -> int:
        """Read access_count from the database row (not the MemoryRecord)."""
        row = self.store.conn.execute(
            "SELECT access_count FROM memories WHERE id = ?",
            (memory.id,),
        ).fetchone()
        if row is None:
            return 0
        return int(row["access_count"]) if row["access_count"] is not None else 0

    def _is_forgotten(self, memory: MemoryRecord) -> bool:
        """Check if the memory is already flagged as forgotten."""
        row = self.store.conn.execute(
            "SELECT forgotten FROM memories WHERE id = ?",
            (memory.id,),
        ).fetchone()
        if row is None:
            return False
        return bool(row["forgotten"])

    def _has_protected_salience(self, memory: MemoryRecord) -> bool:
        """Return True if any linked entity has a protected salience class."""
        rows = self.store.conn.execute(
            "SELECT me.salience_class FROM memory_entities me "
            "WHERE me.memory_id = ?",
            (memory.id,),
        ).fetchall()
        for r in rows:
            if r["salience_class"] in self.config.protected_salience:
                return True
        return False

    def _is_corrected(self, memory: MemoryRecord) -> bool:
        """Return True if the memory has a user correction entry."""
        row = self.store.conn.execute(
            "SELECT 1 FROM corrections WHERE target_id = ? LIMIT 1",
            (memory.id,),
        ).fetchone()
        return row is not None

    # -- bulk maintenance -----------------------------------------------------

    def run_maintenance(
        self,
        namespace: str = "default",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Run a full maintenance cycle: decay check + consolidation.

        Returns::

            {
                "candidates_forgotten": [...],
                "forgotten_count": 0,
                "consolidation": {"pairs_found": 0, "merged": 0},
            }
        """
        rows = self.store.conn.execute(
            "SELECT id, raw_text, summary, source, confidence, "
            "metadata, created_at "
            "FROM memories WHERE namespace = ? AND forgotten = 0",
            (namespace,),
        ).fetchall()

        # Build MemoryRecord list for should_forget inspection.
        candidates: list[str] = []
        for row in rows:
            metadata = self.store._parse_metadata(row["metadata"])
            metadata.setdefault("source", row["source"])
            metadata.setdefault("created_at", row["created_at"])

            rec = MemoryRecord(
                id=row["id"],
                raw_text=row["raw_text"],
                summary=row["summary"],
                confidence=float(row["confidence"]),
                metadata=metadata,
            )
            if self.should_forget(rec):
                candidates.append(rec.id)

        forgotten_count = 0
        if not dry_run:
            for mid in candidates:
                if self.forget(mid):
                    forgotten_count += 1

        cons_stats = {"pairs_found": 0, "merged": 0}
        if not dry_run:
            cons_stats = self.consolidate(namespace=namespace)

        return {
            "candidates_forgotten": candidates,
            "forgotten_count": forgotten_count,
            "consolidation": cons_stats,
        }
