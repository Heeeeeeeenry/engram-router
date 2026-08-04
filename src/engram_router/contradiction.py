"""Contradiction detection via cross-encoder NLI (L2.2).

Compares events extracted from the ``events`` table and detects factual
contradictions. When two events disagree on the same (subject, predicate)
— e.g. "张三 lives in 北京" vs "张三 lives in 上海" — a CONTRADICTS edge
is written with a confidence score from the NLI model.

Design:
  1. Group events by (subject, verb) within a namespace.
  2. For each group with ≥2 distinct objects, pair them and score with
     cross-encoder NLI (entailment/neutral/contradiction).
  3. Write CONTRADICTS edges back to the edges table (old-object →
     new-object, with evidence refs anchored to the respective memory_ids).

The NLI model is lazily loaded (like cross_encoder.py) and
``available = False`` when ``sentence_transformers`` is absent.
Rule-based fallback: if two events mention the same verb + subject
with lexically different objects and no time-expr overlap, flag
as a soft contradiction (confidence 0.5).

Usage::

    from engram_router.contradiction import detect_contradictions
    detect_contradictions(store, namespace="default")
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ── NLI (Natural Language Inference) model ────────────────────────────────

class NLIScorer:
    """Cross-encoder NLI: score premise-hypothesis pairs → contradiction."""

    def __init__(self) -> None:
        self._model: Any = None
        self._load_attempted = False
        self._init_error: str = ""

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                "cross-encoder/nli-deberta-v3-base",
                automodel_args={"torch_dtype": "float16"},
            )
        except ImportError:
            self._init_error = "sentence-transformers not installed"
            logger.debug("NLI unavailable: %s", self._init_error)
        except Exception as exc:
            self._init_error = str(exc)
            logger.debug("NLI model load failed: %s", exc)

    def contradiction_score(self, premise: str, hypothesis: str) -> float:
        """Return P(contradiction) ∈ [0, 1] for premise vs hypothesis.

        Returns 0.0 if unavailable (safe default: assume no contradiction).
        """
        if not self.available:
            return 0.0
        try:
            # NLI label mapping: 0=entailment, 1=neutral, 2=contradiction
            scores = self._model.predict([(premise, hypothesis)])
            # scores is array of logits; softmax to get probabilities
            import numpy as np

            def _softmax(x):
                e = np.exp(x - np.max(x))
                return e / e.sum()

            probs = _softmax(scores[0]) if len(scores.shape) == 2 else _softmax(scores)
            # Index 2 = contradiction
            return float(probs[2]) if len(probs) > 2 else 0.0
        except Exception as exc:
            logger.debug("NLI scoring failed: %s", exc)
            return 0.0


# ── Contradiction detection ───────────────────────────────────────────────

def detect_contradictions(
    store: Any,
    namespace: str = "default",
    threshold: float = 0.7,
) -> dict[str, int]:
    """Scan the events table and write CONTRADICTS edges for detected conflicts.

    Args:
        store: MemoryStore instance.
        namespace: namespace to scope the check to.
        threshold: minimum contradiction probability to write an edge.

    Returns:
        ``{"checked_pairs": N, "contradictions": N, "edges_written": N}``
    """
    stats: dict[str, int] = {
        "checked_pairs": 0,
        "contradictions": 0,
        "edges_written": 0,
    }

    # ── 1. Fetch events, grouped by (subject, verb) ──
    rows = store.conn.execute(
        """SELECT id, memory_id, subject, verb, object, time_expr, confidence
           FROM events
           WHERE namespace = ? AND subject != '' AND verb != ''
           ORDER BY created_at""",
        (namespace,),
    ).fetchall()

    if not rows:
        return stats

    # Group by (subject_normalized, verb)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        key = (r["subject"].strip(), r["verb"].strip())
        groups.setdefault(key, []).append(dict(r))

    # ── 2. Within each group, compare distinct objects ──
    nli = NLIScorer()

    for (subject, verb), events in groups.items():
        # Only care about groups with multiple distinct objects
        objects_seen: dict[str, list[dict[str, Any]]] = {}
        for ev in events:
            obj = ev["object"].strip()
            if obj:
                objects_seen.setdefault(obj, []).append(ev)

        object_list = list(objects_seen.items())
        if len(object_list) < 2:
            continue

        # Compare every pair of distinct objects
        for i in range(len(object_list)):
            for j in range(i + 1, len(object_list)):
                obj_a, events_a = object_list[i]
                obj_b, events_b = object_list[j]

                # Skip if either object is empty
                if not obj_a or not obj_b:
                    continue

                stats["checked_pairs"] += 1

                # ── 2a. Rule-based soft check ──
                # Different time expressions → likely NOT a contradiction
                # (person moved, changed jobs, etc.)
                times_a = {e["time_expr"] for e in events_a if e["time_expr"]}
                times_b = {e["time_expr"] for e in events_b if e["time_expr"]}
                if times_a and times_b and times_a.isdisjoint(times_b):
                    # Different times → temporal change, not contradiction
                    continue

                # ── 2b. Lexical overlap check ──
                # If objects share significant lexical overlap, it's probably
                # just a rephrasing
                overlap = _lexical_overlap(obj_a, obj_b)
                if overlap > 0.6:
                    continue  # too similar to be a contradiction

                # ── 2c. NLI scoring ──
                premise = f"{subject} {verb} {obj_a}"
                hypothesis = f"{subject} {verb} {obj_b}"
                contra_prob = nli.contradiction_score(premise, hypothesis)

                # Fallback: if NLI is unavailable, use a lexical heuristic
                if not nli.available:
                    contra_prob = _heuristic_contradiction(verb, obj_a, obj_b)

                if contra_prob >= threshold:
                    stats["contradictions"] += 1

                    # Write CONTRADICTS edge: older → newer object
                    # (the new info contradicts the old)
                    older_ev = events_a[0]  # first event with obj_a
                    newer_ev = events_b[0]  # first event with obj_b

                    # Use the event IDs as edge references (entities don't exist for events)
                    edge_id = store._next_id("edges", "edge")
                    try:
                        store.conn.execute(
                            """INSERT INTO edges
                               (id, src_id, dst_id, relation, confidence, evidence_ref, namespace)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (
                                edge_id,
                                older_ev["memory_id"],
                                newer_ev["memory_id"],
                                "CONTRADICTS",
                                round(contra_prob, 4),
                                f"event:{older_ev['id']},{newer_ev['id']}",
                                namespace,
                            ),
                        )
                        stats["edges_written"] += 1
                    except Exception as exc:
                        logger.debug("CONTRADICTS edge write failed: %s", exc)

    store.conn.commit()
    logger.info(
        "contradiction scan: %d pairs, %d contradictions, %d edges (ns=%s)",
        stats["checked_pairs"], stats["contradictions"],
        stats["edges_written"], namespace,
    )
    return stats


# ── Helpers ────────────────────────────────────────────────────────────────

def _lexical_overlap(a: str, b: str) -> float:
    """Jaccard-like overlap on CJK character bigrams."""
    import re

    def bigrams(s: str) -> set[str]:
        chars = re.findall(r"[一-鿿]|[A-Za-z0-9]+", s)
        return {f"{chars[i]}{chars[i+1]}" for i in range(len(chars) - 1)} if len(chars) >= 2 else set()

    bg_a = bigrams(a)
    bg_b = bigrams(b)
    if not bg_a or not bg_b:
        return 0.0
    intersection = bg_a & bg_b
    union = bg_a | bg_b
    return len(intersection) / len(union) if union else 0.0


def _heuristic_contradiction(verb: str, obj_a: str, obj_b: str) -> float:
    """Lexical contradiction heuristic when NLI is unavailable.

    Returns a float 0.0-1.0 representing estimated contradiction probability.
    """
    # High-confidence case: exact antonyms or mutually exclusive values
    # e.g. "住/北京" vs "住/上海" → locations are likely contradictory
    # e.g. "是/程序员" vs "是/设计师" → different professions

    # Exact-match common attribute verbs → higher confidence that
    # different objects are a real contradiction
    identity_verbs = {"是", "在", "住", "住在", "工作", "做"}
    if verb in identity_verbs:
        return 0.65

    # Other verbs → lower confidence (could be sequential events)
    return 0.40


# ── Integration helper ────────────────────────────────────────────────────

def check_new_event(store: Any, event: dict[str, Any]) -> list[dict[str, Any]]:
    """Check a new event against existing ones; return conflict list.

    Called immediately after a new event is inserted (optional, for
    real-time contradiction detection rather than batch scan).

    Returns list of conflicting existing events with contra_prob scores.
    """
    conflicts: list[dict[str, Any]] = []

    existing = store.conn.execute(
        """SELECT id, memory_id, subject, verb, object, time_expr
           FROM events
           WHERE subject = ? AND verb = ? AND namespace = ? AND id != ?""",
        (event["subject"], event["verb"], event.get("namespace", "default"), event["id"]),
    ).fetchall()

    if not existing:
        return conflicts

    nli = NLIScorer()
    for row in existing:
        obj_a = event["object"].strip()
        obj_b = row["object"].strip()
        if not obj_a or not obj_b or obj_a == obj_b:
            continue

        # Skip if same time expression
        if event.get("time_expr") and row["time_expr"] and event["time_expr"] == row["time_expr"]:
            continue

        overlap = _lexical_overlap(obj_a, obj_b)
        if overlap > 0.6:
            continue

        premise = f"{event['subject']} {event['verb']} {obj_a}"
        hypothesis = f"{row['subject']} {row['verb']} {obj_b}"
        contra = nli.contradiction_score(premise, hypothesis)
        if not nli.available:
            contra = _heuristic_contradiction(event["verb"], obj_a, obj_b)

        if contra >= 0.5:
            conflicts.append({
                "existing_event_id": row["id"],
                "existing_memory_id": row["memory_id"],
                "existing_object": obj_b,
                "contradiction_probability": round(contra, 4),
            })

    return conflicts
