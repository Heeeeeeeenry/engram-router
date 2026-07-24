"""Graph module: edge indexing, query-relevance scoring, and n-hop BFS edge expansion.

These are free functions designed to work with a ``sqlite3.Connection``
and a ``RecallWeights`` instance so any store-like object can delegate
edge-level operations without carrying the full MemoryStore surface area.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import deque
from typing import Any

from .config import config
from .entities import extract_entities
from .llm_extractor import extract_edges_llm
from .scoring import RecallWeights, _default_weights
from .scoring import base_score as _scoring_base_score
from .scoring import terms as extract_terms

REASON_MARKERS: tuple[str, ...] = ("因为", "原因", "所以", "导致", "出于", "由于", "为了", "生日")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Edge indexing (write typed relations between co-occurring entities)
# ---------------------------------------------------------------------------

def index_edges(
    conn: sqlite3.Connection,
    _next_id,  # callable (table: str, prefix: str) -> str
    weights: RecallWeights | None = None,
    memory_id: str = "",
    indexed: list[dict[str, Any]] | None = None,
    text: str = "",
    llm_extractor: Any = None,
) -> None:
    """Write typed relations between the entities co-occurring in one memory.

    Causal-edge hard boundary:
      - Co-occurrence inside a memory is an *inference*, so it is only ever
        written as CO_OCCURS_WITH at low confidence (0.4) and must never be
        promoted to a fact without evidence (≥3 confirmations or an explicit
        user statement).
      - A *user-stated* cause -- surfaced as a ``reason`` entity from a
        causal marker (因为/由于/...) in this very turn -- is the user
        asserting causation. We honour that as a CAUSED_BY edge at high
        confidence (0.95), pointing from each non-reason entity to the
        reason (effect -> cause).
      - When LLM extraction is enabled, LLM-typed edges (HAS_ATTRIBUTE,
        REPLACES, PREFERS, etc.) are merged in with source=llm confidence.

    Every edge's ``evidence_ref`` points back to the memory it was drawn
    from, so the inference is always auditable / revocable.
    """
    if indexed is None:
        return
    # Deduplicate entity ids within this memory while keeping kind/name.
    # Exclude cjk_ngram entities from edge creation to prevent noise.
    uniq: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for ent in indexed:
        if ent["id"] in seen_ids:
            continue
        if ent.get("kind") == "cjk_ngram":
            continue  # cjk_ngram is for shared-entity scoring only, not edges
        seen_ids.add(ent["id"])
        uniq.append(ent)
    if len(uniq) < 2:
        return

    reasons = [e for e in uniq if e["kind"] == "reason"]
    nonreason = [e for e in uniq if e["kind"] != "reason"]

    written: set[tuple[str, str, str]] = set()

    def _add_edge(src: str, dst: str, relation: str, confidence: float) -> None:
        if src == dst:
            return
        key = (src, dst, relation)
        if key in written:
            return
        written.add(key)
        edge_id = _next_id("edges", "edge")
        conn.execute(
            "INSERT INTO edges (id, src_id, dst_id, relation, confidence, evidence_ref) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (edge_id, src, dst, relation, confidence, memory_id),
        )

    # CO_OCCURS_WITH: every unordered pair, written once as a directed edge
    # from the earlier-indexed entity to the later one (low confidence).
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            _add_edge(uniq[i]["id"], uniq[j]["id"], "CO_OCCURS_WITH", 0.4)

    # CAUSED_BY: user-stated causation (effect -> cause). The reason entity
    # is the cause; connect it only to the nearest non-reason entities
    # (capped at 3 per reason) to avoid the all-to-all noise problem.
    # High confidence because the user said so.
    for cause in reasons:
        nearby = nonreason[:3]
        for effect in nearby:
            _add_edge(effect["id"], cause["id"], "CAUSED_BY", 0.95)

    # DESCRIBES: product→topic (e.g. HHKB → 键盘).  When both a product
    # alias and its topic appear in the same memory, create a high-confidence
    # directed edge so recall can follow the semantic link.
    for ent in uniq:
        topic = config.entities.object_topic_aliases.get(ent["name"])
        if topic is None:
            continue
        topic_ent = next((e for e in uniq if e["name"] == topic), None)
        if topic_ent is not None:
            _add_edge(ent["id"], topic_ent["id"], "DESCRIBES", 0.9)

    # LLM-typed edges: supplement rule-based edges with LLM-annotated
    # relations (HAS_ATTRIBUTE, REPLACES, PREFERS, etc.).  LLM edges use
    # lower confidence (×0.8) and are tagged source=llm in evidence_ref
    # for auditability.
    if llm_extractor is not None and getattr(llm_extractor, "available", False) and text:
        name_to_id: dict[str, str] = {e["name"]: e["id"] for e in uniq}
        for edge in extract_edges_llm(text):
            src_id = name_to_id.get(edge.get("src", ""))
            dst_id = name_to_id.get(edge.get("dst", ""))
            if src_id is None or dst_id is None:
                continue
            relation = edge.get("relation", "CO_OCCURS_WITH")
            confidence = float(edge.get("confidence", 0.8))
            # Don't re-write CO_OCCURS_WITH or CAUSED_BY if already handled.
            if relation in ("CO_OCCURS_WITH", "CAUSED_BY"):
                continue
            edge_id = _next_id("edges", "edge")
            conn.execute(
                "INSERT INTO edges (id, src_id, dst_id, relation, confidence, evidence_ref) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (edge_id, src_id, dst_id, relation, confidence,
                 f"{memory_id}:llm"),
            )


# ---------------------------------------------------------------------------
# Query relevance scoring for entities reached during edge expansion
# ---------------------------------------------------------------------------

def entity_query_relevance(
    entity_name: str,
    entity_kind: str,
    query_terms: list[str],
    query_entities: set[str],
    query_entity_objs: list[dict[str, Any]],
    weights: RecallWeights | None = None,
) -> float:
    """Score how relevant an expanded entity is to the original query.

    SAG-inspired: at each expansion hop, check if the newly reached entity
    still connects to the query's semantic intent. If not, prune the branch.

    Scoring factors:
    1. Direct name overlap with query terms (strongest signal)
    2. Kind-based relevance multiplier (person/topic > generic noun)
    3. Substring / partial match with query entities

    Returns 0.0–1.0, where 1.0 = directly mentioned in query.
    """
    if weights is None:
        weights = _default_weights()
    if not entity_name:
        return 0.0
    name_lower = entity_name.lower()
    score = 0.0

    # 1. Direct term overlap: entity name appears in query terms
    term_overlap = 0
    for t in query_terms:
        if t and (t in name_lower or name_lower in t):
            term_overlap += 1
    if term_overlap > 0:
        # Up to 0.6 for term overlap
        score = min(0.6, 0.2 * term_overlap)

    # 2. Entity set match: entity is in the query's extracted entities
    if entity_name in query_entities:
        score = max(score, 0.9)  # Strong signal: explicitly extracted

    # 3. Kind-based multiplier
    kind_weights = weights.relevance_entity_kind_weights or {
        "person": 0.9, "topic": 0.8, "object": 0.6,
        "location": 0.5, "time": 0.4, "number": 0.3,
        "event": 0.7, "organization": 0.5,
    }
    kind_mult = kind_weights.get(entity_kind, 0.3)
    # Blend: if no term match, fall back to kind weight
    if score < 0.1:
        # When query entities are sparse, rely more on kind relevance.
        # A "time" or "object" entity in a no-entity query still has
        # structural value for edge traversal, just at a lower weight.
        if not query_entities:
            score = kind_mult * 0.5  # Allow traversal through structural entities
        else:
            score = kind_mult * 0.4
    else:
        # Boost existing score by kind relevance
        score = score * (0.5 + 0.5 * kind_mult)

    # 4. Bonus: entity name is a substring of a query entity or vice versa
    for qe in query_entities:
        if qe and (qe in name_lower or name_lower in qe):
            score = max(score, 0.7)
            break

    return min(1.0, score)


# ---------------------------------------------------------------------------
# N-hop edge expansion (BFS spreading activation)
# ---------------------------------------------------------------------------

def edge_expansion(
    conn: sqlite3.Connection,
    weights: RecallWeights | None = None,
    query: str = "",
    terms: list[str] | None = None,
    rows: list[sqlite3.Row] | None = None,
    max_hops: int | None = None,
    decay: float | None = None,
    entity_map: dict[str, list[dict[str, Any]]] | None = None,
    namespace: str = "default",
    base_score_fn: Any = None,
) -> dict[str, tuple[float, str]]:
    """N-hop spreading activation over the typed ``edges`` table.

    Returns ``{memory_id: (bonus, reason)}`` for memories reached by
    following edges out of a *seed* memory (one the direct pass already
    matches on tokens/entities). Edges let recall reach a memory that
    shares no token and no entity with the query but sits a few hops away
    in the association graph (妈妈 → 做饭好吃 → 红烧肉 → 那顿饭).

    Activation model (bounded BFS, re-propagating):
      - seed entities start at activation 1.0;
      - crossing an edge multiplies activation by ``decay × confidence``;
      - a node reached again by a STRONGER path is updated AND re-queued so
        the stronger activation propagates onward (this is the bug the
        0002 draft had — it never re-queued, so multi-path gains died);
      - activation below ``activation_threshold`` stops spreading;
      - ``max_hops`` caps depth.

    Defaults: ``max_hops=2``, ``decay=0.5``. Two hops covers the
    三段联想 the user described while keeping the dense low-confidence
    CO_OCCURS_WITH graph from lighting up the whole store on small data.
    The bonus is the neighbour entity's final activation, so a strong
    CAUSED_BY (0.95) hop pulls far harder than a weak CO_OCCURS_WITH (0.4).
    """
    if weights is None:
        weights = _default_weights()
    if max_hops is None:
        max_hops = weights.max_recall_hops
    if decay is None:
        decay = weights.recall_decay
    if terms is None:
        terms = []
    if rows is None:
        rows = []
    if base_score_fn is None:
        # Scoring.base_score() requires weights/stop_chars/reason_markers
        _stop = frozenset(config.recall.stop_chars)
        base_score_fn = lambda q, t, h: _scoring_base_score(q, t, h, weights, _stop, REASON_MARKERS)

    # 1. Find seed memories: those with a positive direct token/entity match.
    query_entity_objs = [
        e for e in extract_entities(query)
        if e.get("kind") != "cjk_ngram"
    ]
    query_entities = {e["name"] for e in query_entity_objs}
    seed_ids: list[str] = []
    for row in rows:
        haystack = f"{row['raw_text']} {row['summary']}".lower()
        if base_score_fn(query, terms, haystack) > 0:
            seed_ids.append(row["id"])
            continue
        if entity_map is not None:
            mem_entities = {e["name"] for e in entity_map.get(row["id"], [])}
        else:
            mem_entities = _entity_names_for(conn, row["id"])
        if query_entities & mem_entities:
            seed_ids.append(row["id"])
    if not seed_ids:
        return {}

    # 2. Collect the entity ids each seed memory carries (the BFS frontier).
    seed_entity_ids: set[str] = set()
    if seed_ids:
        placeholders = ",".join("?" for _ in seed_ids)
        for r in conn.execute(
            f"SELECT DISTINCT entity_id FROM memory_entities "
            f"WHERE memory_id IN ({placeholders})",
            tuple(seed_ids),
        ).fetchall():
            seed_entity_ids.add(r["entity_id"])
    if not seed_entity_ids:
        return {}

    # 3. Dynamic multi-hop BFS with query-relevance gating (SAG-inspired).
    #    activated[entity_id] = (activation, relation, hop, path_names, relevance)
    #
    #    SAG insight: at query time, dynamically connect events through shared
    #    entities instead of pre-building a global graph. EngramRouter mirrors
    #    this with relevance-gated edge traversal: only expand through entities
    #    that maintain semantic connection to the original query.
    #
    #    Termination is dual-threshold:
    #      - activation < activation_threshold → too attenuated, stop
    #      - relevance < relevance_threshold → lost query connection, stop

    # Preload entity names + kinds for the BFS loop (avoid per-hop DB hits).
    entity_info: dict[str, tuple[str, str]] = {}  # id → (name, kind)
    all_eids = set(seed_entity_ids)
    if all_eids:
        e_ph = ",".join("?" for _ in all_eids)
        for r in conn.execute(
            f"SELECT id, name, kind FROM entities WHERE id IN ({e_ph})",
            tuple(all_eids),
        ).fetchall():
            entity_info[r["id"]] = (r["name"], r["kind"] if "kind" in r.keys() else "")

    def _name(eid: str) -> str:
        if eid not in entity_info:
            r = conn.execute(
                "SELECT name, kind FROM entities WHERE id = ?", (eid,)
            ).fetchone()
            if r:
                entity_info[eid] = (r["name"], r["kind"] if "kind" in r.keys() else "")
            else:
                entity_info[eid] = (eid, "")
        return entity_info[eid][0]

    def _kind(eid: str) -> str:
        _name(eid)  # ensure loaded
        return entity_info.get(eid, ("", ""))[1]

    # Pre-compute query terms for relevance scoring.
    _query_terms = list(extract_terms(query)) if query else []

    activated: dict[str, tuple[float, str, int, list[str], float]] = {}
    queue: deque[tuple[str, float, int, float]] = deque()  # (eid, act, hop, relevance)

    # Seed entities start with relevance=1.0 (they directly match the query).
    for eid in seed_entity_ids:
        queue.append((eid, 1.0, 0, 1.0))

    while queue:
        src_eid, src_act, hop, src_relevance = queue.popleft()
        if hop >= max_hops:
            continue
        next_hop = hop + 1

        for src_col, dst_col in (("src_id", "dst_id"), ("dst_id", "src_id")):
            for e in conn.execute(
                f"SELECT {dst_col} AS nbr, relation, confidence "
                f"FROM edges WHERE {src_col} = ?",
                (src_eid,),
            ).fetchall():
                nbr = e["nbr"]
                if nbr in seed_entity_ids:
                    continue  # never re-activate a seed entity

                # ── Activation decay ──
                new_act = src_act * decay * float(e["confidence"])
                if new_act < weights.activation_threshold:
                    continue

                # ── Query relevance check (SAG-style dynamic gate) ──
                nbr_name = _name(nbr)
                nbr_kind = _kind(nbr)
                nbr_relevance = entity_query_relevance(
                    nbr_name, nbr_kind,
                    _query_terms, query_entities, query_entity_objs,
                    weights=weights,
                )
                # Relevance decays per hop, blended with the neighbor's own relevance.
                nbr_relevance = nbr_relevance * (weights.relevance_decay ** next_hop)

                # ── Adaptive threshold: sparse-entity queries need looser gating ──
                # When the query extracts few/no entities (e.g., "根因是什么？"),
                # entity-level relevance is inherently low. Use a fraction of the
                # standard threshold to avoid over-pruning valid associations.
                _adaptive_threshold = weights.relevance_threshold
                if len(query_entities) <= 1:
                    # Sparse-entity query: halve the threshold so edge expansion
                    # can still reach relevant memories through shared entities.
                    _adaptive_threshold = weights.relevance_threshold * 0.4
                if nbr_relevance < _adaptive_threshold:
                    continue  # Lost query connection — prune this branch

                # ── Effective activation: activation × relevance ──
                # A high-confidence edge to an irrelevant entity should not pull.
                effective = new_act * nbr_relevance

                prev = activated.get(nbr)
                if prev is None or effective > prev[0] * prev[4]:
                    src_path = (
                        activated.get(src_eid, (0.0, "", 0, [_name(src_eid)], 1.0))[3]
                    )
                    activated[nbr] = (
                        new_act, e["relation"], next_hop,
                        src_path + [nbr_name], nbr_relevance,
                    )
                    # Re-queue so improved activation propagates onward.
                    queue.append((nbr, new_act, next_hop, nbr_relevance))
    if not activated:
        return {}

    # 4. Map activated neighbour entities -> the memories carrying them
    #    (excluding seeds), keep the strongest *effective* activation per memory.
    #    Filter by namespace so edge associations never cross tenants.
    #
    #    Uses effective_activation = activation × relevance for final bonus,
    #    so an entity reached through a strong edge but irrelevant to the query
    #    contributes less than a weaker edge to a query-relevant entity.
    result: dict[str, tuple[float, str]] = {}
    seed_set = set(seed_ids)
    activated_ids = list(activated.keys())
    if activated_ids:
        a_ph = ",".join("?" for _ in activated_ids)
        for r in conn.execute(
            "SELECT me.memory_id, me.entity_id FROM memory_entities me "
            "JOIN memories m ON m.id = me.memory_id "
            f"WHERE me.entity_id IN ({a_ph}) AND m.namespace = ?",
            tuple(activated_ids) + (namespace,),
        ).fetchall():
            mid = r["memory_id"]
            if mid in seed_set:
                continue
            entity_id = r["entity_id"]
            act, relation, hop, path, relevance = activated[entity_id]
            # ── Effective bonus: raw activation weighted by query relevance ──
            effective_bonus = act * relevance
            reason = (
                f"edge assoc hop={hop} act={act:.3f} rel={relevance:.3f} "
                f"eff={effective_bonus:.3f} [{' → '.join(path)}] ({relation})"
            )
            existing = result.get(mid)
            if existing is None or effective_bonus > existing[0]:
                result[mid] = (effective_bonus, reason)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entity_names_for(conn: sqlite3.Connection, memory_id: str) -> set[str]:
    rows = conn.execute(
        "SELECT e.name FROM memory_entities me JOIN entities e ON e.id = me.entity_id "
        "WHERE me.memory_id = ?",
        (memory_id,),
    ).fetchall()
    return {r["name"] for r in rows}
