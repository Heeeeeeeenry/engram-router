"""Recall pipeline extracted from store.py (Step 6).

The module contains the core recall logic as free functions that accept a
``store`` object (the MemoryStore instance) as the first parameter.  This
avoids circular imports — recall.py only imports from downstream modules
(scoring, candidates, graph, records, query_intent, entities, fusion, config)
and never from store.py.

The ``recall()`` method on MemoryStore remains as a thin delegating wrapper.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from . import candidates
from . import graph
from . import query_intent
from . import scoring
from ..config import config
from ..entities import extract_entities
from ..fusion import reciprocal_rank_fusion
from .records import (
    MemoryRecord,
    parse_metadata,
    row_to_record,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# should_inject — fast pre-recall gate
# ---------------------------------------------------------------------------

def should_inject(query: str) -> bool:
    """快速判断查询是否需要记忆注入。

    节省上下文：闲聊/通用知识/数学/编程 不注入。
    仅在查询涉及个人信息/历史/偏好时注入。

    规则（在 recall 之前快速判定，不触发召回）：
    - 含"什么/谁/怎么/哪个/多大/之前/说过/记得" → 需要
    - 纯闲聊("你好/哈哈") → 不需要
    - 通用知识("Python/定义/代码") → 不需要
    - 实时数据("天气/新闻/股票") → 不需要
    """
    q = query.strip()

    # 纯闲聊 / 纯表情
    if re.match(r'^[你好嗨嗯哦啊哈哈嘿]{1,4}$', q):
        return False

    # 通用知识 / 编程 / 数学 — LLM 自有能力
    _general = ["python", "代码", "定义", "什么是", "如何", "怎么用",
                "1+1", "等于", "数学", "公式", "算法", "编程",
                "天气", "新闻", "股票", "汇率", "时间"]
    if any(kw in q.lower() for kw in _general):
        return False

    # 依赖个人记忆的关键词
    _needs_memory = ["什么", "谁", "怎么", "为什么", "哪个", "哪里",
                     "多大", "几岁", "之前", "说过", "记得",
                     "上次", "历史", "聊过", "知道",
                     "最近", "罗列", "列出", "回顾", "总结"]
    if any(kw in q for kw in _needs_memory):
        return True

    # 默认不注入（保守，避免浪费上下文）
    return False


# ── Intent helper: use classifier when available, fall back to regex ──

def _asks_identity(store: Any, query: str) -> bool:
    """Check if query is an identity question, using classifier if available.

    L1: cache classifier result per (store, query) so repeated calls in
    recall_single + build_scored_candidates + build_recall_response don't
    trigger multiple LLM invocations for the same query.
    """
    classifier = getattr(store, 'intent_classifier', None)
    if classifier and classifier.available and store.weights.intent_classifier_enabled:
        cache = getattr(store, '_intent_cache', None)
        if cache is None:
            cache = {}
            setattr(store, '_intent_cache', cache)
        if query in cache:
            return cache[query].identity > 0.5
        intent = classifier.classify(query)
        cache[query] = intent
        return intent.identity > 0.5
    return query_intent.asks_identity(query)


# ---------------------------------------------------------------------------
# _apply_salience_decay
# ---------------------------------------------------------------------------

def apply_salience_decay(
    store: Any,
    row: Any,
    score: float,
    reason: str,
    mem_entity_objs: list[dict[str, Any]],
    directly_matched: bool,
) -> tuple[float, str]:
    """Apply associative-reach salience decay to non-directly-matched rows.

    Directly-matched memories are exempt so identity questions still
    surface base-attribute memories at full strength.
    """
    if directly_matched or score <= 0 or not mem_entity_objs:
        return (score, reason)

    reach = min(
        getattr(store.weights, f"assoc_reach_{e.get('salience_class', 'event')}", 1.0)
        for e in mem_entity_objs
    )
    if reach < 1.0:
        score *= reach
        reason = (reason + f"; assoc-reach×{reach:.2f}").lstrip("; ")

    return (score, reason)


# ---------------------------------------------------------------------------
# _identity_subjects / _looks_like_product — static helpers
# ---------------------------------------------------------------------------

def _identity_subjects(entities: list[dict[str, Any]]) -> set[str]:
    """Anchors an identity attribute can belong to.

    Identity boosts must be scoped: a memory with 妈妈's 55岁 should not
    answer 张三/室友/咪咪 age questions. Person entities are the primary
    anchors; named objects are included so pet/device identity questions can
    still bind to specific non-person subjects (咪咪 / HHKB / iPhone).
    """
    subjects: set[str] = set()
    for ent in entities:
        kind = ent.get("kind")
        name = ent.get("name", "")
        if kind in ("person", "object", "topic"):
            subjects.add(name)
    return subjects


def _looks_like_product(name: str) -> bool:
    """A product-like object name is one carrying specific identity, i.e.
    an ASCII/alnum token (HHKB, MX-3) rather than a generic CJK noun (键盘)."""
    return bool(re.search(r"[A-Za-z0-9]", name))


# ---------------------------------------------------------------------------
# _apply_context_boosts
# ---------------------------------------------------------------------------

def apply_context_boosts(
    store: Any,
    query: str,
    row: Any,
    score: float,
    reason: str,
    mem_entity_objs: list[dict[str, Any]],
    directly_matched: bool,
    asks_brand: bool = False,
    asks_identity: bool = False,
    asks_eval: bool = False,
    query_topics: set[str] | None = None,
    query_identity_subjects: set[str] | None = None,
) -> tuple[float, str] | None:
    """Apply brand, identity, and evaluation boosts.

    Returns ``(new_score, new_reason)``, or ``None`` when the row should
    be skipped entirely (identity-subject mismatch).
    """
    if query_topics is None:
        query_topics = set()
    if query_identity_subjects is None:
        query_identity_subjects = set()

    # Identity scope check: skip rows that don't share the query's subject.
    mem_subjects = _identity_subjects(mem_entity_objs) if asks_identity else set()
    # Augment with CJK bigrams from the memory's raw_text that also appear
    # in the query — catches entity-extraction misses like pet names (咪咪).
    if asks_identity and query_identity_subjects and not (query_identity_subjects & mem_subjects):
        raw_text = row["raw_text"]
        for m in re.finditer(r"[\u4e00-\u9fff]{2,3}", raw_text):
            w = m.group()
            if w in query_identity_subjects:
                mem_subjects.add(w)
    if asks_identity and query_identity_subjects and not (query_identity_subjects & mem_subjects):
        return None

    # Brand/product boost.
    if asks_brand and query_topics:
        product_ents = [
            e["name"]
            for e in mem_entity_objs
            if e["kind"] == "object" and _looks_like_product(e["name"])
        ]
        topic_hit = query_topics & {e["name"] for e in mem_entity_objs if e["kind"] == "topic"}
        if product_ents and topic_hit:
            score += store.weights.brand_boost
            reason = (reason + "; brand-bearing product: " + ", ".join(sorted(product_ents))).lstrip("; ")

    # Occupation boost: when query asks about 职业/工作 and memory has an
    # occupation-like topic (教师/医生/...), boost that memory above peers.
    if query_topics & {"职业", "工作"}:
        mem_topics = {e["name"] for e in mem_entity_objs if e["kind"] == "topic"}
        if mem_topics & set(config.recall.occupation_topics):
            score += store.weights.occupation_boost
            reason = (reason + "; occupation topic match").lstrip("; ")

    # Identity base-attribute boost.
    if asks_identity:
        has_base_attr = any(
            e.get("salience_class") == "base_attr" for e in mem_entity_objs
        )
        scoped_identity = bool(query_identity_subjects & mem_subjects)
        if has_base_attr and scoped_identity:
            score += store.weights.identity_base_attr_boost
            reason = (reason + "; identity-question base-attr boost (matched subject)").lstrip("; ")

    # Evaluation sensory boost.
    if asks_eval and directly_matched:
        has_sensory = any(
            e.get("salience_class") == "sensory" for e in mem_entity_objs
        )
        if has_sensory:
            score += store.weights.eval_sensory_boost
            reason = (reason + "; evaluation-question sensory boost").lstrip("; ")

    return (score, reason)


# ---------------------------------------------------------------------------
# _build_scored_candidates
# ---------------------------------------------------------------------------

def build_scored_candidates(
    store: Any,
    query: str,
    terms: list[str],
    rows: list[Any],
    entity_map: dict[str, list[dict[str, Any]]],
    edge_bonus: dict[str, tuple[float, str]],
    fts_ids: set[str] | None = None,
    corrected_ids: set[str] | None = None,
    query_entities: set[str] | None = None,
    query_topics: set[str] | None = None,
    query_identity_subjects: set[str] | None = None,
    query_entity_objs: list[dict[str, Any]] | None = None,
) -> list[tuple[float, str, Any]]:
    """Main scoring loop: term matching + entity hop + composed boosts.

    Returns a list of ``(score, reason, row)`` tuples ready for sorting.
    """
    if corrected_ids is None:
        corrected_ids = set()
    if query_entities is None:
        query_entities = set()
    if query_entity_objs is None:
        query_entity_objs = []

    # Pre-compute query context once for the whole batch.
    # Use LLM intent classifier when available and enabled, fall back to regex.
    # L1 cache: reuse the classifier result if already computed for this query.
    intent_classifier = getattr(store, 'intent_classifier', None)
    if intent_classifier and intent_classifier.available and store.weights.intent_classifier_enabled:
        cache = getattr(store, '_intent_cache', None)
        if cache is None:
            cache = {}
            setattr(store, '_intent_cache', cache)
        if query in cache:
            intent = cache[query]
        else:
            intent = intent_classifier.classify(query)
            cache[query] = intent
        asks_brand = intent.brand > 0.5
        asks_identity = intent.identity > 0.5
        asks_eval = intent.eval > 0.5
    else:
        asks_brand = query_intent.asks_brand(query)
        asks_identity = query_intent.asks_identity(query)
        asks_eval = query_intent.asks_eval(query)

    scored: list[tuple[float, str, Any]] = []
    for row in rows:
        raw_text = row["raw_text"]
        summary = row["summary"] if row["summary"] else ""
        haystack = f"{raw_text} {summary}".lower()
        base = scoring.base_score(
            query, terms, haystack,
            store.weights, store.STOP_CHARS,
            store.REASON_MARKERS,
            ranker=store.ranker if hasattr(store, 'ranker') else None,
            store=store,
        )
        reason = scoring.match_reason(terms, haystack, base)
        score = base
        directly_matched = base > 0  # a real surface-token hit

        # FTS provenance boost.
        if fts_ids is not None and row["id"] in fts_ids:
            score += store.weights.fts_boost
            directly_matched = True
            reason = (reason + "; fts trigram candidate").lstrip("; ")

        # Entity/topic hop: reward memories sharing extracted entities.
        # Prefer the pre-computed ``query_entities`` set when the caller
        # supplied it (avoids O(n) re-filtering per row). Falls back to
        # deriving from query_entity_objs for callers that didn't pass it.
        mem_entity_objs = entity_map.get(row["id"], [])
        mem_entities = {
            e["name"] for e in mem_entity_objs
            if e.get("kind") != "cjk_ngram"
        }
        if query_entities:
            query_ents_filtered = query_entities
        else:
            query_ents_filtered = {
                e["name"] for e in query_entity_objs
                if e.get("kind") != "cjk_ngram"
            }
        shared = query_ents_filtered & mem_entities
        if shared:
            score += store.weights.shared_entity_multiplier * len(shared)
            directly_matched = True
            reason = (reason + "; shared entities: " + ", ".join(sorted(shared))).lstrip("; ")

        # Conflicting-person penalty: when the query asks about a specific
        # person (张三) but the memory is about a different person (妈妈),
        # apply a mild penalty so same-score ties break toward the right person.
        # ALSO: when persons conflict, strip the entity bonus from topic-type
        # entities — "老张开什么车" should NOT boost 小李's car via shared "车".
        # Exception: role words (同事/朋友) are abstract references, not specific
        # individuals — "同事的键盘" should still find 张三's HHKB.
        _ROLE_WORDS = set(config.entities.role_words)
        query_persons = {e["name"] for e in query_entity_objs if e["kind"] == "person"}
        mem_persons = {e["name"] for e in mem_entity_objs if e["kind"] == "person"}
        matching_person = bool(query_persons & mem_persons)
        conflicting_person = (
            query_persons and mem_persons
            and not matching_person
            and not query_persons.issubset(_ROLE_WORDS)
        )

        # Person-isolation scope: when query names a specific person,
        # all memories about other people (or with no person entity)
        # get penalized and have topic-based entity bonuses stripped.
        # This closes the gap where pronoun-only memories like
        # "她特别喜欢猫" escaped person checks entirely.
        person_scoped = bool(query_persons and not query_persons.issubset(_ROLE_WORDS))
        if person_scoped and not matching_person:
            # Isolation penalty is applied ONCE in Phase 2.6 post-filter
            # (after CE/ColBERT).  The scoring loop only strips entity
            # bonuses for wrong-person memories to avoid double-penalty.
            if shared:
                score -= store.weights.shared_entity_multiplier * len(shared)
                shared.clear()
            reason = (reason + "; person-scoped (deferred to Phase 2.6)").lstrip("; ")

        if matching_person:
            # Query specifically names this person — significant boost
            score += store.weights.person_match_boost
            reason = (reason + "; matching person: " + ", ".join(sorted(query_persons & mem_persons))).lstrip("; ")
        if conflicting_person:
            score -= store.weights.conflicting_person_penalty
            reason = (reason + "; conflicting person").lstrip("; ")
            # Recalculate shared entities WITHOUT topic-kind entities
            non_topic_shared = {
                e["name"] for e in query_entity_objs if e["kind"] != "topic"
            } & {e["name"] for e in mem_entity_objs if e["kind"] != "topic"}
            # Remove the topic-contributed entity bonus and replace with non-topic only
            if shared and non_topic_shared != shared:
                score -= store.weights.shared_entity_multiplier * len(shared)
                score += store.weights.shared_entity_multiplier * len(non_topic_shared)
                shared = non_topic_shared

        # Tie-break micro-bonus: when two memories would tie, prefer the one
        # that shares MORE entities with the query (not just a "+1.2 per").
        if shared:
            score += store.weights.entity_tie_break_bonus * len(shared)

        # Context-aware boosts (brand / identity / eval).
        result = apply_context_boosts(
            store, query, row, score, reason, mem_entity_objs, directly_matched,
            asks_brand=asks_brand, asks_identity=asks_identity,
            asks_eval=asks_eval, query_topics=query_topics or set(),
            query_identity_subjects=query_identity_subjects or set(),
        )
        if result is None:
            continue  # identity-subject mismatch → skip this row
        score, reason = result

        # Edge-association bonus (scaled to compete with direct term matches).
        bonus = edge_bonus.get(row["id"])
        if bonus is not None:
            score += bonus[0] * store.weights.edge_assoc_boost
            reason = (reason + "; " + bonus[1]).lstrip("; ")

        # Correction penalty.
        if row["id"] in corrected_ids:
            score *= store.weights.correction_penalty
            reason = (reason + "; user_corrected").lstrip("; ")

        # Salience decay for association-only memories.
        score, reason = apply_salience_decay(
            store, row, score, reason, mem_entity_objs, directly_matched,
        )

        if score > 0 or not terms:
            scored.append((score, reason, row))

    return scored


# ---------------------------------------------------------------------------
# _batch_evidence_refs / _batch_raw_refs
# ---------------------------------------------------------------------------

def batch_evidence_refs(conn: Any, memory_ids: list[str]) -> dict[str, list[str]]:
    """Single-batch fetch of evidence ids for multiple memory rows."""
    if not memory_ids:
        return {}
    out: dict[str, list[str]] = {mid: [] for mid in memory_ids}
    for i in range(0, len(memory_ids), candidates.SQLITE_IN_BATCH):
        batch = memory_ids[i : i + candidates.SQLITE_IN_BATCH]
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT id, memory_id FROM evidence WHERE memory_id IN ({placeholders}) ORDER BY id",
            tuple(batch),
        ).fetchall()
        for r in rows:
            out.setdefault(r["memory_id"], []).append(r["id"])
    return out


def batch_raw_refs(conn: Any, memory_ids: list[str]) -> dict[str, list[str]]:
    """Single-batch fetch of raw-log refs for multiple memory rows."""
    if not memory_ids:
        return {}
    out: dict[str, list[str]] = {mid: [] for mid in memory_ids}
    for i in range(0, len(memory_ids), candidates.SQLITE_IN_BATCH):
        batch = memory_ids[i : i + candidates.SQLITE_IN_BATCH]
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT raw_log_id, memory_id FROM distilled_memories WHERE memory_id IN ({placeholders}) ORDER BY id",
            tuple(batch),
        ).fetchall()
        for r in rows:
            out.setdefault(r["memory_id"], []).append(r["raw_log_id"])
    return out


# ---------------------------------------------------------------------------
# _build_recall_response
# ---------------------------------------------------------------------------
# _apply_reranker — shared helper for CE + ColBERT rerank blocks (M5)
# ---------------------------------------------------------------------------

def _apply_reranker(
    records: list[MemoryRecord],
    query: str,
    reranker: Any,
    reason_label: str,
    score_key: str,
) -> list[MemoryRecord]:
    """Apply a rerank pass and preserve records not in the reranked head.

    Both ``CrossEncoderReranker`` and ``ColBERTReranker`` share this shape:
      1. Build ``cands = [{text, score, id}, ...]``.
      2. Call ``reranker.rerank(query, cands)`` — returns list of dicts
         augmented with ``{score_key}: raw`` + ``score: blended``.
      3. Rebuild ``MemoryRecord`` list with the new scores + label attached
         to match_reason.
      4. Append records that the reranker's head cutoff excluded.

    Any exception silently falls back to the input ``records`` (graceful
    degradation — pipeline never crashes on a reranker failure).
    """
    try:
        cands = [{"text": r.raw_text, "score": r.score, "id": r.id} for r in records]
        reranked = reranker.rerank(query, cands)
        new_order: list[MemoryRecord] = []
        for c in reranked:
            src = next((r for r in records if r.id == c["id"]), None)
            if src is None:
                continue
            reason = src.match_reason
            if score_key in c and reason_label not in reason:
                reason = (reason + " · " + reason_label if reason else reason_label).strip(" ·")
            new_order.append(MemoryRecord(
                id=src.id, raw_text=src.raw_text, summary=src.summary,
                confidence=src.confidence, metadata=src.metadata,
                evidence_refs=src.evidence_refs,
                score=float(c.get("score", src.score)),
                match_reason=reason,
            ))
        seen = {r.id for r in new_order}
        for r in records:
            if r.id not in seen:
                new_order.append(r)
        logger.debug("%s reranked %d records", reason_label, len(cands))
        # N1: stage hit metric
        try:
            from . import metrics as _m
            stage = "cross_encoder" if reason_label == "cross-encoder" else "colbert"
            _m.stage_hit_ratio.labels(stage=stage, outcome="applied").inc()
        except Exception:
            pass
        return new_order
    except Exception as exc:
        logger.debug("%s rerank skipped: %s", reason_label, exc)
        return records


# ---------------------------------------------------------------------------
# _build_recall_response
# ---------------------------------------------------------------------------

def build_recall_response(
    store: Any,
    scored: list[tuple[float, str, Any]],
    top_k: int,
    query: str = "",
    namespace: str = "default",
    _deadline: float = float('inf'),
) -> list[MemoryRecord]:
    """Sort scored candidates, take top-k, and convert to MemoryRecords.

    Evidence refs are batch-fetched (one query each for evidence and
    raw_logs) to fix the N+1 query problem in ``_row_to_record``.

    When keyword results are below top_k, supplements with recent items
    (sorted by created_at DESC) so queries like "罗列对话"/"历史记录"
    don't return empty.
    """
    scored.sort(
        key=lambda item: (item[0], item[2]["created_at"], item[2]["id"]),
        reverse=True,
    )
    top = scored[:top_k]
    if not top:
        # No keyword/vector hits at all — fall back to vector + recent items.
        fb_records: list[MemoryRecord] = []
        existing_ids: set[str] = set()

        # Phase 2 (inline): vector search fallback for keywordless queries
        _vector_enabled = getattr(store, '_vector_enabled', True)
        embedding_engine = getattr(store, 'embedding_engine', None)
        vector_index = getattr(store, 'vector_index', None)
        if query and _vector_enabled and embedding_engine and vector_index:
            try:
                vec = embedding_engine.encode(query)
                if vec is not None:
                    vector_results = vector_index.search(vec, k=top_k * 2)
                    for mid, sim in vector_results:
                        if len(fb_records) >= top_k:
                            break
                        row = candidates.rows_by_ids(store.conn, [mid], namespace=namespace)
                        if row:
                            rw = row[0]
                            rw_summary = rw["summary"] if rw["summary"] else rw["raw_text"][:160]
                            confidence = rw["confidence"] if rw["confidence"] is not None else 1.0
                            fb_records.append(MemoryRecord(
                                id=mid, raw_text=rw["raw_text"], summary=rw_summary,
                                confidence=confidence,
                                metadata=parse_metadata(rw["metadata"]),
                                evidence_refs=[],
                                score=store.weights.vector_fallback_base + sim * store.weights.vector_fallback_sim_scale,
                                match_reason="vector search (semantic)",
                            ))
                            existing_ids.add(mid)
            except Exception as exc:
                logger.debug("Vector fallback skipped: %s", exc)

        # Fill remaining slots with recent items
        if len(fb_records) < top_k:
            recent_rows = store.conn.execute(
                "SELECT * FROM memories WHERE namespace = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (namespace, top_k * 2),
            ).fetchall()
            for row in recent_rows:
                rid = row["id"]
                if rid not in existing_ids and len(fb_records) < top_k:
                    fb_records.append(row_to_record(store.conn, row,
                        score=store.weights.recent_fallback_score,
                        match_reason="recent fallback (no keyword match)"))
                    existing_ids.add(rid)
        return fb_records

    mem_ids = [item[2]["id"] for item in top]
    evidence_map = batch_evidence_refs(store.conn, mem_ids)
    raw_refs_map = batch_raw_refs(store.conn, mem_ids)

    records: list[MemoryRecord] = []
    for score, reason, row in top:
        mid = row["id"]
        records.append(
            row_to_record(
                store.conn, row, score=score, match_reason=reason,
                evidence_refs=evidence_map.get(mid, []),
                raw_refs=raw_refs_map.get(mid, []),
            )
        )

    # ── Phase 2: Three-channel fusion (L1.1) ──
    # When channels module is available and store has vector_index,
    # run independent channels and fuse them.
    _three_channel_records: list[MemoryRecord] | None = None
    _channel_merged_ids: set[str] = set()
    try:
        from . import channels as _channels
        _vector_enabled_ch = getattr(store, '_vector_enabled', True)
        _vector_index_ch = getattr(store, 'vector_index', None)
        _embedding_engine_ch = getattr(store, 'embedding_engine', None)
        if _vector_enabled_ch and _vector_index_ch and _embedding_engine_ch and query:
            channel_results = _channels.run_channels(
                store, query, top_n=store.weights.channel_top_n)
            if channel_results:
                # channel_results is a list of per-channel lists:
                #   [(id, score, reason, metadata), ...]
                # Build result_lists for fusion: each channel becomes a list
                # of (id, score) tuples.
                result_lists: list[list[tuple[str, float]]] = []
                channel_weights: list[float] = []
                for ch_name, ch_list in channel_results:
                    result_lists.append([(cid, cscore) for cid, cscore, _, _ in ch_list])
                    if ch_name == "fts":
                        channel_weights.append(store.weights.channel_fts_weight)
                    elif ch_name == "vector":
                        channel_weights.append(store.weights.channel_vector_weight)
                    elif ch_name == "hyde":
                        channel_weights.append(store.weights.channel_hyde_weight)
                    else:
                        channel_weights.append(1.0)

                if result_lists:
                    from ..fusion import fuse as _fuse
                    fused = _fuse(
                        store.weights.fusion_method,
                        result_lists,
                        weights=channel_weights,
                        k=store.weights.fusion_k,
                    )
                    # Merge channel-fused results with existing keyword-scored records
                    existing_ids = {r.id for r in records}
                    _channel_merged_ids = {mid for mid, _ in fused}
                    _three_channel_records = []
                    # Boost keyword records that also appear in channel fusion
                    for r in records:
                        ch_score = next(
                            (s for mid, s in fused if mid == r.id), None)
                        if ch_score is not None:
                            new_score = r.score + ch_score * 10.0
                            new_reason = (r.match_reason + " · channel-boost"
                                          if r.match_reason else "channel-boost").strip("·")
                            _three_channel_records.append(MemoryRecord(
                                id=r.id, raw_text=r.raw_text, summary=r.summary,
                                confidence=r.confidence, metadata=r.metadata,
                                evidence_refs=r.evidence_refs,
                                score=new_score, match_reason=new_reason,
                            ))
                        else:
                            _three_channel_records.append(r)
                    # Add new candidates from channels not already in keyword results
                    for mid, fused_score in fused:
                        if mid not in existing_ids and len(_three_channel_records) < top_k * 2:
                            row_list = candidates.rows_by_ids(
                                store.conn, [mid], namespace=namespace)
                            if row_list:
                                rw = row_list[0]
                                rw_summary = (
                                    rw["summary"] if rw["summary"]
                                    else rw["raw_text"][:160]
                                )
                                _three_channel_records.append(MemoryRecord(
                                    id=mid, raw_text=rw["raw_text"],
                                    summary=rw_summary,
                                    confidence=rw["confidence"]
                                    if rw["confidence"] is not None else 1.0,
                                    metadata=parse_metadata(rw["metadata"]),
                                    evidence_refs=[],
                                    score=fused_score * 60.0,
                                    match_reason="three-channel fusion",
                                ))
                    _three_channel_records.sort(key=lambda x: x.score, reverse=True)
    except ImportError:
        pass  # channels module not yet available
    except Exception as exc:
        logger.debug("Three-channel fusion skipped: %s", exc)

    # ── Phase 2: Vector-search fusion ──
    # If three-channel fusion produced results, use those and skip the
    # standard vector fusion path.  Otherwise fall through to the existing
    # keyword+vector+hyde RRF path.
    if _three_channel_records is not None:
        records = _three_channel_records
        _skip_rrf = True
    else:
        _skip_rrf = False

    # Pre-compute the query vector once — used by RRF fusion (Phase 2)
    # and post-CE fallback (Phase 3.5) below.  Skipped when three-channel
    # fusion already produced results (avoids double HyDE/vector re-run).
    if _skip_rrf:
        _query_vec: Any = None
    else:
        _query_vec: Any = None
    _vector_enabled = getattr(store, '_vector_enabled', True)
    embedding_engine = getattr(store, 'embedding_engine', None)
    vector_index = getattr(store, 'vector_index', None)
    if query and _vector_enabled and embedding_engine and vector_index:
        # M13: reuse cached vector from multi-variant batch encode if present
        _cache = getattr(store, '_query_vec_cache', None)
        if _cache is not None and query in _cache:
            _query_vec = _cache[query]
        else:
            try:
                _query_vec = embedding_engine.encode(query)
            except Exception:
                _query_vec = None

    # Track hyde/vector ids for hyde-only de-weighting before CE.
    _hyde_ids: set[str] = set()
    _vector_ids: set[str] = set()
    if _query_vec is not None:
        try:
            vec = _query_vec
            if vec is not None:
                vector_results = vector_index.search(vec, k=top_k * 4)
                vector_list = [(mid, score) for mid, score in vector_results]
                keyword_list = [(r.id, r.score) for r in records if r.score > 0]

                # ── HyDE result list (Phase 2 of rerank_and_hyde.md).
                hyde_list: list[tuple[str, float]] = []
                hyde_result = None
                hyde_skipped = False
                hyde = getattr(store, 'hyde', None)
                if hyde and getattr(hyde, "available", False) \
                        and store.weights.hyde_rrf_weight > 0 \
                        and not _check_deadline(_deadline, "HyDE"):
                    try:
                        hyde_list, hyde_result = hyde.expand_and_recall(
                            query, embedding_engine, vector_index,
                            k=store.weights.hyde_top_k,
                        )
                        if hyde_result:
                            if hyde_result.source == "skipped":
                                hyde_skipped = True
                            if hyde_result.hypotheses:
                                logger.debug("HyDE: %d hypotheses, %d candidates, source=%s",
                                             len(hyde_result.hypotheses), len(hyde_list),
                                             hyde_result.source)
                    except Exception as exc:
                        logger.debug("HyDE skipped: %s", exc)
                        hyde_skipped = True

                hyde_ids: set[str] = {mid for mid, _ in hyde_list}
                vector_ids: set[str] = {mid for mid, _ in vector_list}
                _hyde_ids = hyde_ids
                _vector_ids = vector_ids

                result_lists: list[list[tuple[str, float]]] = [keyword_list, vector_list]
                rrf_weights: list[float] = [
                    store.weights.rrf_keyword_weight,
                    store.weights.rrf_vector_weight,
                ]
                if hyde_list:
                    result_lists.append(hyde_list)
                    rrf_weights.append(float(store.weights.hyde_rrf_weight))

                merged = reciprocal_rank_fusion(
                    result_lists, k=60, weights=rrf_weights)
                rrf_scores = dict(merged)
                # Build new list (MemoryRecord is frozen)
                new_records: list[MemoryRecord] = []
                existing_ids = {r.id for r in records}
                for r in records:
                    # Blend: keep keyword score as base, boost if also found by vector
                    rrf_s = rrf_scores.get(r.id, 0)
                    if rrf_s:
                        new_score = r.score + rrf_s * store.weights.rrf_score_boost
                    else:
                        new_score = r.score
                    reason = r.match_reason
                    if r.id in hyde_ids and "HyDE" not in reason:
                        reason = (reason + " · HyDE" if reason else "HyDE").strip(" ·")
                    new_records.append(MemoryRecord(
                        id=r.id, raw_text=r.raw_text, summary=r.summary,
                        confidence=r.confidence, metadata=r.metadata,
                        evidence_refs=r.evidence_refs,
                        score=new_score, match_reason=reason,
                    ))
                for mid, rrf_score in merged:
                    if mid not in existing_ids and len(new_records) < top_k:
                        row_list = candidates.rows_by_ids(store.conn, [mid], namespace=namespace)
                        if row_list:
                            rw = row_list[0]
                            rw_summary = rw["summary"] if rw["summary"] else rw["raw_text"][:160]
                            reason = "HyDE (hypothetical)" if mid in hyde_ids else "vector search (semantic)"
                            # Reduce confidence for vector-only hits when HyDE
                            # was skipped (negative query → wrong direction).
                            base_score = max(store.weights.vector_fallback_base,
                                             rrf_score * store.weights.rrf_new_insert_score_scale)
                            if hyde_skipped and mid not in hyde_ids:
                                base_score *= store.weights.hyde_skip_vector_penalty
                            new_records.append(MemoryRecord(
                                id=mid, raw_text=rw["raw_text"], summary=rw_summary,
                                confidence=rw["confidence"] if rw["confidence"] is not None else 1.0,
                                metadata=parse_metadata(rw["metadata"]),
                                evidence_refs=[], score=base_score,
                                match_reason=reason,
                            ))
                new_records.sort(key=lambda x: x.score, reverse=True)
                records = new_records
        except Exception as exc:
            logger.debug("Vector search skipped: %s", exc)

    # ── Phase 2.35: HyDE-only candidate de-weight ──
    # Candidates found ONLY by HyDE (not in keyword or vector) are lower
    # confidence.  De-weight so CE doesn't accidentally promote a
    # hyde-only match into top-1 and poison rejection accuracy.
    #
    # Prefer id-set arithmetic over match_reason string matching: keyword
    # sources (base scoring pass) are the records with positive fusion
    # score BEFORE HyDE-only insertion; those we subtract from _hyde_ids.
    if _hyde_ids and records:
        _keyword_scored_ids = {r.id for r in records
                               if r.id not in _hyde_ids
                               and r.id not in _vector_ids}
        hyde_only = _hyde_ids - _vector_ids - _keyword_scored_ids
        if hyde_only:
            records = [
                MemoryRecord(
                    id=r.id, raw_text=r.raw_text, summary=r.summary,
                    confidence=r.confidence, metadata=r.metadata,
                    evidence_refs=r.evidence_refs,
                    score=r.score * store.weights.hyde_only_penalty,
                    match_reason=r.match_reason + " · hyde-only deweight",
                ) if r.id in hyde_only else r
                for r in records
            ]

    # ── Phase 2.4: Cross-encoder rerank (skip if deadline expired) ──
    # Save pre-CE signal so Phase 3.5 can decide whether to fall back to
    # vector search.  M3 fix: instead of ``r.score > 0`` (which context
    # boosts inflate — a memory with 0 base but +2.0 brand boost would
    # falsely register as "meaningful"), check for real hit markers in
    # match_reason strings: base term matches, FTS hits, entity matches,
    # channel/CE/ColBERT rerank contributions.
    _MEANINGFUL_MARKERS = (
        "matched terms:", "fts trigram candidate",
        "shared entities:", "matching person:",
        "channel-boost", "rrf-fused",
        "vector search (semantic)",
        "cross-encoder", "colbert",
    )
    _had_meaningful_hits = bool(records) and any(
        any(m in (r.match_reason or "") for m in _MEANINGFUL_MARKERS)
        for r in records
    )
    cross_encoder = getattr(store, 'cross_encoder', None)
    if cross_encoder and getattr(cross_encoder, "available", False) \
            and query and len(records) > 1 \
            and not _check_deadline(_deadline, "cross-encoder"):
        records = _apply_reranker(
            records, query, cross_encoder, "cross-encoder", "ce_score",
        )

    # ── Phase 2.45: ColBERT late-interaction rerank (skip if deadline expired) ──
    colbert = getattr(store, 'colbert_reranker', None)
    if colbert and getattr(colbert, "available", False) and records and len(records) > 1 \
            and not _check_deadline(_deadline, "colbert"):
        records = _apply_reranker(
            records, query, colbert, "colbert", "colbert_score",
        )

    # ── Phase 2.5: LLM reranker (semantic re-rank, skip if deadline expired) ──
    reranker = getattr(store, 'reranker', None)
    if reranker and reranker.available and records and len(records) > 1 \
            and not _check_deadline(_deadline, "LLM-reranker"):
        try:
            candidates_list = [{"text": r.raw_text, "score": r.score} for r in records]
            reranked = reranker.rerank(query, candidates_list)
            # Rebuild MemoryRecord with new scores
            rrmap = {rr.get("text", ""): rr.get("score", 0) for rr in reranked}
            records = [MemoryRecord(
                id=r.id, raw_text=r.raw_text, summary=r.summary,
                confidence=r.confidence, metadata=r.metadata,
                evidence_refs=r.evidence_refs,
                score=rrmap.get(r.raw_text, r.score),
                match_reason=r.match_reason,
            ) for r in records]
            records.sort(key=lambda x: x.score, reverse=True)
            logger.debug("Reranker applied to %d records", len(records))
        except Exception as exc:
            logger.debug("Reranker skipped: %s", exc)

    # ── Phase 2.6: Person-isolation post-filter ──
    # Runs AFTER CE and LLM reranker so their scores don't undo the isolation.
    # When query names a specific person, heavily penalize memories about
    # other people. Uses DB entity_map (covers pronoun-only memories like
    # "她特别喜欢猫" that have no text-level person entity).
    #
    # Query-side person augmentation: when the query has NO explicit person
    # entity ("谁在华为做过"/"他送我的东西"), scan the top-ranked memory's
    # DB entity_map to find candidate persons referenced by the query context.
    from ..config import config as _cfg
    _ROLE_WORDS_POST = set(_cfg.entities.role_words)
    from ..entities import extract_entities as _extract_entities
    _q_persons_post = {e["name"] for e in _extract_entities(query) if e.get("kind") == "person" and e.get("kind") != "cjk_ngram"}

    # Augment: when query lacks a person entity but the query is asking about
    # a person ("谁"/"他"/"她" questions), look at top-ranked memories' DB
    # entity_map to find candidate persons that might be the query's target.
    # This closes the gap for pronoun-only or "谁"-type isolation failures.
    if records and not _q_persons_post:
        _query_has_person_pronoun = any(
            w in query for w in ("谁", "他", "她", "哪位", "哪个人")
        )
        if _query_has_person_pronoun:
            _top_ids = [r.id for r in records[:3]]
            _top_entity_map = candidates.entities_for_memories(store.conn, _top_ids)
            for _mid in _top_ids:
                _top_ents = _top_entity_map.get(_mid, [])
                for e in _top_ents:
                    if e.get("kind") == "person" and e.get("name") not in _ROLE_WORDS_POST:
                        _q_persons_post.add(e["name"])
            # Deduplicate and limit to 3 persons max
            _q_persons_post = set(list(_q_persons_post)[:3])

    if records and _q_persons_post and not _q_persons_post.issubset(_ROLE_WORDS_POST):
        _all_post_ids = [r.id for r in records]
        _post_entity_map = candidates.entities_for_memories(store.conn, _all_post_ids)
        for i, r in enumerate(records):
            _post_ents = _post_entity_map.get(r.id, [])
            _mem_persons_post = {e["name"] for e in _post_ents if e.get("kind") == "person"}
            if not _mem_persons_post:
                _txt_ents_fb = _extract_entities(r.raw_text)
                _mem_persons_post = {e["name"] for e in _txt_ents_fb if e.get("kind") == "person" and e.get("kind") != "cjk_ngram"}
            if _q_persons_post & _mem_persons_post:
                continue  # matching person — no penalty
            if _mem_persons_post:
                # Has a DIFFERENT person entity → clear isolation violation
                records[i] = MemoryRecord(
                    id=r.id, raw_text=r.raw_text, summary=r.summary,
                    confidence=r.confidence, metadata=r.metadata,
                    evidence_refs=r.evidence_refs,
                    score=r.score * store.weights.person_isolation_scope_weight
                          - store.weights.person_isolation_subtract,
                    match_reason=r.match_reason + " · person-isolation (different person)",
                )
            else:
                # No person entity at all → likely pronoun (他/她) coreference,
                # NOT a clear isolation violation. Apply milder penalty.
                records[i] = MemoryRecord(
                    id=r.id, raw_text=r.raw_text, summary=r.summary,
                    confidence=r.confidence, metadata=r.metadata,
                    evidence_refs=r.evidence_refs,
                    score=r.score * store.weights.person_isolation_noentity_weight
                          - store.weights.person_isolation_noentity_subtract,
                    match_reason=r.match_reason + " · person-isolation (no entity)",
                )
        records.sort(key=lambda x: x.score, reverse=True)

    # ── Phase 3: Record access for forgetting engine ──
    candidates.record_access(store.conn, [r.id for r in records])
    store._apply_decay(records)

    # ── Phase 3.5: Vector fallback for queries with no keyword matches ──
    # Uses pre-CE flag (_had_meaningful_hits) so CE/LLM normalisation
    # doesn't cause the threshold to misfire (all scores become [0,1]).
    if (not records or not _had_meaningful_hits) \
            and _query_vec is not None:
        try:
            vec = _query_vec
            if vec is not None:
                vector_results = vector_index.search(vec, k=top_k * 2)
                existing_ids = {r.id for r in records}
                for mid, sim in vector_results:
                    if mid not in existing_ids and len(records) < top_k:
                        row_list = candidates.rows_by_ids(store.conn, [mid], namespace=namespace)
                        if row_list:
                            rw = row_list[0]
                            rw_summary = rw["summary"] if rw["summary"] else rw["raw_text"][:160]
                            confidence = rw["confidence"] if rw["confidence"] is not None else 1.0
                            records.append(MemoryRecord(
                                id=mid, raw_text=rw["raw_text"], summary=rw_summary,
                                confidence=confidence,
                                metadata=parse_metadata(rw["metadata"]),
                                evidence_refs=[],
                                score=store.weights.vector_fallback_base
                                      + sim * store.weights.vector_fallback_sim_scale,
                                match_reason="vector search (semantic)",
                            ))
                            existing_ids.add(mid)
        except Exception as exc:
            logger.debug("Vector fallback skipped: %s", exc)

    # ── Phase 4: Recent fallback ──
    # When keyword/vector recall returns fewer than top_k results,
    # supplement with recently-saved items. This handles meta-queries
    # like "罗列最近对话" where no content keyword matches exist.
    if len(records) < top_k:
        existing_ids = {r.id for r in records}
        recent_limit = top_k - len(records)
        recent_rows = store.conn.execute(
            "SELECT * FROM memories WHERE namespace = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (namespace, recent_limit + top_k),  # extra buffer for dedup
        ).fetchall()
        for row in recent_rows:
            rid = row["id"]
            if rid not in existing_ids and len(records) < top_k:
                records.append(
                    row_to_record(store.conn, row, score=store.weights.recent_fallback_score,
                        match_reason="recent fallback"))
                existing_ids.add(rid)

    return records


# ---------------------------------------------------------------------------
# _recall_single
# ---------------------------------------------------------------------------

def recall_single(
    store: Any,
    query: str,
    terms: list[str],
    query_entity_objs: list[dict[str, Any]],
    namespace: str = "default",
    _deadline: float = float('inf'),
) -> list[MemoryRecord]:
    """Run the standard recall pipeline for a single query variant.

    This is a self-contained recall path that extracts entities,
    performs FTS + edge expansion + scoring, and returns scored records.
    Used by the multi-variant RRF fusion path in ``recall()``.
    """
    query_entities = {e["name"] for e in query_entity_objs}
    query_topics = {e["name"] for e in query_entity_objs if e["kind"] == "topic"}
    query_identity_subjects = _identity_subjects(query_entity_objs)

    if _asks_identity(store, query) and not query_identity_subjects:
        for r in store.conn.execute(
            "SELECT name, kind FROM entities"
        ).fetchall():
            if r["name"] in query:
                query_identity_subjects.add(r["name"])
        if not query_identity_subjects and len(query) <= 8:
            _STOP_CJK = {"什么", "怎么", "这个", "那个", "哪个", "因为",
                         "所以", "还是", "但是", "虽然", "如果", "可以",
                         "没有", "不是", "时候", "几岁", "多大", "多少"}
            for m in re.finditer(r"[\u4e00-\u9fff]{2,3}", query):
                w = m.group()
                if w not in _STOP_CJK:
                    query_identity_subjects.add(w)

    corrected_ids = _get_corrected_ids(store.conn)
    fts_ids = candidates.fts_candidates(store.conn, getattr(store, '_fts_enabled', True),
                                        query, terms, namespace)
    rows = candidates.memory_rows(store.conn, store.weights, fts_ids, namespace)
    entity_map = candidates.entities_for_memories(store.conn, [r["id"] for r in rows])
    edge_bonus = graph.edge_expansion(
        store.conn, weights=store.weights,
        query=query, terms=terms, rows=rows,
        entity_map=entity_map, namespace=namespace,
    )
    missing_edge_ids = sorted(set(edge_bonus) - {r["id"] for r in rows})
    if missing_edge_ids:
        rows.extend(candidates.rows_by_ids(store.conn, missing_edge_ids, namespace=namespace))
        entity_map = candidates.entities_for_memories(store.conn, [r["id"] for r in rows])

    scored = build_scored_candidates(
        store, query=query, terms=terms, rows=rows,
        entity_map=entity_map, edge_bonus=edge_bonus,
        fts_ids=fts_ids, corrected_ids=corrected_ids,
        query_entities=query_entities, query_topics=query_topics,
        query_identity_subjects=query_identity_subjects,
        query_entity_objs=query_entity_objs,
    )
    # Return a larger top-k for RRF to fuse; the final truncation happens
    # after fusion in the caller.
    return build_recall_response(store, scored, max(50, len(scored)), query, namespace=namespace,
                                 _deadline=_deadline)


# ---------------------------------------------------------------------------
# recall (top-level pipeline)
# ---------------------------------------------------------------------------

def recall(
    store: Any,
    query: str,
    top_k: int = 5,
    namespace: str = "default",
) -> list[MemoryRecord]:
    """Recall top-k memories by a weighted composite score.

    The pipeline:
    1. Tokenise the query and extract entities.
    2. Optionally filter candidates through the FTS5 trigram index.
    3. Expand the candidate set with one-hop edge associations.
    4. Score every candidate through the composable pipeline.
    5. Sort, truncate, and convert to MemoryRecord with batched refs.
    """
    # N1: recall_latency_ms observation
    import time as _time
    from . import metrics as _metrics
    _t0 = _time.perf_counter()
    _deadline = _t0 + (store.weights.recall_deadline_ms / 1000.0) if store.weights.recall_deadline_ms > 0 else float('inf')
    try:
        return _recall_impl(store, query, top_k, namespace, _deadline)
    finally:
        _elapsed_ms = (_time.perf_counter() - _t0) * 1000.0
        try:
            _metrics.recall_latency_ms.observe(_elapsed_ms)
        except Exception:
            pass


def _check_deadline(deadline: float, phase: str) -> bool:
    """Return True if the deadline has passed. Log the skipped phase."""
    if time.perf_counter() >= deadline:
        logger.debug("Recall deadline exceeded, skipping phase: %s", phase)
        return True
    return False


def _recall_impl(
    store: Any,
    query: str,
    top_k: int = 5,
    namespace: str = "default",
    _deadline: float = float('inf'),
) -> list[MemoryRecord]:
    """Actual recall implementation, wrapped by recall() for metrics."""
    # ── Phase 2: Query Expansion ───────────────────────────────────
    from ..query_expansion import QueryExpander

    query_expander = getattr(store, 'query_expander', None)
    if query_expander is not None:
        eq = query_expander.expand(query, async_llm=False)

        # 1. Merge synonyms into extra terms for token matching.
        extra_terms: list[str] = []
        for synonyms in eq.synonyms.values():
            extra_terms.extend(synonyms)

        # 2. Extract entities for the base query.
        query_entity_objs = [
            e for e in extract_entities(query)
            if e.get("kind") != "cjk_ngram"
        ]

        # 3. Merge LLM extra entities into query entities.
        existing = {(e["name"], e.get("kind", "")) for e in query_entity_objs}
        for ent in eq.extra_entities:
            key = (ent.get("name", ""), ent.get("kind", ""))
            if key[0] and key not in existing:
                query_entity_objs.append(ent)
                existing.add(key)

        # 4. If there are variants, run multi-path recall → RRF fusion.
        has_multi_query = (
            store.weights.multi_query_enabled
            and query_expander is not None
        )
        # Collect multi-query variants if enabled (separate from LLM rewrite).
        mq_variants: list[str] = []
        if has_multi_query:
            mq_variants = query_expander.expand_multi_query(
                query, num_variants=store.weights.multi_query_num_variants,
            )

        # Build the full variant set: LLM rewrite variants + multi-query variants.
        all_variants = list(dict.fromkeys(
            list(eq.variants) + mq_variants
        ))  # dict.fromkeys preserves order and deduplicates

        if all_variants:
            all_results: list[list[tuple[str, float]]] = []

            # M13: pre-encode query + all variants in a single batch so
            # the embedding backend can amortize model warmup / GPU dispatch.
            # Individual recall_single() calls no longer trigger separate
            # encode() invocations — they consume the pre-computed vectors.
            embedding_engine = getattr(store, 'embedding_engine', None)
            _prefetched_vecs: dict[str, Any] = {}
            if embedding_engine is not None and getattr(embedding_engine, 'available', False):
                _all_query_texts = [query] + all_variants
                try:
                    _batch = embedding_engine.encode_batch(_all_query_texts)
                    if _batch is not None:
                        for _q_text, _vec in zip(_all_query_texts, _batch):
                            _prefetched_vecs[_q_text] = _vec
                        # Cache on store so recall_single/build_recall_response
                        # can pick them up without re-encoding.
                        setattr(store, '_query_vec_cache', _prefetched_vecs)
                except Exception as exc:
                    logger.debug("Multi-variant batch encode skipped: %s", exc)

            # Primary: recall with original query + expanded terms/entities.
            primary_terms = list(dict.fromkeys(scoring.terms(query) + extra_terms))
            primary = recall_single(
                store, query, primary_terms, query_entity_objs, namespace,
                _deadline=_deadline,
            )
            all_results.append([(r.id, r.score) for r in primary])

            # Each variant gets its own recall path.
            for variant in all_variants:
                v_terms = list(dict.fromkeys(scoring.terms(variant)))
                v_entities = [
                    e for e in extract_entities(variant)
                    if e.get("kind") != "cjk_ngram"
                ]
                # Also merge LLM extra entities into variant entities.
                v_existing = {(e["name"], e.get("kind", "")) for e in v_entities}
                for ent in eq.extra_entities:
                    key = (ent.get("name", ""), ent.get("kind", ""))
                    if key[0] and key not in v_existing:
                        v_entities.append(ent)
                        v_existing.add(key)
                v_results = recall_single(
                    store, variant, v_terms, v_entities, namespace,
                    _deadline=_deadline,
                )
                all_results.append([(r.id, r.score) for r in v_results])

            # Clear the per-recall cache to avoid stale reuse across calls
            if _prefetched_vecs:
                try:
                    delattr(store, '_query_vec_cache')
                except AttributeError:
                    pass

            # RRF fuse all recall paths.
            # Primary results get weight 1.0; variant results get lower weight.
            fusion_weights: list[float] = [1.0]  # primary weight
            for _ in all_variants:
                # LLM rewrite variants: weight 0.8; multi-query variants: lower weight
                fusion_weights.append(store.weights.multi_query_fusion_weight)

            merged = reciprocal_rank_fusion(all_results, k=60, weights=fusion_weights)

            # Sort merged results by RRF score.
            scored: list[tuple[float, str, Any]] = []
            for mem_id, rrf_score in merged:
                row = candidates.row_by_id(store.conn, mem_id, namespace=namespace)
                if row is not None:
                    # Boost RRF scores to comparable range
                    scored.append((rrf_score * 10, "rrf-fused", row))

            scored.sort(key=lambda x: x[0], reverse=True)
            return build_recall_response(store, scored, top_k, query, namespace=namespace,
                                         _deadline=_deadline)
        else:
            # No variants: use expanded terms/entities with standard pipeline.
            terms = list(dict.fromkeys(scoring.terms(query) + extra_terms))
            # Fall through to standard pipeline below.
    else:
        terms = scoring.terms(query)
        query_entity_objs = [
            e for e in extract_entities(query)
            if e.get("kind") != "cjk_ngram"
        ]

    # LLM query augmentation: supplement rule-based entities with LLM-
    # extracted ones for better recall (e.g., unlisted brands, topics).
    from ..llm_extractor import extract_entities_llm

    llm_query_extract = getattr(store, 'llm_query_extract', False)
    llm_extractor = getattr(store, 'llm_extractor', None)
    if llm_query_extract and llm_extractor is not None and llm_extractor.available:
        llm_ents = extract_entities_llm(query)
        existing = {(e["name"], e["kind"]) for e in query_entity_objs}
        for ent in llm_ents:
            if (ent["name"], ent["kind"]) not in existing:
                query_entity_objs.append(ent)

    query_entities = {e["name"] for e in query_entity_objs}
    query_topics = {e["name"] for e in query_entity_objs if e["kind"] == "topic"}
    query_identity_subjects = _identity_subjects(query_entity_objs)
    # When an identity question asks about a subject the rule-based
    # extractor missed (pet names like 咪咪, nicknames, rare objects),
    # fall back to scanning the DB for entity names that appear in the
    # query text.  Without this, "咪咪几岁了" would leak 妈妈's age
    # because no query entity is extracted → scope check never fires.
    if _asks_identity(store, query) and not query_identity_subjects:
        for r in store.conn.execute(
            "SELECT name, kind FROM entities"
        ).fetchall():
            if r["name"] in query:
                query_identity_subjects.add(r["name"])
        # If STILL no subjects found AND the query is short (likely a
        # named-entity identity question like "咪咪几岁了"), extract CJK
        # bigrams as fallback subjects.  Skip for long technical queries
        # ("数据库迁移方案是谁做的？") where bigrams would be noise.
        if not query_identity_subjects and len(query) <= 8:
            _STOP_CJK = {"什么", "怎么", "这个", "那个", "哪个", "因为",
                         "所以", "还是", "但是", "虽然", "如果", "可以",
                         "没有", "不是", "时候", "几岁", "多大", "多少"}
            for m in re.finditer(r"[\u4e00-\u9fff]{2,3}", query):
                w = m.group()
                if w not in _STOP_CJK:
                    query_identity_subjects.add(w)

    # Corrections: fetch corrected memory ids so we can down-weight them
    # during scoring. We do NOT hard-delete; the original text and correction
    # history stay in the corrections table for audit.
    corrected_ids = _get_corrected_ids(store.conn)

    # f3: FTS5 trigram candidate selection.
    fts_ids = candidates.fts_candidates(store.conn, getattr(store, '_fts_enabled', True),
                                        query, terms, namespace)
    rows = candidates.memory_rows(store.conn, store.weights, fts_ids, namespace)
    entity_map = candidates.entities_for_memories(store.conn, [r["id"] for r in rows])

    # f2: one-hop edge expansion.
    edge_bonus = graph.edge_expansion(
        store.conn, weights=store.weights,
        query=query, terms=terms, rows=rows,
        entity_map=entity_map, namespace=namespace,
    )
    missing_edge_ids = sorted(set(edge_bonus) - {r["id"] for r in rows})
    if missing_edge_ids:
        # Namespace must be threaded here — edge_bonus may include memory
        # ids whose entities are cross-namespace duplicates; without the
        # filter, hydration would leak rows from other namespaces.
        rows.extend(candidates.rows_by_ids(store.conn, missing_edge_ids, namespace=namespace))
        entity_map = candidates.entities_for_memories(store.conn, [r["id"] for r in rows])

    scored = build_scored_candidates(
        store, query=query, terms=terms, rows=rows,
        entity_map=entity_map, edge_bonus=edge_bonus,
        fts_ids=fts_ids, corrected_ids=corrected_ids,
        query_entities=query_entities, query_topics=query_topics,
        query_identity_subjects=query_identity_subjects,
        query_entity_objs=query_entity_objs,
    )
    return build_recall_response(store, scored, top_k, query, namespace=namespace,
                                 _deadline=_deadline)


# ---------------------------------------------------------------------------
# _get_corrected_ids (was MemoryStore._get_corrected_ids)
# ---------------------------------------------------------------------------

def _get_corrected_ids(conn: Any) -> set[str]:
    """Return the set of memory ids that have been user-corrected.

    Corrections are traced through the ``corrections`` table; the original
    memory stays in ``memories`` so the evidence chain is never severed.
    """
    rows = conn.execute("SELECT DISTINCT target_id FROM corrections").fetchall()
    return {r["target_id"] for r in rows}
