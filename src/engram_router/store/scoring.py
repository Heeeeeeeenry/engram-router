"""Scoring helpers for EngramRouter recall.

Extracted from store.py (Step 3): RecallWeights, term extraction, token
weighting, base/semantic scoring, and match-reason generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..config import config


# ── RecallWeights (centralised recall scoring knobs) ──────────────────────

@dataclass(frozen=True)
class RecallWeights:
    """Centralised recall scoring weights, injectable via MemoryStore.__init__.

    Every number that was previously a hard-coded literal lives here, with its
    current value as the default, so constructing ``RecallWeights()`` is
    behaviourally identical to the old hard-coded world.
    """

    # ── token scoring (_term_weight) ──────────────────────────────────────
    ascii_base: float = 4.0
    ascii_per_char_cap: int = 6
    ascii_per_char: float = 0.5
    cjk_multi_base: float = 2.0
    cjk_multi_per_char: float = 0.5
    stop_char_weight: float = 0.05
    single_cjk_weight: float = 0.4

    # ── _score semantic boosts ────────────────────────────────────────────
    colleague_boost: float = 1.0
    reason_marker_boost: float = 1.5

    # ── recall pipeline (_build_scored_candidates) ────────────────────────
    fts_boost: float = 0.1
    shared_entity_multiplier: float = 1.2
    conflicting_person_penalty: float = 2.2
    person_match_boost: float = 2.0
    entity_tie_break_bonus: float = 0.01

    # ── context boosts (_apply_context_boosts) ────────────────────────────
    brand_boost: float = 2.0
    occupation_boost: float = 1.5
    identity_base_attr_boost: float = 2.0
    eval_sensory_boost: float = 1.5

    # ── correction ────────────────────────────────────────────────────────
    correction_penalty: float = 0.3

    # ── spreading activation (was DEFAULT_MAX_HOPS / DEFAULT_DECAY / _ACT_THRESHOLD)
    max_recall_hops: int = 5
    """Safety cap for BFS depth. Actual termination is dynamic (activation × relevance)."""
    recall_decay: float = 0.7
    """Per-hop activation decay factor."""
    activation_threshold: float = 0.03
    """Minimum activation to keep expanding a branch."""

    # ── dynamic multi-hop relevance (SAG-inspired query-time relevance gate) ──
    relevance_decay: float = 0.6
    """Per-hop relevance decay. Higher = relevance decays slower."""
    relevance_threshold: float = 0.15
    """Minimum query relevance to keep expanding. Below this → prune branch."""
    relevance_entity_kind_weights: dict[str, float] | None = None
    """Entity kind → relevance multiplier. Defaults built-in if None."""
    edge_assoc_boost: float = 5.0
    """Multiplier for edge-association bonus in final scoring.
    Edge expansion produces small raw activation values (0.01-0.20).
    This scales them to compete with direct term matches (0.05-0.50 per term)."""

    # ── associative reach (_apply_salience_decay) ─────────────────────────
    assoc_reach_base_attr: float = 0.15
    assoc_reach_constraint: float = 0.6
    assoc_reach_decision: float = 0.7
    assoc_reach_sensory: float = 1.0
    assoc_reach_event: float = 1.0

    # ── scale protection ──────────────────────────────────────────────────
    full_scan_limit: int = 2000
    """Cap rows loaded into Python when FTS5 returns no candidates.
    At 10K+ memories the full scan would OOM; 2000 rows is a generous
    budget for the Python ranker to find top_k from."""

    # ── cross-encoder reranker (Phase 1 of rerank_and_hyde.md) ────────────
    ce_enabled: bool = True
    """Enable cross-encoder reranking after fusion, before context boosts.
    Falls back gracefully if sentence-transformers is missing or the model
    can't be loaded — recall() continues without CE."""
    ce_model: str = "bge-v2-m3"
    """CrossEncoderReranker model key. See cross_encoder._LOCAL_MODELS."""
    ce_max_candidates: int = 20
    """Only rerank top-N of the fusion output; tail keeps original order."""
    ce_weight: float = 0.6
    """Blend factor: final = ce_weight * ce_norm + (1 - ce_weight) * fusion_norm.
    0.0 disables CE contribution, 1.0 makes CE the sole ordering signal."""

    # ── HyDE query expansion (Phase 2 of rerank_and_hyde.md) ──────────────
    hyde_enabled: bool = False
    """Enable HyDE (Hypothetical Document Embeddings). Off by default because
    it requires an LLM key and adds 300–600 ms per fresh query."""
    hyde_num_hypotheses: int = 3
    """How many hypothetical answers to generate per query."""
    hyde_min_query_chars: int = 10
    """Short queries carry too little signal — skip HyDE below this length."""
    hyde_top_k: int = 20
    """Per-hypothesis vector-search width. The RRF layer folds them together."""
    hyde_rrf_weight: float = 0.25
    """RRF weight for the HyDE-derived result list vs the primary keyword
    list (weight 0.4) and vector list (weight 0.6). Reduced from 0.5 after
    L-fix eval (2026-07-23)."""

    # ── RRF fusion scaling (was inline magic numbers) ─────────────────────
    rrf_keyword_weight: float = 0.4
    """RRF weight for the keyword/FTS recall list."""
    rrf_vector_weight: float = 0.6
    """RRF weight for the bi-encoder vector recall list."""
    rrf_score_boost: float = 10.0
    """Multiplier: keyword + vector RRF boost for existing records."""
    rrf_new_insert_score_scale: float = 60.0
    """Multiplier: RRF score → MemoryRecord.score for new vector-only candidates."""
    vector_fallback_base: float = 0.55
    """Base score for vector-only fallback candidates (above recent fallback)."""
    vector_fallback_sim_scale: float = 0.2
    """Sim multiplier for vector fallback: base + sim * scale."""
    recent_fallback_score: float = 0.5
    """Score for recent-item fallback when no keyword/vector matches."""
    hyde_skip_vector_penalty: float = 0.8
    """Multiplier for vector-only hits when HyDE was skipped (negative query)."""
    hyde_only_penalty: float = 0.7
    """Multiplier for HyDE-only hits (not in keyword or vector) before CE."""

    def __post_init__(self) -> None:
        """Validate weights on construction so misconfiguration is caught early."""
        if not (0 < self.recall_decay <= 1):
            raise ValueError(f"recall_decay must be in (0, 1], got {self.recall_decay}")
        if self.activation_threshold <= 0:
            raise ValueError(
                f"activation_threshold must be > 0, got {self.activation_threshold}"
            )
        if self.full_scan_limit < 1:
            raise ValueError(
                f"full_scan_limit must be >= 1, got {self.full_scan_limit}"
            )
        if self.max_recall_hops < 1:
            raise ValueError(
                f"max_recall_hops must be >= 1, got {self.max_recall_hops}"
            )
        if not (0.0 <= self.ce_weight <= 1.0):
            raise ValueError(f"ce_weight must be in [0, 1], got {self.ce_weight}")
        if self.ce_max_candidates < 1:
            raise ValueError(f"ce_max_candidates must be >= 1, got {self.ce_max_candidates}")
        if self.hyde_num_hypotheses < 1:
            raise ValueError(f"hyde_num_hypotheses must be >= 1, got {self.hyde_num_hypotheses}")
        if self.hyde_min_query_chars < 1:
            raise ValueError(f"hyde_min_query_chars must be >= 1, got {self.hyde_min_query_chars}")
        if self.hyde_top_k < 1:
            raise ValueError(f"hyde_top_k must be >= 1, got {self.hyde_top_k}")
        if not (0.0 <= self.hyde_rrf_weight <= 5.0):
            raise ValueError(f"hyde_rrf_weight must be in [0, 5], got {self.hyde_rrf_weight}")
        if self.rrf_score_boost <= 0:
            raise ValueError(f"rrf_score_boost must be > 0, got {self.rrf_score_boost}")
        if not (0.0 <= self.hyde_skip_vector_penalty <= 1.0):
            raise ValueError(f"hyde_skip_vector_penalty must be in [0,1], got {self.hyde_skip_vector_penalty}")
        if not (0.0 <= self.hyde_only_penalty <= 1.0):
            raise ValueError(f"hyde_only_penalty must be in [0,1], got {self.hyde_only_penalty}")


def _default_weights() -> RecallWeights:
    """Build RecallWeights from config, keeping defaults as fallback."""
    c = config.recall
    return RecallWeights(
        ascii_base=c.ascii_base, ascii_per_char_cap=c.ascii_per_char_cap,
        ascii_per_char=c.ascii_per_char, cjk_multi_base=c.cjk_multi_base,
        cjk_multi_per_char=c.cjk_multi_per_char, stop_char_weight=c.stop_char_weight,
        single_cjk_weight=c.single_cjk_weight,
        colleague_boost=c.colleague_boost, reason_marker_boost=c.reason_marker_boost,
        fts_boost=c.fts_boost, shared_entity_multiplier=c.shared_entity_multiplier,
        conflicting_person_penalty=c.conflicting_person_penalty,
        person_match_boost=c.person_match_boost,
        entity_tie_break_bonus=c.entity_tie_break_bonus,
        brand_boost=c.brand_boost, occupation_boost=c.occupation_boost,
        identity_base_attr_boost=c.identity_base_attr_boost,
        eval_sensory_boost=c.eval_sensory_boost,
        correction_penalty=c.correction_penalty,
        max_recall_hops=c.max_recall_hops, recall_decay=c.recall_decay,
        activation_threshold=c.activation_threshold,
        assoc_reach_base_attr=c.assoc_reach_base_attr,
        assoc_reach_constraint=c.assoc_reach_constraint,
        assoc_reach_decision=c.assoc_reach_decision,
        assoc_reach_sensory=c.assoc_reach_sensory,
        assoc_reach_event=c.assoc_reach_event,
        full_scan_limit=c.full_scan_limit,
    )


# ── Term extraction and weighting ─────────────────────────────────────────

def terms(query: str) -> list[str]:
    """Split a query into scored terms: ASCII tokens, multi-CJK, single-CJK."""
    ascii_terms = re.findall(r"[A-Za-z0-9_]+", query.lower())
    cjk_terms = re.findall(r"[\u4e00-\u9fff]{2,}", query)
    chars = [ch for ch in query if "\u4e00" <= ch <= "\u9fff"]
    return list(dict.fromkeys(ascii_terms + cjk_terms + chars))


def term_weight(term: str, weights: RecallWeights, stop_chars: frozenset[str]) -> float:
    """Weight a term by how much retrieval signal it carries.

    - ASCII / alphanumeric tokens (brands like HHKB, dates, numbers) are
      high-signal and rare -> heaviest weight.
    - Multi-char CJK words (机械键盘, 同事, 腾讯) are meaningful -> medium.
    - Single common CJK chars (我, 的, 是) are noise -> near zero.
    """
    if re.fullmatch(r"[A-Za-z0-9_]+", term):
        return weights.ascii_base + min(len(term), weights.ascii_per_char_cap) * weights.ascii_per_char
    if len(term) >= 2:
        return weights.cjk_multi_base + (len(term) - 2) * weights.cjk_multi_per_char
    if term in stop_chars:
        return weights.stop_char_weight
    return weights.single_cjk_weight


# ── Scoring ───────────────────────────────────────────────────────────────

def score(query: str, terms: list[str], haystack: str,
          weights: RecallWeights, stop_chars: frozenset[str],
          reason_markers: tuple[str, ...]) -> float:
    """Semantic-aware recall score: term matches + colleague / reason boosts."""
    s = 0.0
    for term in terms:
        if term.lower() in haystack:
            s += term_weight(term, weights, stop_chars)
    if "同事" in query and "同事" in haystack:
        s += weights.colleague_boost
    if "为什么" in query and any(marker in haystack for marker in reason_markers):
        s += weights.reason_marker_boost
    return s


def match_reason(terms: list[str], haystack: str, score: float) -> str:
    """Human-readable match reason for recall results."""
    matched = [term for term in terms if term.lower() in haystack]
    if matched:
        return "matched terms: " + ", ".join(matched[:8])
    return f"fallback score={score:.2f}"


def base_score(query: str, terms: list[str], haystack: str,
               weights: RecallWeights, stop_chars: frozenset[str],
               reason_markers: tuple[str, ...],
               ranker: Any = None, store: Any = None) -> float:
    """Base recall score, using LLM reranker if available, else semantic score."""
    if ranker is not None:
        return float(ranker(query, terms, haystack, store))
    return score(query, terms, haystack, weights, stop_chars, reason_markers)
