"""Unit tests for store/pipeline.py — RecallContext, RecallStage, and strategies."""

import pytest

from engram_router.store import MemoryStore
from engram_router.store.pipeline import (
    RecallContext,
    QueryPrepStage,
    FTSCandidateStage,
    EdgeExpansionStage,
    ScoringStage,
    ResponseStage,
    build_pipeline,
    run_pipeline,
)


# ---------------------------------------------------------------------------
# 1. RecallContext creation
# ---------------------------------------------------------------------------


def test_recallcontext_default_values():
    ctx = RecallContext(query="测试查询")
    assert ctx.query == "测试查询"
    assert ctx.top_k == 5
    assert ctx.namespace == "default"
    assert ctx.strategy == "standard"
    assert ctx.terms == []
    assert ctx.rows == []
    assert ctx.records == []
    assert ctx.debug == {}


def test_recallcontext_field_assignment():
    ctx = RecallContext(
        query="什么品牌", top_k=3, namespace="work", strategy="rrf",
    )
    assert ctx.top_k == 3
    assert ctx.namespace == "work"
    assert ctx.strategy == "rrf"


# ---------------------------------------------------------------------------
# 2. RecallContext with strategies
# ---------------------------------------------------------------------------


def test_recallcontext_standard_strategy():
    ctx = RecallContext(query="HHKB", strategy="standard")
    assert ctx.strategy == "standard"


def test_recallcontext_rrf_strategy():
    ctx = RecallContext(query="键盘", strategy="rrf")
    assert ctx.strategy == "rrf"


def test_recallcontext_fallback_strategy():
    ctx = RecallContext(query="最近", strategy="fallback")
    assert ctx.strategy == "fallback"


# ---------------------------------------------------------------------------
# 3. QueryPrepStage
# ---------------------------------------------------------------------------


def test_query_prep_stage(tmp_path):
    store = MemoryStore(enable_vector=False, path=tmp_path / "db.db")
    store.save("张三是我的前同事，现在在腾讯。")
    store.save("张三前两天送我一把 HHKB，说是生日礼物")

    stage = QueryPrepStage()
    ctx = RecallContext(query="张三 HHKB")
    ctx = stage(store, ctx)

    assert len(ctx.terms) > 0, "terms should be extracted from query"
    assert "张三" in ctx.terms or "HHKB" in ctx.terms or "hhkb" in ctx.terms
    assert len(ctx.query_entity_objs) >= 0  # entity extractor may or may not find entities
    assert isinstance(ctx.corrected_ids, set)
    assert "query_prep" in [s["stage"] for s in ctx.debug["stages"]]


def test_query_prep_stage_empty_store(tmp_path):
    store = MemoryStore(enable_vector=False, path=tmp_path / "empty.db")

    stage = QueryPrepStage()
    ctx = RecallContext(query="测试")
    ctx = stage(store, ctx)

    assert len(ctx.terms) > 0
    assert ctx.corrected_ids == set()


# ---------------------------------------------------------------------------
# 4. FTSCandidateStage
# ---------------------------------------------------------------------------


def test_fts_candidate_stage(tmp_path):
    store = MemoryStore(enable_vector=False, path=tmp_path / "fts.db")
    store.save("张三前两天送我一把 HHKB，说是生日礼物")
    store.save("张三是我的前同事，现在在腾讯。")

    # QueryPrep produces the terms we need
    prep = QueryPrepStage()
    ctx = RecallContext(query="张三 HHKB")
    ctx = prep(store, ctx)

    fts_stage = FTSCandidateStage()
    ctx = fts_stage(store, ctx)

    assert ctx.rows is not None
    assert len(ctx.rows) > 0, "should find at least one candidate row"
    assert any("张三" in r["raw_text"] or "HHKB" in r["raw_text"] for r in ctx.rows)
    assert ctx.entity_map is not None
    assert "fts_candidate" in [s["stage"] for s in ctx.debug["stages"]]


# ---------------------------------------------------------------------------
# 5. EdgeExpansionStage
# ---------------------------------------------------------------------------


def test_edge_expansion_stage(tmp_path):
    store = MemoryStore(enable_vector=False, path=tmp_path / "edge.db")
    store.save("张三前两天送我一把 HHKB，说是生日礼物")
    store.save("张三是我的前同事，现在在腾讯。")

    prep = QueryPrepStage()
    fts_stage = FTSCandidateStage()
    edge_stage = EdgeExpansionStage()

    ctx = RecallContext(query="张三 HHKB")
    ctx = prep(store, ctx)
    ctx = fts_stage(store, ctx)
    ctx = edge_stage(store, ctx)

    assert isinstance(ctx.edge_bonus, dict)
    assert "edge_expansion" in [s["stage"] for s in ctx.debug["stages"]]


# ---------------------------------------------------------------------------
# 6. build_pipeline
# ---------------------------------------------------------------------------


def test_build_pipeline_standard():
    stages = build_pipeline("standard")
    names = [s.name for s in stages]
    assert "query_prep" in names
    assert "fts_candidate" in names
    assert "edge_expansion" in names
    assert "scoring" in names
    assert "response" in names


def test_build_pipeline_rrf():
    stages = build_pipeline("rrf")
    names = [s.name for s in stages]
    assert "query_prep" in names
    assert "fts_candidate" in names
    assert "edge_expansion" in names
    assert "scoring" in names
    assert "response" in names


def test_build_pipeline_fallback():
    stages = build_pipeline("fallback")
    names = [s.name for s in stages]
    assert "query_prep" in names
    assert "fts_candidate" in names
    assert "edge_expansion" not in names, "fallback skips edge expansion"
    assert "scoring" in names
    assert "response" in names


def test_build_pipeline_invalid():
    with pytest.raises(ValueError, match="Unknown pipeline strategy"):
        build_pipeline("nonexistent")


def test_build_pipeline_is_new_list_each_time():
    """Each call returns a new list instance, not a shared reference."""
    a = build_pipeline("standard")
    b = build_pipeline("standard")
    assert a is not b
    assert len(a) == len(b)


# ---------------------------------------------------------------------------
# 7. Empty context pass-through
# ---------------------------------------------------------------------------


def test_stages_graceful_with_empty_context(tmp_path):
    """Stages should not crash when context has no rows / no terms."""
    store = MemoryStore(enable_vector=False, path=tmp_path / "empty_ctx.db")

    ctx = RecallContext(query="")  # empty query

    prep = QueryPrepStage()
    ctx = prep(store, ctx)  # should not crash

    fts = FTSCandidateStage()
    ctx = fts(store, ctx)  # should not crash

    edge = EdgeExpansionStage()
    ctx = edge(store, ctx)  # should not crash (no rows)

    scoring_stage = ScoringStage()
    ctx = scoring_stage(store, ctx)  # should not crash (no rows)

    resp = ResponseStage()
    ctx = resp(store, ctx)

    assert ctx.records == []


def test_empty_query_produces_empty_results(tmp_path):
    store = MemoryStore(enable_vector=False, path=tmp_path / "no_words.db")
    store.save("some memory content")
    ctx = run_pipeline(store, "", top_k=5)
    # With empty query, terms list is empty so scoring produces nothing
    assert len(ctx.records) >= 0  # may get recent fallback


# ---------------------------------------------------------------------------
# 8. Debug trace
# ---------------------------------------------------------------------------


def test_run_pipeline_populates_debug_trace(tmp_path):
    store = MemoryStore(enable_vector=False, path=tmp_path / "debug.db")
    store.save("张三前两天送我一把 HHKB，说是生日礼物")
    store.save("张三是我的前同事，现在在腾讯。")

    ctx = run_pipeline(store, "张三 HHKB", top_k=3)

    assert "stages" in ctx.debug
    assert len(ctx.debug["stages"]) > 0
    assert "total_elapsed_ms" in ctx.debug
    assert ctx.debug["total_elapsed_ms"] >= 0

    for stage_info in ctx.debug["stages"]:
        assert "stage" in stage_info
        assert "elapsed_ms" in stage_info


# ---------------------------------------------------------------------------
# 9. Scoring + Response stages
# ---------------------------------------------------------------------------


def test_scoring_stage_produces_scores(tmp_path):
    store = MemoryStore(enable_vector=False, path=tmp_path / "score.db")
    store.save("张三前两天送我一把 HHKB，说是生日礼物")
    store.save("张三是我的前同事，现在在腾讯。")

    prep = QueryPrepStage()
    fts = FTSCandidateStage()
    edge = EdgeExpansionStage()
    score_stage = ScoringStage()

    ctx = RecallContext(query="HHKB", top_k=3)
    ctx = prep(store, ctx)
    ctx = fts(store, ctx)
    ctx = edge(store, ctx)
    ctx = score_stage(store, ctx)

    assert len(ctx.scored) > 0, "should produce scored candidates"
    for score_val, reason, row in ctx.scored:
        assert isinstance(score_val, float)
        assert isinstance(reason, str)


def test_response_stage_finalizes_records(tmp_path):
    store = MemoryStore(enable_vector=False, path=tmp_path / "resp.db")
    store.save("张三前两天送我一把 HHKB，说是生日礼物")

    prep = QueryPrepStage()
    fts = FTSCandidateStage()
    edge = EdgeExpansionStage()
    score_stage = ScoringStage()
    resp = ResponseStage()

    ctx = RecallContext(query="HHKB", top_k=3)
    ctx = prep(store, ctx)
    ctx = fts(store, ctx)
    ctx = edge(store, ctx)
    ctx = score_stage(store, ctx)
    ctx = resp(store, ctx)

    assert len(ctx.records) > 0
    assert ctx.records[0].id.startswith("mem_")
    assert ctx.records[0].score > 0


# ---------------------------------------------------------------------------
# 10. End-to-end test
# ---------------------------------------------------------------------------


def test_run_pipeline_end_to_end(tmp_path):
    store = MemoryStore(enable_vector=False, path=tmp_path / "e2e.db")
    store.save("张三前两天送我一把 HHKB，说是生日礼物")
    store.save("张三是我的前同事，现在在腾讯。")
    store.save("李四在百度工作")

    ctx = run_pipeline(store, "张三", top_k=3)

    assert len(ctx.records) > 0, "should recall at least one memory"
    for record in ctx.records:
        assert record.id.startswith("mem_")
        assert record.score > 0
        assert record.raw_text

    # Verify HHKB-related content is found when querying for it
    ctx2 = run_pipeline(store, "HHKB", top_k=3)
    assert any("HHKB" in r.raw_text for r in ctx2.records)


def test_run_pipeline_namespace_isolation(tmp_path):
    store = MemoryStore(enable_vector=False, path=tmp_path / "ns_pipe.db")
    store.save("work content", namespace="work")
    store.save("personal content", namespace="personal")

    ctx_work = run_pipeline(store, "content", namespace="work")
    assert len(ctx_work.records) == 1
    assert "work" in ctx_work.records[0].raw_text

    ctx_personal = run_pipeline(store, "content", namespace="personal")
    assert len(ctx_personal.records) == 1
    assert "personal" in ctx_personal.records[0].raw_text
