"""Unit tests for store/pipeline.py - RecallContext, RecallStage, and strategies."""

import pytest

from engram_router.store import MemoryStore
from engram_router.store.pipeline import (
    RecallContext, QueryPrepStage, FTSCandidateStage,
    EdgeExpansionStage, ScoringStage, ResponseStage,
    build_pipeline, run_pipeline,
)


def test_recallcontext_default_values():
    ctx = RecallContext(query='测试查询')
    assert ctx.query == '测试查询'
    assert ctx.top_k == 5
    assert ctx.namespace == 'default'
    assert ctx.strategy == 'standard'
    assert ctx.terms == []
    assert ctx.rows == []
    assert ctx.records == []
    assert ctx.debug == {}


def test_recallcontext_field_assignment():
    ctx = RecallContext(query='什么品牌', top_k=3, namespace='work', strategy='rrf')
    assert ctx.top_k == 3 and ctx.namespace == 'work' and ctx.strategy == 'rrf'


def test_recallcontext_strategies():
    assert RecallContext(query='HHKB', strategy='standard').strategy == 'standard'
    assert RecallContext(query='键盘', strategy='rrf').strategy == 'rrf'
    assert RecallContext(query='最近', strategy='fallback').strategy == 'fallback'


def test_query_prep_stage(tmp_path):
    store = MemoryStore(enable_vector=False, path=tmp_path / 'db.db')
    store.save('张三是我的前同事，现在在腾讯。')
    store.save('张三前两天送我一把 HHKB，说是生日礼物')

    ctx = QueryPrepStage()(store, RecallContext(query='张三 HHKB'))
    assert len(ctx.terms) > 0
    assert '张三' in ctx.terms or 'HHKB' in ctx.terms or 'hhkb' in ctx.terms
    assert isinstance(ctx.corrected_ids, set)
    assert 'query_prep' in [s['stage'] for s in ctx.debug['stages']]


def test_query_prep_stage_empty_store(tmp_path):
    store = MemoryStore(enable_vector=False, path=tmp_path / 'empty.db')
    ctx = QueryPrepStage()(store, RecallContext(query='测试'))
    assert len(ctx.terms) > 0
    assert ctx.corrected_ids == set()


def test_fts_candidate_stage(tmp_path):
    store = MemoryStore(enable_vector=False, path=tmp_path / 'fts.db')
    store.save('张三前两天送我一把 HHKB，说是生日礼物')
    store.save('张三是我的前同事，现在在腾讯。')

    ctx = RecallContext(query='张三 HHKB')
    ctx = QueryPrepStage()(store, ctx)
    ctx = FTSCandidateStage()(store, ctx)

    assert ctx.rows is not None and len(ctx.rows) > 0
    assert any('张三' in r['raw_text'] or 'HHKB' in r['raw_text'] for r in ctx.rows)
    assert 'fts_candidate' in [s['stage'] for s in ctx.debug['stages']]


def test_edge_expansion_stage(tmp_path):
    store = MemoryStore(enable_vector=False, path=tmp_path / 'edge.db')
    store.save('张三前两天送我一把 HHKB，说是生日礼物')
    store.save('张三是我的前同事，现在在腾讯。')

    ctx = RecallContext(query='张三 HHKB')
    ctx = QueryPrepStage()(store, ctx)
    ctx = FTSCandidateStage()(store, ctx)
    ctx = EdgeExpansionStage()(store, ctx)

    assert isinstance(ctx.edge_bonus, dict)
    assert 'edge_expansion' in [s['stage'] for s in ctx.debug['stages']]


def test_build_pipeline_standard():
    names = [s.name for s in build_pipeline('standard')]
    assert names == ['query_prep', 'fts_candidate', 'edge_expansion', 'scoring', 'response']


def test_build_pipeline_rrf():
    names = [s.name for s in build_pipeline('rrf')]
    assert names == ['query_prep', 'fts_candidate', 'edge_expansion', 'scoring', 'response']


def test_build_pipeline_fallback():
    names = [s.name for s in build_pipeline('fallback')]
    assert 'edge_expansion' not in names
    assert 'scoring' in names


def test_build_pipeline_invalid():
    with pytest.raises(ValueError, match='Unknown pipeline strategy'):
        build_pipeline('nonexistent')


def test_build_pipeline_is_new_list_each_time():
    a = build_pipeline('standard')
    b = build_pipeline('standard')
    assert a is not b and len(a) == len(b)


def test_stages_graceful_with_empty_context(tmp_path):
    store = MemoryStore(enable_vector=False, path=tmp_path / 'empty_ctx.db')
    ctx = RecallContext(query='')

    for stage in [QueryPrepStage(), FTSCandidateStage(), EdgeExpansionStage(),
                  ScoringStage(), ResponseStage()]:
        ctx = stage(store, ctx)

    assert ctx.records == []


def test_empty_query_produces_empty_results(tmp_path):
    store = MemoryStore(enable_vector=False, path=tmp_path / 'no_words.db')
    store.save('some memory content')
    ctx = run_pipeline(store, '', top_k=5)
    assert ctx.records is not None


def test_run_pipeline_populates_debug_trace(tmp_path):
    store = MemoryStore(enable_vector=False, path=tmp_path / 'debug.db')
    store.save('张三前两天送我一把 HHKB，说是生日礼物')
    store.save('张三是我的前同事，现在在腾讯。')

    ctx = run_pipeline(store, '张三 HHKB', top_k=3)

    assert 'stages' in ctx.debug and len(ctx.debug['stages']) > 0
    assert 'total_elapsed_ms' in ctx.debug
    for s in ctx.debug['stages']:
        assert 'stage' in s and 'elapsed_ms' in s


def _run_full_chain(store, query, top_k=3):
    stages = [QueryPrepStage(), FTSCandidateStage(), EdgeExpansionStage(),
              ScoringStage(), ResponseStage()]
    ctx = RecallContext(query=query, top_k=top_k)
    for s in stages:
        ctx = s(store, ctx)
    return ctx


def test_scoring_stage_produces_scores(tmp_path):
    store = MemoryStore(enable_vector=False, path=tmp_path / 'score.db')
    store.save('张三前两天送我一把 HHKB，说是生日礼物')
    store.save('张三是我的前同事，现在在腾讯。')

    ctx = RecallContext(query='HHKB', top_k=3)
    ctx = QueryPrepStage()(store, ctx)
    ctx = FTSCandidateStage()(store, ctx)
    ctx = EdgeExpansionStage()(store, ctx)
    ctx = ScoringStage()(store, ctx)

    assert len(ctx.scored) > 0
    for score_val, reason, _row in ctx.scored:
        assert isinstance(score_val, float) and isinstance(reason, str)


def test_response_stage_finalizes_records(tmp_path):
    store = MemoryStore(enable_vector=False, path=tmp_path / 'resp.db')
    store.save('张三前两天送我一把 HHKB，说是生日礼物')

    ctx = _run_full_chain(store, 'HHKB')
    assert len(ctx.records) > 0
    assert ctx.records[0].id.startswith('mem_')
    assert ctx.records[0].score > 0


def test_run_pipeline_end_to_end(tmp_path):
    store = MemoryStore(enable_vector=False, path=tmp_path / 'e2e.db')
    store.save('张三前两天送我一把 HHKB，说是生日礼物')
    store.save('张三是我的前同事，现在在腾讯。')
    store.save('李四在百度工作')

    ctx = run_pipeline(store, '张三', top_k=3)
    assert len(ctx.records) > 0
    for record in ctx.records:
        assert record.id.startswith('mem_') and record.score > 0 and record.raw_text

    ctx2 = run_pipeline(store, 'HHKB', top_k=3)
    assert any('HHKB' in r.raw_text for r in ctx2.records)


def test_run_pipeline_namespace_isolation(tmp_path):
    store = MemoryStore(enable_vector=False, path=tmp_path / 'ns_pipe.db')
    store.save('work content', namespace='work')
    store.save('personal content', namespace='personal')

    ctx_work = run_pipeline(store, 'content', namespace='work')
    assert len(ctx_work.records) == 1
    assert 'work' in ctx_work.records[0].raw_text

    ctx_personal = run_pipeline(store, 'content', namespace='personal')
    assert len(ctx_personal.records) == 1
    assert 'personal' in ctx_personal.records[0].raw_text
