# store.py 重构设计

**日期**: 2026-07-21
**作者**: Agent B(设计) · 由主会话代为落盘
**目标文件**: `src/engram_router/store.py`(2641 行,单文件承载 `MemoryRecord` / `RecallWeights` / `MemoryStore` 全部逻辑)
**依赖**: 本重构不改变任何公开 API 语义;完成后 `ENGRAM_SKIP_VECTOR=1 pytest -q` 期望 265 passed / 12 xfailed 不变

---

## 0. 结论提要

- **文档规模**: ~620 行(本文件),含 5 段代码骨架 + 7 步迁移表 + API 保留清单 + 风险矩阵 + dry-run 命令
- **拆分目标**: **8 个** 目标文件(用户给的 7 个 + `records.py`),合计 ~2450 行(比 2641 略降,因为 `pipeline.py` 消化了 recall 里的胶水代码,但公共 API 表面积不变)
- **迁移步数**: **7 步**,叶子先动、根最后动;每一步能独立通过 `pytest -q` 与 `benchmark --gate`
- **最大风险**: `recall()` 主流程的 RRF 早返回分支与标准 pipeline 不能 flatten 成一条 Stage 链;必须由 `RecallContext.strategy` 分派器保底(见 §5.1)

---

## 1. 目标文件结构(函数级)

拆成 **8 个** 目标文件。所有文件目标行数 **< 400 行**。

| 目标文件 | 预估行数 | 承接 `store.py` 中的行区间 |
|---|---|---|
| `store/__init__.py`(兼容 re-export) | ~30 | 无 — 只做 `from .core import MemoryStore, RecallWeights`; `from .records import MemoryRecord` |
| `store/records.py` | ~230 | `store.py:31-44`, `853-869`, `2221-2266`, `2343-2416`(MemoryRecord + `_row_to_record` + `_serialize/_parse_metadata` + `_summarize`/`_clean_sentence`/`_truncate_cjk`) |
| `store/scoring.py` | ~310 | `store.py:47-160`(RecallWeights + `_default_weights`),`918-921`(`_base_score`),`2419-2446`(`_terms`/`_term_weight`/`STOP_CHARS`),`2545-2562`(`_score`/`_match_reason`),`1504-1528`(`_apply_salience_decay`),`1427-1502`(`_apply_context_boosts`) |
| `store/query_intent.py` | ~180 | `store.py:2112-2155`, `2564-2635`(所有 `_asks_*`/`_has_*` + `_looks_like_product` + `_identity_subjects` + `_suggest_question` + `REASON_MARKERS`) |
| `store/candidates.py` | ~330 | `store.py:456-501`(`_init_fts`/`_fts_remove`/`_fts_rebuild`),`430-455`(`_init_indices`),`923-1005`(`_fts_candidates`),`1006-1086`(`_rows_by_ids`/`_row_by_id`/`_memory_rows`/`_entities_for_memories`),`1087-1103`(`_record_access`) |
| `store/graph.py` | ~370 | `store.py:610-712`(`_index_edges`),`1823-1888`(`_entity_query_relevance`),`1889-2101`(`_edge_expansion`) |
| `store/recall.py` | ~340 | `store.py:1104-1140`(`should_inject`),`1141-1294`(`recall`),`1296-1425`(`_build_scored_candidates`),`1530-1583`(`_recall_single`),`1585-1789`(`_build_recall_response`),`1791-1822`(`_batch_evidence_refs`/`_batch_raw_refs`) |
| `store/pipeline.py` | ~280 | 新增 — `RecallStage` 协议 + `RecallContext` dataclass + 7 个 Stage 实现(包裹现有逻辑,不改数值) |
| `store/core.py` | ~380 | `store.py:163-548` 的骨架(`__init__` / `_init_schema` / `_migrate_schema` / `save` / `delete` / lazy 属性 / `_get_or_create_entity` / `_populate_timed_events_for_memory` / `close` / `__enter__/__exit__` / `_next_id` / `_seed_sequence` / `_ID_TABLES` / `consolidate` / `save_raw_log` / `get_raw_log` / `compact` / `entities_for`/`_entity_names_for`/`_entities_for` / `_index_entities` / `_get_corrected_ids` / `gap_check` / `_update_persona` / `_apply_decay` / `MAX_TEXT_BYTES` / `_SQLITE_IN_BATCH`) |

**为什么加一个 `records.py`**: `MemoryRecord` + `_row_to_record` + `_summarize` 语义上不属于 core(它是数据模型),放 core 会把 core 顶破 400 行;放 recall 又跨模块被 candidates/graph 引用。独立最合适。

---

## 2. Pipeline 抽象设计

### 数据结构

```python
# store/pipeline.py
from dataclasses import dataclass, field
from typing import Any, Protocol
import sqlite3

@dataclass
class RecallContext:
    query: str
    terms: list[str]
    namespace: str = "default"
    top_k: int = 5
    strategy: str = "standard"          # "standard" | "rrf" | "fallback"
    query_entity_objs: list[dict[str, Any]] = field(default_factory=list)
    query_entities: set[str] = field(default_factory=set)
    query_topics: set[str] = field(default_factory=set)
    query_identity_subjects: set[str] = field(default_factory=set)
    corrected_ids: set[str] = field(default_factory=set)
    fts_ids: set[str] | None = None
    rows: list[sqlite3.Row] = field(default_factory=list)
    entity_map: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    edge_bonus: dict[str, tuple[float, str]] = field(default_factory=dict)
    scored: list[tuple[float, str, sqlite3.Row]] = field(default_factory=list)
    ranked: list[Any] = field(default_factory=list)          # MemoryRecord 列表
    debug: dict[str, Any] = field(default_factory=dict)

class RecallStage(Protocol):
    name: str
    def run(self, ctx: RecallContext) -> RecallContext: ...
```

### Stage 列表与契约

| Stage | 读 | 写 | 备注 |
|---|---|---|---|
| `QueryPrepStage` | `query` | `terms`, `query_entity_objs`, `query_entities`, `query_topics`, `query_identity_subjects`, `corrected_ids` | 吸收 `store.py:1152-1268` 的 preamble;含 `query_expander` 判断 |
| `FTSCandidateStage` | `terms`, `query` | `fts_ids`, `rows`, `entity_map` | `candidates.fts_candidates` + `memory_rows` |
| `GraphCandidateStage` | `rows`, `entity_map` | `edge_bonus`, `rows`(追加), `entity_map`(刷新) | `graph.edge_expansion`,负责把 missing_edge_ids 拉进 rows |
| `ScoringStage` | `rows`, `entity_map`, `edge_bonus`, `fts_ids`, corrected/query_* | `scored` | 现 `_build_scored_candidates` 主体;`ContextBoost`/`SalienceDecay` 在此 per-row 调用,不单独走 ctx |
| `FusionStage` | `scored`, records | `ranked` | vector fusion + RRF(现 `_build_recall_response` 中段) |
| `RerankStage` | `ranked` | `ranked` | LLMReranker 占位,默认 no-op;**Phase 1 CE 也在这里插** |
| `FallbackStage` | `ranked` | `ranked` | vector fallback + recent fallback |
| `TruncateStage` | `ranked`, `top_k` | `ranked` | 排序 + 截断 + 记录 access |

### recall.recall() 只做三件事

```python
def recall(self, query: str, top_k: int = 5, namespace: str = "default"):
    # 1. 装配 RecallContext(query=..., namespace=..., top_k=..., strategy=...)
    ctx = RecallContext(query=query, namespace=namespace, top_k=top_k)
    ctx.strategy = _detect_strategy(query, self.query_expander)

    # 2. 按 strategy 选择 stage 链
    chain = _standard_pipeline() if ctx.strategy == "standard" else _rrf_pipeline()

    # 3. 顺序 run
    for stage in chain:
        ctx = stage.run(ctx)

    return ctx.ranked
```

### 关键行为等价性

老代码里 `_apply_context_boosts` / `_apply_salience_decay` 是在 `_build_scored_candidates` 的 **per-row 循环内**被调用的(`store.py:1396-1420`),**不是**独立 Stage。因此文档明确写:`ContextBoostStage` / `SalienceDecayStage` 在概念图里存在,但物理上被 `ScoringStage` 融合调用,**避免二次遍历 rows** 引入 O(N) 开销与顺序漂移。

---

## 3. 迁移计划(7 步)

分 **7 步**,按"叶子先动、根最后动"排:

| # | 步骤 | 出走函数 | 保留的 delegate | 影响 test 文件 | 期望新增测试 |
|---|---|---|---|---|---|
| 1 | 抽 `query_intent.py` | 15 个 `_asks_*`/`_has_*`/`_looks_like_product`/`_identity_subjects`/`_suggest_question`(全部 static/class) | `MemoryStore._asks_brand = staticmethod(query_intent.asks_brand)` 之类的 module-level 别名 | 0(测试没直接调) | `tests/test_query_intent.py`(纯函数, ~15 用例) |
| 2 | 抽 `records.py` + `store/__init__.py` re-export | `MemoryRecord`, `_row_to_record`, `_serialize_metadata`, `_parse_metadata`, `_summarize`, `_clean_sentence`, `_truncate_cjk` | `MemoryStore._clean_sentence = staticmethod(records.clean_sentence)`(`test_summary.py` 直接 `MemoryStore._clean_sentence`) | 1(`test_summary.py`:22)、`test_forgetting.py` import `MemoryRecord` | 无(既有测试足够) |
| 3 | 抽 `scoring.py` | `RecallWeights`, `_default_weights`, `_terms`, `_term_weight`, `STOP_CHARS`, `_base_score`, `_score`, `_match_reason`, `_apply_salience_decay`, `_apply_context_boosts` | `RecallWeights` 从 `store` 顶层照常 re-export(`test_store.py:598/646/664` 直接 import);`MemoryStore.weights` 属性签名不变;`_apply_context_boosts`/`_apply_salience_decay` 变为纯函数接收 `weights` + row/reason | 1(`test_store.py` 3 处 import `RecallWeights` + 一组 weight 测试) | `tests/test_scoring.py`(**数值锁死**:同 query 同 weights 应产出与 baseline 完全一致的分数,用现有 corpus 生成 golden JSON) |
| 4 | 抽 `candidates.py` | `_init_fts`, `_fts_remove`, `_fts_rebuild`, `_init_indices`, `_fts_candidates`, `_rows_by_ids`, `_row_by_id`, `_memory_rows`, `_entities_for_memories`, `_record_access` | `MemoryStore._fts_candidates` / `_fts_enabled` / `_terms` 全部保留为 `MemoryStore` 上的 delegate 方法(`test_fts.py:47,108,110` 直接调) | 1(`test_fts.py`:6) | 无 |
| 5 | 抽 `graph.py` | `_index_edges`, `_entity_query_relevance`, `_edge_expansion` | `MemoryStore._index_edges` / `_edge_expansion` 作为 delegate 保留 | 1(`test_edges.py`:8, `test_entities.py`:5 依赖 edge 副作用) | `tests/test_graph.py`(BFS activation 单测,mock conn) |
| 6 | 抽 `pipeline.py` + `recall.py` | `recall`, `_recall_single`, `_build_scored_candidates`, `_build_recall_response`, `_batch_evidence_refs`, `_batch_raw_refs`, `should_inject` | `MemoryStore.recall(...)` 签名 100% 不变,内部改为构造 `RecallContext` → 顺序 `run` 一串 Stage | 8(`test_store.py`:38, `test_recent_fallback.py`:10, `test_edges.py`:8, `test_causal.py`:25, `test_demo_advantages.py`:8, `test_persona.py`:44, `test_forgetting.py`:28, `test_query_expansion.py`:33) | `tests/test_pipeline.py`(每个 Stage 单测 + 端到端组合测试) |
| 7 | 收尾:`core.py` + `store/__init__.py` | 剩余的 `__init__`/schema/save/delete/next_id/consolidate/gap_check/compact/save_raw_log/entities_for + 全部 lazy 属性 | 顶层 `from engram_router.store import MemoryStore, MemoryRecord, RecallWeights` 100% 保留(`__init__.py:3` 依赖) | 全部 16 个 test 文件(端到端 smoke) | `tests/test_store_facade.py`(冒烟测试:所有 API 签名 introspection) |

每步的"独立可测"体现在:每一步只跑到 delegate 层完成时,`pytest -q` 必须 265 passed / 12 xfailed 不变,且 benchmark gate PASS。

---

## 4. 公共 API 保留清单(签名 100% 不变)

`MemoryStore` 上以下方法/属性必须保持签名与行为不变,tests 有直接调用:

| API | 使用位置(测试) |
|---|---|
| `__init__(path, max_recall_hops, recall_decay, weights, llm_extractor, llm_query_extract, reranker, embedding_engine, vector_index, query_expander, enable_vector)` | `test_store.py`, `test_persona.py` 等所有 |
| `save(text, source, metadata, namespace)` | 全部 test 文件 |
| `recall(query, top_k, namespace)` | 全部 test 文件 |
| `delete(memory_id)` | `test_store.py`, `test_id_sequences.py` |
| `persona`(property) | `test_persona.py`, `test_demo_advantages.py` |
| `causal`(property) | `test_causal.py`, `test_demo_advantages.py` |
| `timeline`(property) | `test_demo_advantages.py` |
| `forgetting`(property) | `test_forgetting.py`, `test_demo_advantages.py` |
| `gap_check(query, memories, namespace, scan_all)` | `test_store.py` |
| `entities_for(memory_id)` | `test_entities.py` |
| `should_inject(query)` | `mcp_server.py` |
| `close()`, `__enter__`, `__exit__` | `test_store.py` |
| `consolidate()` | `test_store.py` |
| `save_raw_log`, `get_raw_log`, `compact` | `test_store.py` |
| `MemoryStore._clean_sentence`, `_summarize`, `_truncate_cjk`(static) | `test_summary.py:10,57,98` |
| `MemoryStore._fts_candidates`, `_terms`, `_fts_enabled`(实例) | `test_fts.py:47,108,110` |
| `MemoryStore._next_id`(实例,与 `PersonaStore._next_id` **不同**) | `test_id_sequences.py` |

**模块顶层 re-export**(`from engram_router.store import ...`):
- `MemoryStore`, `MemoryRecord`, `RecallWeights` — `__init__.py:3`、`tests/test_store.py:598/646/664` 依赖

---

## 5. 风险与回滚

按风险从高到低:

### 5.1 [最高] `recall()` 主流程与 Pipeline 装配的数值一致性(Step 6)

老代码在"有 `query_expander` 且 `eq.variants` 非空"时会 **早返回 RRF 融合结果**(`store.py:1176-1217`),完全绕过 `_build_scored_candidates` 的 context/salience/correction 分支;而"无 variants"分支才走标准 pipeline。这两条路径顺序 **不能 flatten 成一条 Stage 链**,否则 RRF 融合分和标准 pipeline 分会互相污染。

**缓解**:
- `RecallContext.strategy` 字段承担 `"rrf" | "standard" | "fallback"` 三态
- `recall.py` 内部保留分派器
- 每一 Stage 都要写"当 strategy 不是我这条路时直接透传 ctx"

**金标准兜底**: Step 3 生成的 `test_scoring.py` 数值 golden JSON 必须在 Step 6 结束后逐字节对齐,一个 0.001 的漂移就回滚。

### 5.2 [高] SQLite 事务原子性(Step 5,`save()` 拆分)

`save()` 现在一路 `INSERT memories → evidence → memory_entities → edges → memories_fts`,最后一次 `self.conn.commit()`(`store.py:530`)。`_index_entities` 里又循环调 `_index_edges`(`store.py:608`),两个模块共用同一个 `self.conn`。拆到 `graph.py` 之后如果 `graph.py` 里误加 `self.conn.commit()`(常见反射:"我自己的模块自己 commit"),会把半个 save 提前落盘,后续 FTS 插入失败时无法回滚。

**缓解**:
- 所有拆出去的模块 **一律不调用 `conn.commit()`**
- commit 只在 `core.save()` / `core.compact()` / `core.delete()` / `core.consolidate()` 这 4 个入口点由 `core.py` 主责
- `pipeline.py` 全部 read-only
- 在 `graph.py` / `candidates.py` 顶部加断言:"module must not commit"(通过 `pytest` fixture 挂 `conn.commit = _guard` 检测)

### 5.3 [高] 循环依赖 `core ↔ graph`

`core._index_entities` 需要 `graph._index_edges`;`graph._index_edges` 需要 `core._get_or_create_entity` + `core._next_id`。

**解耦**:
- `graph.py` 设计为 **函数式**,只接收 `(conn, indexed, memory_id, text, next_id_fn, llm_extractor)`,不 import `core`
- 反向 `core._index_entities` `from .graph import index_edges` 是单向 import
- `_entity_query_relevance` 同理,只依赖 `weights` + `entities` 模块的规则

### 5.4 [中] `RecallWeights.__post_init__` 校验向后兼容

`RecallWeights` 是 `@dataclass(frozen=True)`,`test_store.py:598/646/664` 直接 `RecallWeights(...)` 构造后传入 `MemoryStore(weights=...)`。

**缓解**: `scoring.py` 里 `RecallWeights` 定义搬走,但 `store/__init__.py` re-export `RecallWeights = scoring.RecallWeights`,校验行为完全保留(`__post_init__` 原样迁移)。测试中 `from engram_router.store import RecallWeights` 依然工作。

### 5.5 [中] `MemoryStore._fts_candidates` / `_terms` 是测试私有 API

`test_fts.py:108,110` 直接调 `store._fts_candidates` 和 `store._terms`(而不是走 `recall()`)。这两个必须以 delegate 的形式留在 `MemoryStore` 上:
```python
def _fts_candidates(self, ...):
    return candidates.fts_candidates(self.conn, ..., self._fts_enabled)
```
否则测试会 `AttributeError`。

### 5.6 [低] `store.py` 被 5 处外部 import 引用

`__init__.py:3` / `cli.py:12` / `benchmark.py:31` / `mcp_server.py:23` / `forgetting.py:355` 都写死 `from .store import ...`。

**缓解**: 用 `store/` 目录 + `store/__init__.py` re-export 保持 100% 兼容,不改任何调用点。

### 5.7 回滚机制

每一步开始前打 tag `refactor-store-step-{N}-baseline`,失败时:
```bash
git reset --hard refactor-store-step-{N}-baseline
```

测试挂了但不确定是数值漂移还是接线错误时,先跑 `pytest tests/test_scoring.py --tb=short`(Step 3 生成的 golden):
- 若 golden 不动而端到端挂 → 是接线
- 若 golden 动了 → 是数值语义被破坏,必须回滚

---

## 6. Dry-run 校验命令(每步末尾)

```bash
# 1. 单测(以 conftest.py 的 ENGRAM_SKIP_VECTOR fixture 为准)
ENGRAM_SKIP_VECTOR=1 pytest -q
# 期望: 265 passed / 12 xfailed(基线)

# 2. 回归基准(端到端语义 + 门槛)
engram --db /tmp/reg.db benchmark \
  --conversation examples/regression_corpus.md \
  --cases examples/regression_questions.jsonl \
  --gate --text
# 期望: gate: PASS,recall@5 与 baseline 差异 < 1%

# 3. eval_v2 严格评估(Agent A 已建)
ENGRAM_SKIP_VECTOR=1 python tests/eval_v2.py
# 期望: 与 baseline P@1=0.6087 差异 <2 pp(纯重构不应改动数值)

# 4. 每步 tag(便于回滚)
git tag refactor-store-step-{N}
```

---

## 7. 验收 checklist

7 步都完成后:
- `store.py` 从 2641 行降到 **0 行**(整个文件被 `store/` 包替代)
- `git diff --stat` 期望净变化为 `store.py: -2641`,`store/*.py: +~2450`
- `grep -R "from .store import\|from engram_router.store import" src tests | wc -l` 与迁移前完全一致(**0 差异 → 零调用侧代码修改**)

---

## 关键文件锚点

- `/Users/v_liheng02/work/test_code/for_test/engram-router/src/engram_router/store.py`
- `/Users/v_liheng02/work/test_code/for_test/engram-router/src/engram_router/__init__.py`
- `/Users/v_liheng02/work/test_code/for_test/engram-router/src/engram_router/fusion.py`
- `/Users/v_liheng02/work/test_code/for_test/engram-router/src/engram_router/entities.py`
- `/Users/v_liheng02/work/test_code/for_test/engram-router/src/engram_router/vector_index.py`
- `/Users/v_liheng02/work/test_code/for_test/engram-router/src/engram_router/query_expansion.py`
- `/Users/v_liheng02/work/test_code/for_test/engram-router/tests/conftest.py`
- `/Users/v_liheng02/work/test_code/for_test/engram-router/tests/test_store.py`
- `/Users/v_liheng02/work/test_code/for_test/engram-router/tests/test_fts.py`
- `/Users/v_liheng02/work/test_code/for_test/engram-router/tests/test_summary.py`
- `/Users/v_liheng02/work/test_code/for_test/engram-router/tests/eval_v2.py`(新增,基线锁定)
