# EngramRouter Phase 2 架构审核报告

> **审核日期**: 2026-07-01
> **审核范围**: `LLM_EXTRACTOR_ENHANCEMENT.md` / `query_expansion_design.md` / `phase2_integration_test_plan.md` / `REQUIREMENTS.md` / 现有 v0.1 代码
> **审核维度**: 一致性 · 完整性 · 兼容性 · 设计原则 · 耦合度 · 回退策略

---

## 总体评级

| 维度 | 评级 | 说明 |
|------|:----:|------|
| **接口一致性** | ⚠️ WARN | REQUIREMENTS.md 与代码/设计文档有 3 处不一致 |
| **设计完整性** | ❌ FAIL | 存在 7 个设计空白，其中 3 个为阻塞级 |
| **设计原则合规** | ⚠️ WARN | 8 条原则中 2 条存在合规风险 |
| **模块耦合度** | ⚠️ WARN | MemoryStore 参数膨胀，双向依赖风险 |
| **回退策略** | ✅ PASS | 5 层降级路径完整且可测试 |
| **v0.1 兼容性** | ⚠️ WARN | API 兼容但存在 1 个 Schema 迁移缺失 |

**综合评级**: **⚠️ WARN — 有 4 项阻塞问题需在设计冻结前解决，其中 1 项为双红（当前向量通路完全失效）**

---

## 一、接口一致性审查

### 1.1 MemoryStore 构造函数签名 (🔴 严重)

| 来源 | 参数列表 |
|------|---------|
| **REQUIREMENTS.md §6.1** | `(path, max_recall_hops, recall_decay, weights, llm_extractor, reranker)` |
| **实际代码 store.py L152-162** | `(path, max_recall_hops, recall_decay, weights, llm_extractor, llm_query_extract, reranker, embedding_engine, vector_index)` |
| **query_expansion_design.md** | 追加 `query_expander` 参数 |
| **test_plan.md** | 使用 `(path, embedding_engine, vector_index)` — 与代码一致 |

**问题**: REQUIREMENTS.md 作为"唯一需求来源"，其 MemoryStore 签名已经与实际代码脱节，缺少 `embedding_engine`、`vector_index`、`llm_query_extract` 三个已有参数，也未包含 `query_expander` 设计参数。这违反了 REQUIREMENTS.md 自身声明的权威性。

**建议**: 🔴 **立即更新 REQUIREMENTS.md §6.1**，以实际代码为准对齐签名，并在 Phase 2 设计中追加 `query_expander`。

---

### 1.2 LLM 查询实体提取的双重路径 (🟡 中等)

现有代码 (`store.py L838-843`) 已有 `llm_query_extract` 机制：在 `recall()` 中调用 `extract_entities_llm(query)` 补充规则提取遗漏的实体。

`query_expansion_design.md` 又设计了 `LLMEntitySuppleter`（架构图中的组件 3）来补充实体。

**冲突点**:
```python
# 现有代码 (store.py L838)
if self.llm_query_extract and self.llm_extractor is not None:
    llm_ents = extract_entities_llm(query)

# 新设计 (query_expansion_design.md)
# QueryExpander 也做实体补充，走 LLMQueryRewriter.rewrite()
eq = self.query_expander.expand(query)
# eq.extra_entities 也会合并到 query_entity_objs
```

如果两者同时启用，**同一查询可能触发两次 LLM 调用**做同一件事。

**建议**: 🟡 明确分工或统一：要么 `QueryExpander` 取代 `llm_query_extract`，要么 `QueryExpander` 的实体补充与 `extract_entities_llm` 合并去重。设计文档需给出决策。

---

### 1.3 数据格式一致性 ✅

| 数据类型 | LLM_EXTRACTOR_ENHANCEMENT | query_expansion | test_plan | 实际代码 |
|----------|:---:|:---:|:---:|:---:|
| Entity dict `{name, kind, evidence}` | ✅ | ✅ | — | ✅ |
| RRF `reciprocal_rank_fusion(lists, k, weights)` | — | ✅ 无 weights | ✅ 含 weights | ✅ 含 weights |
| MemoryRecord `(id, raw_text, summary, score, match_reason)` | — | ✅ | ✅ | ✅ |

**结论**: 数据结构、RRF 签名、MemoryRecord 格式在所有文档间一致。

---

## 二、设计完整性审查

### 2.1 阻塞级空白 (🔴)

#### GAP-1: Schema 迁移方案缺失

REQUIREMENTS.md §10 声明: "新增 `memories.embedding_model` 字段"。

**现状**: 
- 实际代码的 `_init_schema()` 中 `memories` 表**没有** `embedding_model` 列
- 没有任何文档定义迁移 SQL（`ALTER TABLE` 或版本化 schema）
- test_plan.md 未包含 schema 迁移测试

**风险**: 已有数据库升级到 Phase 2 会因缺列报错。

**建议**: 🔴 在实施前编写 `migrations/002_add_embedding_model.sql`，并在 test_plan 中添加 TC-M01（空库创建）和 TC-M02（旧库升级）。

---

#### GAP-2: VectorIndex 无 namespace 支持

test_plan.md 的 TC-VR11 测试 `namespace` 隔离下向量搜索结果，但 `VectorIndex`（FAISS）原生不支持 namespace 过滤。

**现状**:
- `VectorIndex.search()` 返回 `[(memory_id, score)]` — 无 namespace 参数
- FAISS 索引中向量与 namespace 无关联
- `store.py` 的 recall 中对向量结果**没有**按 namespace 过滤（L921-935 直接追加）

**风险**: 当多 namespace 并存时，向量搜索会跨 namespace 泄露数据。TC-VR11 会失败。

**建议**: 🔴 方案 A：为每个 namespace 维护独立 FAISS 索引（简单但内存翻倍）；方案 B：search 后通过 SQLite 按 memory_id + namespace 过滤（增加一次 DB 查询）。

---

#### GAP-3: 已存在的代码 Bug — 类型不匹配导致向量通路完全失效 🔴🔴

`store.py L890-938` 的向量融合路径存在**类型不匹配**问题：

**根因**: `_build_scored_candidates()` 返回 `list[tuple[float, str, sqlite3.Row]]`（三元组），但向量路径代码将其当作 `list[MemoryRecord]` 来操作（使用 `.id` / `.score` 属性访问）。

**出错链路** (已验证触发):
```
L890: scored = _build_scored_candidates(...)
        → list[tuple[float, str, Row]]  (三元组)

L908: keyword_list = [(r.id, r.score) for r in scored if r.score > 0]
        → r 是 tuple(2.8, 'matched terms: ...', <Row>)
        → r.score 抛 AttributeError  ← 这里崩溃

L937: except Exception as exc:
        logger.debug("Vector search skipped: %s", exc)
        → 静默吞掉异常，向量路径完全跳过
```

**验证结果**:
```
$ python -c "..."
scored[0] type: <class 'tuple'>
scored[0].id → 'tuple' object has no attribute 'id'
scored[0].score → 'tuple' object has no attribute 'score'

$ store.recall('测试')  
DEBUG: Vector search skipped: 'tuple' object has no attribute 'score'
# 向量路径从未实际执行！
```

**额外类型错误**（即使 L908 被修复也会在后续触发）:
- L917-918: `s.score = ...` → tuple 不可赋值 + 无 `.score` 属性
- L920: `s.id` → tuple 无 `.id`
- L926: `scored.append(MemoryRecord(...))` → 向 tuple 列表中混入 MemoryRecord
- L936: `x.score` → tuple 无 `.score`

**影响**: 🔴🔴 **向量搜索通路完全是死代码**。整个 RRF 融合从未执行过。所有 TC-VR 测试即使编写也无法通过。

**建议**: 🔴 在 Phase 2 实施前彻底重写 L899-938 的向量融合路径，与 `_build_scored_candidates` 的返回类型对齐。建议方案：
1. 在 `_build_recall_response` 中将 tuple 转换为 MemoryRecord 之后再执行 RRF 融合；或
2. 将 `_build_scored_candidates` 的返回类型改为 `list[MemoryRecord]`。

---

### 2.2 中等空白 (🟡)

#### GAP-4: query_expansion 引用了不存在的 `_recall_single`

`query_expansion_design.md §5.1` 伪代码:
```python
primary = self._recall_single(query, terms, query_entity_objs, namespace)
```

`store.py` 中**不存在** `_recall_single` 方法。当前 recall 是单体方法，没有可重用的单路径召回子函数。

**建议**: 🟡 实施 query_expansion 前需要先重构 recall 管道，提取 `_recall_single`。

---

#### GAP-5: Salience 后处理边类型未在测试中覆盖

`LLM_EXTRACTOR_ENHANCEMENT.md` 定义了 4 种新边类型（DECISION_CAUSED_BY, CONSTRAINS, HAPPENED_AT, polarity），但 `phase2_integration_test_plan.md` 只覆盖了现有的 CO_OCCURS_WITH 边。新的边类型在 recall 管道中的行为（是否参与 edge expansion？权重多少？）完全未设计。

**建议**: 🟡 要么将新边类型纳入 test_plan（至少在 edge_expansion 测试中覆盖），要么明确标注为 Phase 3 范围。

---

#### GAP-6: LLM 批量提取与 save() 的集成未定义

`LLM_EXTRACTOR_ENHANCEMENT.md` 提出 `extract_batch(texts: list[str])` 批量模式，但未说明：
- 谁调用批量提取？（MemoryStore？外部调度器？）
- 批量窗口如何形成？（攒够 10 条？定时触发？）
- save() 是同步立即返回还是异步等待批量？

**建议**: 🟡 补充批量模式的触发机制和集成方案。

---

#### GAP-7: 查询扩展不暴露给 MCP

设计原则 5 要求"只通过 MCP 标准接口对外暴露能力"，但 `QueryExpander` 被设计为 `MemoryStore` 的内部组件。Agent 无法通过 MCP 工具控制：
- 是否启用查询扩展
- 预热特定查询的 LLM 缓存
- 查看扩展统计

**建议**: 🟡 至少在 MCP `memory.recall` 工具中增加可选参数 `expand=true/false`。

---

## 三、8 条设计原则审查

| # | 原则 | 合规 | 风险说明 |
|---|------|:----:|---------|
| P1 | 证据优先 — 存原文不摘要 | ✅ | 所有设计保持 raw_text 不可变 |
| P2 | 最小上下文 — 按需召回 top-k | ✅ | top_k 截断在所有路径中保持 |
| P3 | 缺失追问 — 不编造 | ✅ | gap_check 保留；LLM 失败时回退规则引擎 |
| P4 | 可撤销 — 不硬删除 | ✅ | corrections 降权逻辑未改动 |
| P5 | 平台无关 — MCP 标准接口 | ⚠️ | QueryExpander 未暴露给 MCP（见 GAP-7） |
| P6 | 本地优先 — 隐私安全 | ✅ | bge-small-zh 本地模型 + SQLite |
| P7 | 推理标记 — 不可越级 | ⚠️ | Salience 新边类型（DECISION_CAUSED_BY 等）需 evidence_ref，但 LLM_EXTRACTOR_ENHANCEMENT 未规定证据引用格式 |
| P8 | 零依赖内核 — 可选扩展不进核心 | ⚠️ | `LLMQueryRewriter` 导入 `llm_extractor.LLMClient` — 这本身不引入新依赖（复用已有），但 `SynonymTable` 依赖 `config.py` 中的 200+ 硬编码词表（违背零依赖精神） |

**原则 P7 详细**: `DECISION_CAUSED_BY` 边是推断性的（"决策X是由原因Y导致的"），按原则 7 应标记置信度 0.4 并提供 `evidence_ref`。当前设计未指定。

---

## 四、模块耦合度分析

### 4.1 MemoryStore 参数膨胀

```
v0.1:  MemoryStore(path, max_recall_hops, recall_decay, weights)
v0.2a: MemoryStore(..., llm_extractor, reranker)
v0.2b: MemoryStore(..., embedding_engine, vector_index, llm_query_extract)  ← 当前代码
Phase2: MemoryStore(..., query_expander)  ← 设计中
```

**最终**: 10 个构造参数 + 6 个可选扩展对象。MemoryStore 正在变成 God Object。

**建议**: 考虑引入 `RetrievalPipeline` 或 `MemoryStore.Builder` 模式。至少将 `query_expander`、`embedding_engine`、`vector_index` 聚合为一个 `SearchConfig` 对象。

---

### 4.2 双向依赖风险

```
query_expansion.py
  └── LLMQueryRewriter
        └── imports llm_extractor.LLMClient  ← 单向，可接受

store.py
  ├── imports fusion.reciprocal_rank_fusion
  ├── imports entities.extract_entities
  ├── imports llm_extractor (LLMExtractor, extract_entities_llm, extract_edges_llm)
  └── imports query_expansion (QueryExpander)  ← 设计中新增
```

`query_expansion → llm_extractor` + `store → query_expansion` + `store → llm_extractor`。三层依赖链，但均为单向（无循环），尚可接受。

---

### 4.3 缓存重复

| 缓存 | 位置 | 键 | 容量 |
|------|------|-----|:---:|
| LLM 提取结果缓存 | `llm_extractor` (LRU) | `sha256(text)` | 4096 |
| 查询扩展结果缓存 | `ExpansionCache` (LRU) | `normalized(query)` | 256 |

两者均为线程安全 LRU，但实现独立。如果未来需要统一缓存策略（如磁盘持久化、统一淘汰），需要重构两处。

**建议**: 提取公共 `LRUCache` 基类，避免代码重复。

---

## 五、回退策略审查

| 降级场景 | 触发条件 | 降级行为 | 验证 |
|----------|---------|---------|:---:|
| LLM API 不可用 | `LLMClient.available == False` | `QueryExpander`: 仅同义词；`LLMExtractor`: 跳过 | ✅ test_plan |
| Embedding 模型不可用 | `embedding_engine.available == False` | `_vector_enabled = False`，回退纯关键词 | ✅ TC-VR05/06 |
| FAISS 不可用 | ImportError / 初始化失败 | `_vector_enabled = False` | ✅ REQUIREMENTS §5.5 |
| LLM 返回异常 JSON | `json.JSONDecodeError` | 返回空 `RewriteResult` / 空实体列表 | ⚠️ 仅 query_expansion 有点 |
| 缓存满 (LRU 淘汰) | 超过 max_size | 淘汰最旧条目 | ✅ 设计中有 |

**唯一不足**: LLM JSON 解析失败后的重试策略未定义。是否需要指数退避重试？还是静默跳过等下次？

---

## 六、v0.1 兼容性审查

### 6.1 API 兼容性 ✅

| 接口 | v0.1 | Phase 2 | 兼容 |
|------|------|---------|:---:|
| `MemoryStore(path)` | ✅ | ✅ (新参数均为可选) | ✅ |
| `MemoryStore.save(text)` | ✅ | ✅ (内部追加向量编码) | ✅ |
| `MemoryStore.recall(query, top_k, ns)` | ✅ | ✅ (内部追加向量+扩展) | ✅ |
| `MemoryStore.delete(id)` | ✅ | ✅ (需同步删向量) | ⚠️ |
| `MemoryRecord` | frozen dataclass | frozen dataclass | ✅ (但存在 Bug) |
| MCP 工具签名 | 6 工具固定参数 | 6 工具固定参数 | ✅ |
| `config.py` dataclass 结构 | 3 个子配置 | + `ExpansionConfig` | ✅ (YAML 自动忽略未知 key) |

### 6.2 Schema 兼容性 ⚠️

- 新增列 `memories.embedding_model` 使用 `ALTER TABLE ADD COLUMN` 是安全的
- **但**: 迁移脚本未设计（见 GAP-1）

### 6.3 测试回归 ✅

test_plan.md §5.1 明确要求 "现有 114 个通过测试继续保持通过"，设有回归保护门禁。

### 6.4 delete() 向量同步缺失 ⚠️

当前 `store.py` 的 `delete()` 方法（L425-437）删除了 SQLite 记录但**没有**从 `vector_index` 中移除对应向量。`VectorIndex.remove()` 方法存在（test_plan TC-V04 测试了它），但 `store.delete()` 未调用。

**风险**: 删除记忆后向量索引中存在幽灵向量，可能导致召回已删除内容。

---

## 七、汇总建议

### 🔴 阻塞项 (Phase 2 实施前必须解决)

| # | 问题 | 解决方式 |
|---|------|---------|
| 1 | REQUIREMENTS.md 签名过时 | 更新 §6.1，与代码对齐 |
| 2 | **向量通路是完全死代码** — `_build_scored_candidates` 返回 tuple 但向量路径用 `.id`/`.score` 访问 | 重写 L899-938，与现有数据类型对齐 |
| 3 | VectorIndex 无 namespace 隔离 | 实现 namespace 过滤机制 |
| 4 | delete() 未同步清理 vector_index | 在 delete() 中追加 `self.vector_index.remove(id)` |

### 🟡 建议项 (Phase 2 发布前解决)

| # | 问题 | 解决方式 |
|---|------|---------|
| 5 | llm_query_extract 与 QueryExpander 实体补充重叠 | 统一为一条 LLM 调用路径 |
| 6 | `_recall_single` 方法不存在 | 重构 recall 管道，提取子函数 |
| 7 | Schema 迁移脚本缺失 | 编写 `migrations/002_*.sql` |
| 8 | Salience 新边类型测试缺失 | 纳入 test_plan 或推迟到 Phase 3 |
| 9 | LLM 批量提取触发机制未定义 | 补充批量窗口策略 |
| 10 | QueryExpander 未暴露 MCP 控制 | 增加 `memory.recall` 的 `expand` 参数 |

### 🟢 观察项 (不阻塞，后续优化)

| # | 问题 |
|---|------|
| 11 | MemoryStore 参数膨胀 → 考虑 Builder 模式 |
| 12 | 两个 LRU 缓存可提取公共基类 |
| 13 | SynonymTable 内置词表应可配置化而非硬编码 |
| 14 | LLM JSON 解析失败的重试策略 |

---

## 附录 A: 文档交叉引用矩阵

| 需求/功能 | REQUIREMENTS | LLM_EXTRACTOR | query_expansion | test_plan | 代码 |
|-----------|:---:|:---:|:---:|:---:|:---:|
| EmbeddingEngine | ✅ F-13 | — | — | ✅ (8 cases) | ✅ |
| VectorIndex (FAISS) | ✅ F-13 | — | — | ✅ (10 cases) | ✅ |
| RRF 融合 | ✅ F-15 | — | ✅ 提及 | ✅ (15 cases) | ✅ |
| LLM 查询改写 | ✅ F-14 | — | ✅ 详细设计 | ❌ 无测试 | ❌ 无代码 |
| QueryExpander | — | — | ✅ 详细设计 | ❌ 无测试 | ❌ 无代码 |
| LLM 批量提取 | — | ✅ P0 | — | ❌ 无测试 | ❌ 无代码 |
| LLM 提取缓存 | — | ✅ P0 | — | ❌ 无测试 | ❌ 无代码 |
| Salience 后处理 | — | ✅ P2 | — | ❌ 无测试 | ❌ 无代码 |
| 边类型验证 | — | ✅ P1 | — | ❌ 无测试 | ❌ 无代码 |

**关键发现**: query_expansion 模块设计完整但与 test_plan 完全脱节——test_plan 未覆盖查询扩展的任何场景。

---

> **审核结论**: Phase 2 的核心向量通路（embedding + FAISS + RRF）已在代码中实现且设计合理，但存在 1 个运行时 Bug（frozen dataclass）和 1 个功能缺陷（namespace 隔离）。`QueryExpander` 和 `LLM Extractor 增强` 两个子模块设计完整但与现有代码和测试的衔接存在空白。**建议在解决 4 个阻塞项后启动 Phase 2 实施。**
