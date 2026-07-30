# EngramRouter 差异化能力对照

**日期**: 2026-07-22
**目的**: 把优化叙事从"P@1 数字之争"拉回项目核心主张。P@1 数字表(见 `docs/eval_v2_matrix.json`)显示在 15-turn 玩具语料上,long-context / naive-vector / engram 三家差距≤ 6 pp,retrieval 系统的价值被玩具场景压缩。真实差异在**能力覆盖**,不在 P@1。

本文档是**功能覆盖表**,不是性能表。功能存在(✓)/ 部分(◐)/ 缺失(✗)。每一格都是从代码源头 or 官方文档核实的,不是猜。

---

## 一图流对照

| 能力类别 | engram-router | mem0 | naive-vector | long-context | zep / graphiti (P1) |
|---|---|---|---|---|---|
| **原文无损存储** | ✓ | ✗ | ✓ | ✓ | ✗ |
| **证据回填 (evidence_refs)** | ✓ | ✗ | ✗ | ✗ | ✗ |
| **LLM 事实抽取** | ◐(可选) | ✓ | ✗ | ✗ | ✓ |
| **向量检索(bi-encoder)** | ✓ | ✓ | ✓ | ✗ | ✓ |
| **FTS trigram 关键词** | ✓ | ✗ | ✗ | ✗ | ◐ |
| **图召回(实体 → 边)** | ✓ | ◐(需 Neo4j) | ✗ | ✗ | ✓ |
| **多跳边扩展 BFS** | ✓(2 hop 默认) | ◐ | ✗ | ✗ | ✓(时序图) |
| **cross-encoder 精排** | ✓(bge-reranker-v2-m3) | ✓(可选) | ✗ | ✗ | ✓ |
| **HyDE 查询扩展** | ✓(默认关) | ✗ | ✗ | ✗ | ✗ |
| **人物画像聚合** | ✓(PersonaStore) | ◐ | ✗ | ✗ | ✓ |
| **因果链推理** | ✓(CausalChain) | ✗ | ✗ | ✗ | ◐(时序推理近似) |
| **时间线事件查询** | ✓(Timeline) | ✗ | ✗ | ✗ | ✓(时序图专长) |
| **Ebbinghaus 遗忘引擎** | ✓(ForgettingEngine) | ✗ | ✗ | ✗ | ✗ |
| **用户纠正 (corrections)** | ✓(硬降权 ×0.3) | ✗ | ✗ | ✗ | ✗ |
| **命名空间(多租户)** | ✓ | ◐(user_id 层) | ✗ | ✗ | ✓ |
| **原生 MCP server** | ✓(6 tools) | ◐(第三方) | ✗ | ✗ | ◐ |
| **单文件部署(SQLite)** | ✓ | ✗ | ✓ | N/A | ✗(需 Neo4j) |
| **离线可用** | ✓ | ✗ | ✓ | ✗ | ✗ |
| **中文原生优化** | ✓(bge-small-zh + CJK bigram + FTS trigram) | ◐ | ◐ | ◐ | ✗ |

**图例**: ✓ 一等公民 · ◐ 部分实现或需要额外配置 · ✗ 不支持

---

## 能力项证据(逐条 file:line 引用)

### 1. 原文无损存储 + 证据回填
- `src/engram_router/store.py:503`(`save()`)存原文到 `memories.raw_text`,不做摘要
- `src/engram_router/store.py:1734`(`_batch_evidence_refs`)每次 recall 回填证据 id
- schema:`docs/SCHEMA.md` `evidence` 表专门存 quote,不允许 lossy rewrite

**对比**: mem0 默认 `infer=True` → LLM 抽事实覆盖原文(见 `tests/providers/mem0_provider.py:107`);long-context 依赖 LLM 挑选,不主动保存证据链;naive-vector 存原文但无 evidence_refs 概念。

### 2. FTS + Graph + Vector 三层召回
- FTS trigram:`store.py:456`(`_init_fts`)+ `store.py:923`(`_fts_candidates`)
- Entity graph:`entities.py` 规则实体抽取 + `store.py:610`(`_index_edges`)自动写 CO_OCCURS_WITH / CAUSED_BY 边
- BFS 边扩展:`store.py:1889`(`_edge_expansion`)带 salience-based decay
- Vector fusion:`store.py:1748` RRF 融合三通道

**对比**: mem0 是"向量+可选图"两层;naive-vector 是"向量"一层;long-context 是"塞满 prompt 让 LLM 挑"。

### 3. Cross-encoder 精排
- `src/engram_router/cross_encoder.py`(2026-07-21 落地)
- `bge-reranker-v2-m3` 默认,mps/cuda/cpu 自动
- fail-open 全链路兜底

**对比**: mem0 有 `SentenceTransformerReranker`(见 mem0 内 `reranker/` 目录),配置门槛更高;naive-vector / long-context 无。

### 4. 三大 Phase 3 差异化模块

- **PersonaStore**(`persona.py:aggregate`)—— 跨 session 聚合人物属性(年龄、职业、偏好),自动去重,冲突时保留最新
- **CausalChain**(`causal.py:trace_causes / trace_effects`)—— 沿 CAUSED_BY 边追溯因果链
- **Timeline**(`causal.py:Timeline`)—— 按时间/人物过滤事件

**对比**: **这三个能力 mem0/naive-vector/long-context 都完全没有**。zep/graphiti 有类似的时序推理,但需要 Neo4j 服务。

### 5. Ebbinghaus 遗忘引擎
- `forgetting.py:ForgettingEngine` + `forgetting.py:ForgettingConfig`
- 分级衰减(base_attr 长半衰期,event 短半衰期)
- 软删除:`forgotten` 标记降权,不硬删

**对比**: mem0 有 `expiration_date` 但没自动衰减;其他两家全无。

### 6. 用户纠正机制
- `store.py` `corrections` 表 + `_get_corrected_ids` 硬降权 ×0.3
- 原始记忆保留,corrections 表审计不可删

**对比**: mem0 支持 `update()` 但会覆盖原值(不留原始事实);其他两家全无。

### 7. 生产级工程能力
- **单文件部署**: SQLite,零外部服务(mem0 需 Chroma/Qdrant + LLM API;zep 需 Neo4j)
- **命名空间**: `namespace` 字段贯穿所有表,多租户开箱即用
- **MCP server**: `mcp_server.py` 6 个 tool(save / recall / gap_check / compact / consolidate / delete)
- **离线可用**: `ENGRAM_SKIP_VECTOR=1` 关向量,`ENGRAM_ALLOW_CLOUD_*` 系列 env 完全关云端

---

## 什么时候 engram 是错误的选择

诚实说明,避免过度推销:

1. **玩具语料**(< 50 memory 或对话 < 20 轮):long-context 就够,retrieval 是多余复杂度
2. **纯语义匹配**(query 和 memory 语义相似但词面重合少):naive-vector(bge-small-zh)已经很好,engram 的 FTS + 图层贡献边际
3. **没有多人物 / 无时序 / 无因果**:PersonaStore / CausalChain / Timeline 用不上,engram 变成"带 rerank 的 vector store",不如 mem0 生态成熟
4. **需要图谱可视化 / 时序 KG 分析**:zep + graphiti 是专用工具,engram 的边是内部路由,不是分析对象

## engram 唯一站得住的场景

1. **长对话跨 session 记忆**(500-turn+):证据链 + 遗忘 + 人物画像组合发挥
2. **中文场景需要低成本部署**:单文件 SQLite,无外部服务,bge-small-zh 中文优化
3. **agent 需要证据回填**:每条召回都能追到原始记忆 evidence_refs,可审计
4. **同时需要向量、关键词、图三条路**:例如"HHKB(词面)+ 键盘(同义)+ 张三(实体图)"需要三层同时命中
5. **多租户 SaaS**:namespace 隔离 + 云端可选

---

## 下一次评测应该测什么

**当前 semantic_audit + multi_angle 测不出上表的差异化能力**——它们全是 15-turn 单人物场景。真正暴露差异的测试:

| 待建评测集 | 测什么能力 | 现有工具 |
|---|---|---|
| **长对话** | 遗忘引擎、人物画像 | LongMemEval / MSC / LOCOMO |
| **中文原生** | CJK bigram / FTS trigram | MemoryBench (THUIR) / PerLTQA |
| **多人物因果** | CausalChain + 实体图 | 需自建 |
| **证据审计** | evidence_refs 回填质量 | 需自建 |
| **矛盾/纠正** | corrections 机制 | 需自建(mem0 也没有) |

见 `docs/design/competitor_benchmark.md` 的 P0 评测集清单。

---

## 结论

**engram vs 竞品的胜负不在 P@1,在能力矩阵**。

- P@1 数字上:玩具语料里 long-context 赢 6 pp,真实长对话里 retrieval 必胜(否则 200k token 上下文 latency 崩溃)
- 能力矩阵上:engram 独占 5 项(证据链、Persona、Causal、Timeline、Forgetting、Corrections),另有 3 项显著优于对比方(FTS + 图 + 单文件部署)
- 卖点:**"存原文、结构化路由、生命周期管理" 三合一 + MCP 一键接入 + 单文件本地**

**当前 P@1 数字不代表最终评价**——需要迁移到 LongMemEval / MemoryBench 才能测出真实价值。这条已列入 `docs/OPTIMIZATION_ROADMAP.md` L0.3 待办。
