# EngramRouter — 确定性需求文档

> **版本**: v1.0-final | **状态**: 待确认，确认后锁定 | **最后更新**: 2026-07-01
>
> **权威性**: 本文档是 EngramRouter 项目唯一的需求来源。所有代码变更必须对齐本文档。
> 当代码实现与本文档冲突时，以本文档为准。

---

## 目录

1. [核心愿景](#一核心愿景)
2. [不可违背的设计原则](#二不可违背的设计原则)
3. [当前问题诊断](#三当前问题诊断)
4. [功能需求](#四功能需求)
5. [技术架构](#五技术架构)
6. [API 与接口规范](#六api-与接口规范)
7. [非功能需求](#七非功能需求)
8. [NOT in scope](#八not-in-scope)
9. [实施路线图](#九实施路线图)
10. [迁移与兼容性](#十迁移与兼容性)

---

## 一、核心愿景

> EngramRouter 是一个 **Agent 按需无损记忆层**，替代「滚动摘要压缩」。

当前主流的 Agent 记忆方案是**滚动摘要**——把对话历史不断压缩，每次压缩都丢失信息。
EngramRouter 反其道而行：**对话原文完整保存，只在需要时调取最相关的片段**。

核心主张：**记忆不应该被反复摘要，而应该被结构化保存、按需取回。**

---

## 二、不可违背的设计原则

以下 8 条原则是项目的基石。任何设计决策、代码变更、功能取舍，都不得违反。

### 原则 1: 证据优先 → 存原文不摘要

- 记忆**必须**以原始文本 (`raw_text`) 作为第一类持久化对象
- **禁止**对原文做有损压缩（摘要、改写、截断）
- 摘要 (`summary`) 仅作为检索辅助字段，**不得替代原始证据**
- 蒸馏/压缩后的记忆必须保留 `evidence_refs` 指向原始证据

### 原则 2: 最小上下文 → 按需召回

- **不**将所有记忆塞入模型上下文。仅返回 `top-k` 条最相关记忆
- 每条召回必须携带可审计的匹配理由 (`match_reason`)
- 不做预计算上下文包——召回以查询为驱动

### 原则 3: 缺失追问 → 不编造

- 当召回的记忆不足以回答时，系统必须明确告知"证据不足"
- Agent 合同：`recall` 不足 → 调用 `gap_check` 或直接询问用户
- **禁止**基于模糊压缩摘要作答

### 原则 4: 可撤销 → 不硬删除

- 用户纠正通过**降权**处理（不删除原始记录）
- 被纠正的记忆标记为 `user_corrected`，原文保留

### 原则 5: 平台无关 → MCP 标准接口

- **只通过 MCP** 标准接口对外暴露能力
- 不绑定任何特定 Agent 框架

### 原则 6: 本地优先 → 隐私安全

- 默认 SQLite (WAL) 本地持久化
- 不做默认云同步
- 不上传用户数据

### 原则 7: 推理标记 → 不可越级

- 基于共现的推断只能标记为低置信度 (`CO_OCCURS_WITH`, 0.4)
- **禁止**自动将推断提升为事实
- 每条边必须携带 `evidence_ref`

### 原则 8: 零依赖内核 → 可选扩展不进核心

- 核心引擎仅依赖 **Python stdlib + SQLite**
- LLM / Embedding / 向量索引均为**可选扩展**

---

## 三、当前问题诊断

> 以下问题来自 v0.1 的实际表现，是 v0.2 架构升级的直接驱动力。

### 问题 1: 硬编码枚举天花板

`config.py` 维护了 200+ 个硬编码词汇，但真实世界的实体是无限的：

| 用户输入 | 能否提取 | 原因 |
|----------|:--------:|------|
| "我刚买了一台 MacBook Pro" | ❌ | `known_objects` 里没有 |
| "我血压有点高" | ❌ | 没有血压相关 pattern |
| "上次体检说胆固醇偏高" | ❌ | 完全无法结构化 |

**结论**: 规则引擎只能处理"已知的已知"，无法处理"未知的已知"。

### 问题 2: 关键词匹配的置信陷阱

当前 `_score()` 是精确子串匹配：

| 查询 | 存储内容 | 命中 |
|------|----------|:----:|
| "键盘什么牌子" | "张三送了我一把 HHKB" | ❌ |
| "通勤方便吗" | "搬家离公司近，通勤方便" | ✅ 巧合命中 |
| "体检结果怎么样" | "上次验血，各项指标正常" | ❌ |

**结论**: 能工作的场景仅限于"词汇恰好重合"。同义词/近义词/上下位词完全失效。

### 问题 3: 权重参数不可维护

`RecallWeights` 拥有 25+ 个手动调整的参数——每新增一种查询类型就要加一个 boost。

### 问题 4: 向量检索缺失

当前完全依赖 FTS5 trigram + LIKE 回退——没有语义相似度计算。这是系统"无法联想"的根本原因。

---

## 四、功能需求

### 4.1 MUST have（核心功能，不可缺失）

| 编号 | 功能 | 说明 |
|------|------|------|
| **F-01** | 原文持久化存储 | `save(text)` → 不可变持久化，原文可精确回读 |
| **F-02** | 证据锚定 | 每条记忆至少附带一条 evidence 引用 |
| **F-03** | 多路候选检索 | FTS5 trigram + LIKE 回退 + 实体名回退，永不丢结果 |
| **F-04** | 规则实体提取 | 人/物/公司/时间/原因/话题/属性，为基础检索兜底 |
| **F-05** | 实体关联传播 | 共享实体 + topic alias + edge hop → 跨记忆联想 |
| **F-06** | 人物冲突隔离 | 问张三不召回李四 |
| **F-07** | MCP Server | 6 工具，stdlib-only |
| **F-08** | 缺失检测 | `gap_check` → 5 种维度 + 追问建议 |
| **F-09** | namespace 隔离 | 多租户互不干扰 |
| **F-10** | 纠正降权 | 标记错误记忆 ×0.3，不硬删 |
| **F-11** | 原始日志层 | 完整对话轮次/工具输出存入 `raw_logs` |
| **F-12** | 蒸馏压缩 | `compact()` 在不删原文的前提下生成结构化记忆 |

### 4.2 SHOULD have（应该实现）

| 编号 | 功能 | 说明 |
|------|------|------|
| **F-13** | **Embedding 向量检索** | sentence-transformers 本地优先，API 兜底。**这是解决"无法联想"的关键** |
| **F-14** | **LLM 结构化提取** | 用 LLM 替代纯规则提取实体/关系/事件 |
| **F-15** | 混合检索 + RRF 融合 | 向量 + 关键词 + 图传播 → RRF → 精排 |
| **F-16** | LLM 语义重排序 | 对 top-20 候选做语义精排 |
| **F-17** | YAML 配置系统 | `~/.engram/config.yaml` 覆盖所有可配参数 |
| **F-18** | 一键安装接入 | 自动检测并配置 5 种 Agent MCP |

### 4.3 NICE to have（后续 Phase）

| 编号 | 功能 | Phase |
|------|------|:----:|
| **F-19** | 人物画像聚合 | 3 |
| **F-20** | 自动遗忘与衰减 | 3 |
| **F-21** | 因果/时序关系建模 | 3 |
| **F-22** | 可选存储后端 | 3 |

---

## 五、技术架构

### 5.1 三层架构

```
接口层 (Interface)
├── MCP Server (主力) ← 6 工具
├── CLI (辅助)
└── Python API (开发)

检索层 (Retrieval)     ← 核心变化在 Phase 2
├── 向量检索 (bge-small-zh + FAISS) 【新增】
├── 关键词检索 (FTS5 + LIKE)      【保留】
├── 图传播 (edge BFS)              【保留】
├── RRF 融合                       【新增】
└── LLM 重排序                     【增强】

存储层 (Storage)
├── SQLite (原文 + 实体 + 边)
└── FAISS (向量索引)
```

### 5.2 save() 数据流

```
文本输入
  → 校验 (长度/非空)
  → 写入 SQLite (memories + evidence + FTS5)
  → 实体提取 (规则 + LLM，双通道合并)
  → 边构建 (CO_OCCURS_WITH + CAUSED_BY + LLM 边)
  → 向量编码 (bge-small-zh → FAISS)   【Phase 2 新增】
  → commit → 返回 memory_id
```

### 5.3 recall() 数据流

```
查询输入
  → 查询理解 (分词 + 实体提取 + LLM 改写) 【Phase 2 增强】
  → 多路候选召回 (并行)                   【Phase 2 新增】
    ├── 向量检索 (FAISS.search → top-50)
    ├── 关键词检索 (FTS5+LIKE → 候选集)
    └── 图传播 (BFS edges → 关联记忆)
  → RRF 融合 → top-100 候选
  → 精排 (规则打分 + 向量相似度)          【Phase 2 增强】
  → LLM 重排序 (可选)                     【Phase 2 增强】
  → 截断 top-k → 返回 MemoryRecord
```

### 5.4 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| 向量模型 | `BAAI/bge-small-zh-v1.5` | 24MB, 512维, 中英文, MTEB 前列 |
| 向量索引 | FAISS (cpu) | 零服务依赖, 内存效率高 |
| 融合算法 | Reciprocal Rank Fusion (RRF) | 无需调参, 多路结果自然融合 |
| LLM 提取 | OpenAI 兼容 API | DeepSeek / OpenAI / OneAPI 通用 |
| 回退策略 | 纯规则引擎 | 向量/LLM 不可用时自动降级 |

### 5.5 回退策略（渐进增强，不退化）

```python
class HybridMemoryStore:
    def __init__(self):
        try:
            self._embedding = EmbeddingEngine()
            self._vector_index = FAISSIndex()
            self._vector_enabled = True
        except (ImportError, Exception):
            self._vector_enabled = False

    def recall(self, query, top_k=5):
        if not self._vector_enabled:
            return self._keyword_recall(query, top_k)  # 完全回退 v0.1
        return self._hybrid_recall(query, top_k)
```

---

## 六、API 与接口规范

### 6.1 核心 Python API

#### MemoryStore 构造函数

```python
MemoryStore(
    path: str | Path | None = None,         # SQLite 路径, None=内存
    max_recall_hops: int = 2,                # 图谱跳跃深度
    recall_decay: float = 0.5,               # 传播衰减
    weights: RecallWeights | None = None,     # 权重配置, None=读config.yaml
    llm_extractor: LLMExtractor | None = None,# LLM提取器(可选)
    reranker: LLMReranker | None = None,      # LLM重排序器(可选)
)
```

#### 核心方法

| 方法 | 签名 | 返回 |
|------|------|------|
| `save` | `(text, source, metadata, namespace)` | `memory_id` |
| `recall` | `(query, top_k, namespace)` | `list[MemoryRecord]` |
| `delete` | `(memory_id)` | `bool` |
| `gap_check` | `(query, memories?, scan_all?)` | `{sufficient, missing[], suggested_question}` |
| `compact` | `(raw_log_id, distilled_text)` | `distilled_id` |
| `consolidate` | `()` | `{merged_entities, removed_edges, ...}` |

#### MemoryRecord

```python
@dataclass(frozen=True)
class MemoryRecord:
    id: str                    # "mem_1"
    raw_text: str              # 完整原文
    summary: str               # 首句摘要 (≤160 chars)
    score: float               # 召回得分
    match_reason: str           # 可审计的匹配理由
    evidence_refs: list[str]   # 证据ID列表
```

### 6.2 MCP 工具（6 个）

| 工具 | 必填参数 | 返回 |
|------|---------|------|
| `memory.save` | `text` | `{memory_id}` |
| `memory.recall` | `query` | `{memories: [...]}` |
| `memory.gap_check` | `query` | `{sufficient, missing, suggested_question}` |
| `memory.compact` | `raw_log_id, distilled_text` | `{distilled_id}` |
| `memory.consolidate` | (无) | `{status, stats}` |
| `memory.delete` | `memory_id` | `{deleted}` |

### 6.3 CLI

```bash
engram save <text>                          # 保存
engram recall <query> [--top-k N]           # 召回
engram gap-check <query> [--scan-all]       # 缺口检测
engram delete <memory_id>                   # 删除
engram-install install                      # 一键接入所有智能体
engram-install status                       # 查看连接状态
engram-mcp --db ~/.engram/memory.db         # 启动 MCP 服务
```

### 6.4 配置

```
加载优先级: ENGRAM_CONFIG 环境变量 > ~/.engram/config.yaml > 代码默认值
```

```yaml
# ~/.engram/config.yaml
entities:
  known_objects: [...]        # 可扩展的实体词表
  object_topic_aliases: {...} # 对象→话题映射
  # ... (全部可配)

recall:
  shared_entity_multiplier: 1.2
  conflicting_person_penalty: 2.2
  # ... (全部权重可调)
```

---

## 七、非功能需求

### 7.1 性能

| 指标 | v0.1 (当前) | v0.2 (目标) |
|------|:----------:|:----------:|
| save 延迟 (规则) | <5ms | <10ms |
| save 延迟 (含LLM) | ~2s | ~50ms (本地向量) |
| recall top-5 (1000条) | ~15ms | ~50ms |
| recall top-5 (10000条) | ~100ms | ~200ms |
| 向量存储 (10000条) | 0 | ~20MB |

### 7.2 语义覆盖（核心指标）

| 指标 | v0.1 | v0.2 目标 |
|------|:----:|:--------:|
| 语义匹配覆盖率 | ~30% | **>90%** |
| 联想命中率（跨词汇） | ~10% | **>70%** |
| 实体提取覆盖率 | ~30% | **>90%** |

### 7.3 安全

- 本地优先，不上传云端
- 推理标记透明（`CO_OCCURS_WITH` = 0.4）
- 无隐藏删除

### 7.4 可靠性

- FTS5 不可用时自动回退全扫描
- 配置损坏时使用默认值
- 117+ 单元测试（目标 90%+ 覆盖率）

---

## 八、NOT in scope

| 不在范围 | 理由 |
|----------|------|
| 替代 Agent 上下文管理 | Agent 框架职责 |
| 通用知识库 / 文档检索 | 不是 RAG |
| 毫秒级实时性 | 50-200ms 可接受 |
| 全局知识图谱 | 不是 GraphRAG |
| 发明新记忆格式 | 不做 Mem0/Letta/Zep |
| 多进程并发写入 | 单进程模型 |
| 默认云同步 | 用户自行备份 |

---

## 九、实施路线图

### Phase 1: 规则引擎稳定化 ✅ 已交付

- 三级候选检索 (FTS5 → LIKE → 实体名)
- 规则实体提取 + Salience 分类
- 人物冲突隔离 + 纠正降权
- MCP Server + 一键安装
- 117+ 测试通过

### Phase 2: LLM-Native 联想引擎 🚧（当前目标）

**时间**: 2-3 周

| 交付 | 说明 |
|------|------|
| 2.1 Embedding 引擎 | `EmbeddingEngine` (bge-small-zh 本地 + API 兜底) |
| 2.2 FAISS 向量索引 | `VectorIndex` (增量写入, 持久化, 热切换模型) |
| 2.3 RRF 混合融合 | 向量 + 关键词 + 图传播 → 统一排序 |
| 2.4 LLM 查询改写 | 同义词扩展 + 实体补充 |
| 2.5 LLM 提取器增强 | 批量模式 + 缓存 + 自定义 prompt |
| 2.6 回退策略 | 向量不可用时完全退化为 Phase 1 |

**验收标准**:
- embedding recall 在 50+ benchmark 上 precision@5 ≥ 规则引擎
- 混合检索 recall@10 比纯规则提升 ≥ 20%
- 全量 v0.1 测试继续通过（100% 向后兼容）

### Phase 3: True Associative Recall 📋

- 人物画像聚合（跨会话）
- 自动遗忘与衰减
- 因果链推理
- 时序事件线

---

## 十、迁移与兼容性

### Phase 1 → Phase 2

| 维度 | 策略 |
|------|------|
| **Schema** | 新增 `memories.embedding_model` 字段，其余不变 |
| **API** | 所有新增参数/方法 100% 向后兼容 |
| **MCP** | 仅增加可选参数，不修改 required |
| **配置** | 新增 `embedding:` 段，自动忽略未知 key |

### 回滚

```bash
# 禁用向量检索
export ENGRAM_DISABLE_VECTOR=1

# 或直接回退版本
pip install engram-router==0.1.0
```

---

> **锁定声明**: 本需求文档经用户最终确认后，成为项目的唯一需求基准。此后任何功能变更必须先修订本文档，再修改代码。

> **文档维护规则**: 当代码实现与本文档矛盾时，以本文档为需求基准——代码应被修正以满足需求，或需求应被正式修订。
