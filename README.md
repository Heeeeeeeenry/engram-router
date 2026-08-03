# EngramRouter — Lossless Agent Memory

> 结构化记忆层：存原文，按需召回，多层融合 + cross-encoder 精排，自动衰减

## 一句话

现有 Agent 用「摘要压缩」保存对话记忆，压缩会失真。
EngramRouter 把对话原封不动保存，只在需要时召回最相关的片段 —
不相关的绝不进上下文，需要的带证据回来。

## 文档入口

- [项目概要](docs/PROJECT_BRIEF.md): 当前状态、保留原则、可改进点、可修改点、删除点
- [存储结构](docs/SCHEMA.md): SQLite 实际 schema
- [MCP 协议参考](docs/MCP.md): MCP stdio JSON-RPC 接口完整说明
- [贡献说明](CONTRIBUTING.md): 变更约束和协作规则
- [更新日志](CHANGELOG.md): 版本变更记录

## 核心能力

```
用户对话 → 原样存入 SQLite（永不摘要）
       ↓
  多层召回: FTS5(关键词) + 实体图(BFS) + 向量(bge-m3 1024d)
       ↓
  cross-encoder 精排 → LLM rerank → 带证据返回
       ↓
  自动更新人物画像 + 因果链 + 时间线 + Ebbinghaus 衰减
```

| 层 | 技术 | 作用 |
|---|---|---|
| 1 | FTS5 trigram + CJK bigram | 关键词快速命中 |
| 2 | 实体图 BFS 多跳 | "HHKB" → "张三" → "键盘" 联想 |
| 3 | bge-m3 1024d 向量 + FAISS HNSW | 语义召回（"开心"→"高兴"） |
| 精排 | bge-reranker-v2-m3 cross-encoder | P@1 从 0.61 提升到 0.83 (+21.7pp) |
| 重排 | LLM semantic rerank（可选） | 语义级精确排序 |

## Phase 2+ 功能

| 模块 | 属性 | 功能 |
|------|------|------|
| CrossEncoderReranker | `store.cross_encoder` | bge-reranker-v2-m3 精排，实测 MRR 0.74→0.88 |
| HyDEExpander | `store.hyde` | 假想答案向量召回，抽象指代兜底 |
| ColBERTReranker | `store.colbert_reranker` | Late-interaction 精排（Jina ColBERT v2） |
| IntentClassifier | `store.intent_classifier` | 8-class 意图软概率（brand/identity/eval/reason/...） |
| QueryExpander | `store.query_expander` | 同义词 + multi-query 变体 + RRF fusion |
| PersonaStore | `store.persona` | 跨 session 人物画像聚合（年龄/偏好/职业） |
| CausalChain | `store.causal` | 因果链推理（内存不足→变慢→部署新版本） |
| Timeline | `store.timeline` | 时间线事件查询（按人物/时间范围） |
| ForgettingEngine | `store.forgetting` | Ebbinghaus 衰减 + 自动遗忘标记 + consolidate |
| CorefTracker | — | Fastcoref 代词消解（"她特别喜欢猫"→小李） |
| HanLP NER | — | 实体识别（MSRA_NER_ELECTRA_SMALL_ZH） |
| EntityCanonicalizer | — | 实体别名表 + canonical merging |

```python
from engram_router import MemoryStore

store = MemoryStore()
store.save("张三今年30岁，是程序员，喜欢钓鱼", namespace="user_a")
store.save("李四住北京，养了只猫", namespace="user_a")

# 三层 + CE + LLM 重排
records = store.recall("张三的爱好", top_k=5)

# 人物画像
p = store.persona.aggregate("张三")

# 因果链
store.causal.trace_causes("数据库")

# 时间线
store.timeline.get_timeline(person="张三", limit=20)

# 自动衰减（每次 recall 后触发）
# → 访问计数 +1，低活跃记忆自动标记 forgotten
```

## 不是 RAG 向量库

- ❌ 不是 Mem0/Letta/Zep — 不发明新记忆格式、不摘要改写
- ❌ 不是 GraphRAG — 不建全局知识图谱
- ✅ 是 **证据优先 + 结构化路由 + 生命周期管理** 的混合检索引擎
- ✅ 单文件 SQLite 部署，零外部依赖即可运行
- ✅ 独占能力：PersonaStore / CausalChain / Timeline / ForgettingEngine / Corrections

## 多租户

通过 `namespace` 参数实现轻量级多租户隔离：

```python
store.save("Alice 喜欢设计", namespace="tenant_a")
store.save("Bob 喜欢后端", namespace="tenant_b")

# 召回仅返回当前租户的数据
store.recall("Alice 的爱好", namespace="tenant_a")
# → 不会泄漏 tenant_b 的数据

# 删除也受租户保护
store.delete("mem_123", namespace="tenant_b")
# → 租户 a 无法删除租户 b 的记忆
```

## 快速开始

```bash
pip install engram-router[llm]  # 含向量搜索 + cross-encoder

# CLI 测试
engram save "张三送了我一把 HHKB 键盘，因为生日"
engram recall "同事送的键盘什么牌子"
# → "张三送我 HHKB 键盘" — FTS + 实体图 + 向量 + CE 四层命中

# 一键接入支持 MCP 的智能体（Claude / Cursor / Windsurf 等）
engram-install install

# MCP server
engram-mcp --db ~/.engram/memory.db
```

## 环境变量

### 功能开关

| 变量 | 作用 |
|------|------|
| `ENGRAM_SKIP_VECTOR=1` | 跳过向量模型加载（测试 / 纯 FTS 场景） |
| `ENGRAM_SKIP_CE=1` | 跳过 cross-encoder 重排（低延迟场景） |
| `ENGRAM_FORCE_CE=1` | 仅启用 CE 重排，跳过其他通道 |
| `ENGRAM_SKIP_COLBERT=1` | 跳过 ColBERT late-interaction 重排 |
| `ENGRAM_SKIP_COREF=1` | 跳过 Fastcoref 代词消解 |
| `ENGRAM_SKIP_HANLP=1` | 跳过 HanLP NER 实体识别 |
| `ENGRAM_SKIP_CANONICAL=1` | 跳过实体别名 canonicalization |

### 云端控制

| 变量 | 作用 |
|------|------|
| `ENGRAM_ALLOW_CLOUD=1` | 一次性打开所有云端调用 |
| `ENGRAM_ALLOW_CLOUD_LLM=1` | 只允许 LLM 走云端 |
| `ENGRAM_ALLOW_CLOUD_EMBEDDING=1` | 只允许 embedding 走云端 |
| `ENGRAM_ALLOW_CLOUD_RERANKER=1` | 只允许重排走云端 |

### LLM / API

| 变量 | 作用 |
|------|------|
| `DEEPSEEK_API_KEY` | LLM API key（HyDE / LLM 抽取 / 查询改写 / intent classifier） |
| `ENGRAM_LLM_BASE_URL` | 自定义 LLM API 地址（默认 DeepSeek 兼容 endpoint） |
| `ENGRAM_LLM_MODEL` | 自定义 LLM 模型名（默认 deepseek-v4-pro） |
| `ENGRAM_API_KEY` | MCP server 认证密钥 |
| `ENGRAM_EMBEDDING_API_KEY` | 云端 embedding API key（替代本地模型） |
| `ENGRAM_EMBEDDING_API_BASE` | 云端 embedding API 地址 |
| `HF_TOKEN` | HuggingFace token（下载本地模型） |

### 硬件 / 调试

| 变量 | 作用 |
|------|------|
| `ENGRAM_CE_DEVICE` | Cross-encoder 设备（`cpu`/`mps`/`cuda`，默认自动检测） |
| `ENGRAM_COLBERT_DEVICE` | ColBERT 设备（同上） |
| `ENGRAM_SSL_VERIFY=0` | 跳过 SSL 验证（内部代理） |
| `ENGRAM_CONFIG` | 自定义配置文件路径（默认 `~/.engram/config.yaml`） |

默认情况下即便配置了 API key，云端调用也是关闭的（本地优先原则）。以上变量取值 `1 / true / yes / on`（大小写不敏感）才生效，其他值一律视为关闭。

**测试**: `pytest -q` → **600 passed, 4 skipped, 12 xfailed**

## 项目结构

```
engram_router/
├── store/                  # 核心引擎（~4.5K 行）
│   ├── core.py             #   MemoryStore（save/delete/schema/consolidate）
│   ├── recall.py           #   recall pipeline（召回 + 评分 + deadline）
│   ├── candidates.py       #   FTS5 候选检索
│   ├── scoring.py          #   RecallWeights + 打分函数
│   ├── graph.py            #   实体图 + BFS edge expansion
│   ├── channels.py         #   三通道（fts/vector/hyde）调度
│   ├── pipeline.py         #   类型化 pipeline（RecallStage Protocol）
│   ├── records.py          #   MemoryRecord dataclass
│   ├── query_intent.py     #   查询意图检测（_asks_* 系列）
│   ├── metrics.py          #   Prometheus 指标 stub
│   └── trace.py            #   RecallTracer 调试 trace
├── entities.py             # 规则实体提取
├── entity_canon.py         # 实体别名表 + canonical merging
├── coref.py                # Fastcoref 代词消解
├── hanlp_ner.py            # HanLP NER 实体识别
├── embedding.py            # bge-m3 1024d 本地嵌入
├── vector_index.py         # FAISS HNSW 向量索引 + 暴力余弦兜底
├── cross_encoder.py        # bge-reranker-v2-m3 精排
├── colbert_reranker.py     # Jina ColBERT v2 late-interaction
├── hyde.py                 # HyDE 假想答案向量召回
├── fusion.py               # RRF / ISF / learned fusion
├── persona.py              # 人物画像聚合
├── causal.py               # 因果链 + 时间线
├── forgetting.py           # Ebbinghaus 衰减引擎 + consolidate
├── config.py               # 统一配置
├── query_expansion.py      # 查询改写 + multi-query 变体
├── intent_classifier.py    # 8-class LLM 意图分类
├── llm_extractor.py        # LLM 实体/边提取
├── llm_reranker.py         # LLM 语义重排
├── prompt_guard.py         # Prompt 注入防御
├── mcp_server.py           # MCP stdio JSON-RPC 服务（6 tools）
├── benchmark.py            # 基准测试工具
├── install.py              # 一键接入智能体
├── cli.py                  # CLI 工具
└── __init__.py             # 公开 API
```

## 设计原则

1. **证据优先** — 存原文，不摘要，不推断
2. **最小上下文** — 只传 top-k 证据给模型
3. **渐进增强** — LLM 是可选项，离线也能跑
4. **软遗忘** — forgotten 标记降权，不硬删除
5. **平台无关** — MCP 标准接口，任何 Agent 可用
6. **多租户安全** — namespace 隔离，entities/edges/timed_events 全部 scoped
