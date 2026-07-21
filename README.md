# EngramRouter — Lossless Agent Memory

> 结构化记忆层：存原文，按需召回，三层检索，自动衰减

## 一句话

现有 Agent 用「摘要压缩」保存对话记忆，压缩会失真。
EngramRouter 把对话原封不动保存，只在需要时召回最相关的片段——
不相关的绝不进上下文，需要的带证据回来。

## 文档入口

- [项目概要](docs/PROJECT_BRIEF.md): 当前状态、保留原则、可改进点、可修改点、删除点
- [存储结构](docs/SCHEMA.md): SQLite 实际 schema
- [贡献说明](CONTRIBUTING.md): 变更约束和协作规则

## 核心能力

```
用户对话 → 原样存入 SQLite
       ↓
  三层召回: FTS5(关键词) + 实体图(BFS多跳) + 向量(bge-small-zh 512d)
       ↓
  带证据返回 + 自动更新人物画像
```

| 层 | 技术 | 作用 |
|---|------|------|
| 1 | FTS5 trigram + CJK ngram | 关键词快速命中 |
| 2 | 实体图 BFS 多跳 | "HHKB" → "张三" → "键盘" 联想 |
| 3 | bge-small-zh-v1.5 512d 本地向量 | 语义兜底（"开心"→"高兴"） |

## Phase 3 模块（自动启用）

| 模块 | 属性 | 功能 |
|------|------|------|
| PersonaStore | `store.persona` | 跨 session 人物画像聚合（年龄/偏好/职业） |
| CausalChain | `store.causal` | 因果链推理（内存不足→变慢→部署新版本） |
| Timeline | `store.timeline` | 时间线事件查询（按人物/时间范围） |
| ForgettingEngine | `store.forgetting` | Ebbinghaus 衰减 + 自动遗忘标记 |

```python
from engram_router import MemoryStore

store = MemoryStore()
store.save("张三今年30岁，是程序员，喜欢钓鱼")
store.save("李四住北京，养了只猫")

# 人物画像
p = store.persona.aggregate("张三")
print(p.age, p.occupation)  # 自动聚合

# 因果链
store.causal.trace_causes("数据库")

# 自动衰减（每次 recall 后触发）
store.recall("张三的爱好")
# → 访问计数 +1，低活跃记忆自动标记 forgotten
```

## 不是 RAG 向量库

- ❌ 不是 Mem0/Letta/Zep —— 不发明新记忆格式
- ❌ 不是 GraphRAG —— 不建全局知识图谱
- ✅ 是 **元数据优先 + 向量兜底** 的混合检索引擎

## 快速开始

```bash
pip install engram-router[llm]  # 含向量搜索

# CLI 测试
engram save "张三送了我一把 HHKB 键盘，因为生日"
engram recall "同事送的键盘什么牌子"
# → "张三送我 HHKB 键盘" — 关键词 + 实体图 + 向量三层命中

# 一键接入智能体
engram-install install

# MCP server
engram-mcp --db ~/.engram/memory.db
```

## 环境变量

| 变量 | 作用 |
|------|------|
| `ENGRAM_SKIP_VECTOR=1` | 跳过高开销的向量模型加载，测试用 |
| `ENGRAM_EXPANSION_LLM=0` | 关闭 LLM 查询扩展 |
| `DEEPSEEK_API_KEY` | LLM 功能（可选，降级可用） |
| `ENGRAM_ALLOW_CLOUD=1` | 一次性打开所有云端调用（LLM 抽取 / 云端 embedding / LLM 重排） |
| `ENGRAM_ALLOW_CLOUD_LLM=1` | 只允许 LLM 抽取 / 查询改写走云端 |
| `ENGRAM_ALLOW_CLOUD_EMBEDDING=1` | 只允许 embedding 调用云端 API |
| `ENGRAM_ALLOW_CLOUD_RERANKER=1` | 只允许 LLM 语义重排调用云端 |

默认情况下即便配置了 API key，云端调用也是关闭的（本地优先原则）。以上变量取值 `1 / true / yes / on`（大小写不敏感）才生效，其他值一律视为关闭。

测试：`ENGRAM_SKIP_VECTOR=1 pytest -q`  → **265 passed, 12 xfailed（4 秒）**

## 项目结构

```
engram_router/
├── store.py              # 核心引擎 (2561行)
├── entities.py           # 规则实体提取（可配置）
├── embedding.py          # bge-small-zh 本地嵌入
├── vector_index.py       # FAISS 向量索引 + 暴力余弦兜底
├── persona.py            # 人物画像聚合
├── causal.py             # 因果链 + 时间线
├── forgetting.py         # Ebbinghaus 衰减引擎
├── fusion.py             # RRF 融合 + 加权评分
├── config.py             # 统一配置（~/.engram/config.yaml）
├── query_expansion.py    # 查询改写（规则 + 可选 LLM）
├── llm_extractor.py      # LLM 实体提取（可选）
├── llm_reranker.py       # LLM 语义重排序（可选）
├── mcp_server.py         # MCP stdio JSON-RPC 服务
├── install.py            # 一键接入智能体
└── cli.py                # CLI 工具
```

## 设计原则

1. **证据优先** — 存原文，不摘要，不推断
2. **最小上下文** — 只传 top-k 证据给模型
3. **渐进增强** — LLM 是可选项，离线也能跑
4. **软遗忘** — forgotten 标记降权，不硬删除
5. **平台无关** — MCP 标准接口，任何 Agent 可用
