# EngramRouter 项目概要

## 原始目标（始终不变）

构建一个 **Agent 按需无损记忆层**，替代「滚动摘要压缩」。

核心主张：
> 记忆不应该被反复摘要，而应该被结构化保存、按需取回。

## 当前状态（v0.1）

已实现：
- [x] SQLite 持久化，WAL 模式，支持并发
- [x] FTS5 trigram + LIKE 回退 + 实体名回退 三级候选检索
- [x] 规则实体提取（可配置词表）
- [x] 实体关联传播（共享实体 / topic alias / edge hop）
- [x] 人物冲突隔离（penalty + topic 压制 + match boost）
- [x] MCP Server（6 工具，零第三方依赖）
- [x] 一键安装：`engram-install install`（Hermes / Claude / Codex / OpenClaw）
- [x] LLM 语义重排序（可选，需 API key）
- [x] YAML 配置系统（`~/.engram/config.yaml`）
- [x] 277 单元测试，265 passed

已知局限：
- [ ] 纯规则引擎 → 语义覆盖有限（跨人物关联、同义扩展
- [ ] 无向量检索 → 无法捕捉深层语义相似性
- [ ] 实体提取靠枚举 → 维护成本高，覆盖面窄

## 下一阶段方向

### Phase 2: LLM-Native 联想引擎

核心变化：**从「规则提取 + 关键词匹配」升级为「LLM 提取 + 语义检索」**

```
Phase 1 (当前)          →    Phase 2 (目标)
─────────────────────────────────────────────
规则实体提取 (200行)     →    LLM 结构化提取 (entities + relations + events)
FTS5 trigram 关键词      →    Embedding 向量相似度
手动权重调参             →    LLM 打分 / 重排序
```

新增依赖：
- `sentence-transformers` (本地向量)
- 或 OpenAI-compatible embedding API

### Phase 3: True Associative Recall

- 多条记忆间的因果/时序/从属关系
- 人物画像聚合（跨会话学习偏好）
- 自动遗忘（衰减 + 合并）

## 架构

```
Agent Adapters (MCP / CLI / Python API)
        │
   ┌────┴────┐
   │  Recall │  ← FTS5 + LIKE + Entity + LLM Reranker
   │  Ingest │  ← save() → extract entities → store
   └────┬────┘
        │
   SQLite (WAL, FTS5, edges, entities)
```

## 不做的事

- 不替代 Agent 的上下文管理
- 不做通用知识库 / 文档检索
- 不追求实时性（毫秒级）
- 不绑定特定 Agent 框架
