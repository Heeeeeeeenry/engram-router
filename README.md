# EngramRouter — Lossless Agent Memory

> 一种替代滚动摘要的 Agent 记忆层：结构化存储，按需召回，无损证据

## 一句话

现有 Agent 用「摘要压缩」保存对话记忆，压缩会失真。
EngramRouter 把对话原封不动保存，只在需要时召回最相关的片段——
不相关的绝不进上下文，需要的带着证据回来。

## 不是什么

- ❌ 不是 RAG 向量库 —— 不做 embedding
- ❌ 不是 Mem0/Letta/Zep —— 不发明新记忆格式
- ❌ 不是 GraphRAG —— 不建全局知识图谱

## 是什么

```
用户对话 → 原样存入 SQLite → 需要时 FTS5 + 实体关联召回 → 带证据返回
```

- **零压缩失真** —— 存原文，不摘要
- **证据链可追溯** —— 每条记忆记录来源
- **MCP 标准接口** —— 一行配置接入 Hermes / Claude / Codex / OpenClaw
- **零依赖** —— 纯 Python stdlib + SQLite

## 快速开始

```bash
pip install engram-router

# CLI 测试
engram save "张三送了我一把HHKB键盘，因为生日"
engram recall "同事送的键盘什么牌子"

# 一键接入所有智能体
engram-install install

# 作为 MCP server 运行
engram-mcp --db ~/.engram/memory.db
```

## 项目结构

```
├── src/engram_router/
│   ├── store.py          # SQLite 存储 + FTS5 召回引擎
│   ├── entities.py       # 规则实体提取（可配置）
│   ├── config.py         # 统一配置（支持 ~/.engram/config.yaml）
│   ├── mcp_server.py     # MCP stdio JSON-RPC 服务
│   ├── install.py        # 一键接入智能体
│   ├── llm_reranker.py   # 可选 LLM 语义重排序
│   └── cli.py            # CLI 工具
├── tests/                # 127+ 测试
└── docs/                 # 架构 + Schema 文档
```

## 设计原则

1. **证据优先** —— 存原文，不摘要，不推断
2. **最小上下文** —— 只传 top-k 证据给模型
3. **缺失追问** —— 记忆不够时提示补充
4. **可撤销** —— 更正降权不硬删
5. **平台无关** —— MCP 标准接口，任何 Agent 可用
