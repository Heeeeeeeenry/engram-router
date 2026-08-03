# MCP Server 协议参考

EngramRouter 通过 MCP stdio JSON-RPC 协议暴露 6 个 tools，任何支持 MCP 的 Agent
（Claude Code / Cursor / Windsurf 等）都可以直接接入。

## 启动

```bash
# 默认环境变量 DEEPSEEK_API_KEY 认证
engram-mcp --db ~/.engram/memory.db

# 指定 API key
engram-mcp --db ~/.engram/memory.db --api-key "sk-xxx"

# 调试模式（stderr 输出日志）
engram-mcp --db ~/.engram/memory.db --debug
```

## Tools

### `memory.save`

保存一条记忆。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `text` | string | ✅ | 记忆文本（最大 64 KB） |
| `namespace` | string | | 命名空间/租户，默认为 `"default"`，限 64 字符，仅允许 `[a-zA-Z0-9_\-\.]` |
| `source` | string | | 来源标记，默认 `"mcp"` |

**返回**: `{"memory_id": "mem_42"}`

### `memory.recall`

召回相关记忆。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | ✅ | 查询文本（最大 64 KB） |
| `top_k` | integer | | 返回条数（默认 5，范围 [1, 100]） |
| `namespace` | string | | 命名空间，默认 `"default"` |

**返回**: `{"memories": [...]}`  — 每个 memory 含 `id`, `raw_text`, `summary`, `score`, `match_reason`, `evidence_refs`。

### `memory.gap_check`

检查记忆空白。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | ✅ | 查询文本 |
| `namespace` | string | | 命名空间 |
| `scan_all` | boolean | | 全量扫描（默认 false） |

**返回**: `{"gaps": [...], "missing": [...]}`

### `memory.compact`

将原始日志蒸馏为精简记忆，保留证据引用。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `raw_log_id` | string | ✅ | 原始日志 ID（最大 128 字符） |
| `distilled_text` | string | ✅ | 蒸馏文本（最大 64 KB） |

**返回**: `{"distilled_id": "dst_42"}`

### `memory.consolidate`

合并重复实体名（大小写/空白变体），清理孤立边和重复边。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `namespace` | string | | 命名空间（操作仅限一个 namespace） |

**返回**: `{"status": "ok", "stats": {"merged_entities": N, "removed_edges": N, ...}}`

### `memory.delete`

删除一条记忆（带 namespace 隔离保护）。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `memory_id` | string | ✅ | 记忆 ID（最大 128 字符，仅可打印字符） |
| `namespace` | string | | 命名空间（namespace 不匹配则删除失败） |

**返回**: `{"deleted": true}` 或 `{"deleted": false}`

## 认证

通过 `ENGRAM_API_KEY` 环境变量认证。所有 tools/call 请求在不设置此变量的情况下
返回 `-32001` (AUTH_ERROR)。

```bash
ENGRAM_API_KEY="sk-xxx" engram-mcp --db memory.db
```

## JSON-RPC 协议

本 MCP server 实现 MCP 2024-11-05 协议版本的标准 JSON-RPC 2.0 接口：

- `initialize` — 返回 `protocolVersion`, `capabilities`, `serverInfo`
- `initialized` — 通知（无响应）
- `ping` — 心跳
- `tools/list` — 返回 6 个 tools 的 schema
- `tools/call` — 调用 tool

错误码：

| 码 | 含义 |
|---|------|
| -32600 | Invalid Request |
| -32601 | Method not found |
| -32700 | Parse error |
| -32001 | Auth error (ENGRAM_API_KEY) |
| -32002 | Server not initialized |

## 接入 Claude Code

```bash
# 一键安装
engram-install install

# 或手动配置 claude_desktop_config.json / .claude/mcp.json:
{
  "mcpServers": {
    "engram-router": {
      "command": "engram-mcp",
      "args": ["--db", "/path/to/memory.db"],
      "env": {"ENGRAM_API_KEY": "sk-xxx"}
    }
  }
}
```

## 输入安全

所有输入参数经过校验/夹紧：

| 参数 | 限制 |
|------|------|
| `text` / `query` / `distilled_text` | 最大 64 KB |
| `namespace` | 最大 64 字符，仅 `[a-zA-Z0-9_\-\.]`，非法值静默设为 `"default"` |
| `memory_id` | 最大 128 字符，仅可打印字符，非法值抛出 `ValueError` |
| `top_k` | 夹紧到 [1, 100]，非法类型默认为 5 |

FTS5 查询中的 `LIKE` 通配符（`%` `_`）会被 escape 防止注入。
LLM 相关的 prompt 注入由 `prompt_guard` 模块防御。
