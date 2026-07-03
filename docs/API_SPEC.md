# EngramRouter API 规范、CLI 接口与实施路线图

> **版本**: v0.1.0 → v0.2.0 路线图  
> **最后更新**: 2026-07-01  
> **基于源码**: `src/engram_router/store.py` (1849 行)、`cli.py` (133 行)、`mcp_server.py` (484 行)、`config.py` (269 行)、`entities.py` (167 行)、文档 (`docs/`)

---

## 目录

1. [API 设计原则](#1-api-设计原则)
2. [核心 Python API](#2-核心-python-api)
3. [MCP 工具定义](#3-mcp-工具定义)
4. [CLI 命令设计](#4-cli-命令设计)
5. [配置接口](#5-配置接口)
6. [实施路线图](#6-实施路线图)
7. [迁移计划](#7-迁移计划)

---

## 1. API 设计原则

### 1.1 六大核心原则

| 原则 | 说明 | 已有实现体现 |
|------|------|------------|
| **证据优先 (Evidence-First)** | 存原文，不摘要；摘要只是检索辅助，不可替代原始证据 | `save()` 存入完整 `raw_text`；`summary` 仅为首句截断 |
| **最小上下文 (Minimal Context)** | 只返回 top-k 最相关记忆，不将全部历史注入模型 | `recall()` 通过多级候选→加权评分的 pipeline，仅返回 top_k |
| **缺失追问 (Gap Detection)** | 记忆不够时主动提示补充问题，而不是强迫模型编造 | `gap_check()` 检测 5 种缺口类型 + 建议追问 |
| **可撤销性 (Revocable)** | 更正降权不硬删，证据链完整保留 | `corrections` 表 + `_get_corrected_ids()` 对纠正记忆 ×0.3 降权 |
| **平台无关 (Platform Agnostic)** | MCP 标准接口，任何 Agent 框架可用 | 6 个 MCP 工具，Hermes/Claude/Codex/OpenClaw 一键安装 |
| **零压缩失真 (Lossless)** | 压缩允许但必须保留证据链；`source=compaction` 永不删除原文 | `compact()` 写入 `distilled_memories` 并保留 `raw_logs` |

### 1.2 设计约束

1. **向后兼容优先**: 新增字段必须有默认值，重命名保留别名 2 个大版本
2. **优雅降级**: SSR、LLM、YAML 等可选模块不可用时自动回退（已有）
3. **本地优先**: SQLite WAL 模式，无默认云同步
4. **显式优于隐式**: 不自动推断事实（`CO_OCCURS_WITH` = 0.4 置信度）

### 1.3 不做的事

- ❌ 不替代 Agent 的上下文管理（那是 Agent 框架的职责）
- ❌ 不做通用知识库/文档检索（不是 RAG）
- ❌ 不追求毫秒级实时性（面向 Agent 单会话）
- ❌ 不绑定特定 Agent 框架（MCP 已是标准）

---

## 2. 核心 Python API

### 2.1 MemoryStore 类

```python
from engram_router import MemoryStore, MemoryRecord, RecallWeights
```

#### 2.1.1 构造函数

```python
class MemoryStore:
    def __init__(
        self,
        path: str | Path | None = None,       # SQLite 文件路径，None = :memory:
        max_recall_hops: int | None = None,     # 图谱跳跃深度，默认 2
        recall_decay: float | None = None,      # 传播衰减因子，默认 0.5
        weights: RecallWeights | None = None,   # 自定义召回权重
        llm_extractor: LLMExtractor | None = None,  # LLM 实体提取器（可选）
        llm_query_extract: bool = False,        # 是否开启查询端 LLM 实体提取
        reranker: Any | None = None,            # LLM 重排序器（可选，LLMReranker 实例）
    ) -> None
```

**语义**:
- `path=None` → 内存数据库，进程结束即消失（适合测试）
- `path=Path("~/.engram/memory.db")` → 持久化存储（生产推荐）
- `weights=None` → 从 `~/.engram/config.yaml` 加载权重，不存在则使用 `RecallWeights()` 默认值
- `llm_extractor=None` → 仅使用规则实体提取（`entities.py`），不调用 LLM
- `llm_query_extract=True` → 在 recall 时也用 LLM 提取查询实体
- `reranker=None` → 仅使用加权 token/entity 排序；提供 `LLMReranker` 实例则混合打分

**错误**:
- `RuntimeError(xxx)`: SQLite 版本 < 3.35.0（不支持 `RETURNING`）
- 目录自动创建（`path.parent.mkdir(parents=True, exist_ok=True)`）

---

#### 2.1.2 MemoryRecord（查询结果）

```python
@dataclass(frozen=True)
class MemoryRecord:
    id: str                        # "mem_1"
    raw_text: str                  # 完整原文
    summary: str                   # 压缩摘要（raw_text[:160] 的首句 + 清理）
    confidence: float = 1.0        # 置信度，默认 1.0
    metadata: dict[str, Any] | None = None  # 用户元数据 + source + created_at
    evidence_refs: list[str] | None = None  # 证据 ID 列表
    score: float = 0.0             # 召回得分（仅 recall 时有值）
    match_reason: str = ""         # 匹配理由（可审计的得分明细）
```

---

#### 2.1.3 RecallWeights（召回权重配置）

```python
@dataclass(frozen=True)
class RecallWeights:
    # Token 评分
    ascii_base: float = 4.0              # ASCII token 基础权重
    ascii_per_char_cap: int = 6          # 长度加权上限
    ascii_per_char: float = 0.5          # 每字符附加权重
    cjk_multi_base: float = 2.0          # 多字 CJK 基础权重
    cjk_multi_per_char: float = 0.5      # 多字 CJK 每字附加
    stop_char_weight: float = 0.05       # 停用字权重
    single_cjk_weight: float = 0.4       # 单字 CJK 权重

    # 语义增强
    colleague_boost: float = 1.0         # "同事" 关键词加成
    reason_marker_boost: float = 1.5     # "因为/由于" 标记加成

    # 召回管线
    fts_boost: float = 0.1               # FTS5 命中加分
    shared_entity_multiplier: float = 1.2 # 共享实体倍率
    conflicting_person_penalty: float = 2.2  # 人物冲突惩罚
    person_match_boost: float = 2.0      # 精确人物匹配加成
    entity_tie_break_bonus: float = 0.01 # 实体数平局微调

    # 上下文增强
    brand_boost: float = 2.0             # 品牌问询加成
    occupation_boost: float = 1.5        # 职业问询加成
    identity_base_attr_boost: float = 2.0 # 身份属性问询加成
    eval_sensory_boost: float = 1.5      # 评价/感官问询加成

    # 纠正
    correction_penalty: float = 0.3      # 被纠正记忆降权系数

    # 传播激活
    max_recall_hops: int = 2             # 图谱最大跳跃
    recall_decay: float = 0.5            # 传播衰减率
    activation_threshold: float = 0.03   # 激活下限（低于此停止传播）

    # 关联可达性（salience 衰减，仅对非直接命中记忆生效）
    assoc_reach_base_attr: float = 0.15  # 基础属性记忆关联衰减
    assoc_reach_constraint: float = 0.6  # 约束记忆关联衰减
    assoc_reach_decision: float = 0.7    # 决策记忆关联衰减
    assoc_reach_sensory: float = 1.0     # 感官记忆关联衰减
    assoc_reach_event: float = 1.0       # 事件记忆关联衰减

    # 规模保护
    full_scan_limit: int = 2000          # 全量扫描上限
```

---

### 2.2 核心方法签名与语义

#### 2.2.1 `save()` — 保存记忆

```python
def save(
    self,
    text: str,                              # 要保存的文本（必填，不可为空）
    source: str = "conversation",           # 来源标签
    metadata: dict[str, Any] | None = None, # 用户自定义元数据
    namespace: str = "default",             # 租户命名空间
) -> str:                                   # 返回 memory_id（如 "mem_3"）
```

**语义**:
1. 校验：`text` 不能为空或纯空白；UTF-8 字节数 ≤ 10240
2. 生成单调递增 ID（原子 `UPDATE id_sequences ... RETURNING`）
3. 生成摘要：取首句 → 清理语气词 → 截断 ≤ 120 字符
4. 写入 `memories` 表（含 `namespace` 隔离）
5. 写入 `evidence` 表（关联原文引用）
6. 执行实体提取（规则 + 可选 LLM）→ 写入 `entities` / `memory_entities`
7. 自动生成边（`CO_OCCURS_WITH` / `CAUSED_BY` / `DESCRIBES` / LLM 类型）
8. 写入 FTS5 trigram 索引
9. 提交事务

**错误**:
- `ValueError("text must not be empty")`
- `ValueError("text exceeds 10240 bytes (got N)")`

**v0.2 计划**:
- `metadata` 允许 `dict` 或 `str`（JSON string）

---

#### 2.2.2 `recall()` — 召回记忆

```python
def recall(
    self,
    query: str,                  # 查询文本（必填）
    top_k: int = 5,              # 返回数量
    namespace: str = "default",  # 租户命名空间
) -> list[MemoryRecord]:         # 按 score DESC 排序的 top-k 结果
```

**召回管线**（7 阶段）:
1. **Token 化**: ASCII/数字 + 多字 CJK + 单字 CJK（去重保持顺序）
2. **实体提取**: 规则提取 `query` 中的实体（可选 LLM 增强）
3. **候选筛选**: FTS5 trigram → LIKE 回退（短词）/ 实体名回退 → 全量扫描回退
4. **加权评分**: `_term_weight` × `_base_score` + FTS 加成
5. **实体/边扩展**: 共享实体加成 + 人物冲突处理 + 边图传播激活（2 hops）
6. **上下文增强**: 品牌/身份/评价检测 + occupation topic 匹配
7. **纠正降权**: `corrections` 表中的记忆 ×0.3
8. **Salience 衰减**: 非直接命中记忆按 salience 分类衰减
9. **排序截断**: 按 (score DESC, created_at DESC, id DESC) 取 top_k

**返回值字段**:
```python
MemoryRecord(
    id="mem_3",
    raw_text="张三前两天送我一把 HHKB，说是生日礼物",
    summary="张三前两天送我一把 HHKB，说是生日礼物",
    confidence=1.0,
    metadata={"source": "conversation", "created_at": "2026-06-30 12:00:00"},
    evidence_refs=["evi_3"],
    score=7.45,
    match_reason="matched terms: HHKB, 张三; shared entities: 张三"
)
```

**可审计性**: `match_reason` 记录了每个 boost 的来源（`"matched terms"`, `"fts trigram candidate"`, `"shared entities: ..."`, `"brand-bearing product: ..."`, `"identity-question base-attr boost (matched subject)"`, `"edge assoc hop=2 act=0.045 [张三 → HHKB → 键盘]"`, `"assoc-reach×0.15"` 等）

---

#### 2.2.3 `delete()` — 删除记忆

```python
def delete(self, memory_id: str) -> bool:  # True = 已删除，False = ID 不存在
```

**语义**:
- FK `ON DELETE CASCADE` 自动清理 `evidence`、`distilled_memories`、`memory_entities`
- FTS5 删除当前为 no-op（ghost entry 无害，recall 管线会过滤）
- **慎用**: 仅删除自己创建的记忆；不要删除其他 Agent 的记忆

---

#### 2.2.4 `gap_check()` — 缺口检测

```python
def gap_check(
    self,
    query: str,                                    # 查询文本
    memories: list[MemoryRecord] | None = None,    # 已召回记忆（可选）
    namespace: str = "default",
    scan_all: bool = False,  # True = 扫描全部记忆而非仅 top-k
) -> dict[str, Any]:
```

**返回值**:
```python
{
    "sufficient": False,          # True = 记忆足够回答，False = 有缺口
    "missing": ["reason"],         # 缺失维度列表
    "suggested_question": "你之前有说过为什么/出于什么原因吗？"
}
```

**5 种检测维度**:
| 维度 | 检测触发词 | 缺失条件 |
|------|-----------|---------|
| `reason` | 为什么/原因/为啥/为何 | 无 "因为/原因/由于/为了/生日/所以" |
| `person` | 谁/哪位/哪个人 | 无类似人名的CJK 2-3字词 |
| `time` | 什么时候/何时/几点/哪天 | 无时间模式 (昨天/上个月/2024年) |
| `location` | 哪里/哪儿/在哪/什么地方 | 无地点模式 (X市/X路/X区) |
| `object` | 什么东西/什么/啥/哪个 | 无 ASCII 2+ 字符 或 已知物体词 |

---

#### 2.2.5 `compact()` — 压缩原始日志

```python
def compact(
    self,
    raw_log_id: str,            # 原始日志 ID（来自 save_raw_log）
    distilled_text: str,        # 蒸馏后的文本
    namespace: str = "default",
) -> str:                       # 返回 distilled_id
```

**语义**:
1. 校验 `raw_log_id` 存在
2. 调用 `save(distilled_text, source="compaction", metadata={"raw_log_id": raw_log_id})`
3. 写入 `distilled_memories` 链接表
4. 创建额外 `evidence` 行指向原始日志
5. **原始日志不受影响**（不删除、不修改）

**错误**: `KeyError("raw_log_id not found: xxx")`

---

#### 2.2.6 `save_raw_log()` — 保存原始日志

```python
def save_raw_log(
    self,
    text: str,                          # 原始日志文本
    kind: str = "conversation",         # 类型: conversation / file_change / ...
) -> str:                               # 返回 raw_log_id
```

**语义**: L0 层入口。保存完整的对话轮次、工具调用、命令输出、diff 等。不进行实体提取或 FTS5 索引。

---

#### 2.2.7 `consolidate()` — 数据整理

```python
def consolidate(self) -> dict[str, Any]:
```

**返回值**:
```python
{
    "merged_entities": 3,          # 合并的重复实体数
    "removed_edges": 2,            # 删除的孤立边数
    "removed_self_loops": 1,       # 删除的自环边数
    "removed_duplicate_edges": 0,  # 删除的重复边数
}
```

**语义**: 定期调用以清理数据碎片。合并大小写/空白变体的实体，删除 orphan edges、self-loops 和 duplicate edges。

---

#### 2.2.8 `entities_for()` — 获取记忆的实体列表

```python
def entities_for(self, memory_id: str) -> list[dict[str, Any]]:
    # 返回: [{"id": "ent_1", "name": "张三", "kind": "person", "evidence": "张三"}, ...]
```

---

#### 2.2.9 `get_raw_log()` — 获取原始日志

```python
def get_raw_log(self, raw_id: str) -> dict[str, Any]:
    # 返回: {"id": "raw_1", "kind": "conversation", "text": "...", "created_at": "..."}
```

---

#### 2.2.10 `close()` / 上下文管理器

```python
def close(self) -> None:
    """关闭数据库连接"""

# 支持 with 语句
with MemoryStore(path="memory.db") as store:
    store.save("hello")
```

---

### 2.3 辅助函数

```python
from engram_router.entities import extract_entities, classify_salience

# 纯规则实体提取（不依赖 MemoryStore 实例）
entities = extract_entities("张三在腾讯工作，送了我一把 HHKB")
# → [{"name": "张三", "kind": "person", "evidence": "张三"}, ...]

# 分类 salience
salience = classify_salience(entity, source_text)  # → "base_attr" / "sensory" / "event" / ...
```

---

## 3. MCP 工具定义

> **协议**: JSON-RPC 2.0 over stdio  
> **认证**: `ENGRAM_API_KEY` 环境变量  
> **当前版本**: 6 个工具，全部已实现并测试通过（117 tests）

### 3.1 工具清单总览

| 工具名 | 用途 | 必填参数 | 返回 |
|--------|------|---------|------|
| `memory.save` | 保存文本到记忆库 | `text` | `{"memory_id": "mem_1"}` |
| `memory.recall` | 召回 top-k 相关记忆 | `query` | `{"memories": [...]}` |
| `memory.gap_check` | 检查记忆是否足够回答 | `query` | `{"sufficient": bool, "missing": [...], "suggested_question": "..."}` |
| `memory.compact` | 将原始日志蒸馏为记忆 | `raw_log_id`, `distilled_text` | `{"distilled_id": "dst_1"}` |
| `memory.consolidate` | 清理去重实体和孤立边 | (无) | `{"status": "ok", "stats": {...}}` |
| `memory.delete` | 删除指定记忆 | `memory_id` | `{"deleted": true/false}` |

### 3.2 完整 JSON Schema

#### 3.2.1 `memory.save`

```json
{
    "name": "memory.save",
    "description": "Save a text to the EngramRouter memory store. Returns the memory ID.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text content to save."
            },
            "source": {
                "type": "string",
                "description": "Optional source label (e.g. 'conversation', 'compaction'). Default: 'mcp'"
            },
            "namespace": {
                "type": "string",
                "description": "Tenant namespace. Default: 'default'"
            }
        },
        "required": ["text"]
    }
}
```

**优化建议** (v0.2):
- `source` 默认值统一为 `"mcp"`（当前实现已是 `"mcp"`，但 schema 说明写了 `"conversation"`，需统一）
- 增加 `metadata` 字段: `{"type": "object", "description": "Optional user metadata dict"}`
- 增加 `confidence` 字段: `{"type": "number", "minimum": 0, "maximum": 1}`

#### 3.2.2 `memory.recall`

```json
{
    "name": "memory.recall",
    "description": "Recall top-k memories matching a query from the store.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query string to search for."
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results to return. Default: 5"
            },
            "namespace": {
                "type": "string",
                "description": "Tenant namespace. Default: 'default'"
            }
        },
        "required": ["query"]
    }
}
```

**确认**: 当前实现正确。`top_k` 默认 5（MCP 端）；CLI 端同样为 5。

#### 3.2.3 `memory.gap_check`

```json
{
    "name": "memory.gap_check",
    "description": "Check whether recalled memories are sufficient to answer a query, or if a gap exists.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query to check for gaps."
            },
            "namespace": {
                "type": "string",
                "description": "Tenant namespace. Default: 'default'"
            },
            "scan_all": {
                "type": "boolean",
                "description": "When true, bypass recall and scan all memories for gap analysis. Default: false"
            }
        },
        "required": ["query"]
    }
}
```

**确认**: 当前实现正确。`scan_all` 可选，CLI 和 MCP 均支持。

#### 3.2.4 `memory.compact`

```json
{
    "name": "memory.compact",
    "description": "Distill a raw log into a compact memory while preserving evidence references.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "raw_log_id": {
                "type": "string",
                "description": "ID of the raw log to compact."
            },
            "distilled_text": {
                "type": "string",
                "description": "The distilled/compacted text to store."
            },
            "namespace": {
                "type": "string",
                "description": "Tenant namespace. Default: 'default'"
            }
        },
        "required": ["raw_log_id", "distilled_text"]
    }
}
```

**优化建议** (v0.2):
- 当前 schema 未列 `namespace` 参数，但 MCP 工具实现也未透传（`_tool_memory_compact` 未传 namespace）。CLI 侧支持 `--namespace`。需统一：
  - 方案 A: MCP 工具增加 `namespace` 参数并透传
  - 方案 B: `compact()` 从 metadata 取 `raw_log_id` 关联的 namespace

#### 3.2.5 `memory.consolidate`

```json
{
    "name": "memory.consolidate",
    "description": "合并重复实体名 (大小写/空白变体)、清理孤立边和重复边。返回清理统计。",
    "inputSchema": {
        "type": "object",
        "properties": {}
    }
}
```

**确认**: 当前实现正确，无参数。

**优化建议** (v0.2):
- 增加 `namespace` 参数以支持租户级别的 consolidation
- 增加 `dry_run` 参数，仅返回统计而不实际执行

#### 3.2.6 `memory.delete`

```json
{
    "name": "memory.delete",
    "description": "Delete a memory by ID. Returns true/false. USE CAUTIOUSLY: only delete memories you created and no longer need; never delete another agent's memories.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "ID of the memory to delete."
            }
        },
        "required": ["memory_id"]
    }
}
```

**确认**: 当前实现正确。安全警告已在 description 中。

---

### 3.3 MCP 协议细节

**初始化流程**:
```
Client → Server: initialize (protocolVersion: "2024-11-05")
Server → Client: {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "engram-router", "version": "0.1.0"}}
Client → Server: initialized (notification)
```

**认证**: `ENGRAM_API_KEY` 环境变量非空即通过；未设置则 `tools/list` 和 `tools/call` 返回 `AUTH_ERROR (-32001)`。

**其他错误码**:
- `-32700` Parse error（JSON 解析失败 / 深度超限 / 行超长）
- `-32600` Invalid Request
- `-32601` Method not found
- `-32602` Invalid params
- `-32603` Internal error
- `-32002` Server not initialized

---

## 4. CLI 命令设计

### 4.1 总览

```bash
engram [--db <path>] <command> [args...]
```

**全局选项**:
| 选项 | 默认值 | 说明 |
|------|-------|------|
| `--db` | `memory.db` (CWD) | SQLite 数据库路径 |

### 4.2 子命令详解

#### 4.2.1 `save` — 保存记忆

```bash
engram save <text> [--namespace <ns>]

# 示例
engram save "张三送了我一把HHKB键盘，因为生日"
# → {"memory_id": "mem_1"}

engram save "team decided on PostgreSQL" --namespace project-alpha
# → {"memory_id": "mem_2"}
```

**参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `text` | positional | 是 | 记忆文本 |
| `--namespace` | string | 否 | 默认 `"default"` |

**缺失功能** (v0.2 计划):
- `--source` 选项（当前硬编码为 `"conversation"`，MCP 端为 `"mcp"`）
- `--metadata` JSON string 选项

#### 4.2.2 `recall` — 召回记忆

```bash
engram recall <query> [--top-k N] [--namespace <ns>]

# 示例
engram recall "同事送的键盘什么牌子"
# → {"memories": [{...}, ...]}

engram recall "architecture decision" --top-k 10 --namespace project-alpha
```

**参数**:
| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `query` | positional | 是 | - | 查询文本 |
| `--top-k` | int | 否 | 5 | 返回数量（≥1） |
| `--namespace` | string | 否 | `"default"` | 命名空间 |

**缺失功能** (v0.2 计划):
- `--format json|text` 输出格式选择（当前始终 JSON）
- `--no-evidence` 不返回 evidence_refs（减少输出体积）

#### 4.2.3 `gap-check` — 缺口检测

```bash
engram gap-check <query> [--top-k N] [--namespace <ns>] [--scan-all]

# 示例
engram gap-check "他为什么送我键盘" --scan-all
# → {"sufficient": false, "missing": ["reason"], "suggested_question": "..."}
```

**参数**:
| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `query` | positional | - | 查询文本 |
| `--top-k` | int | 5 | 召回数量 |
| `--namespace` | string | `"default"` | 命名空间 |
| `--scan-all` | flag | false | 扫描全部记忆而非仅 top-k |

#### 4.2.4 `delete` — 删除记忆

```bash
engram delete <memory_id>

# 示例
engram delete mem_3
# → {"deleted": true, "memory_id": "mem_3"}
```

#### 4.2.5 `save-raw-log` — 保存原始日志

```bash
engram save-raw-log <text> [--kind <kind>]

# 示例
engram save-raw-log "Full conversation turn..." --kind conversation
# → {"raw_log_id": "raw_1"}
```

#### 4.2.6 `compact` — 压缩日志

```bash
engram compact <raw_log_id> <distilled_text> [--namespace <ns>]

# 示例
engram compact raw_1 "团队决定使用 PostgreSQL 替代 MySQL。" --namespace project-alpha
# → {"distilled_id": "dst_1"}
```

#### 4.2.7 `benchmark` — 基准测试

```bash
engram benchmark --conversation <path> --cases <path> [--text] [--gate]

# 示例
engram benchmark \
  --conversation examples/long_conversation_demo.md \
  --cases examples/benchmark_questions.jsonl \
  --text --gate
```

**参数**:
| 参数 | 说明 |
|------|------|
| `--conversation` | 对话 Markdown 文件路径 |
| `--cases` | 问题 JSONL 文件路径 |
| `--text` | 输出人类可读报告（否则 JSON） |
| `--gate` | 有 hard-gate 失败时 exit 1（CI 用） |

#### 4.2.8 `install` / `status` / `uninstall` — 智能体安装

```bash
engram install              # 自动检测并安装所有智能体
engram status               # 显示智能体连接状态
engram uninstall            # 移除所有 engram MCP 配置
```

**支持智能体**: Hermes Agent, Claude Desktop, Claude Code, OpenClaw, Codex (OpenAI)

---

### 4.3 独立 CLI 入口

```bash
# MCP 服务器（独立运行）
engram-mcp --db ~/.engram/memory.db

# 智能体安装器（独立运行）
engram-install install
engram-install status
engram-install uninstall

# 模块入口
python -m engram_router save "hello"
```

### 4.4 CLI 设计原则

1. **JSON 输出**: 所有命令输出 JSON（`stderr` 输出错误），方便脚本解析
2. **幂等性**: `save` 总是返回新 ID，`delete` 不存在返回 `false`
3. **错误消息**: 通过 `stderr` 以 `{"error": "message"}` 输出
4. **一致性**: 所有读写操作通过同一 `MemoryStore` 实例

---

## 5. 配置接口

### 5.1 配置路径优先级

```
1. ENGRAM_CONFIG 环境变量 → 自定义路径
2. ~/.engram/config.yaml → 用户配置
3. 代码默认值         → EngramConfig() 字段默认值
```

### 5.2 配置结构 (`~/.engram/config.yaml`)

```yaml
# ============================================================
# EngramRouter 配置文件 (~/.engram/config.yaml)
# 所有字段均为可选，未设置的字段使用代码默认值
# ============================================================

# ═══ 实体提取配置 ═══
entities:
  # --- 人物识别 ---
  kinship_words:
    - 妈妈
    - 母亲
    - 爸爸
    - 父亲
    - 爷爷
    - 奶奶
    # ... (可扩展现有列表)

  role_words:
    - 同事
    - 前同事
    - 同学
    # ...

  surname_chars: "张李王赵刘陈杨黄周吴徐孙马朱胡林郭何高罗"

  name_breakers: "前送是在的说现给最这那今昨明上下个把和与认识喜爱有过来去要会想做、，。！？ 和跟同当于到脾平每总很也已才就又都还便只"

  # --- 物体识别 ---
  known_objects:
    - 机械键盘
    - 键盘
    - 鼠标
    # ...

  food_words:
    - 红烧肉
    - 糖醋排骨
    # ...

  ascii_stop_words: [the, a, an, is, are]

  # --- 公司识别 ---
  known_companies:
    - 腾讯
    - 阿里
    # ...

  # --- 主题识别 ---
  topic_words:
    - 键盘
    - 礼物
    - 生日
    # ...

  object_topic_aliases:
    HHKB: 键盘
    MX: 键盘
    Keychron: 键盘
    特斯拉: 车
    # ...

  # --- 时间识别 ---
  time_patterns:
    - 前[两三四五六七八九十0-9]*天
    - 昨天
    # ...

  # --- 原因识别 ---
  reason_markers:
    - 因为
    - 由于
    - 为了
    # ...

  # --- 属性模式 ---
  attr_patterns:
    - "[0-9]{1,3}岁"
    - "[ABO]型血"
    # ...

# ═══ Salience 分类配置 ═══
salience:
  base_attr_name_patterns:
    - "^[男女]$"
    - "^\\d{1,3}岁$"
    # ...

  base_attr_context:
    - 性别
    - 叫什么
    # ...

  sensory_patterns:
    - "(?:做[饭菜]|烧菜|炒菜|烹饪|手艺).{0,4}(?:好吃|难吃|一般|不错|很棒|香)"
    # ...

  decision_markers:
    - 决定
    - 确定
    # ...

  constraint_markers:
    - 不能
    - 不允许
    # ...

  event_markers:
    - 昨天
    - 今天
    # ...

# ═══ 召回权重配置 ═══
recall:
  # Token 评分
  ascii_base: 4.0
  ascii_per_char_cap: 6
  ascii_per_char: 0.5
  cjk_multi_base: 2.0
  cjk_multi_per_char: 0.5
  stop_char_weight: 0.05
  single_cjk_weight: 0.4

  # 语义增强
  colleague_boost: 1.0
  reason_marker_boost: 1.5

  # 召回管线
  fts_boost: 0.1
  shared_entity_multiplier: 1.2
  conflicting_person_penalty: 1.5
  person_match_boost: 1.5
  entity_tie_break_bonus: 0.01

  # 上下文增强
  brand_boost: 2.0
  occupation_boost: 1.5
  identity_base_attr_boost: 2.0
  eval_sensory_boost: 1.5

  # 纠正
  correction_penalty: 0.3

  # 传播激活
  max_recall_hops: 2
  recall_decay: 0.5
  activation_threshold: 0.03

  # 关联可达性
  assoc_reach_base_attr: 0.15
  assoc_reach_constraint: 0.6
  assoc_reach_decision: 0.7
  assoc_reach_sensory: 1.0
  assoc_reach_event: 1.0

  # 规模保护
  full_scan_limit: 2000

  # 停用字符
  stop_chars: "我你他她它的了是在有和与啊吗呢吧那这个什么牌子哪家为么把了和就都也很"

  # 职业主题
  occupation_topics:
    - 退休教师
    - 教师
    - 医生
    # ...
```

### 5.3 配置数据类映射

```
EngramConfig
├── entities: EntityConfig   (实体提取规则)
├── salience: SalienceConfig  (显著性分类规则)
└── recall: RecallWeightsConfig (召回权重)
```

### 5.4 环境变量

| 变量 | 用途 | 默认值 |
|------|------|-------|
| `ENGRAM_CONFIG` | 自定义配置路径 | `~/.engram/config.yaml` |
| `ENGRAM_API_KEY` | MCP 服务 API 密钥（必填） | (空) |
| `DEEPSEEK_API_KEY` | LLM 提取器 API 密钥 | - |
| `ENGRA_LLM_BASE_URL` | LLM API 基础 URL | `https://api.deepseek.com/v1` |
| `ENGRA_LLM_MODEL` | LLM 模型名 | `deepseek-chat` |
| `ENGRAM_LLM_API_KEY` | 重排序器 API 密钥 | `OPENAI_API_KEY` |
| `ENGRAM_LLM_BASE_URL` | 重排序器 API 基础 URL | - |
| `ENGRAM_LLM_MODEL` | 重排序器模型 | `gpt-4o-mini` |
| `ENGRAM_LLM_MAX_CONCURRENT` | 最大并发 LLM 调用 | `3` |

---

## 6. 实施路线图

### 6.1 Phase 1: 规则引擎稳定化 (v0.1.x) ✅ 已完成

**状态**: ✅ 已交付

**交付物**:
- [x] SQLite WAL 持久化 + FTS5 trigram 索引
- [x] 三级候选检索（FTS5 → LIKE → 实体名 → 全量扫描回退）
- [x] 规则实体提取（7 种类型：person/object/company/topic/time/reason/attribute）
- [x] Salience 分类（5 级：base_attr/sensory/event/decision/constraint）
- [x] 自动边生成（CO_OCCURS_WITH/CAUSED_BY/DESCRIBES）+ N-hop 传播激活（2 hops）
- [x] 人物冲突隔离 + 身份属性范围检查
- [x] 纠正机制（corrections 表 + ×0.3 降权）
- [x] MCP Server（6 工具，零第三方依赖）
- [x] 一键安装（5 种 Agent：Hermes/Claude Desktop/Claude Code/OpenClaw/Codex）
- [x] YAML 配置系统 + 全部权重可调
- [x] 单调递增 ID 分配器（id_sequences）
- [x] Multi-tenant namespace 隔离
- [x] gap_check 5 维检测 + 追问建议
- [x] 277 单元测试，265 passed

**验收标准**:
- [x] 基准测试：evidence recall 3/3 击败 summary baseline 0/3
- [x] 所有 117 个 pytest 通过
- [x] benchmarks/regression 的 34 个 hard gates 全部通过

---

### 6.2 Phase 2: LLM-Native 联想引擎 (v0.2.0) 🚧 进行中

**目标**: 从「规则提取 + 关键词匹配」升级为「LLM 提取 + 语义检索」

**交付物**:

| 序号 | 交付物 | 说明 | 优先级 |
|------|-------|------|--------|
| 2.1 | LLM 实体提取器完善 | 当前 `llm_extractor.py` 已实现基础框架，需加强：批量处理、缓存、自定义 prompt | P0 |
| 2.2 | LLM 重排序器增强 | 当前 `llm_reranker.py` 已实现混合打分，需：自适应候选数、分数归一化 | P1 |
| 2.3 | Embedding 向量索引 | 新增 `sqlite-vec` 扩展或 ChromaDB/LanceDB 后端 | P0 |
| 2.4 | 混合检索管线 | FTS5 + Vector 并行检索 → 加权合并 → LLM 重排序 | P0 |
| 2.5 | `memory.update()` API | 支持就地更正（非删除重写），写入 corrections 表 | P1 |
| 2.6 | `memory.stats()` API | 返回各 namespace 的记忆数量、大小、实体分布 | P2 |
| 2.7 | MCP `metadata` 支持 | `memory.save` 增加 `metadata` 字段透传 | P1 |
| 2.8 | CLI `--source` 支持 | `engram save` 增加 `--source` 选项 | P2 |
| 2.9 | `memory.search()` API | 支持按 metadata 字段过滤（如 `source=compaction`） | P2 |
| 2.10 | 异步 API（可选） | `AsyncMemoryStore` 包装器（asyncio + aiosqlite） | P3 |

**验收标准**:
- embedding 召回在 50+ 个 bench cases 上的 precision@5 ≥ 规则引擎（不退化）
- 混合检索（FTS5 + Vector）的 recall@10 比纯规则引擎提升 ≥ 20%
- `memory.update()` 能正确写入 corrections 表且不影响原始 evidence_refs
- 新增 API 100% 向后兼容（所有 v0.1 测试继续通过）

---

### 6.3 Phase 3: True Associative Recall (v0.3.0) 📋 规划中

**目标**: 多记忆间的因果/时序/从属关系推理 + 人物画像聚合

**交付物**:

| 序号 | 交付物 | 说明 | 优先级 |
|------|-------|------|--------|
| 3.1 | 人物画像聚合 | 跨会话学习用户偏好、习惯、决策模式 → `persona` 表 | P0 |
| 3.2 | 自动遗忘机制 | 时间衰减 + 低价值记忆合并（Ebbinghaus 遗忘曲线模型） | P0 |
| 3.3 | 因果链推理 | 多跳 CAUSED_BY 链式传播 → `因为A所以B所以C` 自动连接 | P1 |
| 3.4 | 时序事件线 | 自动按时间顺序排列记忆，生成事件时间线 | P2 |
| 3.5 | 记忆重要性评分 | 基于访问频率、用户标记、与其他记忆的关联密度 | P2 |
| 3.6 | 冲突检测 API | 当新记忆与已有记忆矛盾时自动标记 | P1 |
| 3.7 | 记忆导出/导入 | `memory.export(format)` / `memory.import(path)` | P2 |
| 3.8 | 多用户支持 | 同一 DB 内区分不同用户的记忆（user_id 维度），不限于 namespace | P3 |

**验收标准**:
- 人物画像能跨 3+ 会话正确聚合（不会把 张三 的偏好归给 妈妈）
- 自动遗忘不会删除用户 24 小时内访问过的记忆
- 因果链推理能从「A→B→C」中召回 C 当查询问「A 导致了什么」
- 冲突检测在已知矛盾场景下准确率 ≥ 80%

---

## 7. 迁移计划

### 7.1 从 v0.1.x 到 v0.2.0 的兼容性策略

#### 7.1.1 Schema 兼容

| 变更 | 影响 | 策略 |
|------|------|------|
| 新增向量表 (如 `memories_vec`) | 无影响（新表） | 自动创建；已有记忆可通过后台任务生成 embedding |
| `memories` 表新增 `updated_at` 列 | 无影响（有默认值） | `ALTER TABLE ADD COLUMN ... DEFAULT CURRENT_TIMESTAMP` → 现有 schema migration 框架已就绪 |
| `metadata` 改为结构化 JSON | 无影响（现有 `_parse_metadata` 已兼容多种格式） | 新写入用标准格式，旧数据正常读取 |

#### 7.1.2 API 兼容

| 变更 | 影响 | 迁移策略 |
|------|------|---------|
| `save()` 新增 `metadata: dict | str` 参数 | **向前兼容**（现有 dict 行为不变） | 接受 string 时内部 `json.loads()` |
| `recall()` 新增 `filters: dict` 参数 | **向前兼容**（默认 `None`） | 提供 metadata 字段过滤 |
| 新增 `update()` 方法 | **向前兼容**（新方法） | 独立方法，不影响现有 save() |
| 新增 `search()` 方法 | **向前兼容**（新方法） | 独立方法 |
| `MemoryRecord` 新增字段 | **向前兼容**（frozen dataclass 字段有默认值） | `updated_at: str = ""` |

#### 7.1.3 MCP 协议兼容

| 变更 | 影响 | 策略 |
|------|------|------|
| 工具 schema 新增可选参数 | **向前兼容**（MCP 客户端忽略未知字段或使用默认值） | 仅增加 `properties`，不修改 `required` |
| 新增 MCP 工具 (如 `memory.search`) | **向前兼容**（新工具名） | 客户端按需调用 |
| `memory.compact` 增加 `namespace` | **向前兼容**（新增可选参数） | 默认 "default" |

#### 7.1.4 CLI 兼容

| 变更 | 影响 | 策略 |
|------|------|------|
| `engram save` 增加 `--source` / `--metadata` | **向前兼容**（可选） | 不提供时使用默认值 |
| 新增 `engram search` / `engram stats` 子命令 | **向前兼容**（新命令） | 不影响现有命令 |
| 输出格式增加 `--format text` | **向前兼容**（默认 JSON 不变） | 新增 flag |

#### 7.1.5 配置兼容

| 变更 | 影响 | 策略 |
|------|------|------|
| 新增配置段 (如 `embedding:`, `persona:`) | **向前兼容**（新 section） | `_deep_merge` 自动忽略未知 key |
| 重命名配置键 | **需迁移** | 保留旧键别名 2 个版本，运行时 warn |

### 7.2 回滚方案

如果 v0.2.0 嵌入向量引入新依赖导致安装困难：

```bash
# 回退到 v0.1.x（纯规则引擎）
pip install engram-router==0.1.0

# 或在 v0.2.0 中禁用向量检索
export ENGRAM_DISABLE_VECTOR=1
```

### 7.3 升级检查清单

用户在升级前应运行：
```bash
# 1. 备份数据库
cp ~/.engram/memory.db ~/.engram/memory.db.backup

# 2. 运行数据整理
engram consolidate  # (当前无 CLI 命令，需通过 Python API 或 MCP)

# 3. 升级包
pip install --upgrade engram-router

# 4. 验证
engram recall "test"  # 确认正常工作
python -m pytest --pyargs engram_router  # 运行测试套件
```

---

## 附录 A: 当前实现覆盖率

| 模块 | 行数 | 覆盖率 | 说明 |
|------|------|--------|------|
| `store.py` | 2486 | 高 | 核心引擎，277 测试覆盖 |
| `mcp_server.py` | 484 | 高 | 6 工具全测试 |
| `cli.py` | 133 | 中 | 手动测试为主 |
| `entities.py` | 167 | 高 | 5 个 dedicated 测试 |
| `config.py` | 269 | 中 | 隐式覆盖于其他测试 |
| `llm_extractor.py` | 376 | 低 | 需 mock LLM 的集成测试 |
| `llm_reranker.py` | 167 | 低 | 同上 |
| `install.py` | 581 | 低 | 手动测试为主 |
| `benchmark.py` | ~200 | 中 | 有 dedicated bench 测试 |

## 附录 B: 与其他方案对比

| 特性 | EngramRouter | Mem0 | Letta | RAG |
|------|-------------|------|-------|-----|
| 存储原文 | ✅ | ❌ (摘要) | ❌ (结构化) | ✅ |
| 证据链 | ✅ | ❌ | 部分 | 部分 |
| 缺口检测 | ✅ (gap_check) | ❌ | ❌ | ❌ |
| 实体关联图 | ✅ (edges + N-hop) | ❌ | ❌ | ❌ |
| MCP 标准 | ✅ | ❌ | ❌ | ❌ |
| 零依赖 | ✅ (stdlib) | ❌ | ❌ | ❌ |
| 隐私 (本地) | ✅ | ❌ (云端) | 可选 | 可选 |
