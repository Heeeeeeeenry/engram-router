# EngramRouter 项目概要

## 原始目标

构建一个 **Agent 按需无损记忆层**，替代滚动摘要压缩。

核心主张：
> 记忆不应该被反复摘要，而应该被结构化保存、按需取回。

## 当前实现

EngramRouter 现在已经不是纯概念稿，而是一个可用的本地记忆路由器：

- SQLite 持久化，WAL 模式，保存原始证据
- 召回链路包含 FTS5 / LIKE / 实体名回退 / 实体传播 / 打分排序
- `recall()` 已接入查询扩展、边扩展、top-k 截断和证据回填
- CLI、MCP Server、安装入口都已存在
- 支持配置系统、LLM 语义重排、向量索引等增强能力
- 项目测试已覆盖主流程，当前仓库内的测试与基准脚本仍在持续演进

## 现在应该保留的文档

| 文档 | 角色 |
|---|---|
| `README.md` | 入口说明，快速了解项目是什么 |
| `docs/PROJECT_BRIEF.md` | 当前状态、收敛结论、后续改进点 |
| `docs/SCHEMA.md` | SQLite 实际 schema 的源码级说明 |
| `CONTRIBUTING.md` | 变更约束与协作规则 |

## 当前状态

### 已落地

- [x] SQLite 持久化与原始证据保存
- [x] 召回链路的候选检索、实体扩展、加权排序、top-k 返回
- [x] CLI 和 MCP Server
- [x] 一键安装入口
- [x] 配置系统
- [x] LLM 语义重排的可选接入
- [x] 向量与查询扩展相关模块已进入主仓库

### 仍可改进

- 需要把“设计提案”和“实现现状”彻底分层，避免同一目录里同时放当前规范和阶段提案
- 需要减少重复文档，把主线收敛到少数权威文件
- 需要把生成产物和手写文档分开，避免审计报告长期占据 docs 目录
- 需要在文档里明确哪些是稳定接口，哪些仍然会变

### 需要修改

- `README.md` 增加文档索引，作为入口页
- `CONTRIBUTING.md` 只引用真实存在的文档
- `SCHEMA.md` 的交叉引用改成当前项目文档，不再指向不存在的旧架构文件
- 以后如果要新增架构变更，优先更新这里，再补单独设计稿

### 需要删除

下面这些文件属于过时或重复的阶段性产物，已经不再适合作为主文档：

- `docs/ARCHITECTURE_V2.md`
- `docs/LLM_EXTRACTOR_ENHANCEMENT.md`
- `docs/query_expansion_design.md`
- `docs/performance_audit.md`
- `docs/security_audit.md`
- `docs/API_SPEC.md`
- `docs/REQUIREMENTS.md`

### 需要保留但不当作文档源码

- `docs/semantic_audit_report.json`：这是测试/审计产物，当前仍被测试脚本引用，不应当作手写文档处理

## 不做的事

- 不把项目改成通用 RAG 文档检索器
- 不把原始证据改成摘要替代品
- 不把一次性的阶段提案长期放进主文档区
- 不让文档覆盖代码作为事实来源的地位
