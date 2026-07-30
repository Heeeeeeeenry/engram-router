# 竞品与公开评测集调研

**日期**: 2026-07-21
**作者**: Agent D(子代理数据) + 主会话整合(因 API 错误,原代理未能自行落盘)
**目的**: 为 engram-router 建立**外部**参照系,替代自造的 SummaryBaseline 弱对照;为多方案对比 benchmark 提供依据
**数据来源**: 2025-2026 论文、官方 repo、mem0/zep/letta 官方 blog(URL 全部附于文末)

---

## 0. 结论提要

- **P0 首批接入**(2 个):**mem0** + **zep**。理由:两家都发论文 + 有 pip 库 + 有 LOCOMO/LongMemEval 官方分数可直接对照;engram-router 与它们竞位置最直接
- **P0 首批评测集**(2 个):**LongMemEval**(MIT · 500 Q · 5 类记忆能力 · GPT-judge)+ **PerLTQA / MemoryBench(THUIR)**(**中文原生**,engram-router 的实际场景)
- **P1 加入**:**letta**(实现门槛稍高但学界地位)、**LongMemEval-V2**(2026 新版,25M-115M token web-agent 轨迹)、**LOCOMO**(注意:**CC BY-NC 4.0,不可商用**;只做技术对照,不落进商用发布)
- **P2 参考**:cognee / motorhead / graphiti / langchain memory / MSC / HotpotQA / MuSiQue / LOFT
- **一次完整对照矩阵成本**:6 系统 × 4 评测集 × ~500 query × $0.02/query GPT-4o judge ≈ **$240 一次全跑**,约 **6-10 小时墙钟**(mem0/zep 云端 API + engram 本地)
- **最大惊讶**:mem0 论文 LOCOMO **68.44%** vs **Zep 65.99%** vs **OpenAI Memory 52.90%**;而 Zep 官方 blog 又自报 90.2%——**同一评测集不同 protocol 的分数完全不能横比**,必须固定 judge/protocol 自己重跑

---

## 1. 开源记忆系统对照

| 项目 | Repo | 核心思路 | Backend | MCP | 公开 benchmark(2026) | 集成难度 |
|---|---|---|---|---|---|---|
| **mem0** | mem0ai/mem0 | 提取事实 + 向量检索 + 图(可选) | 多种向量库(qdrant/chroma/pgvector) | ✅ | LOCOMO 66.88 / 92.5(token-efficient 版);LongMemEval 94.4 | **低**(pip install mem0ai) |
| **zep** | getzep/zep + getzep/graphiti | 时序知识图谱(Graphiti)+ 向量 | Neo4j / FalkorDB | ✅ | LOCOMO 65.99 / 94.7(marketing);LongMemEval 71.2(GPT-4o) | 中(需 Neo4j 服务) |
| **letta**(原 MemGPT) | letta-ai/letta | 分层记忆 + 文件系统 + LLM 管理器 | SQLite / Postgres | ✅ | LOCOMO 74.0(GPT-4o-mini,filesystem);LongMemEval 未公开 | 中(有 server) |
| **graphiti** | getzep/graphiti | 纯时序知识图(zep 的图层剥离) | Neo4j / FalkorDB | ✅ | 共 zep 分数 | 中(需图数据库) |
| **cognee** | topoteretes/cognee | 知识图 + AI 记忆抽取 | 多种 | ✅ | 无公开 SOTA(自跑 LOCOMO) | 中 |
| **motorhead** | getmetal/motorhead | 简单 Redis + 摘要 | Redis | ❌ | 无 | 低但过时 |
| **langchain memory** | langchain-ai/langchain | Buffer / Summary / VectorStore memory | 各种 | ❌ | 常作为 baseline 出现,分数偏低 | 极低但功能弱 |

**与 engram-router 差异**:

| 维度 | engram-router | mem0 | zep/graphiti | letta |
|---|---|---|---|---|
| **强项** | FTS + 图 + 向量三层,无损原文,SQLite 单文件,MCP 原生,CN 优化(bge-small-zh + CJK bigram) | 事实抽取质量高,token efficiency 强 | 时序推理和知识变更处理最强 | 分层管理成熟,长上下文管理器 |
| **弱项** | 中文优化好但排序层弱(P@1 = 0.61);未接 cross-encoder | 依赖 LLM 事实提取(成本);中文相对弱 | 需要 Neo4j,部署重;英文优化为主 | 复杂度高,单机部署重 |
| **在 LOCOMO/LongMemEval 的位置** | **未测**(本调研目标) | LOCOMO 66-92 · LME 94 | LOCOMO 65-94 · LME 71 | LOCOMO 74 |

**结论**: engram-router 在"中文 + 单文件 + 免部署服务"这条赛道**无直接竞品**。竞品都在英文长对话赛道,一旦跑英文评测集大概率吃亏(bge-small-zh 英文差),这个必须诚实说。

---

## 2. 公开评测集清单

### 2A. 直接可用(记忆场景)

| 名称 | 论文/repo | 规模 | 语言 | Metric | License | 与 engram 匹配度 |
|---|---|---|---|---|---|---|
| **LongMemEval v1** | arXiv 2410.10813 · xiaowu0162/LongMemEval | 500 Q,~40 或 ~500 sessions/Q | English | GPT-4-judge accuracy | **MIT** ✅ 可商用 | **高**——5 类能力覆盖了 engram 的核心宣称 |
| **LongMemEval v2** | arXiv 2605.12493 · LongMemEval-V2 | 451 Q,25M-115M tok 轨迹 | English | 同上 | CC BY 4.0 | 高——2026 新版,更严 |
| **LOCOMO** | arXiv 2402.17753 · snap-research/locomo | 10 对话,~1540 QA,300 turns/对话 | English | LLM-as-judge(J-score) | **CC BY-NC 4.0** ⚠️ 仅非商用 | 高但许可证受限 |
| **MSC** | arXiv 2107.07567 · ParlAI/msc | 237k 训练样本,5 sessions/episode | English | PPL/F1/human | ParlAI/FAIR 条款不清 ⚠️ | 中——预 LLM-judge 时代,主要作训练数据 |
| **MemoryBench(THUIR)** | arXiv 2510.17281 · THUIR/MemoryBench | 4063 rows,28 subsets,190 MB | **EN + ZH** ✅ | satisfaction 1-9 / implicit_actions | **MIT** ✅ | **最高**——原生中文 + 商用 OK |
| **MemBench(不同项目!)** | arXiv 2506.21605 · Membench(注意与上一个不同名) | 未细核 | 主要 EN | 效果/效率/容量 | 未确认 | 中 |
| **MemoryAgentBench** | arXiv 2507.05257 · HUST-AI-HYZ/MemoryAgentBench | 聚合多集(EventQA/RULER/DetectiveQA 等) | English | EM/Recall@5/LLM-judge | **MIT** ✅ | 中——4 大能力框架好,数据源杂 |
| **PerLTQA** | Baidu(arXiv 2203.05797v2) | 中文长期个性化记忆 | **ZH** | Recall@K / MAP | 需核实 | **高**——中文原生 |
| **Convomem** | arXiv 2511.10523 | 75,336 QA | English | 多种 | 未细核 | 中 |
| **BEAM** | mem0.ai blog | 100 对话,10M tokens,2000 Q | English | 多能力 | 未公开 | 高但数据未公开 |
| **LoCoMo-Plus** | arXiv 2602.10715 | LOCOMO 变种(cognitive) | English | 同 LOCOMO | 未细核 | 中 |

### 2B. 相关但非记忆场景(可复用测多跳/长上下文)

| 名称 | 论文 | 规模 | 匹配度 |
|---|---|---|---|
| **HotpotQA** | EMNLP 2018 | 112k QA,2 跳 | 低——静态文档,英文,leaderboard 已停 |
| **MuSiQue** | TACL 2022 | 25k Q,2-4 跳 | 低——同上,但多跳质量比 HotpotQA 好 |
| **LOFT** | arXiv 2406.13121 · DeepMind | 35 datasets,32k-1M tok | 低——长上下文单发,非对话 |
| **RULER / OneRuler** | 多语言 RULER | 多语言含中文 | 中——但是合成数据,不是对话 |

---

## 3. 评测方法与指标共识

**主流指标(2025-2026 社区共识)**:
- **Answer-level**: LLM-as-judge accuracy(GPT-4o 或 Claude 3.5 Sonnet 作 judge)—— LongMemEval / LOCOMO 都用
- **Retrieval-level**: Recall@k, MRR, nDCG@k —— PerLTQA / MemoryAgentBench
- **系统级**: 平均延迟 / p95 / 每 query token 消耗(mem0 论文特别强调 token efficiency)
- **拒答**: abstention accuracy —— LongMemEval 特有的 `_abs` 后缀问题

**LLM-as-judge 已知问题**(论文引用):
1. **自打分偏见**: judge model 与被测 memory 系统若来自同一厂商,会互相高估。**必须选独立第三方 judge**——mem0 论文用 GPT-4o,zep 官方 blog 也用 GPT-4o,但 zep 自评数字比 mem0 论文里的 zep 高 30 pt,说明 protocol 差异极大
2. **Prompt 敏感**: 同一 LLM 换 judge prompt 分数漂 5-15 pt
3. **复现性差**: temperature > 0 时同 query 多次评分不一致

**engram-router 的选择**:
- Judge 用 **DeepSeek-V3 或 Claude Haiku 4.5**(与项目内 llm_extractor 常用的模型解耦,避免自打分)
- Prompt 版本化,写入 `tests/eval_v2.py` 常量
- 每评测集固定 seed,同一 seed 至少复现 3 次取均值

---

## 4. Benchmark harness 集成方案

### 4.1 MemoryProvider 抽象接口

新建 `tests/providers/base.py`:

```python
from abc import ABC, abstractmethod

class MemoryProvider(ABC):
    """Uniform interface for engram / mem0 / letta / zep / naive / long-context."""

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def save(self, text: str, metadata: dict | None = None) -> str:
        """Return memory id."""

    @abstractmethod
    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """Return [{id, text, score, ...}] in rank order."""

    @property
    @abstractmethod
    def name(self) -> str: ...
```

### 4.2 6 个 adapter

| 文件 | Provider | 依赖 | 部署 |
|---|---|---|---|
| `tests/providers/engram.py` | 复用 `MemoryStore` | 无 | 本地 |
| `tests/providers/mem0.py` | `mem0ai` pip | Chroma/Qdrant | 本地 |
| `tests/providers/zep.py` | `zep-python` + Neo4j | Neo4j 服务 | Docker |
| `tests/providers/letta.py` | `letta` server | Postgres | Docker |
| `tests/providers/naive_vector.py` | 手写 bge + FAISS | 无 | 本地 |
| `tests/providers/long_context.py` | 直接把 corpus 全塞进 LLM | LLM API | 云端 |

### 4.3 复用现有 `tests/eval_v2.py`

`run_eval_v2()` 目前 hardcode 用 `MemoryStore`,改成接收 `provider: MemoryProvider`,分别对 6 个 provider 跑同一批 EvalScenario,产出对比矩阵。

### 4.4 部署与成本预估

| Provider | 单次 500 Q 用时 | 单次成本 | 备注 |
|---|---|---|---|
| engram-router | ~2 min | ~$0(本地) | ENGRAM_SKIP_VECTOR=1 更快 |
| naive-vector | ~2 min | ~$0(本地) | 只用 bge |
| mem0 | ~15 min | $2-5 | LLM 提取 + LLM 判 |
| zep | ~20 min | $2-5 | 图入库慢 |
| letta | ~30 min | $3-8 | 大量 LLM 内部调用 |
| long-context | ~10 min | $10-20 | 每 query 塞全 corpus |
| **judge**(GPT-4o) | 每 500 Q ~10 min | $5-10 | 6 systems × 4 sets × 500 = **12k judge calls** ≈ $60-120 |
| **全矩阵一次跑** | **6-10 小时墙钟** | **~$100-250** | 首次 baseline;之后单点更新 $10-30 |

---

## 5. 结果矩阵模板

```
                | LongMemEval | LongMemEval-V2 | LOCOMO(仅内部) | MemoryBench(THUIR, ZH) | 我们的 daily_life
                | (EN, MIT)   | (EN, CC-BY)    | (EN, NC)       | (EN+ZH, MIT)           | (ZH, 自建)
----------------|-------------|----------------|----------------|------------------------|------------------
engram-router   |  P@1=?      |  ?             |  ?             |  ?                     |  0.61 (baseline)
mem0            |  P@1=?      |  ?             |  ?             |  ?                     |  ?
zep             |  P@1=?      |  ?             |  ?             |  ?                     |  ?
letta           |  P@1=?      |  ?             |  ?             |  ?                     |  ?
naive-vector    |  P@1=?      |  ?             |  ?             |  ?                     |  ?
long-context    |  P@1=?      |  ?             |  ?             |  ?                     |  ?
```

外加两张辅助表:
- **延迟对照**(cold / warm / p95)
- **Token 消耗对照**(单 query 平均 token)—— mem0 论文特别爱强调这条

---

## 6. P0/P1/P2 优先级

### P0 · 首批必接
- **mem0**:pip 安装最容易,LOCOMO/LongMemEval 都有官方分数可校准,最直接的对照
- **zep(via graphiti)**:时序图这个方向 engram 最欠缺,是对面镜子
- **LongMemEval**(500 Q,MIT):记忆能力 5 大类的公认基准
- **MemoryBench(THUIR)** 或 **PerLTQA**:唯一原生中文,不接就没法证明中文场景优势

### P1 · 下一批
- **letta**(部署重但学界地位)
- **LongMemEval-V2**(2026 新版,更严)
- **naive-vector RAG**(自写,揭示"多层结构 vs 纯向量"的边际)
- **long-context baseline**(揭示"记忆系统 vs 塞满上下文"的边际)

### P2 · 参考不接
- cognee / motorhead / langchain memory(功能覆盖不完整或已过时)
- LOCOMO(许可证限制,只做技术对照不进商用材料)
- HotpotQA / MuSiQue / LOFT(与记忆场景弱相关)
- MSC(pre-LLM-judge 时代)

---

## 7. 风险清单

### 7.1 我们没有、竞品有的能力

- **时序知识图 + 事实变更追踪** —— zep/graphiti 强项。engram 的 edges 没有 valid_from/valid_to,不能回答"张三 2024 年是程序员,2026 年是产品"
- **LLM-based 事实抽取** —— mem0 强项。engram 的 entities.py 是规则 + CJK bigram,对未登录词兜底不足
- **分层记忆管理器** —— letta 强项。engram 无对应模块

### 7.2 我们有、竞品少见的能力

- **中文场景优化**(bge-small-zh + CJK bigram 兜底 + trigram FTS)
- **单文件部署**(SQLite,零外部服务)
- **无损原文 + evidence_refs 强证据链**
- **MCP server 原生支持**
- **本地优先 + 云端可选**(privacy 双闸门 `ENGRAM_ALLOW_CLOUD_*`)

### 7.3 评测集偏见

- LongMemEval 偏向"信息抽取 + 时序推理",对纯语义联想弱(engram 短板)测得不充分
- LOCOMO 偏向长对话,10 个对话样本量小,评测方差大
- MemoryBench(THUIR) 偏向"用户反馈迭代",对一次性问答测得不多
- PerLTQA 偏向 QA 形式,对隐式召回 / gap-check 类场景测不到

---

## 8. 关键"惊讶点"

1. **同一评测集 protocol 差异导致 30 pt 分数漂移**:mem0 论文里 Zep 是 65.99;Zep 自己 blog 是 90.2。这意味着**别人的分数只能参考,必须自己在固定 judge/prompt/temperature 下重跑才有意义**。这条会写进 `tests/eval_v2.py` 的注释顶部。
2. **MemBench(ACL 2025)与 MemoryBench(THUIR 2026)是两个不同的 benchmark**,同名易混。THUIR 版是**中文原生 + MIT**,更值得优先接。
3. **LongMemEval-V2 已经出现在 arXiv**(2605.12493 是占位号),2026 年 5 月投稿。基线数据比 v1 严格得多(SOTA 只有 72.5%)。
4. **letta 内部实际上是"文件系统 + LLM 管理"**,不是传统 memory API 语义,adapter 需要额外抽象一层,是 P1 而非 P0 的核心原因。
5. **mem0 2026 的 token-efficient 算法**声称 LOCOMO 92.5 / LongMemEval 94.4 的同时把单 query token 降到 ~7k,这是 engram 需要重点学习的方向(engram 现在单 query 消耗未测)。

---

## Sources

### Memory systems
- [mem0 repo](https://github.com/mem0ai/mem0)
- [mem0 paper (ECAI 2025)](https://arxiv.org/abs/2504.19413)
- [mem0 2026 state of memory](https://mem0.ai/blog/state-of-ai-agent-memory-2026)
- [mem0 token-efficient algorithm](https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm)
- [zep repo](https://github.com/getzep/zep) · [graphiti](https://github.com/getzep/graphiti)
- [zep temporal-KG paper](https://arxiv.org/abs/2501.13956)
- [zep SOTA blog](https://blog.getzep.com/state-of-the-art-agent-memory/)
- [zep test-memory guide](https://www.getzep.com/ai-agents/how-to-test-agent-memory/)
- [letta](https://github.com/letta-ai/letta) · [letta filesystem benchmark](https://letta.com/blog/benchmarking-ai-agent-memory)
- [cognee](https://github.com/topoteretes/cognee)
- [motorhead](https://github.com/getmetal/motorhead)

### Evaluation datasets
- [LongMemEval arXiv](https://arxiv.org/abs/2410.10813) · [repo](https://github.com/xiaowu0162/LongMemEval) · [HF](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
- [LongMemEval-V2](https://arxiv.org/abs/2605.12493) · [repo](https://github.com/xiaowu0162/LongMemEval-V2)
- [LOCOMO arXiv](https://arxiv.org/abs/2402.17753) · [page](https://snap-research.github.io/locomo/) · [repo](https://github.com/snap-research/locomo)
- [LoCoMo-Plus](https://arxiv.org/abs/2602.10715)
- [MSC arXiv](https://arxiv.org/abs/2107.07567) · [ACL 2022](https://aclanthology.org/2022.acl-long.356/) · [ParlAI](https://parl.ai/projects/msc/)
- [MemoryBench (THUIR)](https://arxiv.org/abs/2510.17281) · [repo](https://github.com/THUIR/MemoryBench) · [HF](https://huggingface.co/datasets/THUIR/MemoryBench)
- [MemBench (ACL 2025)](https://arxiv.org/abs/2506.21605) · [ACL page](https://aclanthology.org/2025.findings-acl.989/) · [repo](https://github.com/import-myself/Membench)
- [MemoryAgentBench](https://arxiv.org/abs/2507.05257) · [repo](https://github.com/HUST-AI-HYZ/MemoryAgentBench)
- [PerLTQA (Baidu)](https://arxiv.org/pdf/2203.05797v2)
- [Convomem](https://arxiv.org/abs/2511.10523)
- [Evo-Memory](https://arxiv.org/abs/2511.20857)
- [BEAM (mem0 blog)](https://mem0.ai/blog/ai-memory-benchmarks-in-2026)

### General benchmarks
- [HotpotQA](https://arxiv.org/abs/1809.09600) · [page](https://hotpotqa.github.io/)
- [MuSiQue](https://arxiv.org/abs/2108.00573) · [repo](https://github.com/StonyBrookNLP/musique)
- [LOFT](https://arxiv.org/abs/2406.13121) · [repo](https://github.com/google-deepmind/loft)
- [QUEST-LOFT re-eval](https://arxiv.org/abs/2511.06125)

### Related
- [LightMem](https://arxiv.org/html/2510.18866v1)
- [Vectorize Hindsight](https://vectorize.io/benchmarks)
