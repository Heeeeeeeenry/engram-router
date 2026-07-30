# Cross-encoder 精排 + HyDE 集成设计

**日期**: 2026-07-21
**作者**: Agent C(设计) · 由主会话落盘
**前置依赖**: Agent A 的 `tests/eval_v2.py` 已就绪(P@1 / MRR / nDCG@k / Contamination@k),否则本文档所有指标提升数字均无法验证——旧 `semantic_audit.py`("top-3 内含关键词即通过")下 cross-encoder 只有边际提升,只有换成 P@1 / Contamination 才能测出真实收益(预期 +40 pp)。

---

## 0. 结论提要

- **默认 cross-encoder 模型**: `BAAI/bge-reranker-v2-m3` (fp16, 568M 参数, XLM-RoBERTa-large 底座, 8192 tok 上下文, CPU 单对 batch=16 约 180–350 ms, M 系列 Metal 60–120 ms)
- **低配退路**: `BAAI/bge-reranker-base` (278M, ~430 MB)
- **高精可选**: `bge-reranker-v2-m3` fp32(GPU),或云端 `cohere-rerank-multilingual-v3.0`(仅当 `ENGRAM_ALLOW_CLOUD_RERANKER=1`)
- **HyDE LLM 依赖**: 复用现有 `LLMClient`(`src/engram_router/llm_extractor.py:297`),受 `config.privacy.allow_cloud_llm` + `env_allows_cloud("llm")` 双闸门控。**离线时自动跳过**,不阻塞召回。本地 LLM 通道:设 `ENGRAM_LLM_BASE_URL=http://localhost:11434/v1` 走 Ollama。
- **Phase 1 CE 单项预期**: P@1 从 **0.61 → 0.80(±0.05)**,+19 pp;Contamination@3 约 -20 pp;MRR 约 0.74 → 0.85;延迟增量 warm < 150 ms,cold < 400 ms

---

## 1. 模型选型对比

| 模型 | 语言 | 参数量 | 上下文 | CPU latency (batch=16, ≤200 chars) | Metal latency | 中文效果(C-MTEB reranking) | 结论 |
|---|---|---|---|---|---|---|---|
| **`BAAI/bge-reranker-v2-m3`** | 100+ 语言 | 568M | 8192 | 180–350 ms | 60–120 ms | nDCG@10 顶尖梯队 | **默认** |
| `BAAI/bge-reranker-large` | 中/英 | 560M | 512 | 220–400 ms | 80–130 ms | 略低于 v2-m3 于短查询-短候选 | 备选 |
| `BAAI/bge-reranker-base` | 中/英 | 278M | 512 | 90–160 ms | 40–70 ms | 较 v2-m3 低 3–5 pt | 低配退路 |
| `mixedbread-ai/mxbai-rerank-base-v1` | 多语言 | 184M | 512 | 70–140 ms | 30–60 ms | 中文略弱于 bge-base | 参考 |
| `mixedbread-ai/mxbai-rerank-large-v1` | 多语言 | 435M | 512 | 200–380 ms | 70–120 ms | 与 bge-large 接近 | 参考 |
| `cohere-rerank-multilingual-v3.0` | 100+ 语言 | 云端 | 4096 | 网络 RTT 300–600 ms | 同 | 榜单第一但依赖联网 | 云端可选 |
| 现有 `LLMReranker`(`src/engram_router/llm_reranker.py:25`) | 依 LLM | — | 依 LLM | 300–1500 ms(LLM 依赖) | 同 | 依 model | A/B 基线保留 |

**默认理由**(逐条对应"必须能 CPU/Metal 本地跑 + 中文效果 + 现有依赖栈"):
- **中文短查询效果**: `v2-m3` 相比 `bge-reranker-large` 在混合中英/短查询上 nDCG@10 高 3–5 pt。项目 semantic_audit 里 S4/S7/S8/S10 属于短查询–短候选,正是 v2-m3 甜蜜点。
- **上下文足够**: 8192 tok 能容纳 top-20 候选的 raw_text(项目单条记忆 <200 字,远够)。
- **本地跑**: Phase 1 只对 fusion 后 top-20 精排,总增量 < 400 ms cold,< 150 ms warm。
- **依赖**: `sentence-transformers>=2.7.0`(项目 `pyproject.toml:19` 已有 `>=2.2.0`,需上调下限,`CrossEncoder` API 2.7 之后更稳);`torch>=2.1.0`(Metal MPS 稳定线)。**只加到 `[llm]` extra,不进主依赖**。

---

## 2. Cross-encoder 集成点

### 插入位置

在 `store.py:1667-1714` 的 "Phase 2: Vector-search fusion" **之后**、`store.py:1716-1733` 的 LLM reranker **之前**。

**为什么放在融合后**: CE 需要评估 fusion 出的 top-N(20)而非单通道 top-N,否则会遗漏图路径带来的候选。

**为什么放在 boost 之前**: `_apply_context_boosts`(`store.py:1427`) 与 `_apply_salience_decay`(`store.py:1504`) 是**业务规则**,应作用在语义排序之后作为最终调整。否则 CE 会把 brand-mismatch 但语义高的错误结果推到前排,与 context boost 冲突。

**与 LLM reranker 关系**: 新 CE 上线后默认关掉 LLM reranker;通过 flag 允许同时开启用于 A/B。

### 新模块骨架 `src/engram_router/cross_encoder.py`

```python
class CrossEncoderReranker:
    _MODELS = {
        "bge-v2-m3": {"name": "BAAI/bge-reranker-v2-m3", "size_mb": 1100},
        "bge-base":  {"name": "BAAI/bge-reranker-base",  "size_mb": 430},
    }

    def __init__(
        self,
        model: str = "bge-v2-m3",
        max_candidates: int = 20,
        device: str | None = None,           # auto: cuda > mps > cpu
        allow_cloud: bool | None = None,
        remote_provider: str | None = None,  # "cohere" 走云端 API
    ): ...

    @property
    def available(self) -> bool: ...

    def score(self, query: str, memories: list[dict]) -> list[float]: ...

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Score & sort by CE relevance, preserving fusion metadata."""
```

**行为约定**:
- lazy 加载:首次 `score()` 才 `CrossEncoder(model_name)`,pattern 对齐 `embedding.py:148 _init_backend`
- fallback:import 失败 / 加载失败 → `available=False`,recall 跳过 CE
- 云端分支:`allow_cloud and remote_provider == "cohere"` → 调用 Cohere Rerank API;env 门 `ENGRAM_ALLOW_CLOUD_RERANKER`(`config.py:235`)复用,不新增
- 分数融合:`final_score = ce_weight * ce_score + (1 - ce_weight) * fusion_score_normalized`,新增 `RecallWeights.ce_weight: float = 0.6`

### 接入点

- `MemoryStore.__init__` 新增可选参数 `cross_encoder: Any | None = None`,默认懒实例化 `CrossEncoderReranker()`,受与 `enable_vector` 同款开关控制
- `store.py:1717` 之前插入 CE 分支,与现有 `self.reranker` 互斥:CE 可用则走 CE,否则走 LLM reranker

---

## 3. HyDE 集成点

### 新模块 `src/engram_router/hyde.py`

不塞进 `query_expansion.py`(避免 595 行文件继续膨胀)。

```python
class HyDEExpander:
    def __init__(
        self,
        client: LLMClient | None = None,      # 复用 llm_extractor.LLMClient
        num_hypotheses: int = 3,
        max_len: int = 80,
        min_query_chars: int = 10,
        allow_cloud: bool | None = None,
    ): ...

    @property
    def available(self) -> bool: ...

    def generate(self, query: str) -> list[str]:
        """LLM 生成 N 段假想答案,禁编造具体人名/数字。"""

    def expand_and_recall(
        self,
        query: str,
        embedding: EmbeddingEngine,
        vector_index: VectorIndex,
        k: int = 20,
    ) -> list[tuple[str, float]]:
        """假想答案 → embed → vector.search → 返回 (id, score) 列表。"""
```

### Prompt 模板(中文优化)

```
system: 你是查询扩写助手。给定一个用户对个人记忆的提问,
生成 3 段简短的"可能记忆",每段一句,只描述结构与关键概念,
禁止编造具体人名/数字/日期/地点。用 JSON 数组返回。

user: 提问: "{query}"
返回格式: ["...", "...", "..."]
```

### 流程与融合

1. query → LLM 生成 N=3 假想句
2. 各自 embed → `vector_index.search(k=20)`
3. 与原 query 的 vector 结果一起 RRF 融合,权重 `[1.0, 0.5, 0.5, 0.5]`(原 query 更重)
4. 融合结果加入 `all_results` 参与最终 RRF

### 触发闸门

- `len(query.strip()) < 10` → skip
- `store.should_inject(query) is False` → skip(闲聊/编程/天气)
- negative pattern(`不是|没有|别|不`+疑问) → skip(HyDE 会强化错误方向)
- `HyDEExpander.available is False` → skip
- 缓存:复用 `ExpansionCache`(`query_expansion.py:314`)模式,key 前缀 `hyde:<query_sha>`,TTL 与现有一致

### 与 `LLMQueryRewriter` 关系

**并存不替代**。前者产 keyword-style 变体喂 FTS/entity 通道;HyDE 产 statement-style 假想答案**只喂向量通道**——不进 FTS(假想句含具象化描述,会污染关键词匹配)。

### 离线可用性

- HyDE 只在生成假想答案阶段依赖 LLM。完全离线时 `HyDEExpander.available = False`,recall 直接跳过 HyDE,回退到"原 query + 同义词展开"
- **本地 LLM 通道**: 项目 `LLMClient` 是 OpenAI-compatible HTTP,天然支持 Ollama / vLLM / LMStudio。设 `ENGRAM_LLM_BASE_URL=http://localhost:11434/v1` 后 HyDE 纯本地可用

### 接入点

`store.recall`(`store.py:1141`)进入变体 RRF 分支之前,把 HyDE 向量结果作为额外一个 result_list 加入 `all_results`。

---

## 4. Multi-query 与 Intent classifier

### Multi-query(升级现有 `LLMQueryRewriter`)

**结论**: 升级 `LLMQueryRewriter.rewrite()`(`query_expansion.py:280`)的 prompt,不新增模块。将 `max_variants=4` 提到 5,新增变体类型覆盖:

- (a) 同义替换
- (b) 抽象化(HHKB → 键盘)
- (c) 具象化(键盘 → HHKB/机械键盘)
- (d) 反向表述
- (e) 目的意图化("多大" → "年龄多少")

**融合策略**: 每变体独立走完 `_recall_single`,RRF 融合,权重按变体来源:
- rule-based: 1.0
- synonym: 0.7
- LLM: 0.8
- HyDE: 0.6

### Intent classifier(替代 `_asks_brand / _asks_identity / _asks_eval`)

替代 `store.py:2136-2149` 的三个正则,输出软概率。

**方案 A(小模型)**: 复用项目里的 `bge-small-zh-v1.5` 做 embedding,训一个 30 KB 的 8-way logistic regression 分类头。类别: `brand · identity · eval · reason · time · location · person · other`。冷启动无标注 → 用 semantic_audit 场景 + LLM 生成 500 条弱标注 seed。CPU < 1 ms。

**方案 B(LLM zero-shot)**: prompt 直接问 "输出 JSON: {brand: 0.9, identity: 0.1, ...}",延迟 200–500 ms + 云依赖,配合 `ExpansionCache` 命中率高时可用。

**方案 C(混合,推荐)**: 现有正则先跑;若正则命中 >1 条或全 0 → 走 LLM 精分类;LLM 结果 cache 一份。

**接口**:
```python
class IntentClassifier:
    def classify(self, query: str) -> dict[str, float]: ...
```

**破坏性变更**: `_build_scored_candidates`(`store.py:1296`)里 `asks_brand/asks_identity/asks_eval`(1322-1324)从 bool 变 float,`_apply_context_boosts`(`store.py:1427`)的 boost 相应变成 `weight * probability`。**需要过渡开关**: `RecallWeights.use_intent_probs: bool = False`,默认关。

---

## 5. 每项改动的评估计划

依赖: 等 `tests/eval_v2.py` 就绪(已完成,含 P@1/MRR/nDCG@k/Contamination@3/拒答率;稳定性 & 延迟直方图待补 —— `OPTIMIZATION_ROADMAP.md` L0.1 尾条)。**eval_v2 未就绪前不 merge。**

### 配置矩阵(对每项 CE / HyDE / MultiQ / Intent 重复)

`{off, on}` × `{semantic_audit 10 场景, multi_angle 集}` × `{cold, warm}`

### 指标增量

- P@1 / MRR / Contamination@3
- 平均延迟(ms)/ p95 延迟
- 内存 peak(RSS)

### 期望上限(基于 semantic_audit 场景分析)

| 改动 | P@1 增量 | Contam@3 变化 | 延迟增量 warm | 备注 |
|---|---|---|---|---|
| CE (`bge-reranker-v2-m3`) | +30~40 pp | -20 pp | +80~150 ms | 相对当前 P@1=0.61 → **0.80±0.05** |
| HyDE | +5~10 pp | 微降 | +300~600 ms | 仅 S8/S9 类零词面场景受益(2/10) |
| Multi-query | +3~8 pp | ~0 | +200~400 ms | 已有 rewriter,边际 |
| Intent classifier(C) | brand/identity +10 pp | 稳定性提升 | +50~200 ms | 误分类会掉分 |

### 反例场景(每项都必须列出降低情形)

- **CE 降低**: 候选集完全不含正例时,CE 会把语义相近但错误的候选推到 top-1
- **HyDE 降低**: query 是精确关键词(型号号/人名)时,假想答案会把召回稀释到相关领域一般项
- **Multi-query 降低**: 长 query(>20 字)已足够精确,LLM 变体反而引入 drift
- **Intent 降低**: 长尾意图分类误分类会把 `brand_boost` 加到非 brand 记忆上

### semantic_audit 逐场景 CE 预测

| 场景 | 当前 rank | 干扰机制 | CE 能否 promote | 置信度 |
|---|---|---|---|---|
| S1 妈妈多大 | 1 | — | — | 保留 |
| S2 HHKB | 1 | — | — | 保留 |
| S3 张三同事 | 1 | — | — | 保留 |
| S4 李四特斯拉 | 3 | 词面"特斯拉降价"占位 | 语义强区分实体动作 | **高** |
| S5 去年发生 | 2 | 时间词并列 | 弱(需时间推理) | 中 |
| S6 谁送键盘原因 | 1 | — | — | 保留 |
| S7 我最近胖了 | 3 | 词面"最近"命中跑步/手机 | "胖 ↔ 体重增加"经典 CE 优势 | **高** |
| S8 心情不好 | 3 | 全 0.5 平票 | 无词面依赖,CE 直接判语义因果 | **高** |
| S9 之前说的计划 | 3 | 全 0.15 平票 | 弱(指代太抽象) | 低 |
| S10 他送我的东西 | 2 | "键盘不好用"占位 | 人称消解+对象匹配 | **中高** |

- 基线 P@1(仅 semantic_audit) = 4/10 = 0.40
- 高置信提升:S4/S7/S8/S10 → +4
- 中/中高置信部分回收:S5 期望 0.5 概率 → +0.5
- 低置信:S9 保持 0
- **Phase 1 semantic_audit P@1 期望:0.75–0.85**

**警告**: CE 对"含实体但语义翻转"仍会犯错("李四**没**买特斯拉");S9 抽象指代 CE 不解决 → HyDE 的领地。Phase 1 单独上 CE **不会**把 P@1 打到 1.0。

---

## 6. 分阶段 rollout

| Phase | 功能 | 门禁(`tests/eval_v2.py` 必须通过) | 预计时长 |
|---|---|---|---|
| 0 | eval_v2 就绪 + baseline 落库 | 稳定跑 3 次,方差 < 2% | ✅ 已完成 |
| 1 | CE (`bge-reranker-v2-m3`) | P@1 ≥ 0.70,Contam@3 ≤ 15%,p95 延迟 ≤ 500 ms | 3–4 天 |
| 2 | HyDE(本地/云端可选) | Phase 1 指标不回落 + S8/S9 P@1 ≥ 0.5 | 4–5 天 |
| 3 | Multi-query 增强 | P@1 再 +3 pp 或不动;稳定性 Jaccard ≥ 0.85 | 3 天 |
| 4 | Intent classifier(方案 C) | 不引入回归,brand/identity 场景 +5 pp | 5–7 天 |

每阶段用 `RecallWeights` / config 开关控制,rollback 一条 env var 就能关。

---

## 7. 向后兼容

### `RecallWeights`(`store.py:47`)新增字段(默认值保持 `RecallWeights()` 行为不变)

- `ce_enabled: bool = True`
- `ce_model: str = "bge-v2-m3"`
- `ce_max_candidates: int = 20`
- `ce_weight: float = 0.6`
- `hyde_enabled: bool = False`(Phase 2 打开)
- `hyde_num_hypotheses: int = 3`
- `multi_query_max_variants: int = 5`
- `use_intent_probs: bool = False`(Phase 4 打开)
- `RecallWeightsConfig`(`config.py:154`)镜像补上同名字段

### `config.py`

- `PrivacyConfig.allow_cloud_reranker`(`config.py:215`)已有,复用
- HyDE 复用 `allow_cloud_llm`,**不新增 env var**

### `MCPServer`(`mcp_server.py:191`)

- `memory.recall` 工具签名不变(`store.py:1141` 只换内部 pipeline)
- 新增可选参数 `use_hyde: bool | None = None`,None 时读全局

### `cli.py`(`cli.py:26` recall 子命令)

- 新增 flag: `--rerank {ce, llm, off}`(默认 `ce`,若 CE 不可用回退 `off`)
- 新增 flag: `--hyde` / `--no-hyde`(默认 off)

### 依赖(`pyproject.toml`)

- `[llm]` extra: `sentence-transformers>=2.7.0`(上调下限)
- `torch>=2.1.0`(Metal MPS 稳定线)
- CE 模型不走 `faiss` 通道

---

## 关键文件锚点

- `src/engram_router/store.py` — recall pipeline 主入口
- `src/engram_router/embedding.py` — lazy 加载 pattern 参考
- `src/engram_router/query_expansion.py` — LLMQueryRewriter / ExpansionCache
- `src/engram_router/config.py` — Privacy + RecallWeightsConfig
- `src/engram_router/llm_reranker.py` — A/B 基线
- `src/engram_router/llm_extractor.py:297` — LLMClient(HyDE 复用)
- `tests/eval_v2.py` — 评估门禁
