# EngramRouter LLM+向量检索新架构设计文档

> 技术架构师分析报告 | 2026-07-01 | 基于 v0.1.0 源码分析

---

## 一、当前架构的问题：为什么纯规则引擎达不到联想效果

### 1.1 当前架构回顾

当前 EngramRouter（v0.1）的召回管线如下：

```
用户查询 → _terms() 分词 → _fts_candidates() FTS5 trigram候选
         → extract_entities() 规则实体提取
         → _edge_expansion() 图传播扩展
         → _build_scored_candidates() 加权打分
         → 排序截断 → 返回 MemoryRecord
```

核心组件：

| 组件 | 文件 | 实现方式 |
|------|------|----------|
| 实体提取 | `entities.py` (~167行) | 正则 + 硬编码词表 |
| 关键词匹配 | `store.py::_score()` | 精确子串匹配 + 手动权重 |
| 图传播 | `store.py::_edge_expansion()` | BFS over edges 表，最多2跳 |
| 语义增强 | `llm_reranker.py` | 可选，LLM重排序 top-N |

### 1.2 规则引擎的根本局限

#### 问题一：硬编码枚举不可能覆盖所有实体

`config.py` 中维护了 **200+ 个硬编码词汇**：

```python
# 亲属词 (22个)
kinship_words = ["妈妈", "母亲", "爸爸", "父亲", "爷爷", ...]

# 已知公司 (8个)  
known_companies = ["腾讯", "阿里", "阿里巴巴", "字节", ...]

# 话题词 (13个)
topic_words = ["键盘", "礼物", "生日", "机械键盘", ...]

# 食物词 (18个)
food_words = ["红烧肉", "糖醋排骨", "西红柿炒蛋", ...]

# 对象-话题别名映射 (19条)
object_topic_aliases = {"HHKB": "键盘", "特斯拉": "车", ...}
```

**致命缺陷**：每遇到一个新领域，就需要人工维护词表。用户说 "我刚买了一台 MacBook Pro"，系统完全不知道这是什么——没有 `known_objects` 记录。规则引擎 **只能处理已知的已知，无法处理未知的已知**。

#### 问题二：正则模式的覆盖率天花板

`entities.py` 的提取函数全部基于正则：

```python
# 提取人名：只能匹配预定义的 surname_chars
rf"(?:^|[，。、！？\s和跟与])([{surname}]...)"

# 提取时间：只能匹配预定义的 time_patterns
time_patterns = [r"前[两三四五六七八九十0-9]*天", r"昨天", ...]

# 提取属性：只能匹配预定义的 attr_patterns
attr_patterns = [r"[0-9]{1,3}岁", r"[ABO]型血", ...]
```

**问题**：正则无法理解语义边界——
- "我奶奶是山东人" → 能识别"奶奶"（在 kinship_words 里），但不能识别"山东人"是什么（不在 attr_patterns 里）
- "我血压有点高" → 无法提取（没有血压相关的 pattern）
- "上次体检说胆固醇偏高" → 完全无法结构化

#### 问题三：关键词匹配置信度陷阱

当前的 `_score()` 方法本质是**精确子串匹配**：

```python
def _score(self, query, terms, haystack):
    score = 0.0
    for term in terms:
        if term.lower() in haystack:  # <-- 纯子串匹配
            score += self._term_weight(term)
    return score
```

**关键场景失败**：

| 查询 | 存储内容 | 能否命中 | 原因 |
|------|----------|----------|------|
| "我那个同事送我的键盘什么牌子" | "张三前两天送了我一把 HHKB" | ❌ 命中失败 | 查询中"键盘"和"HHKB"无共同子串 |
| "我妈手艺如何" | "妈妈做的红烧肉特别好吃" | ⚠️ 部分命中 | "妈妈"命中，但"手艺"不存在于原文 |
| "最近的体检结果怎么样" | "上次体检验血，各项指标正常" | ❌ 完全失败 | "体检结果"和"验血"是语义相关但无子串重合 |
| "通勤方便吗" | "搬家是因为离公司近，通勤方便" | ✅ 命中 | "通勤"和"方便"刚好都在原文中（巧合） |

规则引擎能工作的场景只限于**词汇恰好重合**的情况，一旦出现同义词、近义词、上下位词或跨语言的语义关联，就完全失效。

#### 问题四：硬编码权重参数难以调优

`RecallWeights` 包含 **30+ 个可调参数**：

```python
ascii_base: float = 4.0
cjk_multi_base: float = 2.0
fts_boost: float = 0.1
shared_entity_multiplier: float = 1.2
conflicting_person_penalty: float = 1.5
brand_boost: float = 2.0
occupation_boost: float = 1.5
# ... 总共 30+ 个参数
```

这些参数值是人工试探出来的，缺乏系统性的调优方法。每增加一种新的 boost 逻辑，就需要反复微调以避免破坏已有场景。

#### 问题五：语义联想链完全依赖预定义关系

当前 edge 传播能实现一定的联想：

```
"妈妈" → CO_OCCURS_WITH → "红烧肉" → edge传播 → "妈妈做的红烧肉特别好吃"
```

但这只在实体被正确提取且建立边的情况下才有效。如果"妈妈做饭手艺怎么样"这个查询中，LLM 能理解"手艺"= "做菜水平"= "好吃程度"，但规则引擎完全无法理解这种等价关系。

---

### 1.3 规则引擎 vs. 向量检索的能力边界

| 能力 | 规则引擎 (当前) | 向量检索 (目标) |
|------|----------------|-----------------|
| 精确关键词匹配 | ✅ 高精度 | ⚠️ 取决于嵌入质量 |
| 同义词匹配 (手艺↔厨艺) | ❌ | ✅ |
| 上下位词匹配 (键盘↔HHKB) | ⚠️ 需手动维护 alias | ✅ 自动 |
| 跨语言匹配 (keyboard↔键盘) | ❌ | ✅ 多语言模型 |
| 语义距离排序 | ❌ | ✅ |
| 实体提取覆盖率 | ~30% (仅硬编码) | >90% (LLM 提取) |
| 维护成本 | 持续增长 | 一次配置 |
| 推理速度 | <10ms | ~50-200ms（本地向量）/ ~200-500ms（远程LLM） |

---

## 二、目标架构设计：三层架构

### 2.1 总体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                       接口层 (Interface Layer)                    │
│                                                                  │
│   MCP Server (stdio JSON-RPC)  │  CLI  │  Python API (直接调用)  │
│   ┌──────────────┐            │       │                          │
│   │ memory.save   │            │       │                          │
│   │ memory.recall │            │       │                          │
│   │ memory.search │ (新增)      │       │                          │
│   └──────────────┘            │       │                          │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
┌──────────────────────────────────┼──────────────────────────────┐
│                     检索层 (Retrieval Layer)                      │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                   混合检索管道 (Hybrid Pipeline)              │ │
│  │                                                              │ │
│  │  召回阶段                    精排阶段                         │ │
│  │  ┌──────────┐              ┌──────────────┐                 │ │
│  │  │ 向量检索  │──┐           │   LLM 重排序  │                 │ │
│  │  │(语义Top-K)│  │           │  (语义精排)    │                 │ │
│  │  └──────────┘  │  ┌───────┐│  └──────────────┘                 │ │
│  │                 ├─►│ 融合  ├┤                                  │ │
│  │  ┌──────────┐  │  │(RRF)  ││  ┌──────────────┐                 │ │
│  │  │ 关键词检索 │──┘  └───────┘│  │  规则 Boost   │                 │ │
│  │  │(FTS5 保留) │              │  │(同当前逻辑)    │                 │ │
│  │  └──────────┘              └──────────────┘                 │ │
│  │                                                              │ │
│  │  ┌──────────────────────────────────────────────────────┐   │ │
│  │  │  查询理解 (Query Understanding)                        │   │ │
│  │  │  · LLM 查询改写 (query expansion)                      │   │ │
│  │  │  · LLM 实体提取 (补规则引擎不足)                        │   │ │
│  │  └──────────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
┌──────────────────────────────────┼──────────────────────────────┐
│                     存储层 (Storage Layer)                        │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   SQLite     │  │  向量索引     │  │   LLM 提取器          │   │
│  │  (原 schema) │  │  (FAISS/     │  │  (sentence-          │   │
│  │              │  │   ChromaDB)  │  │   transformers /     │   │
│  │  · memories  │  │              │  │   OpenAI API)        │   │
│  │  · entities  │  │  · memory    │  │                      │   │
│  │  · edges     │  │    embeddings│  │  · save 时自动提取    │   │
│  │  · evidence  │  │  · entity    │  │  · recall 时查询改写  │   │
│  │  · FTS5      │  │    embeddings│  │                      │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 三层职责

#### 存储层 (Storage Layer)

| 模块 | 职责 | 新增/保留 |
|------|------|-----------|
| SQLite | 原文存储、结构化实体/边、FTS5 关键词索引 | 保留现有 schema |
| FAISS/ChromaDB | 存储 memory embedding 向量，支持 ANN 检索 | **新增** |
| sentence-transformers | 文本→向量的本地编码 | **新增** |
| LLMExtractor（已有） | 增强：save 时自动调用 LLM 提取实体+边 | 增强 |

#### 检索层 (Retrieval Layer)

| 模块 | 职责 | 新增/保留 |
|------|------|-----------|
| 向量检索 | 基于余弦相似度的语义 Top-K 召回 | **新增** |
| 关键词检索 | FTS5 + LIKE 精确匹配（保留） | 保留 |
| 混合融合 | Reciprocal Rank Fusion (RRF) 合并两路结果 | **新增** |
| 查询理解 | LLM 查询改写+实体提取（增强规则引擎） | **新增** |
| 图传播 | Edge 扩展（保留当前 BFS 逻辑） | 保留 |
| LLM 重排序 | 对 Top-N 候选进行语义精排 | 保留增强 |

#### 接口层 (Interface Layer)

| 模块 | 职责 | 新增/保留 |
|------|------|-----------|
| MCP Server | 6 个现有工具 + `memory.search`（语义搜索） | 增强 |
| CLI | 命令行工具 | 保留 |
| Python API | `MemoryStore` 直接调用 | 保留增强 |

---

## 三、核心技术选型

### 3.1 向量模型

```
优先级：本地 sentence-transformers > OpenAI 兼容 API 兜底
```

#### 方案 A：sentence-transformers 本地模型（推荐默认）

```python
# 轻量级中文模型选型
候选模型：
  1. BAAI/bge-small-zh-v1.5     (24MB, 512维, 中英文)
  2. shibing624/text2vec-base-chinese  (400MB, 768维, 纯中文)
  3. intfloat/multilingual-e5-small   (118MB, 384维, 多语言)

推荐：BAAI/bge-small-zh-v1.5
  - 体积小，适合本地部署
  - 中英文混合支持（HHKB/iPhone 等英文品牌词）
  - MTEB 中文榜单前列
  - 首次下载后缓存，后续零延迟
```

```python
# 嵌入接口设计
class EmbeddingEngine:
    def __init__(self, model_name="BAAI/bge-small-zh-v1.5", 
                 backend="local", api_base=None, api_key=None):
        self.backend = backend
        if backend == "local":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
        else:
            self._client = OpenAIClient(api_base, api_key)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return [N, dim] float32 array."""
        ...

    @property
    def dim(self) -> int:
        """向量维度"""
        return 512  # for bge-small-zh
```

#### 方案 B：OpenAI 兼容 API 兜底

当本地模型不可用时（Python 版本限制、无 pip 安装权限等），自动降级到 OpenAPI 兼容的 embeddings API：

```python
# 兼容 OpenAI、DeepSeek、OneAPI 等
POST /v1/embeddings
{
    "model": "text-embedding-3-small",
    "input": ["张三送了我一把 HHKB 键盘"]
}
# → { "data": [{"embedding": [0.123, -0.456, ...], "index": 0}] }
```

### 3.2 向量索引方案

#### FAISS (推荐)

```python
import faiss
import numpy as np

class VectorIndex:
    def __init__(self, dim: int = 512):
        self.dim = dim
        # IVF 索引：先聚类再搜索，适合 1K~100K 规模
        quantizer = faiss.IndexFlatIP(self.dim)  # 内积 = 余弦相似度（归一化后）
        self.index = faiss.IndexIVFFlat(quantizer, self.dim, 
                                         min(100, max(4, int(np.sqrt(expected_size)))))
        
    def add(self, memory_id: str, embedding: np.ndarray):
        """添加一条 embedding"""
        ...
        
    def search(self, query_embedding: np.ndarray, k: int = 20) -> list[tuple[str, float]]:
        """返回 [(memory_id, similarity_score), ...]"""
        ...
```

**为什么选 FAISS 而不是 ChromaDB/Qdrant？**

| 方案 | 优点 | 缺点 | 适合场景 |
|------|------|------|----------|
| FAISS | 零服务依赖、内存效率极高、Facebook 维护 | 需手动管理索引文件 | **本地嵌入式首选** |
| ChromaDB | API 友好、自动持久化 | 有服务依赖、性能不如FAISS | 原型阶段 |
| Qdrant | 生产级、分布式 | 需要 Docker/服务 | 大规模部署 |

> 推荐 FAISS：与 EngramRouter "零依赖" 的设计理念一致。向量索引作为 SQLite 旁边的 `.faiss` 文件存储。

### 3.3 LLM 提取器

当前已有 `llm_extractor.py`，它已经实现了基本的 LLM 实体+关系提取。新架构中需要：

1. **从"可选"升级为"默认推荐"** —— 如果用户配置了 API key，save() 时自动调用
2. **增加批量模式** —— 支持一次 LLM 调用处理多条记忆，减少 API 调用次数
3. **缓存提取结果** —— 同一条记忆的 LLM 提取结果缓存，避免重复调用

```python
# 增强后的 LLMExtractor
class LLMExtractor:
    def extract_batch(self, texts: list[str]) -> list[dict]:
        """一次 LLM 调用处理多条文本，减少延迟和成本"""
        ...
    
    def extract_with_cache(self, text: str) -> dict:
        """基于文本哈希的缓存，避免重复提取"""
        ...
```

### 3.4 依赖清单

```toml
# pyproject.toml 新增依赖
[project.optional-dependencies]
llm = [
    "sentence-transformers>=2.2.0",  # 本地向量编码
    "faiss-cpu>=1.7.0",             # 向量索引（无 GPU 版本）
    "numpy>=1.24.0",                # 向量计算
]
# 已有依赖全为 stdlib (无新增必需依赖)
# LLM API 调用仍使用 stdlib urllib（零第三方 HTTP 库依赖）
```

---

## 四、数据流设计

### 4.1 save() 完整流程

```
用户调用 save(text, source="conversation")
          │
          ▼
┌─────────────────────────────────────────────────────┐
│ Step 1: 文本预处理                                   │
│   · 长度校验 (≤10KB)                                 │
│   · 生成 summary（保持现有取首句逻辑）                  │
│   · 分配 memory_id (monotonic id_sequences)          │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│ Step 2: 写 SQLite（现有逻辑不变）                      │
│   · INSERT INTO memories (raw_text, summary, ...)    │
│   · INSERT INTO evidence                             │
│   · INSERT INTO memories_fts (FTS5 trigram)          │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│ Step 3: 实体提取 (双通道并行)                          │
│                                                      │
│  ┌──────────────────┐    ┌──────────────────────┐    │
│  │ 规则引擎 (保留)    │    │ LLM 提取器 (推荐)     │    │
│  │ entities.py      │    │ llm_extractor.py     │    │
│  │ · 人名/公司/时间  │    │ · 实体名+类型+证据    │    │
│  │ · 对象/食物/属性  │    │ · Salience分类       │    │
│  │ · 因果/话题       │    │ · 关系边 (CAUSED_BY, │    │
│  │                   │    │   HAS_ATTRIBUTE,...) │    │
│  └────────┬─────────┘    └──────────┬───────────┘    │
│           │                         │                 │
│           └─────────┬───────────────┘                 │
│                     ▼                                 │
│           合并去重（规则优先，LLM补充）                   │
│           INSERT INTO entities / memory_entities       │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│ Step 4: 边构建（保留现有逻辑 + LLM边）                  │
│   · CO_OCCURS_WITH (同一记忆中的实体对)                │
│   · CAUSED_BY (因果标记)                             │
│   · DESCRIBES (产品→话题别名)                         │
│   · LLM 关系边 (HAS_ATTRIBUTE, OWNS, PREFERS, ...)   │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│ Step 5: 向量编码 & 索引 【新增】                       │
│                                                      │
│   text_to_encode = f"[summary] {raw_text}"            │
│   embedding = embedding_engine.encode(text_to_encode) │
│   vector_index.add(memory_id, embedding)             │
│   # FAISS: index.add_with_ids(embedding, id_mapping)  │
│   # 或 ChromaDB: collection.add(ids, embeddings, ...) │
│                                                      │
│   ⚠ 注意：如果 LLM 不可用，这里用规则引擎提取的实体名   │
│      拼接文本增强编码信号                               │
└─────────────────────┬───────────────────────────────┘
                      ▼
              commit → 返回 memory_id
```

### 4.2 recall() 完整流程

```
用户调用 recall(query, top_k=5)
          │
          ▼
┌─────────────────────────────────────────────────────┐
│ Step 1: 查询理解 (Query Understanding) 【增强】        │
│                                                      │
│  1a. 分词 _terms(query)      → ["同事","键盘","牌子"]  │
│  1b. 规则实体提取              → [{name:"键盘",kind:..}] │
│  1c. LLM 查询改写 (如果可用)    → 扩展同义词/近义词       │
│      "我那个同事送我的键盘什么牌子"                      │
│      → ["同事送的键盘品牌", "张三送的 HHKB 型号"]        │
│  1d. LLM 实体补充              → 发现"张三"等关联实体    │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│ Step 2: 多路候选召回 (并行) 【增强】                    │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ 向量检索       │  │ 关键词检索     │  │ 图传播扩展   │ │
│  │              │  │              │  │            │ │
│  │ query_vec =  │  │ FTS5 trigram │  │ BFS over  │ │
│  │ embed(query) │  │ + LIKE回退    │  │ edges表    │ │
│  │              │  │ + 实体名查询   │  │            │ │
│  │ FAISS.search │  │              │  │            │ │
│  │ → top-50     │  │ → 候选集      │  │ → 关联记忆  │ │
│  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘ │
│         │                 │                 │        │
│         └────────┬────────┴────────┬────────┘        │
│                  ▼                 ▼                  │
│          Reciprocal Rank Fusion (RRF)                 │
│          score_rrf = Σ 1/(k + rank_i)                │
│          → top-100 融合候选集                          │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│ Step 3: 精排 (Refinement) 【保留 + 增强】              │
│                                                      │
│  对 top-100 逐条打分：                                 │
│  ┌─────────────────────────────────────────────┐     │
│  │  规则打分 (保留)                               │     │
│  │  · 关键词匹配 _score()                        │     │
│  │  · 实体共享 boosted                           │     │
│  │  · 上下文 boost (brand/identity/eval)         │     │
│  │  · 人物冲突 penalty                           │     │
│  │  · 修正 penalty                               │     │
│  │  · 关联衰减 (salience decay)                   │     │
│  └─────────────────────────────────────────────┘     │
│                        +                              │
│  ┌─────────────────────────────────────────────┐     │
│  │  向量相似度打分 (新增)                         │     │
│  │  · cosine_sim(query_vec, memory_vec)          │     │
│  │  · 归一化到 [0, 1]                            │     │
│  └─────────────────────────────────────────────┘     │
│                        =                              │
│  final_score = α × rule_score + β × vector_score      │
│  (默认 α=0.4, β=0.6，可通过 config 调整)              │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│ Step 4: LLM 重排序 (可选) 【保留】                     │
│                                                      │
│  取 top-20 → LLMReranker.rerank() → 语义精排          │
│  (如果配置了 LLM API key)                             │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│ Step 5: 输出                                          │
│                                                      │
│  · 排序 top-k                                         │
│  · 批量加载 evidence_refs / raw_refs                  │
│  · 构造 MemoryRecord 列表                             │
│  · 返回                                               │
└─────────────────────────────────────────────────────┘
```

### 4.3 RRF 融合公式

```python
def reciprocal_rank_fusion(
    vector_results: list[tuple[str, float]],   # [(id, similarity)]
    keyword_results: list[tuple[str, float]],  # [(id, score)]
    k: int = 60,
) -> list[tuple[str, float]]:
    """RRF: Reciprocal Rank Fusion
    
    score_rrf(doc) = Σ_{i in results} 1 / (k + rank_i(doc))
    """
    scores: dict[str, float] = {}
    
    for rank, (doc_id, _) in enumerate(vector_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    for rank, (doc_id, _) in enumerate(keyword_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

---

## 五、Schema 设计

### 5.1 当前 Schema 评估

当前 SQLite schema 共有 **8 张表**：

```sql
memories          -- 核心记忆表 (raw_text + summary)
evidence          -- 证据引用
raw_logs          -- 原文日志
distilled_memories -- 压缩后的记忆
entities          -- 实体字典
memory_entities   -- 记忆-实体关联
edges             -- 实体间关系
id_sequences      -- 单调 ID 分配器
```

**评估结论**：当前 schema 设计良好，满足 Phase 2 需求，**不需要修改表结构**。

理由：
1. 向量索引独立于 SQLite（存储在 FAISS 文件或 ChromaDB 中）
2. Embedding 不需要存入 SQLite——向量索引自己有 ID 映射
3. 如果需要追溯，可以在 `memories` 表加一个 `embedding_version` 字段标识编码模型版本

### 5.2 唯一建议的扩展

```sql
-- 可选：记录 embedding 元信息（极小成本，极大可追溯性）
ALTER TABLE memories ADD COLUMN embedding_model TEXT NOT NULL DEFAULT '';
ALTER TABLE memories ADD COLUMN embedding_version TEXT NOT NULL DEFAULT '';
```

当用户切换向量模型时，可以用 `embedding_model = ''` 筛选出需要重新编码的记忆。

### 5.3 向量索引文件布局

```
~/.engram/
├── memory.db          # SQLite 数据库（不变）
├── memory.faiss       # FAISS 向量索引（新增）
├── memory.idmap       # FAISS ID → memory_id 映射（新增）
└── config.yaml        # 配置文件（增强：添加 embedding 配置）
```

或者使用单个 pickle 文件：

```
~/.engram/
├── memory.db
├── memory.vectors.pkl  # {"embeddings": [...], "id_map": {...}, "model": "..."}
└── config.yaml
```

---

## 六、性能预估

### 6.1 recall() 延迟对比

| 场景 | 当前 (v0.1) | Phase 2 (向量增强) | 说明 |
|------|------------|-------------------|------|
| 10条记忆 | ~1ms | ~5ms | 向量编码开销 |
| 100条记忆 | ~3ms | ~10ms | FAISS 微乎其微 |
| 1000条记忆 | ~15ms | ~50ms | 向量检索 <10ms |
| 10000条记忆 | ~100ms | ~200ms | FTS5 + FAISS 并行 |
| 100000条记忆 | ~500ms (full scan) | ~300ms (FAISS) | 向量检索不随数据量线性增长 |

### 6.2 save() 延迟对比

| 场景 | 当前 | Phase 2 (本地向量) | Phase 2 (远程向量) |
|------|------|-------------------|-------------------|
| 规则提取 | ~1ms | ~1ms | ~1ms |
| LLM 提取 | ~500-2000ms (可选) | ~500-2000ms (可选) | ~500-2000ms |
| 向量编码 | 无 | ~10-50ms (本地) | ~200-500ms (API) |
| 写入 SQLite | ~1ms | ~1ms | ~1ms |
| 写入向量索引 | 无 | ~1ms | ~1ms |
| **总计** | ~2ms (规则) / ~1-2s (LLM) | ~15ms (规则) / ~3-50ms (本地LLM+向量) | ~200-500ms (API) |

### 6.3 存储空间

| 数据规模 | SQLite | FAISS (512维 f32) |
|----------|--------|-------------------|
| 100条记忆 | ~50KB | ~200KB |
| 1000条记忆 | ~500KB | ~2MB |
| 10000条记忆 | ~5MB | ~20MB |
| 100000条记忆 | ~50MB | ~200MB |

> 向量存储开销 = 记忆数 × 维度(512) × 4字节(f32) × 2(FAISS索引开销) ≈ 4KB/条

### 6.4 关键指标对比

| 指标 | 当前 (v0.1) | Phase 2 | 提升 |
|------|------------|---------|------|
| 语义匹配覆盖率 | ~30% | >90% | **3x** |
| 实体提取覆盖率 | ~30% (仅硬编码) | >90% (LLM+规则) | **3x** |
| 联想命中率 (跨词汇) | ~10% | >70% | **7x** |
| 维护成本 | 持续增长 | 一次配置 | **显著降低** |
| API 调用次数/save | 0-1次 | 0-1次 (不变) | — |
| API 调用次数/recall | 0-1次 | 0-2次 (新增查询改写) | 略增 |
| 内存占用 | ~10MB | ~50MB (加载向量模型) | 增加 |

---

## 七、实施路线图

### Phase 2a: 向量检索基础（1-2周）

```
□ 集成 sentence-transformers + FAISS
□ 实现 EmbeddingEngine (本地优先，API兜底)
□ 实现 VectorIndex (FAISS 封装)
□ save() 增加向量编码步骤
□ recall() 增加向量检索通路
□ 实现 RRF 融合
□ 兼容已有数据库（对旧记忆批量编码）
```

### Phase 2b: 查询理解增强（1周）

```
□ LLM 查询改写（query expansion）
□ LLM 查询实体提取补充
□ 混合打分公式调优（α/β 参数）
```

### Phase 2c: 生产优化（1周）

```
□ 批量编码优化（save_batch）
□ 向量索引持久化/加载
□ 增量索引更新（只编码新记忆）
□ 模型热切换（embedding_version 字段）
□ 回退策略（向量不可用时降级到纯规则引擎）
```

---

## 八、回退策略

新架构的核心原则：**渐进增强，绝不退化**。

```python
class HybridMemoryStore(MemoryStore):
    """继承当前 MemoryStore，增强向量检索能力"""
    
    def __init__(self, path=None, embedding_config=None, ...):
        super().__init__(path=path, ...)
        self._vector_enabled = False
        
        try:
            self._embedding_engine = EmbeddingEngine(...)
            self._vector_index = VectorIndex.load_or_create(path)
            self._vector_enabled = True
        except ImportError:
            logger.warning("sentence-transformers/faiss not available; "
                          "vector search disabled. Install with: "
                          "pip install engram-router[llm]")
        except Exception:
            logger.warning("Vector engine init failed, "
                          "falling back to keyword-only mode")
    
    def recall(self, query, top_k=5, ...):
        if not self._vector_enabled:
            return super().recall(query, top_k, ...)  # 完全回退到当前逻辑
        
        # 向量增强的混合召回
        ...
```

即使所有向量/LLM 组件都不可用，系统完全退化为当前的 v0.1 行为——不会变差。

---

## 九、总结

当前 EngramRouter v0.1 是一个设计精巧的规则引擎，在**已知词汇匹配**场景下表现优秀。但其核心局限——纯正则+硬编码词表——使得它无法处理语义联想、同义词替换、跨语言匹配等真实场景。

Phase 2 架构引入**向量检索作为语义匹配层**，配合已有的关键词+FTS5+图传播，形成混合检索管道。核心变化：

1. **实体提取**：规则引擎保留为安全网，LLM 提取器作为主力
2. **候选召回**：FAISS 向量检索 + FTS5 关键词检索并行，RRF 融合
3. **打分排序**：规则评分（保留）与向量相似度加权融合
4. **零退化**：任何新组件不可用时，完全回退到当前行为

这将使 EngramRouter 从"精确字符串匹配引擎"进化为"语义理解记忆引擎"，同时保持其"证据优先、零压缩失真"的核心设计理念。
