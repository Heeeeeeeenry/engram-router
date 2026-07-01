# Query Expansion 模块设计方案

> **EngramRouter Phase 2 — 查询理解子系统**
>
> 作者：查询理解工程师  
> 模块文件：`src/engram_router/query_expansion.py`  
> 日期：2026-07-01

---

## 一、背景与动机

当前 `MemoryStore.recall()` 只对原始查询做表面分词（`_terms` 提取 ASCII 词 + CJK 双字/单字），没有做语义级别的查询扩展。典型问题：

| 用户输入 | 当前行为 | 期望行为 |
|----------|----------|----------|
| "我那个同事送我的键盘什么牌子" | 分词 → "同事" "键盘" "牌子" | 扩展出多个搜索变体：`["同事送的键盘品牌", "同事 送的 键盘", "键盘 品牌 同事", "机械键盘 品牌"]` |
| "张三那个贼好用的HHKB" | 分词 → "张三" "HHKB" | 补充实体关系：张三 -[OWNS]→ HHKB，同义词扩展 HHKB → ["HHKB", "HHKB键盘", "机械键盘"] |
| "上次和老王吃的那个馆子" | 分词 → "老王" "馆子" | 扩展出：`["和老王去的馆子", "老王 餐厅", "老王 吃饭"]` |

**核心原则**：查询扩展的目标不是"改查询"，而是"产生更多召回路径"，由 RRF 融合统一排序。

---

## 二、架构总览

```
┌──────────────────────────────────┐
│         MemoryStore.recall()     │
│   query = "我同事送的键盘什么牌子" │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│         QueryExpander            │  ← 本模块
│  ┌───────────────────────────┐   │
│  │ 1. SynonymExpander        │   │  : 同义词表（零依赖，<1ms）
│  │ 2. LLMQueryRewriter       │   │  : LLM 查询改写（可选，异步）
│  │ 3. LLMEntitySuppleter     │   │  : LLM 实体补充（可选，异步）
│  │ 4. ExpansionCache         │   │  : 缓存层（LRU）
│  └───────────────────────────┘   │
└──────────────┬───────────────────┘
               │
               ▼
    ExpandedQuery(
        variants=["同事送的键盘", "键盘品牌同事", "同事 机械键盘 品牌"],
        extra_entities=[{"name": "HHKB", "kind": "object"}, ...],
        synonyms={"HHKB": ["机械键盘", "键盘"]}
    )
               │
               ▼
    MemoryStore 将每个 variant 送入 recall 管道
    → RRF 融合多路径结果
```

---

## 三、模块接口设计

### 3.1 数据结构

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ExpandedQuery:
    """查询扩展结果，由 QueryExpander.expand() 返回。"""

    original: str
    """原始查询字符串（不变）"""

    variants: list[str] = field(default_factory=list)
    """改写后的搜索变体列表。
       例: ["同事送的键盘", "键盘品牌同事", "同事 HHKB 机械键盘"]
       每条变体都会独立送入 recall 管道 → RRF 融合。
    """

    extra_entities: list[dict[str, Any]] = field(default_factory=list)
    """从查询中额外提取的实体，格式同 entities.extract_entities()。
       例: [{"name": "HHKB", "kind": "object", "evidence": "HHKB"}]
       这些实体会合并到 query_entities 用于 entity hop 打分。
    """

    synonyms: dict[str, list[str]] = field(default_factory=dict)
    """词级同义词映射（用于 term 扩展）。
       例: {"HHKB": ["机械键盘", "键盘"], "车": ["特斯拉", "电动车"]}
       recall 管道的 _terms() 会将这些同义词也加入 term 列表。
    """

    source: str = "none"
    """扩展来源: "none" | "synonym-only" | "llm-cached" | "llm-fresh"
       用于性能监控和降级决策。
    """

    latency_ms: float = 0.0
    """本次扩展耗时（毫秒），用于监控。"""
```

### 3.2 核心接口：QueryExpander

```python
class QueryExpander:
    """查询扩展编排器。

    职责：
    - 编排 3 个扩展策略，按延迟优先级执行
    - 管理缓存（同义词表内存常驻 + LLM 结果 LRU）
    - 降级策略：LLM 不可用时自动退化为纯同义词模式
    - 异步预加载：LLM 结果返回后缓存，下次查询直接命中

    使用方式：
        expander = QueryExpander()

        # 方式 1：同步调用（同义词 + 缓存命中，<1ms）
        eq = expander.expand("我同事送的键盘什么牌子")

        # 方式 2：带异步触发的调用（首次调用，同义词立即返回，
        #         后台触发 LLM 扩展，缓存供下次用）
        eq = expander.expand("我同事送的键盘什么牌子", async_llm=True)

    集成到 MemoryStore：
        class MemoryStore:
            def __init__(self, ..., query_expander=None):
                self.query_expander = query_expander

            def recall(self, query, top_k=5, namespace="default"):
                # Phase 2: 查询扩展
                if self.query_expander:
                    eq = self.query_expander.expand(query)
                    variants = [query] + eq.variants if eq.variants else [query]
                    # 将 extra_entities 合并到 query_entity_objs
                    # 将 synonyms 合并到 terms
                else:
                    variants = [query]
                # ... 现有 recall 管道
    """

    def __init__(
        self,
        synonym_table: SynonymTable | None = None,
        llm_client: LLMClient | None = None,
        cache_size: int = 256,
        enable_llm: bool = True,
    ):
        ...

    def expand(
        self,
        query: str,
        async_llm: bool = False,
    ) -> ExpandedQuery:
        """执行查询扩展。

        Args:
            query: 原始查询字符串。
            async_llm: 是否触发异步 LLM 扩展（后台执行，结果缓存供下次用）。

        Returns:
            ExpandedQuery，包含所有可用的扩展结果。
            延迟保证：同步路径 < 200ms；首次查询仅走同义词 + 缓存，
            LLM 结果在后续查询中生效。

        执行流程：
            1. 【同步 <1ms】检查缓存 → 命中则直接返回
            2. 【同步 <1ms】应用同义词表扩展
            3. 【条件触发】如果 async_llm=True 且缓存未命中，
               启动后台线程调用 LLM，当前请求返回纯同义词结果
            4. LLM 结果返回后写入缓存，下次查询自动命中
        """
        ...

    def expand_sync(self, query: str) -> ExpandedQuery:
        """同步扩展（阻塞等待 LLM，仅用于测试/调试）。"""
        ...

    def prewarm(self, queries: list[str]) -> None:
        """预热缓存：批量异步查询 LLM 并缓存结果。"""
        ...
```

### 3.3 子模块接口

```python
class SynonymTable:
    """同义词映射表（零依赖核心）。

    数据来源：
    - 内置默认表（基于 object_topic_aliases 扩展）
    - ~/.engram/config.yaml 中的 `expansion.synonyms` 字段
    - 运行时通过 `add()` 动态注入

    性能：O(1) 字典查找，< 0.1ms
    """

    def __init__(self, extra_synonyms: dict[str, list[str]] | None = None):
        """初始化同义词表。

        Args:
            extra_synonyms: 额外的同义词映射，会与内置表合并。
        """
        ...

    def expand(self, text: str) -> dict[str, list[str]]:
        """扫描文本中匹配的 token，返回同义词映射。

        Args:
            text: 输入文本（通常是查询字符串）。

        Returns:
            {matched_token: [synonym1, synonym2, ...]}

        Example:
            >>> st = SynonymTable()
            >>> st.expand("我的HHKB键盘坏了")
            {"HHKB": ["机械键盘", "键盘", "HHKB键盘"],
             "键盘": ["按键", "外设"]}
        """
        ...

    def add(self, term: str, synonyms: list[str]) -> None:
        """动态添加同义词。"""
        ...


class LLMQueryRewriter:
    """LLM 查询改写器。

    将口语化查询扩展为多个搜索变体。

    原理：
    - 发送 system prompt + 用户查询到 LLM
    - LLM 返回 JSON：多个改写变体 + 提取的实体
    - 结果缓存到 ExpansionCache

    注意：改写变体 NOT 修改原始查询，而是作为额外的召回路径。
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        max_variants: int = 4,
    ):
        ...

    def rewrite(self, query: str) -> RewriteResult:
        """调用 LLM 改写查询。

        Returns:
            RewriteResult(variants, entities)

        仅在 self.available == True 时有效，
        否则返回空 RewriteResult。
        """
        ...

    @property
    def available(self) -> bool:
        """LLM API 是否可用。"""
        ...


@dataclass
class RewriteResult:
    variants: list[str] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)


class ExpansionCache:
    """查询扩展结果缓存（内存 LRU）。

    键：query 字符串（normalized）
    值：ExpandedQuery（不含 original 字段的变体）
    """

    def __init__(self, max_size: int = 256):
        ...

    def get(self, query: str) -> ExpandedQuery | None: ...
    def put(self, query: str, eq: ExpandedQuery) -> None: ...
    def clear(self) -> None: ...
    def size(self) -> int: ...
```

---

## 四、执行流程（伪代码）

### 4.1 主流程：`QueryExpander.expand()`

```python
def expand(self, query: str, async_llm: bool = False) -> ExpandedQuery:
    """
    ┌──────────────────────────────────────────────────┐
    │               QueryExpander.expand()             │
    │                                                  │
    │  输入: "我那个同事送我的键盘什么牌子"              │
    │                                                  │
    │  Step 1: 查缓存                                  │
    │    if cache.get(query) → return cached  (0.1ms)  │
    │                                                  │
    │  Step 2: 同义词扩展 (同步, <1ms)                  │
    │    synonyms = synonym_table.expand(query)         │
    │    → {"HHKB": ["机械键盘", "键盘"], "同事": ...}  │
    │                                                  │
    │  Step 3: 判断是否需要 LLM                         │
    │    if not enable_llm → return (同义词结果)        │
    │    if cached → return (同义词+缓存LLM)            │
    │                                                  │
    │  Step 4: LLM 扩展                                 │
    │    if async_llm:                                  │
    │      启动后台线程: llm_rewriter.rewrite(query)    │
    │      结果写入缓存后自动生效                         │
    │      当前返回: 同义词结果 (延迟 <1ms)              │
    │    else:                                          │
    │      同步等待 LLM (仅测试用)                       │
    │    return merged_result                           │
    └──────────────────────────────────────────────────┘
    """
    import time
    t0 = time.perf_counter()

    # Step 1: 缓存命中
    cached = self._cache.get(query)
    if cached is not None:
        cached.latency_ms = (time.perf_counter() - t0) * 1000
        return cached

    # Step 2: 同义词扩展（零依赖，永不走LLM）
    synonyms = self._synonym_table.expand(query)
    extra_entities: list[dict[str, Any]] = []

    # Step 3: LLM 扩展
    llm_variants: list[str] = []
    llm_entities: list[dict[str, Any]] = []
    source = "synonym-only"

    if self._enable_llm and self._rewriter.available:
        if async_llm:
            # 异步触发：后台线程执行 LLM，当前立即返回同义词结果
            self._trigger_async_rewrite(query)
        else:
            # 同步等待（仅测试/调试用）
            result = self._rewriter.rewrite(query)
            llm_variants = result.variants
            llm_entities = result.entities
            source = "llm-fresh"

    # Step 4: 组装结果
    eq = ExpandedQuery(
        original=query,
        variants=llm_variants,     # 改写变体（可能为空）
        extra_entities=llm_entities, # LLM 额外实体
        synonyms=synonyms,          # 同义词映射
        source=source,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )

    # 缓存结果（仅同义词结果也可以缓存，LLM 结果异步更新）
    self._cache.put(query, eq)

    return eq
```

### 4.2 同义词扩展：`SynonymTable.expand()`

```python
# 内置同义词表（从现有 object_topic_aliases 扩展而来）
_DEFAULT_SYNONYMS: dict[str, list[str]] = {
    # ── 键盘 ──
    "HHKB":       ["机械键盘", "键盘", "HHKB键盘", "静电容键盘"],
    "Keychron":   ["机械键盘", "键盘", "客制化键盘"],
    "MX":         ["机械键盘", "键盘", "Cherry键盘"],
    "机械键盘":    ["键盘", "外设"],
    "键盘":        ["按键", "外设"],

    # ── 车 ──
    "特斯拉":      ["车", "电动车", "Model 3", "Model Y"],
    "Model 3":     ["特斯拉", "车", "电动车"],
    "电动车":       ["车", "电车"],

    # ── 手机 ──
    "iPhone":      ["手机", "苹果手机"],
    "iPad":        ["平板", "iPad平板"],

    # ── 食品 ──
    "宫保鸡丁":     ["菜", "川菜", "炒菜"],
    "红烧肉":       ["菜", "肉菜", "炒菜"],
    "糖醋排骨":     ["菜", "排骨"],

    # ── 人物关系 ──
    "同事":        ["同僚", "工友"],
    "朋友":        ["好友", "哥们", "闺蜜"],
    "老板":        ["领导", "上司", "Boss"],

    # ── 属性 ──
    "什么牌子":     ["品牌", "型号", "哪个牌子"],
    "多少钱":       ["价格", "多少钱", "价位", "费用"],
    "怎么样":       ["评价", "体验", "好用吗"],
}

class SynonymTable:
    def __init__(self, extra_synonyms=None):
        self._map: dict[str, list[str]] = dict(_DEFAULT_SYNONYMS)
        if extra_synonyms:
            for k, v in extra_synonyms.items():
                self._map[k] = v

    def expand(self, text: str) -> dict[str, list[str]]:
        """扫描文本，返回命中的同义词映射。

        算法：
        1. 按 key 长度降序排列（长词优先匹配，避免 "键盘" 吃掉 "机械键盘"）
        2. 遍历，命中则加入结果
        3. O(k * n) where k = 同义词条目数(~100), n = 文本长度(~50)
           → 实际 < 0.1ms
        """
        result: dict[str, list[str]] = {}
        # 按 key 长度降序：长词优先
        sorted_keys = sorted(self._map.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if key in text and key not in result:
                result[key] = self._map[key]
        return result
```

### 4.3 LLM 查询改写：`LLMQueryRewriter.rewrite()`

```python
_QUERY_REWRITE_SYSTEM_PROMPT = """\
You are a query expansion engine for a personal memory retrieval system.
Given a colloquial Chinese user query, produce search variants and extra entities to improve recall.

Output ONLY valid JSON. No explanation, no markdown fences.

Schema:
{
  "variants": ["改写变体1", "改写变体2", ...],
  "entities": [
    {"name": "实体名", "kind": "person|object|company|topic|...", "evidence": "原文"}
  ]
}

Rules:
1. variants: 2~4 条搜索变体。
   - 每条变体是完整的搜索短语（不是分词）。
   - 保留关键实体（人名、品牌）不变。
   - 去除口语化噪音（"那个""什么""这个""啊"）。
   - 加入可能的同义词替换（HHKB → 机械键盘）。
   例: "我那个同事送我的键盘什么牌子"
     → ["同事送的键盘品牌", "张三送的HHKB型号", "同事 机械键盘 品牌"]

2. entities: 从查询中提取的关键实体。
   - 只提取查询中明确存在的实体（不推测数据库中有什么）。
   - 不要重复 rule-based 已覆盖的实体（同事/朋友/键盘 等）。
   - 重点：品牌名简称、口语化指代、隐含话题。
   例: "我那个同事送我的键盘什么牌子"
     → [{"name": "键盘", "kind": "object", "evidence": "键盘"}]
     （注：同事/牌子已被 rule-based 覆盖，不再重复）

3. 如果查询很简短（<6个字符）或无需改写，variants 可以为空数组 []。
"""

_QUERY_REWRITE_USER_TEMPLATE = "User query: {query}"


class LLMQueryRewriter:
    def __init__(self, client=None, max_variants=4):
        # 复用 llm_extractor.LLMClient（零额外依赖）
        from .llm_extractor import LLMClient
        self._client = client or LLMClient()
        self._max_variants = max_variants

    @property
    def available(self) -> bool:
        return self._client.available

    def rewrite(self, query: str) -> RewriteResult:
        if not self.available:
            return RewriteResult()

        # 短查询不需要改写
        if len(query) < 6:
            return RewriteResult()

        messages = [
            {"role": "system", "content": _QUERY_REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": _QUERY_REWRITE_USER_TEMPLATE.format(query=query)},
        ]

        try:
            raw = self._client.chat(messages, temperature=0.0)
            parsed = _parse_json_response(raw)
            variants = parsed.get("variants", [])[:self._max_variants]
            entities = parsed.get("entities", [])
            return RewriteResult(variants=variants, entities=entities)
        except Exception:
            logger.exception("LLM query rewrite failed")
            return RewriteResult()
```

### 4.4 缓存层：`ExpansionCache`

```python
from collections import OrderedDict
import threading

class ExpansionCache:
    """线程安全的 LRU 缓存，专为查询扩展设计。

    特点：
    - 键归一化：去除多余空格、统一大小写
    - 支持异步更新：LLM 结果到达后通过 update() 追加到已有条目
    """

    def __init__(self, max_size: int = 256):
        self._max_size = max_size
        self._data: OrderedDict[str, ExpandedQuery] = OrderedDict()
        self._lock = threading.Lock()

    def _normalize(self, query: str) -> str:
        return " ".join(query.lower().split())

    def get(self, query: str) -> ExpandedQuery | None:
        key = self._normalize(query)
        with self._lock:
            return self._data.get(key)

    def put(self, query: str, eq: ExpandedQuery) -> None:
        key = self._normalize(query)
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = eq
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)

    def update(self, query: str, variants: list[str], entities: list[dict]) -> bool:
        """异步更新：将 LLM 结果合并到已有缓存条目。

        Returns:
            True 如果更新成功，False 如果条目不存在（缓存已过期）。
        """
        key = self._normalize(query)
        with self._lock:
            existing = self._data.get(key)
            if existing is None:
                return False
            existing.variants = variants
            existing.extra_entities = entities
            existing.source = "llm-cached"
            return True

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._data)
```

### 4.5 异步触发机制

```python
import threading

class QueryExpander:
    # ... (前面的接口)

    def _trigger_async_rewrite(self, query: str) -> None:
        """启动后台线程执行 LLM 改写，结果写入缓存。

        这个方法是 fire-and-forget 的：
        - 当前请求不等待，直接返回同义词结果
        - LLM 结果到达后写入缓存
        - 下一次相同查询会命中缓存，获得完整扩展结果
        """
        def _run():
            try:
                result = self._rewriter.rewrite(query)
                if result.variants or result.entities:
                    self._cache.update(query, result.variants, result.entities)
            except Exception:
                logger.exception("Async query rewrite failed for: %s", query)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
```

---

## 五、集成到 `MemoryStore.recall()`

### 5.1 现有 recall 管道的修改点

```python
class MemoryStore:
    def __init__(
        self,
        ...,
        query_expander: QueryExpander | None = None,   # ← 新增参数
    ):
        ...
        self.query_expander = query_expander

    def recall(self, query: str, top_k: int = 5,
               namespace: str = "default") -> list[MemoryRecord]:

        # ================================================================
        # Phase 2 新增：Query Expansion
        # ================================================================
        if self.query_expander is not None:
            eq = self.query_expander.expand(query, async_llm=True)

            # 1. 将同义词合并到 term 列表
            extra_terms: list[str] = []
            for synonyms in eq.synonyms.values():
                extra_terms.extend(synonyms)
            terms = self._terms(query) + list(dict.fromkeys(extra_terms))

            # 2. 将 LLM 额外实体合并到 query_entities
            query_entity_objs = extract_entities(query)
            existing = {(e["name"], e["kind"]) for e in query_entity_objs}
            for ent in eq.extra_entities:
                if (ent["name"], ent.get("kind", "")) not in existing:
                    query_entity_objs.append(ent)

            # 3. 对每个 variant 执行独立的召回 → RRF 融合
            if eq.variants:
                all_results: list[list[tuple[str, float]]] = []
                # 原始查询的召回
                primary = self._recall_single(query, terms, query_entity_objs, namespace)
                all_results.append([(r.id, r.score) for r in primary])

                # 每个 variant 的召回
                for variant in eq.variants:
                    v_terms = self._terms(variant)
                    v_entities = extract_entities(variant)
                    v_results = self._recall_single(variant, v_terms, v_entities, namespace)
                    all_results.append([(r.id, r.score) for r in v_results])

                # RRF 融合所有路径
                merged = reciprocal_rank_fusion(all_results, k=60)

                # ... 后续排序、截断、构建 MemoryRecord
        else:
            # 回退到原始行为（无查询扩展）
            terms = self._terms(query)
            query_entity_objs = extract_entities(query)
            # ... 现有逻辑

        # ... 其余 recall 管道不变
```

### 5.2 性能预算

| 路径 | 延迟 | 说明 |
|------|------|------|
| 缓存命中 | < 0.5ms | 跳过所有计算 |
| 同义词扩展 | < 1ms | 只做 O(n×k) 字符串匹配 |
| 缓存未命中 + 异步 LLM | < 2ms | 同义词立即返回，LLM 后台执行 |
| 缓存未命中 + 同步 LLM | ~500ms–2s | 仅测试/调试用，生产不推荐 |
| 后续相同查询 | < 0.5ms | 缓存命中（含 LLM 结果） |

**核心保证**：首次查询延迟 < 200ms（同义词路径 < 1ms，其余为现有 recall 管道开销），LLM 结果在下一次查询时自动生效。

---

## 六、配置集成

### 6.1 在 `config.py` 中新增 `ExpansionConfig`

```python
@dataclass
class ExpansionConfig:
    """查询扩展配置。"""

    # ── 同义词表 ──
    synonyms: dict[str, list[str]] = field(default_factory=dict)
    """用户自定义同义词表，会与内置表合并。
       例:
         synonyms:
           HHKB: [机械键盘, 键盘, 静电容]
           Mac: [苹果电脑, MacBook]
    """

    # ── LLM 查询改写 ──
    llm_enabled: bool = True
    """是否启用 LLM 查询改写。设为 False 则仅使用同义词表。"""

    llm_max_variants: int = 4
    """每次改写最多生成的变体数。"""

    # ── 缓存 ──
    cache_size: int = 256
    """查询扩展结果缓存大小（条目数）。"""

    async_llm: bool = True
    """是否使用异步 LLM 模式（推荐）。设为 False 则首次查询同步等待 LLM。"""


@dataclass
class EngramConfig:
    # ... 现有字段 ...
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)
    #          ↑ 新增
```

### 6.2 配置示例（`~/.engram/config.yaml`）

```yaml
expansion:
  synonyms:
    MyProduct: [产品A, 产品B]
    Mac: [苹果电脑, MacBook, 笔记本]
  llm_enabled: true
  llm_max_variants: 3
  cache_size: 512
  async_llm: true
```

---

## 七、测试策略

### 7.1 单元测试

```python
# tests/test_query_expansion.py

class TestSynonymTable:
    def test_exact_match(self):
        st = SynonymTable()
        result = st.expand("我的HHKB键盘坏了")
        assert "HHKB" in result
        assert "机械键盘" in result["HHKB"]

    def test_long_match_first(self):
        """验证长词优先匹配：'机械键盘' 不因 '键盘' 被跳过。"""
        st = SynonymTable()
        result = st.expand("机械键盘")
        assert "机械键盘" in result  # 长词优先
        assert "键盘" in result      # 短词也命中（不同 entry）

    def test_no_match(self):
        st = SynonymTable()
        result = st.expand("今天天气真好")
        assert result == {}

    def test_dynamic_add(self):
        st = SynonymTable()
        st.add("AB测试", ["A/B测试", "灰度实验"])
        result = st.expand("我们做了AB测试")
        assert "AB测试" in result


class TestExpansionCache:
    def test_put_and_get(self):
        cache = ExpansionCache(max_size=10)
        eq = ExpandedQuery(original="测试", variants=["v1"])
        cache.put("测试", eq)
        cached = cache.get("测试")
        assert cached is not None
        assert cached.variants == ["v1"]

    def test_lru_eviction(self):
        cache = ExpansionCache(max_size=2)
        for i in range(5):
            cache.put(f"query_{i}", ExpandedQuery(original=f"query_{i}"))
        assert cache.size == 2

    def test_normalization(self):
        cache = ExpansionCache()
        eq = ExpandedQuery(original="  Hello    World ")
        cache.put("  Hello    World ", eq)
        assert cache.get("hello world") is not None


class TestQueryExpander:
    def test_synonym_only(self, expander_no_llm):
        """无 LLM 时，仅走同义词路径。"""
        eq = expander_no_llm.expand("我的HHKB键盘")
        assert eq.source == "synonym-only"
        assert "HHKB" in eq.synonyms
        assert eq.latency_ms < 10  # 应该很快

    def test_cache_hit(self, expander):
        """第二次相同查询应命中缓存。"""
        eq1 = expander.expand("同事送的键盘")
        eq2 = expander.expand("同事送的键盘")
        assert eq2.source == "synonym-only"  # 缓存命中后不再调 LLM
        assert eq2.latency_ms < 1

    @pytest.mark.slow
    def test_llm_rewrite(self, expander_with_llm):
        """端到端 LLM 改写测试。"""
        eq = expander_with_llm.expand_sync("我那个同事送我的键盘什么牌子")
        assert len(eq.variants) >= 1
        assert any("键盘" in v for v in eq.variants)
```

### 7.2 集成测试（对 `MemoryStore.recall()`）

```python
def test_recall_with_expansion():
    """验证查询扩展确实提升了召回率。"""
    store = MemoryStore(path=":memory:", query_expander=QueryExpander())
    store.save("张三昨天送了小王一把HHKB Professional 2键盘作为生日礼物")
    store.save("李四的MacBook Pro是公司配的")

    results = store.recall("小王收到的是什么键盘")
    # 期望：即使原始查询不含"张三"/"HHKB"，也能通过同义词和改写召回正确结果
    assert any("HHKB" in r.raw_text for r in results)
```

---

## 八、降级与监控

### 8.1 降级路径

```
LLM 不可用？
├── Yes → 仅使用 SynonymTable（零依赖模式）
│         expanded.source = "synonym-only"
│         延迟 < 1ms
│
└── No  → LLM 可用
    ├── 缓存命中 → 直接返回（含上次 LLM 结果）
    │     source = "synonym-only" 或 "llm-cached"
    │     延迟 < 0.5ms
    │
    └── 缓存未命中
        ├── async_llm=True  → 同义词立即返回 + 后台LLM
        │     source = "synonym-only"
        │     延迟 < 2ms（User 无感知）
        │
        └── async_llm=False → 同步等待 LLM
              source = "llm-fresh"
              延迟 ~500ms-2s（不推荐生产）
```

### 8.2 监控指标

```python
# 在 QueryExpander 中暴露监控指标
@dataclass
class ExpansionStats:
    total_calls: int = 0
    cache_hits: int = 0
    synonym_only: int = 0
    llm_cached: int = 0
    llm_fresh: int = 0
    llm_errors: int = 0
    avg_latency_ms: float = 0.0
    cache_size: int = 0

class QueryExpander:
    @property
    def stats(self) -> ExpansionStats:
        """返回累计统计信息，用于监控和调优。"""
        ...
```

### 8.3 日志

```python
logger.info("Query expanded: %r → %d variants, %d entities, source=%s, %.1fms",
            query, len(eq.variants), len(eq.extra_entities), eq.source, eq.latency_ms)
logger.warning("LLM rewrite failed for %r (will retry on next call)", query)
logger.debug("Cache hit for %r (age=%ds)", query, age_seconds)
```

---

## 九、文件清单

| 文件 | 说明 | 状态 |
|------|------|------|
| `src/engram_router/query_expansion.py` | 本模块主文件 | 待创建 |
| `src/engram_router/config.py` | 新增 `ExpansionConfig` | 待修改 |
| `src/engram_router/__init__.py` | 导出 `QueryExpander` | 待修改 |
| `src/engram_router/store.py` | `MemoryStore.__init__` 新增 `query_expander` 参数<br>`recall()` 集成扩展逻辑 | 待修改 |
| `tests/test_query_expansion.py` | 单元测试 | 待创建 |

---

## 十、总结

### 设计原则回顾

| 原则 | 实现方式 |
|------|----------|
| **零依赖核心** | `SynonymTable` 是纯 Python 字典，不需要任何外部依赖 |
| **LLM 增强可选** | `LLMQueryRewriter` 仅在有 `DEEPSEEK_API_KEY` 时激活；无 key 时静默降级 |
| **延迟 < 200ms** | 同义词路径 < 1ms；LLM 异步执行，首次查询不阻塞；缓存命中 < 0.5ms |
| **不修改原始查询** | 扩展结果作为额外召回路径，原始查询始终保留 |
| **渐进增强** | 首次查询 → 同义词（快）；后续查询 → 同义词 + LLM（全） |
| **可观测** | 统计指标 + 日志 + source 标记，便于排查和调优 |

### 与现有架构的兼容性

- **`LLMClient`**：复用 `llm_extractor.LLMClient`，不使用额外 HTTP 客户端
- **`LLMExtractor`**：查询实体补充可复用 `extract_entities_llm()`，也可由 `LLMQueryRewriter` 一并返回
- **`reciprocal_rank_fusion`**：多路径召回结果通过已有的 RRF 融合
- **配置系统**：遵循 `config.py` 的 dataclass 风格和 YAML 加载机制
- **无新依赖**：`query_expansion.py` 的零依赖模式完全不引入新包；LLM 模式复用已有的 `urllib` 客户端

### 预计代码量

- `query_expansion.py`：~350 行
- `config.py` 改动：~20 行
- `store.py` 改动：~80 行（`__init__` + `recall` 集成点）
- `tests/test_query_expansion.py`：~200 行
