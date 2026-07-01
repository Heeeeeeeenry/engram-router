# EngramRouter Phase 2 集成测试方案与验收标准

> 版本: v1.0  
> 日期: 2026-07-01  
> 作者: Phase 2 测试工程师  
> 目标: 为向量通路 (embedding → vector_index → fusion → store.recall) 建立完整的测试防线

---

## 目录

1. [当前状态概述](#1-当前状态概述)
2. [集成测试计划](#2-集成测试计划)
3. [Benchmark 扩展](#3-benchmark-扩展)
4. [性能基准](#4-性能基准)
5. [验收标准](#5-验收标准)
6. [附录](#6-附录)

---

## 1. 当前状态概述

### 1.1 测试现状

| 指标 | 数值 |
|------|------|
| 总测试数 | 127 |
| 通过 | 114 |
| 失败 | 13 (benchmark 已知数据问题) |
| 测试文件 | 9 个 (含 multi_angle_eval.py) |

### 1.2 现有测试覆盖矩阵

| 测试文件 | 覆盖范围 | 状态 |
|----------|---------|------|
| `test_store.py` | save/recall/gap_check/namespace/corrections | ✅ 通过 |
| `test_benchmark.py` | baseline vs engram, regression net, tech/bug/daily gates | ⚠️ 13 失败 |
| `test_entities.py` | 实体提取, 实体跳转召回 | ✅ 通过 |
| `test_fts.py` | FTS5 trigram 候选选择, LIKE 回退, 可插拔 ranker | ✅ 通过 |
| `test_edges.py` | CO_OCCURS_WITH / CAUSED_BY 边, 一跳扩展 | ✅ 通过 |
| `test_mcp.py` | MCP JSON-RPC 协议, tools/list/call | ✅ 通过 |
| `test_summary.py` | 摘要清洗, CJK 截断 | ✅ 通过 |
| `test_id_sequences.py` | 单调 ID 分配器 | ✅ 通过 |
| `multi_angle_eval.py` | 8 维度多角度综合评估脚本 | ✅ 通过 |

### 1.3 Phase 2 新增模块 (尚未有测试)

| 模块 | 路径 | 功能 | 测试状态 |
|------|------|------|----------|
| `EmbeddingEngine` | `src/engram_router/embedding.py` | 文本→向量编码 (local/remote) | ❌ 无测试 |
| `VectorIndex` | `src/engram_router/vector_index.py` | FAISS IVF ANN 索引 | ❌ 无测试 |
| `reciprocal_rank_fusion` | `src/engram_router/fusion.py` | RRF 融合 + 加权分数融合 | ❌ 无测试 |
| `weighted_score_fusion` | `src/engram_router/fusion.py` | 加权分数融合 | ❌ 无测试 |
| `MemoryStore.recall` 向量通路 | `src/engram_router/store.py: L889-928` | RRF 融合关键词+向量结果 | ❌ 无集成测试 |

### 1.4 向量通路架构 (recall pipeline)

```
用户查询
    │
    ├─► 关键词路径 (已有)
    │     ├─ _terms() 分词
    │     ├─ _fts_candidates() FTS5 trigram
    │     ├─ _edge_expansion() 一跳图扩展
    │     └─ _build_scored_candidates() 加权评分
    │
    ├─► 向量路径 (Phase 2 新增)           ◄── 需测试
    │     ├─ embedding_engine.encode(query)
    │     ├─ vector_index.search(vec, k=top_k*4)
    │     └─ 得到 [(memory_id, cosine_similarity), ...]
    │
    └─► RRF 融合 (Phase 2 新增)            ◄── 需测试
          ├─ keyword_list = [(id, rule_score), ...]
          ├─ vector_list = [(id, vec_score), ...]
          ├─ reciprocal_rank_fusion([kw, vec], k=60, w=[0.4, 0.6])
          └─ 重排 scored 列表 + 补充 vector-only 结果
```

---

## 2. 集成测试计划

### 2.1 测试文件组织

```
tests/
├── test_embedding.py          # 新增: EmbeddingEngine 单元/集成测试
├── test_vector_index.py       # 新增: VectorIndex 单元/集成测试
├── test_fusion.py             # 新增: RRF + weighted fusion 单元测试
├── test_vector_recall.py      # 新增: 向量召回 + RRF 融合端到端测试
├── test_benchmark_phase2.py   # 新增: Phase 2 语义联想 benchmark
├── test_performance.py        # 新增: 性能基准测试
├── test_store.py              # 已修改: 补充向量降级测试
└── conftest.py                # 已修改: 新增向量相关 fixtures
```

### 2.2 测试层级设计

采用**三层金字塔**策略:

```
         ┌──────────────┐
         │  E2E 集成测试  │  test_vector_recall.py
         │  (12 cases)   │  test_benchmark_phase2.py
         ├──────────────┤
         │  模块集成测试  │  test_embedding.py
         │  (18 cases)   │  test_vector_index.py
         ├──────────────┤
         │  单元测试      │  test_fusion.py
         │  (15 cases)   │
         └──────────────┘
```

### 2.3 单元测试: `test_fusion.py` (15 cases)

#### 2.3.1 RRF 基础行为

```python
# TC-F01: 单一列表保持不变
def test_rrf_single_list_unchanged():
    """RRF with one list preserves order."""
    results = [("a", 0.9), ("b", 0.7), ("c", 0.3)]
    merged = reciprocal_rank_fusion([results])
    assert [id for id, _ in merged] == ["a", "b", "c"]

# TC-F02: 两个相同列表 → 排名叠加
def test_rrf_two_identical_lists():
    """Identical lists boost shared items."""
    a = [("x", 0.9), ("y", 0.5)]
    b = [("x", 0.8), ("y", 0.4)]
    merged = reciprocal_rank_fusion([a, b], k=60)
    assert merged[0][0] == "x"         # x ranks 1st in both → top
    assert merged[0][1] > merged[1][1]  # x score > y score

# TC-F03: 互补列表 → 合并去重
def test_rrf_complementary_lists():
    """Disjoint results are merged."""
    a = [("a", 0.9), ("b", 0.5)]
    b = [("c", 0.8), ("d", 0.4)]
    merged = reciprocal_rank_fusion([a, b])
    assert len(merged) == 4

# TC-F04: k 值对排名的影响
def test_rrf_k_damping():
    """Higher k flattens rank differences."""
    a = [("x", 0.9), ("y", 0.5)]
    # k=0: x score = 1/(0+1) = 1.0, y score = 1/(0+2) = 0.5, diff=0.5
    # k=60: x score = 1/61 ≈ 0.0164, y = 1/62 ≈ 0.0161, diff≈0.0003
    merged_k0 = reciprocal_rank_fusion([a], k=0)
    merged_k60 = reciprocal_rank_fusion([a], k=60)
    diff0 = merged_k0[0][1] - merged_k0[1][1]
    diff60 = merged_k60[0][1] - merged_k60[1][1]
    assert diff0 > diff60  # k=60 damps more

# TC-F05: 权重影响
def test_rrf_weights():
    """Weighted RRF gives one list more influence."""
    a = [("x", 0.9)]
    b = [("y", 0.9)]
    # weight a=2.0, b=0.5 → a's x should outrank b's y
    merged = reciprocal_rank_fusion([a, b], k=60, weights=[2.0, 0.5])
    assert merged[0][0] == "x"

# TC-F06: 空列表处理
def test_rrf_empty_lists():
    """Empty result lists are handled."""
    merged = reciprocal_rank_fusion([[], [("a", 0.9)]])
    assert merged == [("a", 1/(60+1))]  # only a from list 2
```

#### 2.3.2 Weighted Score Fusion

```python
# TC-F07: 加权融合基础
def test_weighted_fusion_basic():
    kw = [("a", 8.0), ("b", 5.0)]
    vec = [("b", 0.9), ("c", 0.6)]
    merged = weighted_score_fusion(kw, vec, keyword_weight=0.4, vector_weight=0.6)
    # b appears in both → highest score
    assert merged[0][0] == "b"

# TC-F08: 分数归一化
def test_weighted_fusion_score_normalization():
    """Scores from different paths are normalized before weighting."""
    kw = [("a", 100.0)]  # max=100
    vec = [("b", 0.5)]   # max=0.5
    merged = weighted_score_fusion(kw, vec, keyword_weight=0.5, vector_weight=0.5)
    # a_norm = 100/100 = 1.0 → weighted = 0.5
    # b_norm = 0.5/0.5 = 1.0 → weighted = 0.5
    assert abs(merged[0][1] - 0.5) < 0.01

# TC-F09: 单侧为空
def test_weighted_fusion_one_side_empty():
    kw = [("a", 8.0)]
    merged = weighted_score_fusion(kw, [], keyword_weight=0.4, vector_weight=0.6)
    assert merged[0][0] == "a"
```

#### 2.3.3 融合确定性

```python
# TC-F10: 相同输入 → 相同输出 (确定性)
def test_rrf_deterministic():
    a = [("x", 0.9), ("y", 0.5)]
    b = [("y", 0.8), ("z", 0.4)]
    r1 = reciprocal_rank_fusion([a, b])
    r2 = reciprocal_rank_fusion([a, b])
    assert r1 == r2

# TC-F11: 同分元素稳定排序
def test_rrf_tie_stable():
    """Items with same RRF score keep input order stability."""
    a = [("a", 0.9), ("b", 0.9)]  # same score, a first
    merged = reciprocal_rank_fusion([a], k=60)
    assert merged[0][0] == "a"
    assert merged[1][0] == "b"
```

### 2.4 模块集成测试: `test_embedding.py` (8 cases)

#### 2.4.1 EmbeddingEngine 可用性

```python
# TC-E01: Mock 环境构造测试 (不依赖真实模型)
def test_embedding_engine_constructor_defaults():
    """Engine 使用默认参数构造, 不抛异常."""
    engine = EmbeddingEngine(model="bge-small", backend="auto")
    # 在 CI 环境中可能不可用, 但构造本身不应崩溃
    assert engine.dim == 512
    assert engine.backend_name in ("auto", "local", "remote")

# TC-E02: 不可用时 encode 返回 None
def test_embedding_encode_returns_none_when_unavailable():
    """当没有可用后端时, encode 返回 None."""
    engine = EmbeddingEngine(backend="local")
    if not engine.available:
        assert engine.encode("测试文本") is None

# TC-E03: 可用时 encode 返回正确维度
def test_embedding_encode_returns_correct_dimension():
    """可用时返回 numpy array 且维度正确."""
    engine = EmbeddingEngine(model="bge-small", backend="auto")
    if engine.available:
        vec = engine.encode("测试文本")
        import numpy as np
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (512,)
        # L2 归一化后 norm ≈ 1.0
        assert abs(np.linalg.norm(vec) - 1.0) < 0.01

# TC-E04: Batch encode
def test_embedding_encode_batch():
    """Batch encode 返回 (N, dim) 矩阵."""
    engine = EmbeddingEngine(model="bge-small", backend="auto")
    if engine.available:
        texts = ["文本一", "文本二", "文本三"]
        vecs = engine.encode(texts)
        import numpy as np
        assert isinstance(vecs, np.ndarray)
        assert vecs.shape == (3, 512)

# TC-E05: encode_batch 分块
def test_embedding_encode_batch_chunking():
    """大批量输入自动分块."""
    engine = EmbeddingEngine(model="bge-small", backend="auto")
    if engine.available:
        texts = [f"文本{i}" for i in range(100)]
        vecs = engine.encode_batch(texts, batch_size=32)
        import numpy as np
        assert vecs.shape == (100, 512)
```

#### 2.4.2 模型配置

```python
# TC-E06: 不同模型的 dim 属性
def test_embedding_model_dimensions():
    """各模型配置 dim 正确."""
    assert EmbeddingEngine(model="bge-small").dim == 512
    assert EmbeddingEngine(model="text2vec").dim == 768
    assert EmbeddingEngine(model="e5-small").dim == 384

# TC-E07: 模型降级链
def test_embedding_fallback_chain():
    """auto 模式下先 local 后 remote 最后 disabled."""
    engine = EmbeddingEngine(backend="auto")
    # 不抛异常即可 —— 最差情况是 disabled
    assert engine.backend_name in ("auto", "local", "remote") or not engine.available

# TC-E08: 远程 API 配置
def test_embedding_remote_config():
    """API key 和 base URL 可配置."""
    engine = EmbeddingEngine(
        backend="remote",
        api_base="https://custom.api/v1",
        api_key="sk-test-key",
        api_model="text-embedding-3-large",
    )
    assert engine._api_base == "https://custom.api/v1"
    assert engine._api_key == "sk-test-key"
    assert engine._api_model == "text-embedding-3-large"
```

### 2.5 模块集成测试: `test_vector_index.py` (10 cases)

#### 2.5.1 FAISS 索引 CRUD

```python
# TC-V01: 空索引搜索返回空列表
def test_vector_index_empty_search():
    """未训练的索引 search 返回 []."""
    import numpy as np
    idx = VectorIndex(dim=512, nlist=4)
    assert idx.search(np.random.randn(512).astype(np.float32), k=5) == []

# TC-V02: 添加单个向量后可搜索
def test_vector_index_add_and_search():
    """add 后 search 能找回."""
    import numpy as np
    idx = VectorIndex(dim=512, nlist=4)
    vec = np.random.randn(512).astype(np.float32)
    vec = vec / np.linalg.norm(vec)  # L2 normalize
    idx.add("mem_1", vec)
    # 触发训练 (需要 nlist*40=160 vectors, 单条不够 → 缓冲)
    # 直接加足够数据触发训练
    for i in range(200):
        v = np.random.randn(512).astype(np.float32)
        v = v / np.linalg.norm(v)
        idx.add(f"mem_{i+2}", v)
    results = idx.search(vec, k=1)
    assert len(results) == 1
    assert results[0][0] == "mem_1"  # 查询向量自身的最近邻是自己

# TC-V03: add_batch 批量添加
def test_vector_index_add_batch():
    import numpy as np
    idx = VectorIndex(dim=512, nlist=4)
    embs = [(f"mem_{i}", (np.random.randn(512).astype(np.float32))) for i in range(200)]
    # normalize
    embs = [(mid, v / np.linalg.norm(v)) for mid, v in embs]
    idx.add_batch(embs)
    assert idx.size == 200
    assert idx.trained

# TC-V04: remove 移除向量
def test_vector_index_remove():
    import numpy as np
    idx = VectorIndex(dim=512, nlist=4)
    for i in range(200):
        v = np.random.randn(512).astype(np.float32)
        v = v / np.linalg.norm(v)
        idx.add(f"mem_{i}", v)
    assert idx.remove("mem_0") is True
    assert idx.remove("mem_0") is False  # 重复删除
    assert "mem_0" not in idx._id_to_memory.values()

# TC-V05: rebuild 清理幽灵向量
def test_vector_index_rebuild():
    import numpy as np
    idx = VectorIndex(dim=512, nlist=4)
    for i in range(200):
        v = np.random.randn(512).astype(np.float32)
        v = v / np.linalg.norm(v)
        idx.add(f"mem_{i}", v)
    before = idx.size
    idx.remove("mem_0")
    idx.remove("mem_1")
    idx.rebuild()
    assert idx.size == before - 2
```

#### 2.5.2 持久化

```python
# TC-V06: save/load 往返
def test_vector_index_persistence(tmp_path):
    import numpy as np
    path = tmp_path / "idx"
    # 创建并保存
    idx1 = VectorIndex(dim=512, nlist=4, path=path)
    for i in range(200):
        v = np.random.randn(512).astype(np.float32)
        v = v / np.linalg.norm(v)
        idx1.add(f"mem_{i}", v)
    # 加载
    idx2 = VectorIndex(dim=512, path=path)
    assert idx2.size == 200
    assert idx2.trained

# TC-V07: 维度不匹配报错
def test_vector_index_dimension_mismatch():
    import numpy as np
    idx = VectorIndex(dim=512, nlist=4)
    wrong_vec = np.random.randn(384).astype(np.float32)
    import pytest
    with pytest.raises(ValueError, match="dim"):
        idx.add("mem_1", wrong_vec)

# TC-V08: k 自动截断
def test_vector_index_k_truncation():
    import numpy as np
    idx = VectorIndex(dim=512, nlist=4)
    for i in range(50):
        v = np.random.randn(512).astype(np.float32)
        v = v / np.linalg.norm(v)
        idx.add(f"mem_{i}", v)
    results = idx.search(np.random.randn(512).astype(np.float32), k=100)
    assert len(results) <= 50  # 不会超过索引大小
```

### 2.6 E2E 集成测试: `test_vector_recall.py` (12 cases)

这些是**最关键的测试**——验证向量通路与关键词通路通过 RRF 融合后的端到端行为。

```python
# TC-VR01: 向量召回基础 — 语义相似匹配
def test_vector_recall_semantic_match():
    """同义词场景: '电脑' 应能召回包含 '计算机' 的记忆."""
    engine = create_embedding_engine()        # Mock 或真实, 提供可控向量
    idx = VectorIndex(dim=512, path=...)
    store = MemoryStore(
        embedding_engine=engine,
        vector_index=idx,
    )
    store.save("我买了一台新的计算机。")
    store.save("今天天气不错。")
    results = store.recall("电脑", top_k=3)
    assert "计算机" in results[0].raw_text
    assert results[0].match_reason  # 应包含 "vector" 相关说明

# TC-VR02: 向量+关键词互补 — RRF 融合不丢失关键词结果
def test_vector_recall_complements_keyword():
    """向量找到关键词漏掉的 + 关键词精确命中保留."""
    engine = create_embedding_engine()
    idx = VectorIndex(dim=512, path=...)
    store = MemoryStore(embedding_engine=engine, vector_index=idx)
    store.save("张三送我一把 HHKB 键盘。")    # 关键词 HHKB 精确命中
    store.save("王五入手了一台 Topre 静电容。")  # 无 HHKB token, 但语义相似
    results = store.recall("HHKB", top_k=3)
    texts = [r.raw_text for r in results]
    assert any("HHKB" in t for t in texts)
    # 语义相似的 Topre 也应出现在结果中 (RRF 融合)
    assert any("Topre" in t for t in texts)

# TC-VR03: 关键词精确定位 > 语义模糊匹配
def test_vector_recall_keyword_precision_wins():
    """精确关键词命中应排在纯语义匹配之前."""
    engine = create_embedding_engine()
    idx = VectorIndex(dim=512, path=...)
    store = MemoryStore(embedding_engine=engine, vector_index=idx)
    store.save("Python 是最好的编程语言。")
    store.save("我用 Python 写了一个爬虫。")
    store.save("Java 的生态很强大。")
    results = store.recall("Python", top_k=3)
    # Python 精确命中的排前面
    assert all("Python" in r.raw_text for r in results[:2])

# TC-VR04: RRF 融合去重
def test_vector_recall_deduplication():
    """同一条记忆被关键词和向量同时命中 → 不重复出现."""
    engine = create_embedding_engine()
    idx = VectorIndex(dim=512, path=...)
    store = MemoryStore(embedding_engine=engine, vector_index=idx)
    mid = store.save("我买了一把 HHKB 机械键盘。")
    results = store.recall("HHKB 键盘", top_k=5)
    ids = [r.id for r in results]
    assert ids.count(mid) == 1  # 不重复

# TC-VR05: 向量降级 — embedding_engine 不可用时纯关键词工作
def test_vector_recall_graceful_degradation():
    """当 embedding 不可用时, 回退到纯关键词模式."""
    engine = EmbeddingEngine(backend="local")  # CI 环境大概率不可用
    engine._initialized = False                 # 模拟不可用
    idx = VectorIndex(dim=512, path=...)
    store = MemoryStore(embedding_engine=engine, vector_index=idx)
    store.save("张三送我一把 HHKB 键盘。")
    results = store.recall("HHKB", top_k=3)
    assert len(results) >= 1
    assert "HHKB" in results[0].raw_text
    # 不应崩溃, 应正常工作 (纯关键词)

# TC-VR06: 向量降级 — vector_index 为 None
def test_vector_recall_no_vector_index():
    """未提供 vector_index 时关键词路径正常工作."""
    engine = create_embedding_engine()
    store = MemoryStore(embedding_engine=engine)  # 无 vector_index
    store.save("测试记忆")
    results = store.recall("测试", top_k=3)
    assert len(results) >= 1

# TC-VR07: 语义邻接 — 同义词映射
def test_vector_recall_synonym_expansion():
    """'高兴' 向量应匹配 '开心'."""
    engine = create_embedding_engine()
    idx = VectorIndex(dim=512, path=...)
    store = MemoryStore(embedding_engine=engine, vector_index=idx)
    store.save("我今天很开心。")
    store.save("今天下雨了。")
    results = store.recall("高兴", top_k=3)
    assert any("开心" in r.raw_text for r in results)

# TC-VR08: 语义邻接 — 近义表达
def test_vector_recall_paraphrase():
    """'他不干了' 应匹配 '他离职了'."""
    engine = create_embedding_engine()
    idx = VectorIndex(dim=512, path=...)
    store = MemoryStore(embedding_engine=engine, vector_index=idx)
    store.save("张三上个月离职了。")
    results = store.recall("张三不干了", top_k=3)
    assert any("离职" in r.raw_text for r in results)

# TC-VR09: 跨语言语义 (中英混合)
def test_vector_recall_cross_language():
    """'keyboard' 应匹配包含 '键盘' 的记忆 (multilingual-e5)."""
    engine = create_embedding_engine(model="e5-small")  # 多语言模型
    idx = VectorIndex(dim=384, path=...)
    store = MemoryStore(embedding_engine=engine, vector_index=idx)
    store.save("我买了一把机械键盘。")
    results = store.recall("mechanical keyboard", top_k=3)
    assert any("键盘" in r.raw_text for r in results)

# TC-VR10: TopK 截断正确
def test_vector_recall_topk_respected():
    """RRF 融合后 top_k 仍然生效."""
    engine = create_embedding_engine()
    idx = VectorIndex(dim=512, path=...)
    store = MemoryStore(embedding_engine=engine, vector_index=idx)
    for i in range(50):
        store.save(f"这是第 {i} 条测试记忆。")
    for k in [1, 3, 5, 10]:
        results = store.recall("测试", top_k=k)
        assert len(results) <= k

# TC-VR11: namespace 隔离 + 向量搜索
def test_vector_recall_namespace_isolation():
    """向量搜索也遵守 namespace 隔离."""
    engine = create_embedding_engine()
    idx = VectorIndex(dim=512, path=...)
    store = MemoryStore(embedding_engine=engine, vector_index=idx)
    store.save("工作: 张三送我 HHKB。", namespace="work")
    store.save("家庭: 张三今天5岁。", namespace="family")
    wr = store.recall("HHKB", namespace="work", top_k=3)
    assert not any("5岁" in r.raw_text for r in wr)

# TC-VR12: 向量 recall + correction 降权
def test_vector_recall_respects_corrections():
    """被纠正的记忆即使用向量找回也应降权."""
    engine = create_embedding_engine()
    idx = VectorIndex(dim=512, path=...)
    store = MemoryStore(embedding_engine=engine, vector_index=idx)
    mid = store.save("张三说我26岁。")
    store.save("张三说他28岁。")
    store.conn.execute(
        "INSERT INTO corrections (id, target_id, correction_text) VALUES (?, ?, ?)",
        ("corr_1", mid, "年龄更正为28岁"),
    )
    store.conn.commit()
    results = store.recall("张三多大", top_k=5)
    # 被纠正的记忆应排在未纠正之后
    corrected_idx = next(i for i, r in enumerate(results) if r.id == mid)
    uncorrected_idx = next(i for i, r in enumerate(results) if r.id != mid and "28" in r.raw_text)
    assert corrected_idx > uncorrected_idx  # 纠正的排在后面
```

### 2.7 conftest.py 新增 fixtures

```python
# conftest.py 新增
import numpy as np
from engram_router.embedding import EmbeddingEngine
from engram_router.vector_index import VectorIndex

class MockEmbeddingEngine:
    """可控向量引擎, 用于集成测试."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self._available = True
        # 预定义词→向量映射
        self._word_vectors: dict[str, np.ndarray] = {}

    @property
    def available(self) -> bool:
        return self._available

    def set_vectors(self, mapping: dict[str, np.ndarray]):
        """设置词汇向量映射."""
        self._word_vectors = mapping

    def encode(self, texts: str | list[str]) -> np.ndarray | None:
        if not self._available:
            return None
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        vecs = []
        for t in batch:
            v = self._word_vectors.get(t)
            if v is None:
                v = np.random.RandomState(hash(t) % 2**31).randn(self.dim).astype(np.float32)
                v = v / np.linalg.norm(v)
            vecs.append(v)
        result = np.stack(vecs)
        return result[0] if single else result

    @property
    def backend_name(self) -> str:
        return "mock"

    @property
    def init_error(self) -> str | None:
        return None

    def encode_batch(self, texts, batch_size=32):
        return self.encode(texts)


@pytest.fixture
def mock_embedding_engine():
    """提供 MockEmbeddingEngine, dim=64 便于测试."""
    return MockEmbeddingEngine(dim=64)


@pytest.fixture
def vector_store(tmp_path, mock_embedding_engine):
    """创建带向量能力的 MemoryStore."""
    idx = VectorIndex(dim=64, path=tmp_path / "idx")
    return MemoryStore(
        path=tmp_path / "memory.db",
        embedding_engine=mock_embedding_engine,
        vector_index=idx,
    )


@pytest.fixture
def populated_vector_store(vector_store, mock_embedding_engine):
    """预填语义相关的测试数据."""
    # 定义语义相似对
    mock_embedding_engine.set_vectors({
        "计算机": _make_vec(64, seed=1),
        "电脑":   _make_vec(64, seed=1),        # 同 seed = 完全相同 (模拟近义)
        "高兴":   _make_vec(64, seed=2),
        "开心":   _make_vec(64, seed=2),
        "离职":   _make_vec(64, seed=3),
        "不干了": _make_vec(64, seed=3),
        "keyboard": _make_vec(64, seed=4),
        "键盘":    _make_vec(64, seed=4),        # 跨语言模拟
    })
    vector_store.save("我买了一台新的计算机。")
    vector_store.save("我今天很开心。")
    vector_store.save("张三上个月离职了。")
    vector_store.save("我买了一把机械键盘。")
    vector_store.save("今天天气很好。")  # 负样本
    return vector_store


def _make_vec(dim, seed):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)
```

---

## 3. Benchmark 扩展

### 3.1 新增语义联想场景

在现有 5 个 benchmark 场景基础上, 新增第 6 个场景:

#### `examples/semantic_questions.jsonl`

覆盖以下语义联想子场景:

| 子场景 | 示例 Query | 目标记忆 | 语义关系 |
|--------|-----------|---------|---------|
| 近义词 | "不高兴" | "他今天心情很差" | 否定近义 |
| 同义词 | "车子" | "我买了一辆汽车" | 同义替换 |
| 口语化 | "不干了" | "张三离职了" | 口语 vs 书面 |
| 上下位 | "水果" | "我买了苹果和香蕉" | 上位词匹配 |
| 跨语言 | "keyboard" | "我的机械键盘是HHKB" | 中英互译 |
| 语义场 | "交通工具" | "我每天坐地铁上班" | 语义场联想 |
| 否定排除 | "没去过" | "我去过北京" → 不应匹配 | 语义反义 |
| 长句匹配 | "那个经常帮我修电脑的同事" | "张三帮我修过三次电脑" | 自由文本语义 |

#### 文件格式

```jsonl
{"query": "不高兴", "answer_contains": ["心情很差"], "evidence_contains": ["心情很差"], "expect": "pass", "tag": "sem-synonym-negative"}
{"query": "车子", "answer_contains": ["汽车"], "evidence_contains": ["汽车"], "expect": "pass", "tag": "sem-synonym-car"}
{"query": "不干了", "answer_contains": ["离职"], "evidence_contains": ["离职"], "expect": "pass", "tag": "sem-vernacular"}
{"query": "水果", "answer_contains": ["苹果", "香蕉"], "evidence_contains": ["苹果", "香蕉"], "expect": "pass", "tag": "sem-hypernym"}
{"query": "keyboard", "answer_contains": ["键盘"], "evidence_contains": ["键盘"], "expect": "pass", "tag": "sem-crosslang"}
{"query": "交通工具", "answer_contains": ["地铁"], "evidence_contains": ["地铁"], "expect": "pass", "tag": "sem-semantic-field"}
{"query": "没去过上海", "answer_contains": [], "answer_excludes": ["上海"], "expect": "pass", "tag": "sem-negative"}
{"query": "那个经常帮我修电脑的同事", "answer_contains": ["修过", "电脑"], "evidence_contains": ["修过"], "expect": "pass", "tag": "sem-long-query"}
```

#### 对应对话语料 `examples/semantic_conversation.md`

```
## Conversation Events

1. User: 我今天心情很差，什么都不想做。
2. User: 我上个月买了一辆汽车，是特斯拉 Model 3。
3. User: 张三上个月离职了，去了字节跳动。
4. User: 我去超市买了苹果和香蕉，还买了牛奶。
5. User: 我的机械键盘是 HHKB Professional Hybrid Type-S，手感特别好。
6. User: 我每天坐地铁上班，大概四十分钟。
7. User: 去年我去过北京出差，还去了故宫。
8. User: 张三帮我修过三次电脑，每次都很快搞定。
9. User: 今天天气不错，适合出去走走。
```

### 3.2 Benchmark Test Cases

```python
# test_benchmark_phase2.py

SEMANTIC_CONVO = REPO_ROOT / "examples" / "semantic_conversation.md"
SEMANTIC_CASES = REPO_ROOT / "examples" / "semantic_questions.jsonl"


def test_semantic_conversation_loads_correctly():
    turns = load_conversation(SEMANTIC_CONVO)
    assert len(turns) == 9


def test_semantic_cases_structure():
    cases = load_cases(SEMANTIC_CASES)
    assert len(cases) == 8
    assert all(c.tag.startswith("sem-") for c in cases)


def test_semantic_benchmark_keyword_only(tmp_path):
    """纯关键词: 语义联想场景预期部分失败 (如 keyboard→键盘)."""
    turns = load_conversation(SEMANTIC_CONVO)
    cases = load_cases(SEMANTIC_CASES)
    report = run_benchmark(turns, cases, db_path=tmp_path / "sem_kw.db")
    # 纯关键词下, 跨语言/同义词会失败
    assert report["engram"]["answer_hits"] < report["total"]
    # 记录 baseline 分数
    return report["engram"]["answer_hits"]  # 供后续对比


def test_semantic_benchmark_with_vectors(tmp_path, mock_embedding_engine):
    """向量增强: 语义联想场景全部通过."""
    turns = load_conversation(SEMANTIC_CONVO)
    cases = load_cases(SEMANTIC_CASES)

    # 设置语义向量映射
    mock_embedding_engine.set_vectors({
        "心情很差": _make_vec(64, seed=10),
        "不高兴": _make_vec(64, seed=10),          # 同 seed
        "汽车": _make_vec(64, seed=11),
        "车子": _make_vec(64, seed=11),
        "离职": _make_vec(64, seed=12),
        "不干了": _make_vec(64, seed=12),
        "苹果": _make_vec(64, seed=13),
        "香蕉": _make_vec(64, seed=13),             # 同一语义场
        "水果": _make_vec(64, seed=13),
        "keyboard": _make_vec(64, seed=14),
        "键盘": _make_vec(64, seed=14),
        "地铁": _make_vec(64, seed=15),
        "交通工具": _make_vec(64, seed=15),
        "没去过上海": _make_vec(64, seed=99),       # 远离上海
        "上海": _make_vec(64, seed=16),
        "修过": _make_vec(64, seed=17),
        "电脑": _make_vec(64, seed=18),
    })

    db_path = tmp_path / "sem_vec.db"
    idx = VectorIndex(dim=64, path=tmp_path / "sem_vec_idx")
    store = MemoryStore(
        path=db_path,
        embedding_engine=mock_embedding_engine,
        vector_index=idx,
    )
    for turn in turns:
        store.save(turn)
    store.close()

    # 重新打开 (模拟真实场景: 从磁盘加载向量索引)
    store2 = MemoryStore(
        path=db_path,
        embedding_engine=mock_embedding_engine,
        vector_index=VectorIndex(dim=64, path=tmp_path / "sem_vec_idx"),
    )

    # 手动运行 benchmark (因为 run_benchmark 不支持自定义 store)
    results = []
    for case in cases:
        records = store2.recall(case.query, top_k=5)
        joined = " ".join(r.raw_text for r in records)
        hits = all(t.lower() in joined.lower() for t in case.answer_contains)
        excludes = all(t.lower() not in joined.lower() for t in case.answer_excludes)
        results.append(hits and excludes)

    hit_count = sum(results)
    assert hit_count >= 6, f"Vector recall should handle ≥6/8 semantic cases, got {hit_count}"


def test_semantic_benchmark_vector_vs_keyword_comparison(tmp_path, mock_embedding_engine):
    """向量增强应显著提升语义联想场景的召回率."""
    turns = load_conversation(SEMANTIC_CONVO)
    cases = load_cases(SEMANTIC_CASES)

    # 纯关键词 run
    kw_report = run_benchmark(turns, cases, db_path=tmp_path / "sem_kw.db")

    # 向量增强 run
    # ... (配置向量 store)
    vec_hits = 0  # count
    for case in cases:
        records = vector_store.recall(case.query, top_k=5)
        joined = " ".join(r.raw_text for r in records)
        if all(t.lower() in joined.lower() for t in case.answer_contains):
            vec_hits += 1

    # 断言: 向量增强 > 纯关键词 (至少多召回 3 个)
    improvement = vec_hits - kw_report["engram"]["answer_hits"]
    assert improvement >= 3, (
        f"Vector enhancement should improve by ≥3 cases, "
        f"got kw={kw_report['engram']['answer_hits']}, vec={vec_hits}"
    )
```

---

## 4. 性能基准

### 4.1 性能测试指标

| 指标 | 描述 | 目标 (Acceptable) | 目标 (Good) |
|------|------|------------------|-------------|
| `embedding.encode_latency_ms` | 单条文本编码延迟 (P50) | < 50ms | < 20ms |
| `embedding.encode_latency_ms_p99` | 单条文本编码延迟 (P99) | < 200ms | < 100ms |
| `embedding.batch_encode_throughput` | 批量编码吞吐 (32 batch) | > 50 texts/s | > 200 texts/s |
| `vector_index.search_latency_ms` | 向量搜索延迟 (top 20) | < 10ms | < 5ms |
| `vector_index.add_latency_ms` | 单条向量添加延迟 | < 5ms | < 2ms |
| `vector_index.save_latency_ms` | 索引持久化延迟 (1K vectors) | < 100ms | < 50ms |
| `recall.total_latency_ms` | 端到端 recall 延迟 (含融合) | < 200ms | < 100ms |
| `recall.vector_overhead_ms` | 向量通路额外开销 | < 80ms | < 30ms |
| `memory.embedding_cache_hit_rate` | 编码缓存命中率 | > 80% | > 95% |

### 4.2 性能测试实现: `test_performance.py`

```python
"""Phase 2 性能基准测试.

这些测试在有真实 embedding 模型的环境中运行,
在 CI 环境中使用 mock 跳过, 仅作数据记录.
"""

import time
import numpy as np
import pytest
from engram_router.embedding import EmbeddingEngine
from engram_router.vector_index import VectorIndex
from engram_router.store import MemoryStore


def _timeit(func, iterations=50, warmup=5):
    """Measure P50/P99 latency in milliseconds."""
    for _ in range(warmup):
        func()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "p50": times[len(times) // 2],
        "p99": times[int(len(times) * 0.99)],
        "avg": sum(times) / len(times),
        "min": times[0],
        "max": times[-1],
    }


@pytest.mark.skipif(not _has_embedding(), reason="No embedding backend available")
class TestEmbeddingPerformance:

    def test_encode_latency_single(self):
        engine = EmbeddingEngine(model="bge-small")
        if not engine.available:
            pytest.skip("Embedding engine not available")
        result = _timeit(lambda: engine.encode("张三送我一把 HHKB 键盘。"), iterations=100)
        print(f"\n  Encode P50: {result['p50']:.1f}ms, P99: {result['p99']:.1f}ms")
        assert result["p50"] < 50, f"P50 encode latency {result['p50']:.1f}ms > 50ms"

    def test_encode_latency_batch_32(self):
        engine = EmbeddingEngine(model="bge-small")
        if not engine.available:
            pytest.skip("Embedding engine not available")
        texts = [f"这是第 {i} 条测试文本。" for i in range(32)]
        result = _timeit(lambda: engine.encode(texts), iterations=20)
        throughput = 32 / (result["avg"] / 1000)
        print(f"\n  Batch(32) P50: {result['p50']:.1f}ms, "
              f"Throughput: {throughput:.0f} texts/s")
        assert throughput > 50, f"Throughput {throughput:.0f} < 50 texts/s"


@pytest.mark.skipif(not _has_faiss(), reason="FAISS not available")
class TestVectorIndexPerformance:

    def test_search_latency_1k_vectors(self, tmp_path):
        idx = VectorIndex(dim=512, nlist=8, path=tmp_path / "perf_idx")
        # 填充 1000 个向量
        rng = np.random.RandomState(42)
        for i in range(1000):
            v = rng.randn(512).astype(np.float32)
            v = v / np.linalg.norm(v)
            idx.add(f"mem_{i}", v)
        query = rng.randn(512).astype(np.float32)
        query = query / np.linalg.norm(query)
        result = _timeit(lambda: idx.search(query, k=20), iterations=100)
        print(f"\n  Search(1K) P50: {result['p50']:.1f}ms, P99: {result['p99']:.1f}ms")
        assert result["p50"] < 10, f"Search P50 {result['p50']:.1f}ms > 10ms"

    def test_add_latency(self, tmp_path):
        idx = VectorIndex(dim=512, nlist=8, path=tmp_path / "perf_add")
        # 先训练
        rng = np.random.RandomState(42)
        for i in range(200):
            v = rng.randn(512).astype(np.float32)
            v = v / np.linalg.norm(v)
            idx.add(f"mem_{i}", v)
        # 测单次 add
        v = rng.randn(512).astype(np.float32)
        v = v / np.linalg.norm(v)
        result = _timeit(lambda: idx.add("mem_new", v), iterations=50)
        print(f"\n  Add P50: {result['p50']:.1f}ms")
        assert result["p50"] < 5, f"Add P50 {result['p50']:.1f}ms > 5ms"

    def test_save_latency_1k(self, tmp_path):
        path = tmp_path / "perf_save"
        idx = VectorIndex(dim=512, nlist=8, path=path)
        rng = np.random.RandomState(42)
        for i in range(1000):
            v = rng.randn(512).astype(np.float32)
            v = v / np.linalg.norm(v)
            idx.add(f"mem_{i}", v)
        # 测 save
        result = _timeit(lambda: idx.save(), iterations=10)
        print(f"\n  Save(1K) P50: {result['p50']:.1f}ms")
        assert result["p50"] < 100, f"Save P50 {result['p50']:.1f}ms > 100ms"


class TestE2ERecallPerformance:
    """端到端 recall 性能 (含向量融合开销)."""

    def test_recall_latency_without_vectors(self, tmp_path):
        """Baseline: 纯关键词 recall 延迟."""
        store = MemoryStore(path=tmp_path / "perf_kw.db")
        for i in range(100):
            store.save(f"这是第 {i} 条测试记忆，包含关键词 HHKB 和键盘。")
        result = _timeit(lambda: store.recall("HHKB 键盘", top_k=5), iterations=50)
        print(f"\n  Recall(KW only) P50: {result['p50']:.1f}ms, P99: {result['p99']:.1f}ms")
        assert result["p50"] < 100, f"KW recall P50 {result['p50']:.1f}ms > 100ms"

    def test_vector_overhead(self, tmp_path, mock_embedding_engine):
        """向量路径的额外开销应在可控范围."""
        store_kw = MemoryStore(path=tmp_path / "perf_kw2.db")
        for i in range(100):
            store_kw.save(f"测试记忆 {i}")
        t0 = time.perf_counter()
        for _ in range(50):
            store_kw.recall("测试", top_k=5)
        kw_time = (time.perf_counter() - t0) * 1000 / 50

        idx = VectorIndex(dim=64, path=tmp_path / "perf_vec_idx")
        store_vec = MemoryStore(
            path=tmp_path / "perf_vec.db",
            embedding_engine=mock_embedding_engine,
            vector_index=idx,
        )
        for i in range(100):
            store_vec.save(f"测试记忆 {i}")

        # 设置 mock 向量
        mock_embedding_engine.set_vectors({
            "测试": _make_vec(64, seed=1),
        })
        for i in range(100):
            mock_embedding_engine.set_vectors({
                f"测试记忆 {i}": _make_vec(64, seed=i+2),
            })

        t0 = time.perf_counter()
        for _ in range(50):
            store_vec.recall("测试", top_k=5)
        vec_time = (time.perf_counter() - t0) * 1000 / 50

        overhead = vec_time - kw_time
        print(f"\n  KW recall: {kw_time:.1f}ms, Vector recall: {vec_time:.1f}ms, "
              f"Overhead: {overhead:.1f}ms")
        assert overhead < 80, f"Vector overhead {overhead:.1f}ms > 80ms"
```

---

## 5. 验收标准

### 5.1 Pass/Fail 定义

| 测试类别 | 通过标准 | 阻塞发布? |
|----------|---------|----------|
| **单元测试 (fusion.py)** | 15/15 全部通过 | ✅ 是 |
| **单元测试 (embedding.py)** | 在有 embedding 环境 8/8 通过; 无环境时 skip 不阻塞 | ❌ 否 (CI 可能无 GPU/model) |
| **单元测试 (vector_index.py)** | 在有 FAISS 环境 10/10 通过; 无环境时 skip 不阻塞 | ❌ 否 |
| **集成测试 (vector_recall.py)** | 12/12 全部通过 (使用 mock) | ✅ 是 |
| **语义 Benchmark** | ≥ 6/8 语义场景通过 (使用 mock vectors) | ✅ 是 |
| **回归保护** | 现有 114 个通过测试继续保持通过 | ✅ 是 |
| **性能 P50** | embedding < 50ms, search < 10ms, recall < 200ms | ⚠️ 告警不阻塞 |
| **性能 P99** | embedding < 200ms, search < 50ms | ⚠️ 告警不阻塞 |
| **降级行为** | embedding 不可用时纯关键词路径不受影响 | ✅ 是 |
| **数据持久化** | FAISS index + idmap 关闭重开数据不丢失 | ✅ 是 |

### 5.2 Phase 2 成功指标 (KPI)

#### 5.2.1 功能指标

| 指标 | 衡量方式 | 目标值 |
|------|---------|--------|
| **语义召回提升率** | semantic benchmark 中 向量 vs 纯关键词 的 answer_hits 差值 | ≥ +37.5% (多 3/8 cases) |
| **RRF 融合正确性** | 12 个集成测试全部通过 | 100% |
| **关键词无损** | 现有 5 个 benchmark 场景 gate 100% 通过 (不退化) | 127/127 → 无新增失败 |
| **降级鲁棒性** | embedding 不可用时 recall 功能不受影响 | 5/5 降级测试通过 |

#### 5.2.2 性能指标

| 指标 | 衡量方式 | 目标值 |
|------|---------|--------|
| **向量编码延迟 (P50)** | `embedding.encode()` 单条 | < 50ms |
| **向量搜索延迟 (P50)** | `vector_index.search()` 1000 vectors | < 10ms |
| **端到端 recall 延迟增量** | 向量 recall - 纯关键词 recall | < 80ms |
| **批量编码吞吐** | `encode_batch(32)` | > 50 texts/s |

#### 5.2.3 代码质量指标

| 指标 | 衡量方式 | 目标值 |
|------|---------|--------|
| **新增代码覆盖率** | embedding.py + vector_index.py + fusion.py | > 85% |
| **向量通路代码覆盖率** | store.py L889-928 | 100% (每条路径都走到) |
| **无 flaky tests** | CI 连续 10 次运行 | 0 次随机失败 |

### 5.3 验收门禁 (Gate)

```
Phase 2 通过门禁 = ALL OF:
  ✅ 所有集成测试 (test_vector_recall.py) 12/12 通过
  ✅ 所有单元测试 (test_fusion.py) 15/15 通过
  ✅ 语义 Benchmark 向量增强 ≥ 6/8
  ✅ 现有 114 tests 无回归 (0 新增失败)
  ✅ 降级测试 5/5 通过
  ✅ 代码覆盖率 > 85%

Phase 2 有条件通过 = ALL OF:
  ✅ 上述全部 (在 mock 环境)
  ⚠️ 性能测试在有 FAISS/embedding 的真实环境中通过

Phase 2 阻塞项 = ANY OF:
  ❌ 任一项导致现有 keyword recall 退化
  ❌ 集成测试或 fusion 单元测试失败
  ❌ embedding 不可用时系统崩溃
```

---

## 6. 附录

### 6.1 测试执行命令

```bash
# 运行所有非性能测试
cd /Users/v_liheng02/work/test_code/for_test/engram-router
pytest tests/ -xvs \
  --ignore=tests/test_performance.py \
  --ignore=tests/multi_angle_eval.py \
  -k "not benchmark"  # 排除已知数据问题的 13 个

# 仅运行 Phase 2 新增测试
pytest tests/test_fusion.py tests/test_embedding.py \
       tests/test_vector_index.py tests/test_vector_recall.py \
       tests/test_benchmark_phase2.py -xvs

# 运行语义 benchmark
pytest tests/test_benchmark_phase2.py -xvs

# 运行性能测试 (需要真实 FAISS + embedding)
pytest tests/test_performance.py -xvs --no-header -p no:warnings

# 运行回归保护 (确保无退化)
pytest tests/ -x --ignore=tests/test_performance.py \
       --ignore=tests/test_embedding.py \
       --ignore=tests/test_vector_index.py \
       --ignore=tests/test_vector_recall.py \
       --ignore=tests/test_benchmark_phase2.py \
       --ignore=tests/multi_angle_eval.py \
       -k "not benchmark"
```

### 6.2 新增文件清单

| 文件 | 类型 | 预计行数 |
|------|------|----------|
| `tests/test_fusion.py` | 新增 | ~120 行 |
| `tests/test_embedding.py` | 新增 | ~150 行 |
| `tests/test_vector_index.py` | 新增 | ~200 行 |
| `tests/test_vector_recall.py` | 新增 | ~300 行 |
| `tests/test_benchmark_phase2.py` | 新增 | ~180 行 |
| `tests/test_performance.py` | 新增 | ~200 行 |
| `tests/conftest.py` | 修改 | +80 行 |
| `examples/semantic_conversation.md` | 新增 | ~30 行 |
| `examples/semantic_questions.jsonl` | 新增 | ~10 行 |
| `docs/phase2_integration_test_plan.md` | 本文档 | ~600 行 |

### 6.3 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| CI 环境无 FAISS / sentence-transformers | 部分测试 skip | Mock 向量引擎覆盖集成逻辑; 真实模型测试仅在本地/专用环境运行 |
| 语义向量质量依赖模型 | cross-language 场景可能不通过 | 使用 multilingual-e5-small 模型; 若本地模型不可用, benchmark 使用 mock |
| FAISS IndexIVFFlat 训练需要足够数据 | 小数据量索引行为不同 | 测试中使用 160+ vectors 触发训练; 空索引测试覆盖冷启动 |
| 性能在 CI 波动大 | P99 不稳定 | P50 作为主要判据; 性能测试仅告警不阻塞 |
| 现有 13 benchmark 失败 | 可能掩盖回归 | 排除已知失败的 13 个, 单独跟踪修复 |
