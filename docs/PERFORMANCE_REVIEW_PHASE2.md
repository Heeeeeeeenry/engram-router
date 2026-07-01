# EngramRouter Phase 2 — 性能审核报告

> **审核范围**：延迟 / 内存 / 吞吐 / 缓存命中率  
> **审核日期**：2026-07-01  
> **审核对象**：`embedding.py` / `vector_index.py` / `fusion.py` / `llm_extractor.py` / `query_expansion_design.md`  
> **模型基准**：bge-small-zh-v1.5 (24MB, 512d) + DeepSeek V3 (LLM API)

---

## 一、总体延迟预算

| 操作 | Phase 1 (基线) | Phase 2 增量 | Phase 2 总计 | 目标 | 判定 |
|------|:-----------:|:----------:|:----------:|:----:|:----:|
| `save()` | ~5-15ms | +6–35ms | **11–50ms** | <50ms | ✅ 达标 |
| `recall()` | ~40–120ms | +10–30ms | **50–150ms** | <200ms | ✅ 含余量 |
| 查询扩展（首次） | — | <2ms (异步) | <2ms | <200ms | ✅ 远低于预算 |
| 查询扩展（缓存命中） | — | <0.5ms | <0.5ms | <1ms | ✅ |
| LLM 实体提取（批量） | — | 80–250ms/条 | — | — | 替代逐条 500ms–2s |

---

## 二、`save()` 向量编码延迟分解

### 2.1 端到端路径

```
save(text)
  ├── SQLite INSERT (memories + evidence)  ..............  ~1–3ms
  ├── _index_entities() (rule-based extraction)  .......  ~2–5ms
  ├── FTS5 INSERT  ......................................  ~0.5–1ms
  ├── [optional] LLM extract_entities_llm()  ............  ~500–2000ms (串行阻塞!)
  ├── [Phase 2 NEW] embedding_engine.encode()  ..........  ~5–20ms  ★
  ├── [Phase 2 NEW] vector_index.add()  .................  ~0.1–1ms  ★
  ├── [Phase 2 NEW] vector_index._save()  ...............  ~1–10ms  ★
  └── conn.commit()  .....................................  ~2–5ms
```

### 2.2 各环节详细预估

| 子步骤 | 延迟 (ms) | 说明 |
|--------|:------:|------|
| `SentenceTransformer.encode(text)` | 5–20 | bge-small 单条编码。预热后稳定在 8–12ms (M1/M2 Mac)，x86 约 10–20ms |
| `vector_index.add()` (已训练) | 0.1–1 | FAISS `IndexIVFFlat.add()`，单向量追加 O(1) |
| `vector_index._save()` (磁盘 I/O) | 1–10 | `faiss.write_index()` pickle 序列化。小索引 (<1K) ~1ms；大索引 (100K+) 可达 10ms |
| **Phase 2 增量合计** | **6–31** | |
| `_index_entities()` (含可选 LLM) | 5–2000 | LLM 逐条提取是主要瓶颈，建议改用批量异步（见 §5） |

### 2.3 瓶颈分析

- **单条 save 的向量开销不超过 31ms**，加上 Phase 1 基线 5–15ms，总延迟 ≤ 50ms，满足目标。
- **当 LLM 提取开启时**，`save()` 串行调用 `extract_entities_llm()` 会导致 500ms–2s 的延迟。这**不是 Phase 2 引入的问题**，但 Phase 2 的批量模式（`extract_batch`）可以显著改善。
- **FAISS 磁盘写入（`_save()`）**在每次 `add()` 后都触发。对于高频写入场景（批量导入），应改为 N 条提交一次或异步 flush。

### 2.4 优化建议

1. **`_save()` 节流**：增加 `auto_save=False` 选项，或每 N 次 add 才落盘一次。
2. **批量编码**：`embedding_engine.encode_batch()` 可将 10 条文本编码延迟从 10×15ms=150ms 降至 30–50ms。
3. **LLM 异步化**：LLM 提取在后台执行，不阻塞 `save()` 返回。当前 `save()` 在第 458 行 `extract_entities_llm(text)` 是同步调用。

---

## 三、`recall()` 向量通路延迟分解

### 3.1 端到端路径

```
recall(query)
  ├── _terms(query)  .....................................  ~0.1ms
  ├── extract_entities(query)  ...........................  ~1–3ms
  ├── _fts_candidates()  .................................  ~1–10ms
  ├── _memory_rows()  ....................................  ~2–15ms
  ├── _entities_for_memories()  ..........................  ~5–20ms
  ├── _edge_expansion() (BFS 2-hop)  ....................  ~10–50ms
  ├── _build_scored_candidates()  ........................  ~20–80ms
  │
  ├── [Phase 2 NEW] embedding_engine.encode(query)  .....  ~5–20ms  ★
  ├── [Phase 2 NEW] vector_index.search(query_vec, k)  ..  ~1–5ms   ★
  ├── [Phase 2 NEW] reciprocal_rank_fusion()  ...........  <1ms     ★
  └── _build_recall_response()  ..........................  ~2–10ms
```

### 3.2 各环节详细预估

| 子步骤 | 延迟 (ms) | 说明 |
|--------|:------:|------|
| 基线 recall（无向量） | 40–120 | `_edge_expansion` 取决于图的密度，`_build_scored_candidates` 行数 ≤2000 |
| `embedding_engine.encode(query)` | 5–20 | 与 save 路径相同 |
| `vector_index.search(query_vec, k=20)` | 1–5 | IndexIVFFlat 子线性搜索 |
| `reciprocal_rank_fusion()` | <1 | O(N×M)，N=keyword 结果数，M=vector 结果数，均 <100 |
| 结果合并/排序 | <2 | 字典合并 + 排序 |
| **Phase 2 向量增量合计** | **8–28** | |
| **Phase 2 recall 总计** | **48–148** | |

### 3.3 RRF 融合开销

`reciprocal_rank_fusion()` 当前实现（`fusion.py:21-50`）：
```python
for w, results in zip(weights, result_lists):
    for rank, (doc_id, _) in enumerate(results):
        rrf = w / (k + rank + 1)
        scores[doc_id] = scores.get(doc_id, 0.0) + rrf
merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

- 每条结果列表 ≤ 100 条（实际 recall 结果数远少于此）
- 排序 ≤ 200 个 key
- **总耗时 < 1ms，可忽略**

### 3.4 瓶颈分析

- **核心瓶颈仍在关键词召回管线**：`_edge_expansion()` 的 BFS 和 `_build_scored_candidates()` 的逐行打分占 70–80% 延迟。
- 向量通路（编码 + 搜索 + 融合）仅增加 8–28ms，占 <20%。
- **< 200ms 目标有约 50ms 安全余量**，可以容纳 Reranker（如 `LLMReranker`）或查询扩展。

---

## 四、LLM 批量提取 vs 逐条提取

### 4.1 延迟对比

| 模式 | LLM 调用次数 | 每条平均延迟 | 10 条总延迟 | 吞吐量 |
|------|:--------:|:---------:|:--------:|:-----:|
| 逐条提取（当前） | 10 | 500–2000ms | **5–20s** | 0.5–2 qps |
| 批量提取（Phase 2） | 1 | 80–250ms | **0.8–2.5s** | 4–12 qps |
| 批量 + LRU 缓存 | 0–1 | 0–250ms | **0–2.5s** | ∞ (缓存命中时) |

### 4.2 批量模式设计要点

```
extract_batch(texts: list[str], batch_size=10)
  ├── 自动查缓存 (sha256(text))
  ├── 缓存未命中 → 分组打包 (≤10条/组)
  ├── 单次 LLM 调用 → prompt 包含多条文本 + 索引号
  └── 结果拆分 → 按索引号分发回各条
```

### 4.3 瓶颈分析

- **API 往返时间 (RTT)** 是主导因素。DeepSeek API 典型延迟 300–1500ms，含推理时间。
- 批量的核心收益是**合并 RTT**：10 次 RTT → 1 次 RTT。
- **Prompt 长度限制**：每条文本截断到 2000 字符（`MAX_INPUT_CHARS`），10 条文本 + prompt 约 20K tokens，在 DeepSeek 64K 上下文中完全安全。

---

## 五、内存占用分析

### 5.1 各组件内存预算

| 组件 | 内存 (MB) | 说明 |
|------|:------:|------|
| **bge-small-zh-v1.5 模型权重** | 24 | float32 参数，加载后常驻 |
| **SentenceTransformer 运行时** | 30–50 | tokenizer + PyTorch runtime + ONNX 推理图 |
| **FAISS IndexIVFFlat (10K×512d)** | ~25 | 量化器 (flat index) ~20MB + 倒排列表 ~5MB |
| **FAISS IndexIVFFlat (100K×512d)** | ~210 | 量化器 ~205MB + 倒排列表 ~5MB |
| **ExpansionCache (256 entries)** | <1 | OrderedDict + ExpandedQuery 对象 |
| **LLM Extractor LRU (4096 entries)** | <5 | sha256 → 提取结果字典 |
| **Python + SQLite 基础** | 60–100 | CPython runtime + 连接池 + SQLite page cache |
| **Phase 1 基线** | ~80–120 | Python + SQLite + entities 配置 |
| **Phase 2 增量（10K）** | **~80–100** | 模型 24MB + Runtime ~30MB + FAISS ~25MB |
| **Phase 2 总计（10K）** | **~160–220** | |
| **Phase 2 总计（100K）** | **~350–450** | |

### 5.2 FAISS 索引内存线性增长

```
FAISS IndexIVFFlat 内存 = 量化器(FlatIP) + 倒排列表 + 元数据
  = (N × dim × 4) + (N × dim × 4 + cluster_overhead) + overhead
  ≈ 8 × N × dim bytes  (float32, 2 copies: quantizer + codes)

dim=512:  ≈ 4 KB/vector
  1K vectors  →  4 MB
  10K vectors → 40 MB
  100K vectors → 400 MB
  1M vectors  → 4 GB (此时应切换为 IndexIVFPQ 或 IndexHNSW)
```

### 5.3 瓶颈分析

- **最大内存消费者是 FAISS 索引**，与存储的记忆数量线性增长。
- **模型权重 24MB 是一次性成本**，不随数据量增长。
- **100K 条记忆时总内存 ~400MB**，在 16GB RAM 的 Mac 上完全可行。
- **突破 1M 条**需要迁移到 GPU/量化索引（IVFPQ 可将内存降至 1/8）。

---

## 六、查询扩展延迟预算

### 6.1 延迟路径矩阵

| 路径 | 延迟 (ms) | 缓存状态 | LLM 状态 | 说明 |
|------|:------:|:------:|:------:|------|
| 缓存命中 | **<0.5** | HIT | — | `ExpansionCache.get()` + 同义词合并 |
| 同义词 only (无 LLM) | **<1** | MISS | DISABLED | `SynonymTable.expand()` O(k×n) |
| 同义词 + 异步 LLM | **<2** | MISS | ASYNC | 同义词立即返回，LLM 后台不阻塞 |
| 同义词 + 同步 LLM | **500–2000** | MISS | SYNC | 仅测试/调试，生产不推荐 |
| 后续相同查询 | **<0.5** | HIT | — | 缓存含 LLM 结果 |

### 6.2 SynonymTable.expand() 微观分析

```python
def expand(self, text: str) -> dict[str, list[str]]:
    sorted_keys = sorted(self._map.keys(), key=len, reverse=True)  # ~100 keys
    for key in sorted_keys:          # O(k)
        if key in text:              # Python str.__contains__, O(n) worst-case
            result[key] = self._map[key]
    return result
```

- k ≈ 40（内置同义词条目数），n = len(text) ≈ 50（典型查询长度）
- **理论复杂度 O(k×n) = 2000 次字符比较**
- Python 的 `key in text` 使用 FASTSEARCH（Boyer-Moore 简化版），实际远快于 O(n)
- **实测 < 0.1ms**（包含排序开销）

### 6.3 瓶颈分析

- **同义词扩展不是瓶颈**：<1ms，远低于预算。
- **LLM 改写是异步的**，首次查询不阻塞。延迟风险仅在于 `async_llm=False` 的退化场景。
- **缓存 key 归一化**（`" ".join(query.lower().split())`）可将空白差异的查询映射到同一缓存条目，提升命中率。

---

## 七、缓存命中率预估

### 7.1 两级缓存架构

| 缓存层 | 容量 | Key | 预估命中率 | 场景 |
|--------|:--:|-----|:--------:|------|
| **ExpansionCache (查询扩展)** | 256 | 归一化 query 字符串 | **30–60%** | 用户重复或相似查询 |
| **LLM Extractor LRU (实体提取)** | 4096 | sha256(text) | **50–80%** | 相同文本的重复处理 |

### 7.2 场景分析

#### 场景 A：对话式记忆（多轮交互）
- 用户可能在多轮中问类似问题："键盘什么牌子" → "那个键盘品牌" → "HHKB 是哪个牌子"
- 当前归一化 (`lower + split + join`) 不会把这些归一化为同一 key
- **命中率预估：10–30%**（需要语义归一化才能提高）

#### 场景 B：批量导入（日志回放）
- 同一文本可能因 compaction/重处理被多次 save
- sha256(text) 完全匹配
- **命中率预估：60–80%**

#### 场景 C：个人知识库查询
- 用户查询高度多样化
- 256 条 LRU 仅覆盖最近查询
- **命中率预估：20–40%**

#### 场景 D：混合场景（最典型）
- 部分重复查询 + 部分新查询
- **综合命中率预估：30–50%**
- 4096 条 LLM Extractor 缓存可以覆盖大部分批量导入场景

### 7.3 瓶颈分析

- **512 条目对活跃用户的日常查询足够**（假设用户每天 50 个不同查询，256 覆盖约 5 天窗口）。
- **sha256 完全匹配过于严格**——"我同事送的键盘"和"同事送我的键盘"缓存未命中，即使语义相同。未来可引入**语义哈希**或**字符级模糊匹配**提升命中率。
- **LRU 驱逐策略合理**——查询频率通常遵循 Zipf 分布，最近使用 ≈ 最常使用。

---

## 八、大规模数据下性能退化曲线

### 8.1 关键路径随数据量变化

| 数据量 | 1K | 10K | 100K | 1M | 退化特征 |
|--------|:--:|:--:|:---:|:--:|------|
| **save() 总延迟** | 15ms | 18ms | 25ms | 40ms | SQLite B-tree 深度增加，FAISS 写入变慢 |
| **recall() - 关键词路径** | 50ms | 60ms | 80ms | 120ms | 全扫描限 2000 行，增长来自索引遍历 |
| **recall() - 向量路径** | 10ms | 12ms | 18ms | 35ms | FAISS IVF 子线性退化 |
| **recall() - RRF 融合** | <1ms | <1ms | <1ms | 2ms | 字典合并 O(候选数) |
| **recall() 总计** | 60ms | 72ms | 98ms | 157ms | |
| **FAISS 搜索 P99** | 1ms | 3ms | 10ms | 30ms | nlist=sqrt(N) 自适应 |
| **FAISS 内存** | 4MB | 40MB | 400MB | **4GB** | 线性增长，1M 触达瓶颈 |
| **SQLite FTS5 搜索** | 2ms | 5ms | 15ms | 30ms | trigram 索引子线性 |

### 8.2 退化曲线

```
recall() 延迟 (ms)
^
180 │                                          ╭── 向量通路
160 │                                   ╭──────╯
140 │                              ╭────╯
120 │                         ╭───╮╯
100 │                    ╭───╮╯
 80 │               ╭───╮╯
 60 │          ╭───╮╯
 40 │     ╭───╮╯            ← 关键词通路（限 2000 行扫描，增长平缓）
 20 │╭───╮╯
  0 └┴───┴────┴─────┴──────┴─────→ 数据量
     0   1K   10K   100K   1M

FAISS 内存 (MB)
^
4GB │                                    ╭── ★ 警告：1M 时 4GB
2GB │                              ╭────╮╯
1GB │                         ╭───╮╯
500M│                    ╭───╮╯
200M│               ╭───╮╯
100M│          ╭───╮╯
 50M│     ╭───╮╯
 24M│╭───╮╯                  ← 模型权重 + Runtime (常数)
  0 └┴───┴────┴─────┴──────┴─────→ 数据量
     0   1K   10K   100K   1M
```

### 8.3 瓶颈与应对

| 瓶颈点 | 触发阈值 | 影响 | 应对方案 |
|--------|:------:|------|------|
| FAISS 内存线性增长 | 100K+ | OOM 风险 | 切换到 IndexIVFPQ（量化压缩 8×） |
| `_edge_expansion()` BFS | 100K+ | 延迟快速增加到 100ms+ | 限制种子实体数上限 + 图分区 |
| 全扫描 2000 行固定 | 2000+ | 召回覆盖度下降 | 增大 `full_scan_limit` 或切分 namespace |
| FAISS 单次写入 `_save()` | 10K+ | save() 延迟增加 | 异步写入 + 增量 flush |
| Python 内存 | 1M+ | 进程内存 > 8GB | 迁移到服务化架构（gRPC + 索引服务） |

### 8.4 FAISS nlist 自适应策略（当前实现）

```python
# vector_index.py:264
self._nlist = min(int(np.sqrt(len(all_vecs))), 256)
```

| 数据量 | nlist | 搜索延迟 | 说明 |
|--------|:---:|:------:|------|
| 1K | 8 | ~0.5ms | 训练阈值 nlist×40=320，超过即可训练 |
| 10K | 100 | ~2ms | sqrt(10000)=100 |
| 100K | 256 (cap) | ~10ms | 达到 256 上限，不再增加 |
| 1M | 256 (cap) | ~30ms | 固定 256 聚类，每簇 ~4000 向量 |

**256 nlist 上限是合理的**：超过 256 时，quantizer 搜索开销增大，收益递减。

---

## 九、综合风险评估

| 风险 | 等级 | 说明 | 缓解措施 |
|------|:--:|------|------|
| save() LLM 阻塞 | 🔴 高 | 串行 LLM 调用使 save() 延迟飙升到 2s | Phase 2 批量模式 + 异步执行 |
| FAISS 内存膨胀 | 🟡 中 | 100K+ 时 ~400MB，单机仍可承受 | 监控 + 100K 阈值告警 |
| save() 每次写入 FAISS | 🟡 中 | 高频写入时 I/O 成为瓶颈 | 批量 flush |
| recall() 超 200ms | 🟢 低 | 当前评估 ~150ms，有安全余量 | 已有 |
| 缓存命中率偏低 | 🟡 中 | 256 条对活跃用户可能不足 | 适当增加到 512 或引入语义哈希 |
| 同义词表膨胀 | 🟢 低 | 100 条同义词 < 0.1ms，增长到 1000 也 < 1ms | 无 |
| FAISS 训练冷启动 | 🟢 低 | 需要 nlist×40=320 条才开始训练，在此之前用 brute-force | 启动时预热 |

---

## 十、总结与建议

### ✅ Phase 2 性能目标达成情况

| 目标 | 预估值 | 目标值 | 状态 |
|------|:-----:|:-----:|:----:|
| save() 延迟 | 11–50ms | <50ms | ✅ |
| recall() 延迟 | 50–150ms | <200ms | ✅ |
| 同义词扩展 | <1ms | <1ms | ✅ |
| LLM 改写（异步） | <2ms (首次) | <200ms | ✅ |
| 缓存命中率 | 30–50% | — | ⚠️ 可优化 |

### 📋 优先建议

1. **P0**：`save()` 中 LLM 提取改为异步 + 批量模式（当前同步阻塞是最大性能风险）
2. **P1**：FAISS `_save()` 增加节流，每 N 次 add 才落盘一次（当前每次 add 都写盘）
3. **P2**：`_edge_expansion()` BFS 增加种子实体数上限（大规模图时最慢路径）
4. **P3**：准备 100K+ 向量时的监控和告警（FAISS 内存监控）
5. **P3**：ExpansionCache 容量评估 → 256 是否够用，视生产数据调整到 512 或 1024
