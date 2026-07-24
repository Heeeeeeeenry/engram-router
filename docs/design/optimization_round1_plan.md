# 优化执行计划（基于全面审计）

## 背景

审计发现 30+ 个具体问题，按修复顺序排列。本轮不做大重构（P1 store.py 拆解）和大评估（P0 接 LongMemEval `_s`），只修**P0 级 bug + 高频 ROI 优化**。

---

## 本轮改动（6 项）

### Fix 1: `ENGRAM_SKIP_VECTOR` 不应禁用 CE（store.py:268）
**Bug**: `ENGRAM_SKIP_VECTOR=1` 通过 `or` 也禁用了 CE，两个独立模块不应耦合。
**修复**: 删除 `or os.environ.get("ENGRAM_SKIP_VECTOR") == "1"`，只保留 `ENGRAM_SKIP_CE` 控制 CE。
**影响**: 1 行删改。

### Fix 2: `ForgettingEngine._days_since_accessed()` 始终返回 0.0（forgetting.py:260）
**Bug**: `_days_since_accessed()` 读 `memory.metadata.get("accessed_at")`，但 `accessed_at` 是 SQL 列不是 metadata JSON 字段。导致时间衰减永远不生效。
**修复**: 优先从 `memory.accessed_at` 属性（SQL column）读取，fallback 到 metadata。
**影响**: ~5 行修改 + 1 个测试用例验证。

### Fix 3: 三次重复 `embedding_engine.encode(query)`（store.py:1697, 1756, 1922）
**Bug**: 同一 query 在三次不同路径中被重复编码，浪费 2× 推理时间。
**修复**: 在 Phase 2 开始处编码一次，将 vec 传给下游三个阶段复用。
**影响**: ~10 行重构。

### Fix 4: CE 后向量 fallback 阈值语义错误（store.py:1918）
**Bug**: `score <= 0.1` 判断在 CE 归一化后无意义——[0,1] 区间的 0.1 可能是第 2 名。
**修复**: 改为 CE 可用时使用 CE 前的原始 score map 判断，或移到 CE 之前。
**影响**: ~5 行修改。

### Fix 5: HyDE 浪费 LLM 调用在 embedding 不可用时（hyde.py:341-345）
**Bug**: `expand_and_recall()` 先调 LLM（300-600ms），再检查 `if embedding is None`。
**修复**: 检查前置到 `generate()` 之前。
**影响**: 2 行挪动。

### Fix 6: 6 个魔法数字提升到 RecallWeights（store.py 多处）
**位置**: RRF keyword/vector weights `[0.4, 0.6]`、vector fallback `0.55`、recent fallback `0.5`、rrf_score 缩放 `*10`/`*60`、HyDE-skipped 降权 `*0.8`。
**修复**: 新增 6 个 RecallWeights 字段 + 默认值，替换 inline literal。
**影响**: ~30 行。

---

## 预期效果

| 修复 | 预期改善 |
|---|---|
| Fix 1 | CE 解耦后 `ENGRAM_SKIP_VECTOR=1` 不再误关 CE |
| Fix 2 | Forgetting 时间衰减正常运作 |
| Fix 3 | recall latency -10-20ms（省 2× encode） |
| Fix 4 | vector fallback 不再误判 |
| Fix 5 | HyDE 不浪费 LLM token |
| Fix 6 | 权重可配置、A/B 可调 |

## 不在此次范围

- store.py 拆解（P1，需 1-2 天独立工作）
- 换 bge-m3 + ColBERT（P3，需模型下载 + FAISS 重建）
- LongMemEval `_s` split（需跑 ~1 小时 eval）
- Phase 3 打通到 recall（P4，需设计 + 实现）
