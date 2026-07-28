# EngramRouter 阶段性目标重评估

**评估日期**: 2026-07-28
**基于**: `OPTIMIZATION_ROADMAP.md` 所有 checkbox + 最新 eval_v2 数字 + 代码库审计

## 当前真实基线

| 指标 | 值 | 判定 |
|---|---|---|
| P@1 (49 case, 不含 CE) | 0.8261 | ✅ CI gate 通过 |
| P@5 | 0.9348 | 尾部覆盖 OK |
| Contamination@3 | 0.0136 | ✅ 极低污染 |
| Rejection Accuracy | 1.0 | ✅ 负样本全部正确 |
| MRR | 0.8795 | 排序质量好 |
| Precision (multi_angle, 39 case) | 89.7% | ✅ |
| **Isolation (multi_angle)** | **75.0%** | ❌ **核心缺口** |
| 稳定性 (J@1/J@3/rank_var) | 全部 1.0 | ✅ 确定性输出 |
| Regression (pytest) | 529 passed | ✅ |

## 已完成能力清单

### L0 评估体系 — 100% 完成
- 7 个指标 (P@1/MRR/nDCG/Contam/Rejection/Stability)
- 3 个 judge 层 (faithfulness + CE-GT + cross-model 解耦)
- 5 个数据集 (held-out/adversarial/multilingual/agent-traces/long-tail-500)
- 4 个竞品 (naive/mem0/long-context + 对比矩阵)
- 3 个 CI gate + PR shadow diff

### L1 召回架构 — 核心 100% 完成，中期 80% 完成
- ✅ CE 精排 (P@1 0.61→0.83)
- ✅ 三通道 RRF fusion
- ✅ HyDE + Multi-query + Intent classifier
- ✅ bge-m3 替换 (512d→1024d)
- ✅ ColBERT late-interaction (Phase 2.45)
- ✅ HNSW + PQ 量化 (FAISS 升级)
- ✅ HanLP NER 补充规则
- ✅ 代词消解 (coref.py)
- ✅ 实体标准化 (alias 表)
- ✅ 关系类型扩展 (IS_A/OWNS/LOCATED_IN/...)

### L4 工程可观测 — 100% 完成
- ✅ store.py 拆解 (7 steps)
- ✅ 类型化 pipeline (5 stages)
- ✅ 可观测性 (trace/metrics/snapshots)

## 缺失但应先做的能力

### 1. Isolation 从 75% → 85%（P0，阻塞 CI gate-multi）
- **根因**: 剩余 2 例 ("谁之前在华为做过"、"他送我的东西好用吗")
  - 本质是 pronoun-first queries 的 entity 缺失问题
  - Coref 已解决 save 侧（"他→老张"、"她→小李"），但 query 侧代词未解决
- **修复**: recall 阶段对 query 也做 coref（语境来自最近查询历史或 DB entity 最新匹配）

### 2. 时间归一化（P0，影响 temporal-reasoning 召回）
- 当前 `_TIME_SORT_ORDER` 缺 "刚才/现在/前一会儿/昨天" 等常见口语时间词
- `_extract_time` 纯规则，需扩展模式表
- 5 例 SA-S4/SA-S5/SA-S8 回归中 3 例是时间推理

### 3. 5000-turn 数据生成（P1，评估覆盖）
- _m split 数据已下载 (2.7 GB)，converter 已就绪
- `make longtail-5000` 立即可用

### 4. edges 时间戳深度应用（P1）
- `valid_from/valid_to` 列已创建，但 BFS edge_expansion 未使用
- 应让 recall 在 edge traversal 时过滤失效边

## 应延后或暂缓的能力

| 能力 | 原因 |
|---|---|
| PPR 替代 BFS | 当前 BFS 已含 multi-hop relevance-gating，PPR 需要 teleport 向量 + PageRank 预计算，复杂度高且当前召回指标不急需 |
| letta/zep 对照 | 部署重，ROI 低 — mem0 对照已有说服力 |
| L2 全量数据层 | event 表/矛盾检测/Postgres 迁移都是大工程，当前 P@1=0.83 不急需 |
| L3 记忆生命周期 | 分级衰减/离线巩固/冷存储 — 当前 forgetting 基础可用 |
| L5 前沿 | 多模态/代码记忆/主动记忆 — 过早 |

## 修正后的 3-step 最小路径

### Phase 1: 修核心漏洞（今天）
1. **query-side coref** — 解决最后 2 例 isolation（"谁"/"他送我的东西"）
2. **时间归一化扩展** — 修复 SA-S4/S5/S8 回归
3. **edges 时间戳 BFS 过滤** — 让 temporal edges 生效

### Phase 2: 评估补全
4. **5000-turn 生成 + 快速基线**
5. **eval_v2 完整报告**（含 bge-m3 + ColBERT + HanLP → 新基线）

### Phase 3: L2 起步（下次迭代）
6. **L1 event 表** — 结构化的记忆事实拆解
7. **矛盾检测** — NLI-based CONTRADICTS 升级（当前仅 heuristic 0.3）

## 判断矩阵

当前项目处于一个真实的拐点：
- 评测体系完整，能说实话（P@1=0.83 不是虚报的）
- 多层召回 + CE + ColBERT 是行业标准配置
- **但最大瓶颈不是"还能加什么模块"**，而是：
  1. **Isolation 75% → 85%**：这是唯一阻塞 CI gate 的指标
  2. **缺少真实大规模跑分**（500-turn/5000-turn 上一个完整 CE+ColBERT 跑分）
  3. **时间推理精度不足**（5 例老回归中 3 例是时序）

**核心建议**: 不做新模块，做 3 个针对性修复 + 1 个大规模评测，用真实数字验证这 2 周的工作成果。
