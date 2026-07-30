# EngramRouter 修正后阶段性目标

**日期**: 2026-07-28  
**基于**: 全能力审计 + 当前基线 + 剩余待办分析  
**前序评估**: `docs/design/phase2_reassessment.md`

---

## 当前真实基线

| 指标 | 值 | 判定 |
|---|---|---|
| P@1 (49 case) | 0.8043 | ✅ CI gate PASS |
| Contamination@3 | 0.0136 | ✅ 极低污染 |
| Rejection Accuracy | 1.0 | ✅ 负样本全对 |
| Precision (multi_angle) | 89.7% | ✅ 超 85% 门禁 |
| Isolation (multi_angle) | 75.0% | ❌ 未达 85%，阻塞唯一门禁 |
| Stability (J@1) | 1.0 | ✅ 完全确定性 |
| Regression (529 tests) | 529 passed | ✅ 零破坏 |

## 已完成能力（45/71 = 63%）

| 阶段 | 完成率 | 关键交付 |
|---|---|---|
| L0 评估体系 | **100%** (19/19) | 7 指标 + 3 judge + 5 数据集 + 4 竞品 + CI gate |
| L1 召回核心 | **100%** (5/5) | CE + 三通道 + HyDE + Multi-query + Intent |
| L1 召回中期 | **80%** (8/10) | bge-m3 / ColBERT / HNSW+PQ / HanLP / Coref / Alias / Relations / 时间归一化 |
| L4 工程 | **100%** (7/7) | store 拆解 + pipeline + trace + metrics + snapshots |

## 剩余的 2 项短期目标（优先）

### P0: Isolation 75% → 85%

**原因**: 这是唯一阻塞 CI gate-multi 的指标。

**剩余失败 2 例**:
- "谁之前在华为做过？" → `小李` 泄漏。query 含 `谁` 无人名 entity
- "他送我的东西好用吗？" → `不好用` 泄漏。query 用 `他` 代词

**修复方案**: recall 入口对 query 文本查最近 DB 中匹配的人名实体列表，试探性注入 query entity。工作量 ~50 行。

### P1: 5000-turn 完整跑分

**原因**: 项目中最大的评测缺口——长尾数据已有（243K memories），但从未用 CE+ColBERT 完整跑分。

**做法**: `eval_v2.py --scenarios longtail_5000.json --limit 10`（快速基线验证），逐步扩大到 100。

## 按新目标执行的一次迭代

| 步骤 | 内容 | 文件 | 行数 |
|---|---|---|---|
| 1 | query-side person detection | recall.py Phase 2.6 | ~40 行 |
| 2 | 5000-turn smoke test | eval_v2 run | — |
| 3 | Update roadmap + commit | ROADMAP.md | — |
