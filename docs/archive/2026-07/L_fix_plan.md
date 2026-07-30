# L 步骤遗留问题修复 & 优化计划

## 背景回顾

L 步骤 (2026-07-23) 已完成:
- `lme_judge.py`: CONFLICT RESOLUTION Rule 5 + `memory_timestamps` 参数 ✅
- `engram_provider.py`: `created_at` 传递 ✅
- `eval_v2_longmemeval.py`: timestamp 透传 ✅

**结果**: "both X and Y" 冲突全部消除 (0 例)，judge accuracy 0.52 → 0.58 (+6.7pp)
**遗留**: engram 仍输 naive 8 pp (0.58 vs 0.67)，主因是 "I don't know" (召回失败 14/25)

## 诊断

三个独立问题，共用一个修复窗口：

### 问题 1: CE + vector fusion 互相拉扯 → 召回覆盖率下降
- **位置**: `store.py:1771-1851` (Phase 2 vector fusion) + `store.py:1856-1891` (Phase 2.4 CE rerank)
- **根因**: RRF 先 blend keyword + vector → CE 在已 blend 的列表上重排 → ce_weight 单调调大只能部分纠正，已丢失的候选不在列表里
- **证据**: roadmap CE sweep 显示 ce_weight 0.6→0.85 只涨 4 pp P@1 但掉 12 pp Recall@5
- **修复**: CE 启用时跳过 RRF blending，将所有候选 (keyword + vector, dedup) 直接交给 CE 打分

### 问题 2: HyDE 默认关闭 → 抽象指代/时序推理场景缺召回增强
- **位置**: `store.py:1780-1808` (HyDE integration) + `hyde.py` (HyDEExpander)
- **根因**: HyDE 三项缺陷 (权重过高、negative 污染、hyde-only 候选混入 CE)
- **修复**: 三项针对性改动，使 HyDE 可安全默认开启或至少能用于 recall 增强

### 问题 3: 最新 engram 60 题 knowledge-update judge 结果缺失
- **位置**: `docs/eval_v2_longmemeval_oracle_judge.json` 只有 naive-vector 结果
- **修复**: 重新跑 engram 60 题 judge 评估，验证改动效果

---

## 改动清单

### 改动 1: 解耦 CE + vector fusion (`store.py` ~15 行改动)

**当前逻辑** (store.py:1771-1851):
1. keyword recall → records
2. vector search → vector_list + keyword_list → RRF blend → 覆盖 records 的 score
3. CE rerank blended records → blend again with `ce_weight`

**新逻辑**:
1. keyword recall → records
2. **如果 CE 可用**: vector search → 补充不在 keyword 结果中的 vector-only 候选 → 全部交给 CE 纯打分 (ce_weight=1.0 语义)
3. **如果 CE 不可用**: 保持现有的 RRF blend 路径

具体改动:
- 在 `store.py:1771` Phase 2 向量融合段，检测 `self.cross_encoder and self.cross_encoder.available`
  - CE 可用: vector search 结果只补充新候选 (不在 keyword records 中的)，以低初始分插入列表
  - CE 不可用: 保持现有 RRF blend
- 在 `store.py:1861` CE 段，CE 可用时使用 `ce_weight` 直接控制 blend (已经是现有逻辑)

### 改动 2: HyDE 三项修复 (`store.py` + `hyde.py` ~40 行改动)

**a) 权重降级**: `RecallWeights.hyde_rrf_weight` 默认 0.5 → 0.25
  - 位置: `store.py:145`

**b) Negative case 隔离**: HyDE 检测到 negative 时，同时降低 vector fallback 置信度
  - 位置: `hyde.py:_looks_negative` 已检测，`store.py:1784` 已有 `hyde_result.source == "skipped"` 分支
  - 新增: skipped 时 `hygiene_flag = True` → vector_score 打 0.7 折

**c) HyDE-only 候选在 CE 前降权**: 只有 hyde+vector 双通道命中的候选才保留完整 CE 分
  - 位置: `store.py:1861` CE 段之前
  - hyde-only (不在 keyword_list 也不在 vector_list) → score × 0.7
  - hyde+vector 双命中 → 保持原分

### 改动 3: 重新跑 engram 60 题 knowledge-update judge

命令:
```bash
ENGRAM_ALLOW_CLOUD_LLM=1 python tests/eval_v2_longmemeval.py \
  --split oracle --limit 60 --question-types knowledge-update \
  --providers engram,naive-vector --use-judge
```

---

## 预期效果

| 改动 | 预期影响 |
|---|---|
| CE + vector 解耦 | engram substring P@1 保持 (0.45)，Recall@5 提升 (0.77→0.82+)，减少 IDK |
| HyDE 三项修复 | HyDE 可安全开启做 A/B，rejection accuracy 恢复至 1.0 |
| 综合 judge accuracy | engram judge 0.58 → 0.63+ (缩小与 naive 的差距) |

## 不在此次范围的项

- HyDE 默认开启 — 需额外一轮 eval 验证，本次只修缺陷
- LongMemEval `_s` split (500 题含 distractor) — 下个步骤
- store.py 拆解 (L4.1) — 背景债务
- Latency 优化 — CE 冷启动是已知瓶颈，需要架构改动
