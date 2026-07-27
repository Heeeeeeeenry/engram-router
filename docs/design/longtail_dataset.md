# L0.3 Long-tail 数据集方案：500-turn / 5000-turn

**创建**: 2026-07-27
**状态**: 方案设计
**关联**: `OPTIMIZATION_ROADMAP.md` L0.3 长尾长度两档

## 决策: 复用 LongMemEval 而非从头构造

### 为什么不自己生成对话

- hand-write 500-turn 对话不可行：15-turn 已花了大量人工，500-turn 不可能
- LLM self-play 生成对话 → 记忆注入 → QA 自动生成：LLM 生成的"记忆"缺乏真实知识冲突和噪音，过于干净
- LongMemEval (_s/_m split) 恰好是真实 ChatGPT 日志，500 题 × 6 题型，论文验证过难度区分度

### 数据来源

| 档位 | LongMemEval split | turns/case | memories/case | 用例数 | 文件大小 |
|---|---|---|---|---|---|
| **500-turn** | `_s` (short, with distractors) | ~470-620 | ~230-305 | 500 | 277 MB |
| **5000-turn** | `_m` (medium, with distractors) | ~5000 | ~2500 | 500 | 969 MB |

- `_s` split: ~40-55 session / case，含大量 distractor（不含答案的 session）
- `_m` split: ~500 session / case，retrieval 压力级联增大
- 都是 real ChatGPT 多轮对话，英文为主

### 为什么 _s 比 oracle 更适合

回顾之前的对比实验（roadmap 2026-07-23 · H）：

- oracle split (15 memory, no distractor): engram ≈ naive-vector，双方都轻松
- **_s split** (~250 memory, 含 distractor): **engram P@1 +10 pp** (0.53 vs 0.43)

500-turn 的 distractor 场景正是 engram 多层召回证明价值的战场。

## 实现方案

### Step 1: 写 converter `tests/longtail_converter.py`

```
LMECase (现有 loader) → EvalCase (eval_v2 格式)
                        → EvalScenario
                        → JSON 文件写入 tests/data/
```

**核心转换逻辑**:

```
for each LMECase:
  EvalCase:
    id = case.question_id
    query = case.question
    expected_ids_or_texts = [case.answer]
    forbidden_ids_or_texts = []
    is_negative = False
    description = f"{case.question_type} | {case.num_sessions}s | {case.num_turns}t"

  EvalScenario:
    id = case.question_id
    source = f"longmemeval_{split}"
    memories = case.memories       # loader 已经 flatten 为 user-turn 文本
    cases = [eval_case]
```

### Step 2: 输出两个 JSON 文件

- `tests/data/longtail_500.json`: `_s` split converter → 500 scenarios
- `tests/data/longtail_5000.json`: `_m` split converter → 500 scenarios

JSON 格式与 `held_out_cases.json` 一致，eval_v2 的 `_build_scenarios_from_json` 直接可用。

### Step 3: Makefile target

```makefile
# Build long-tail datasets from LongMemEval (requires network first time)
longtail-500:
	python tests/longtail_converter.py --split s --output tests/data/longtail_500.json --limit 500

longtail-5000:
	python tests/longtail_converter.py --split m --output tests/data/longtail_5000.json --limit 100

# Run eval on long-tail datasets
eval-longtail-500:
	ENGRAM_SKIP_VECTOR=1 python tests/eval_v2.py --scenarios tests/data/longtail_500.json --judge

eval-longtail-5000:
	ENGRAM_SKIP_VECTOR=1 python tests/eval_v2.py --scenarios tests/data/longtail_5000.json --judge
```

### Step 4: 评测方法

500-turn: 用 `eval_v2.py` 标准指标 (substring P@1 + LLM judge accuracy)，与 `make eval` 做法一致。

5000-turn: 面临延迟和内存瓶颈:
- 单个 case ~2500 memory → FAISS 训练 OOM 风险高
- CE 冷启动 15s + 2500 candidates 重排不可行
- **建议**: 5000-turn 先做 sampler（随机抽 500 memory 做 per-case），P@1 设为"检索是否覆盖包含答案的那条 session"

## 判断矩阵

| 方案 | 优点 | 缺点 | 决定 |
|---|---|---|---|
| 复用 LongMemEval | 数据已就绪，论文审过，含 distractor | 英文为主，非中文原生 | **采用** |
| 自己 LLM self-play 生成 500-turn | 完全可控，中文定制 | 质量不可靠，缺乏真实噪音 | 不采用 |
| THUIR MemoryBench Locomo | 学术基准，含中文 | schema 不统一，集成成本高 | P1 备用 |

## 工作量估算

| 步骤 | 工作量 |
|---|---|
| converter 脚本 | ~80 行，30 分钟 |
| JSON 文件生成 (_s 500) | 脚本自动，5 分钟 |
| JSON 文件生成 (_m 100) | 脚本自动（需先下载 _m 全量），15 分钟 |
| eval_v2 对接测试 | 改动现有 driver，30 分钟 |
| 首轮 500-turn 评测 | 慢（每 case ~250 mem × CE），预估 ~2-4 小时 |
| 首轮 5000-turn 评测 | 仅采样模式可行 |

## 关联

- [[engram-optimization-roadmap]] L0.3
- `tests/longmemeval_loader.py`: 现有 loader，本方案复用
- `tests/eval_v2_longmemeval.py`: 现有 LongMemEval eval driver
- `tests/lme_judge.py`: LLM faithfulness judge，本方案复用
