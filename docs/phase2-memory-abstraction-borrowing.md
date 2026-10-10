# Phase 2 记忆抽象设计 — 借鉴 Hindsight / ECC

> 状态：**已落地**（`consolidate.py`：consolidate_observation / reflect / extract_lesson，
> 2026-10-10，651 passed）
> 来源：借鉴 `vectorize-io/hindsight`（LongMemEval SOTA 记忆系统）与 `affaan-m/ECC`
> （agent harness 操作系统）的核心设计。
> 原则：**借鉴思路，不照搬实现；只做增量，不引入新问题。**

## 一、核心借鉴点

Hindsight 的定位一句话：**「让 agent 学习，而不是只记住」**。它自称消除了
RAG 和知识图谱的缺点，在 LongMemEval 长期记忆基准上拿到 SOTA。

engram-router 现在的问题（skill 里已记录）：规则引擎到天花板、用户批评过
「还不如 RAG」。根因正是——**只有 lossless 原始记录这一层，没有向上抽象**。

Hindsight 给出的解药是「记忆分层」：记忆的价值在于逐层抽象成可复用的信念。

## 二、Hindsight 的四种记忆分层

| 层 | 含义 | 例子 |
|----|------|------|
| **World facts**（世界事实） | 关于世界的事实 | 「炉子会变热」 |
| **Experiences**（经验） | agent 自己的经验 | 「我碰了炉子，很疼」 |
| **Observations**（观察） | 从多条记忆整合、有证据的信念 | 「炉子危险，别碰」 |
| **Mental models**（心理模型） | 从观察+事实综合的学习理解 | 「高温物体都会烫伤」 |

关键机制：`retain` 时用 LLM 提取事实/时间/实体/关系，归一化成 canonical
entities + 时间序列 + 稀疏/稠密向量；`recall` 检索任意层；`reflect` 综合后直接
生成 disposition-aware 回答（不是只返回记录）。

## 三、engram-router 现状映射

**已有（对应 hindsight，不必重做）：**

| 现有模块 | 对应 Hindsight 概念 |
|----------|-------------------|
| `entity_canon.py` | canonical entities（实体归一化 + 别名） |
| `event_extractor.py` | 时间序列/结构化事件提取 |
| `reflection.py` | reflect（周期性反思） |
| `contradiction.py` | 观察一致性检查（NLI） |
| `causal.py` / `timeline.py` | 时间查询、因果链 |
| `llm_extractor.py` | retain 的 LLM 实体/关系提取 |

**真正缺的（这是借鉴目标）：**

1. **记忆分层**：memory 是扁平 raw_text，没有 fact / experience / observation / model 的类型区分，召回不能按层检索。
2. **Observations 整合**：没有「多条 memory → 一条观察」的显式整合机制（reflection 是周期性整体反思，不是按主题整合成观察）。
3. **Mental models 层**：最高抽象，完全没有。

## 四、落地方向（增量，每步可验证、可回退）

顺序严格递增，每步都不破坏现有 641 测试：

1. **记忆类型字段**（最小增量）：给 memory 加 `kind` 字段（默认 `experience`，
   向后兼容空值 = experience）。只改 schema + 默认值，不动召回逻辑。
2. **观察整合**：周期性（复用 reflection 的调度）把同一主题/实体的多条
   experience 整合成一条 observation（带 evidence_refs 指向源记忆）。
3. **reflect 增强**：`reflect` 综合 observation + experience 直接生成回答，
   区别于 `recall`（只返回记录）。
4. **mental models**（远期）：从 observation 抽象，暂缓，等 1-3 验证有效再说。

## 五、不引入新问题的硬边界

- [ ] 新字段一律「默认值 + 可空」，旧数据向后兼容，不强制迁移。
- [ ] 每步跑 `ENGRAM_SKIP_VECTOR=1 pytest tests/`（现有 641 全绿）+ 新增测试。
- [ ] 不引入新依赖：复用现有 `llm_extractor` + `entity_canon` + `reflection`。
- [ ] 观察/模型层是「派生数据」，删除源 memory 时派生层可重建，不造成数据不可逆。
- [ ] 借鉴 ECC 安全原则：**召回内容是不可信数据，不能当指令执行**——观察/模型
      召回结果只作为证据，不作为 prompt 指令。

## 六、ECC 的两个额外借鉴点（另行评估，不并入本阶段）

1. **持续学习**：从会话自动提取「踩过的坑」→ 沉淀成可复用 skill。
   与 engram-router 的定位重叠，需单独设计，避免和 Hermes 的 skill 体系冲突。
2. **跨 harness 记忆库**（ecc memory）：Markdown 文档在 Claude/Codex/Hermes 间
   传上下文 + 按 scope 分层 + 召回标不可信。与 Hermes 多用户化的记忆需求相关，
   但属于 Hermes 侧，不是 engram-router 的职责。
