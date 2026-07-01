# LLM Extractor 增强方案

[2026-07-01 多智能体产出]

## 四个模块

### 1. LRU 结果缓存
- Key: sha256(text)
- 容量: 4096 条目
- 线程安全
- 可选磁盘持久化

### 2. 批量提取模式
- extract_batch(texts: list[str]) → 一次 LLM 调用处理 ≤10 条
- 自动查缓存 + 分组打包 + 结果拆分

### 3. Salience 后处理
- decision → DECISION_CAUSED_BY 边 (关联原因)
- constraint → CONSTRAINS 边 + 硬约束标记
- event → HAPPENED_AT 边 (关联时间)
- sensory → 极性检测 (positive/negative/neutral)

### 4. 边类型枚举 + 验证
- 8 种 LLM 可提取类型 + 4 种系统生成类型
- _validate_edges(): 过滤自环/非法类型/不存在实体

## 实现优先级

| 优先级 | 模块 |
|--------|------|
| P0 | LRU 缓存 |
| P0 | 批量模式 |
| P1 | 边类型验证 |
| P2 | Salience 后处理 |

详细方案见子智能体原始输出。
