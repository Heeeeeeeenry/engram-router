# Phase 2 安全审核报告

[2026-07-01 安全审核员子智能体产出]

## 发现汇总

| 级别 | 数量 | 关键问题 |
|------|:----:|----------|
| **CRITICAL** | 3 | SSL验证关闭、pickle RCE、记忆无条件上云 |
| **HIGH** | 5 | API Key回退、Prompt注入(x3)、默认境外API |
| **MEDIUM** | 6 | 缓存投毒、Key空串、输出未校验、元数据泄漏、降级静默 |
| **LOW** | 3 | pickle协议、异步错误吞没、FTS5扫描 |

## CRITICAL 详情

### C-1: SSL证书验证被关闭
`llm_extractor.py:172` — `ctx.verify_mode = ssl.CERT_NONE`
修复: 加载内部CA证书而非关闭验证

### C-2: pickle反序列化 (RCE风险)
`vector_index.py:308` — `pickle.load(f)`
修复: 改用JSON存储ID映射

### C-3: 用户记忆无条件发送到云端LLM
三个模块均将完整记忆文本发送到API，无用户同意机制
修复: 增加 `privacy.allow_cloud_llm` 配置 (默认False)

详见子智能体完整输出。
