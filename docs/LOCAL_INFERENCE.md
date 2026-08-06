# 全本地推理 (L5)

EngramRouter 设计原则之一是**本地优先**（local-first）：所有 LLM 调用默认走本地通道，
云端只在用户显式授权后才会使用。

本文档说明如何用 [llama.cpp](https://github.com/ggerganov/llama.cpp) 作为本地
LLM 后端，让 engram-router 完全脱离云端 API 运行。

## 快速开始

### 1. 安装 llama.cpp

```bash
# macOS (Homebrew)
brew install llama.cpp

# 或从源码编译
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && make -j
```

### 2. 下载 GGUF 模型

推荐中文能力较好的模型（任选一个）：

```bash
# Qwen2.5 (推荐)
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_K_M.gguf \
  --local-dir ~/models/

# DeepSeek
huggingface-cli download unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF \
  DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf --local-dir ~/models/
```

> **模型选择建议**：
> - **HyDE / 意图分类 / 查询改写**：7B Q4_K_M 即可，中文能力足够。
> - **矛盾检测 (check_new_event)**：需要推理能力，7B+ 建议。
> - **<7B 模型 (Qwen2.5-3B 等)**：轻量任务可用，但中文复杂推理（因果/隐含情感）质量
>   下降明显，不建议用于生产级 contradiction/persona enhancement。

### 3. 启动 llama.cpp server

```bash
llama-server \
  --model ~/models/qwen2.5-7b-instruct-q4_K_M.gguf \
  --port 8080 \
  --n-gpu-layers 33  # M1/M2/M3 Mac: layers offload to Metal, skip for CPU-only
```

验证 server 正常：

```bash
curl http://localhost:8080/health
# → {"status": "ok"}
```

### 4. 配置 engram-router

```bash
export ENGRAM_LOCAL_LLM=1
# 可选：如果端口不是默认 8080
export ENGRAM_LLM_BASE_URL=http://localhost:1234/v1
# 可选：非 llama.cpp，比如 vllm / oobabooga / text-generation-webui
export ENGRAM_LLM_MODEL=your-model-name
```

`ENGRAM_LOCAL_LLM=1` 的效果：
- 自动设置 `base_url=http://localhost:8080/v1`（llama.cpp 默认地址）
- 使用 dummy API key（本地服务通常不需要认证），使 `LLMClient.available=True`
- **不影响** cloud privacy gate（`ENGRAM_ALLOW_CLOUD=1` 等仍默认关闭）

如果你已经设置了 `ENGRAM_LLM_BASE_URL`，`ENGRAM_LOCAL_LLM=1` 不会覆盖它——显式
变量优先级更高。

### 5. 验证

```bash
engram save "测试本地推理" --namespace test
engram recall "测试" --namespace test
```

观察日志输出（没有 401/403 错误即表示请求成功打到本地 server）。

## 架构说明

### 本地优先路线图

| 组件 | 默认行为 | 本地可运行？ |
|------|----------|------------|
| **Embedding (文本向量)** | `sentence-transformers` 本地加载 bge-m3 | ✅ 默认本地 |
| **LLM Client** | 请求 DeepSeek 兼容 endpoint（需 API key） | ✅ `ENGRAM_LOCAL_LLM=1` |
| **Cross-encoder 重排** | 本地加载 bge-reranker-v2-m3 | ✅ 默认本地 |
| **ColBERT 重排** | 本地加载 Jina ColBERT v2 | ✅ 默认本地（GPU optional） |
| **HanLP NER** | 本地加载 MSRA_NER_ELECTRA_SMALL_ZH | ✅ 默认本地 |
| **Fastcoref 代词消解** | 本地 lib | ✅ 默认本地 |

**结论**：设置 `ENGRAM_LOCAL_LLM=1` 后，整个 engram-router 栈可以完全脱离云端运行。
Vector/CE/ColBERT/HanLP/Fastcoref 本来就是纯本地组件，LLM 是唯一需要远端调用的通路。

### 哪些功能需要 LLM

`ENGRAM_LOCAL_LLM=1` 用本地模型替代以下 LLM 调用：

- **HyDE 假想答案生成**（`hyde.py`）— token 量中等，7B 模型 1-3 秒
- **查询改写**（`query_expansion.py`）— token 量中等
- **意图分类**（`intent_classifier.py`）— token 量小，< 100 tok
- **LLM 实体/边抽取**（`llm_extractor.py`）— token 量中-大
- **LLM 重排**（`llm_reranker.py`）— token 量大，可能慢
- **矛盾检测**（`contradiction.py`）— token 量中-大，需要推理能力
- **Reflection 离线 job**（`reflection.py`）— 批量、异步

### 已知限制

1. **速度**：CPU-only 本地推理比云端 API 慢 5-20 倍。Metal/CUDA offload
   （`--n-gpu-layers`）可以大幅改善，但 7B 模型仍需要数秒 per 请求。
2. **质量**：4-bit 量化 + 7B 模型的推理质量在矛盾检测、情感解读、复杂因果推理
   上可能不及 DeepSeek-V4/GPT-4 等大模型。建议对 production 场景保留云端 LLM
   fallback。
3. **HyDE recall 延迟**：本地 LLM 的 HyDE 生成需要 2-5 秒，而 recall deadline
   默认 5000ms。如果本地模型太慢，HyDE 可能被 skip。可调整
   `RecallWeights.recall_deadline_ms` 放宽预算（比如 15000ms）。
4. **无 API key 的第三方兼容**：`ENGRAM_LOCAL_LLM=1` 用 dummy key `"local"`。
   如果本地 server 确实需要认证（少数场景），仍需显式设置
   `DEEPSEEK_API_KEY=your-key` 或 `ENGRAM_LLM_BASE_URL` 对应的 API key env。

## 本地 server 替代方案

除了 llama.cpp，任何 OpenAI-compatible 本地推理 server 都能用：

```bash
# vLLM
ENGRAM_LLM_BASE_URL=http://localhost:8000/v1 ENGRAM_LLM_MODEL=Qwen2.5-7B

# Ollama (需 curl wrapper 或 OpenAI-compat 插件)
# 注意：Ollama 原生 API 不兼容 OpenAI，部分插件可用

# LM Studio
ENGRAM_LLM_BASE_URL=http://localhost:1234/v1
```

只要 endpoint 实现了 `/v1/chat/completions`，就能直接用 `ENGRAM_LLM_BASE_URL`。
不需要 `ENGRAM_LOCAL_LLM=1`。

## FAQ

### 我不想装 llama.cpp，能用云端 LLM 吗？

当然。不设 `ENGRAM_LOCAL_LLM=1`，只设置：

```bash
export DEEPSEEK_API_KEY=sk-xxx
export ENGRAM_ALLOW_CLOUD_LLM=1
```

### ENGRAM_LOCAL_LLM 和 ENGRAM_ALLOW_CLOUD_LLM 能同时设吗？

可以。`ENGRAM_LOCAL_LLM` 只影响 `base_url` 和 API key，不影响 cloud privacy gate。
两者同时设置时，LLM 请求会发到本地 URL（因为 `base_url` 被 local preset 覆盖了），
cloud gate 是开的但实际不会用到（base_url 已指向本地）。

如果你希望本地 LLM **fallback 到云端**，目前不支持自动切换——需要手动
unset `ENGRAM_LOCAL_LLM` 并设置 `ENGRAM_ALLOW_CLOUD_LLM=1`。
