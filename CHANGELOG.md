# CHANGELOG

All notable changes to EngramRouter.

## [1.0.0] — 2026-08-03

### Security
- **N3: namespace-scoped cross-tenant isolation.** Add `namespace` column to
  `entities`, `edges`, `timed_events`. `delete()` verifies namespace ownership
  before mutating FTS/vector indexes. `consolidate()` scoped to namespace.
  `_get_or_create_entity` uses (name, kind, namespace) identity key. (#7a10bed)
- **N5: prompt-injection defence.** New `prompt_guard.py` module wraps all
  user-supplied text in `<user_content>` XML tags before it reaches LLM prompts.
  Wires into 4 LLM surfaces: HyDE, LLM extractor, query expander, intent
  classifier. Defends against delimiter confusion, instruction hijacking, and
  token-bomb DoS. (#4cafdad)
- **N4: MCP input clamping.** Validate/coerce: namespace (safe-char regex, max
  64 chars), memory_id (printable, max 128 chars), text (max 64 KB), top_k
  (clamped to [1, 100]). FTS5 LIKE wildcards `%`/`_` escaped to prevent SQL
  LIKE injection. (#9fa264e)

### Reliability
- **N6: atomic FAISS writes.** Write `.faiss` and `.idmap` files to temp +
  `os.replace()` for crash-safe persistence. `add()`/`add_batch()` no longer
  auto-save — caller commits once. `save_batch()` calls `vector_index.save()`
  exactly once instead of O(n). (#36be74f)
- **N7: recall query deadline.** `RecallWeights.recall_deadline_ms` (default
  5000ms). Expensive phases (HyDE LLM, cross-encoder, ColBERT, LLM-rerank)
  skip gracefully when deadline exceeded. Cheaper phases (FTS, entity
  extraction, edge expansion, base scoring) always complete. (#746b5ce)

### Observability
- **N1: Prometheus metrics wiring.** `recall_latency_ms`, `stage_hit_ratio`,
  `save_dedup_ratio` histograms/counters with graceful no-op when
  `prometheus_client` is unavailable. `/metrics` HTTP exposition endpoint.
  (#c54e97f)
- **N2: dim mismatch counter.** Vector dimension mismatch on model swap
  increments `dropped_vector_count{reason="dim_mismatch"}` instead of
  silent data loss. (#c54e97f)

### Infrastructure
- `engram_router/store/` subpackage: core.py / recall.py / candidates.py /
  scoring.py / graph.py / channels.py / pipeline.py / records.py /
  query_intent.py / metrics.py / trace.py (store.py拆解，L4.1)
- `recall_snapshots` table for offline replay (L4.3)

### Features
- Cross-encoder reranking (bge-reranker-v2-m3): P@1 0.61→0.83 (+21.7pp)
- HyDE query expansion with negative-pattern gate
- Multi-query variant RRF fusion
- LLM intent classifier (8-class soft probability)
- FAISS IndexHNSWFlat default + IVFPQ opt-in (PQ quantization)
- ColBERT late-interaction reranker (Jina ColBERT v2)
- HanLP NER entity extraction (MSRA_NER_ELECTRA_SMALL_ZH)
- Fastcoref pronoun coreference resolution
- Entity canonicalization + alias table
- Edge relations expanded: 8 LLM + 4 system types
- Recall pipeline with typed stages + debug tracing (L4.2)
- Ebbinghaus forgetting engine + consolidate
- CJK bigram FTS5 tokenizer
- LongMemEval `_s` split driver + GPT-4 judge
- Multi-provider eval matrix (engram / naive-vector / mem0 / long-context)

### Tests
- 600 passed (+52 N4/N5/N6/N3), 4 skipped (GPU-only), 12 xfailed (optional deps)
- PersonaStore / CausalChain / Timeline / ForgettingEngine / Corrections
  capability tests (40 cases)
- LongMemEval oracle + `_s` split test harness
- CI gate: P@1≥0.8, contam≤0.05, rejection≥0.9
- Multi-angle eval with isolation gate
- PR shadow diff mode
