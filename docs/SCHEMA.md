# EngramRouter Storage Schema

This document records the **actual** SQLite schema as implemented in
`src/engram_router/store.py` (`MemoryStore._init_schema`), plus how the tables
map onto the current memory-routing pipeline described in `docs/PROJECT_BRIEF.md` and the source code in `src/engram_router/store.py`.

It is the source of truth for the on-disk shape. If the code and this file ever
disagree, the code wins — update this file to match.

## Design principle

Raw evidence is the source of truth. Summaries and distilled memories are
retrieval aids that **point back** to evidence; they never replace it. No row
is deleted on a normal write path. This is why almost every "derived" table
carries a foreign key back to either `memories` or `raw_logs`.

## Layer mapping

The architecture defines four compaction layers. The tables realise them as:

```text
L0 Raw Log        -> raw_logs
L1 Event          -> (not yet a dedicated table; events are implicit in entities/edges)
L2 Distilled Mem  -> distilled_memories (+ the memories row it produces)
L3 Context Pkg    -> not persisted; assembled per-query by recall(), returned as MemoryRecord list
```

`memories` + `evidence` are the durable backbone that every layer anchors to.
Routing handles (`entities`, `memory_entities`, `edges`, `memories_fts`) sit
beside the backbone to let recall hop across turns that share no surface tokens:
`edges` carry typed associations written on `save` and traversed **up to 2 hops**
on `recall` via spreading activation; `memories_fts` is an FTS5 trigram candidate
index. `salience_class` on entities drives associative decay:
`base_attr` attributes attenuate to ×0.15 through edges, while `event` memories
travel at full strength. `id_sequences` provides monotonic, concurrency-safe
ID allocation via atomic `UPDATE ... RETURNING`.

## Tables

### memories
The durable record. One row per saved turn (or per distilled memory).

| column      | type | notes |
|-------------|------|-------|
| id          | TEXT PK | `mem_N`, N = count+1 at insert time |
| raw_text    | TEXT NOT NULL | full original text — never lossily rewritten |
| summary     | TEXT NOT NULL | lightweight aid; currently `raw_text[:160]` |
| source      | TEXT NOT NULL | default `conversation`; `compaction` for distilled |
| confidence  | REAL NOT NULL | default `1.0` |
| metadata    | TEXT NOT NULL | stringified dict, default `'{}'` |
| namespace   | TEXT NOT NULL | default `'default'`; multi-tenant isolation |
| created_at  | TEXT NOT NULL | `CURRENT_TIMESTAMP` |

### evidence
The anchor that prevents compression drift. Each memory gets at least one
evidence row quoting its source text.

| column          | type | notes |
|-----------------|------|-------|
| id              | TEXT PK | `evi_N` |
| memory_id       | TEXT NOT NULL | FK -> memories(id) |
| quote           | TEXT NOT NULL | exact quoted text |
| source_location | TEXT NOT NULL | default `''`; holds source tag or raw_log id |
| created_at      | TEXT NOT NULL | `CURRENT_TIMESTAMP` |

### raw_logs
L0 raw log. Full turns / tool output / diffs / test logs. Not normally injected
into model context; kept so distillation can always be re-derived and audited.

| column     | type | notes |
|------------|------|-------|
| id         | TEXT PK | `raw_N` |
| kind       | TEXT NOT NULL | e.g. `conversation`, `file_change` |
| text       | TEXT NOT NULL | full raw text |
| created_at | TEXT NOT NULL | `CURRENT_TIMESTAMP` |

### distilled_memories
L2 link table. A distilled memory is itself stored as a `memories` row; this
table records the provenance edge from that distilled memory back to the raw log
it was distilled from. `compact()` writes the row and an extra `evidence` row,
and **never** touches the originating `raw_logs` entry.

| column         | type | notes |
|----------------|------|-------|
| id             | TEXT PK | `dst_N` |
| raw_log_id     | TEXT NOT NULL | FK -> raw_logs(id) |
| memory_id      | TEXT NOT NULL | FK -> memories(id) (the distilled memory) |
| distilled_text | TEXT NOT NULL | the distilled statement |
| created_at     | TEXT NOT NULL | `CURRENT_TIMESTAMP` |

### entities
Routing handles extracted conservatively (rule-based, no LLM) by
`entities.extract_entities`. Deduped on `(name, kind)` via `_get_or_create_entity`.
Salience is classified post-extraction by `entities.classify_salience()` using a
7-level priority rule chain.

| column | type | notes |
|--------|------|-------|
| id     | TEXT PK | `ent_N` |
| name   | TEXT NOT NULL | surface name, e.g. `张三`, `HHKB`, `腾讯` |
| kind   | TEXT NOT NULL | `person` / `object` / `company` / `time` / `reason` / `topic`; default `unknown` |
| salience_class | TEXT NOT NULL | `base_attr` / `sensory` / `event` / `decision` / `constraint`; default `event` |

### memory_entities
Many-to-many link between a memory and the entities found in it, with the
evidence substring the entity was drawn from.

| column        | type | notes |
|---------------|------|-------|
| id            | TEXT PK | `me_N` |
| memory_id     | TEXT NOT NULL | FK -> memories(id) |
| entity_id     | TEXT NOT NULL | FK -> entities(id) |
| salience_class| TEXT NOT NULL | denormalised from entities.salience_class for fast recall filtering; default `event` |
| evidence      | TEXT NOT NULL | default `''`; substring justifying the link |

### edges
Typed, confidence-bearing links between entities, **written automatically on
`save`** and used by `recall` for one-hop graph expansion. From the entities
extracted out of a single memory, `save` writes:

- a `CO_OCCURS_WITH` edge (confidence `0.4`) between every unordered pair of
  entities in that memory — an *inferred* association, deliberately
  low-confidence;
- a `CAUSED_BY` edge (confidence `0.95`) from each concrete entity to a
  user-stated cause (a `reason` entity surfaced from a causal marker such as
  因为/由于). High confidence because the user asserted the causation.

The causal-edge hard boundary: an inferred relation is only ever
`CO_OCCURS_WITH` at low confidence; it is never promoted to a fact (CAUSED_BY)
without a user-stated marker or accumulated evidence. `evidence_ref` points back
to the originating memory id. There is no separate `source`/`revocable` column —
provenance is carried by the relation name + confidence + `evidence_ref`.

`recall` reads these edges to make a one-hop jump: a memory directly matched by
the query exposes its entities; edges out of those entities reach a *second*
memory that shares no surface token with the query, and that memory is pulled in
with a confidence-weighted bonus (see "Recall" below).

| column       | type | notes |
|--------------|------|-------|
| id           | TEXT PK | `edge_N` |
| src_id       | TEXT NOT NULL | entity id |
| dst_id       | TEXT NOT NULL | entity id |
| relation     | TEXT NOT NULL | `CO_OCCURS_WITH` \| `CAUSED_BY` (extensible) |
| confidence   | REAL NOT NULL | default `1.0`; `0.4` co-occurs, `0.95` caused-by |
| evidence_ref | TEXT NOT NULL | default `''`; originating memory id |

### corrections
User corrections preserved as first-class evidence (safety model rule 5).

**实现状态**：完整可用。corrections 表通过 `_get_corrected_ids()` 接入 `recall()` 管线：
- 被纠正的记忆 ×0.3 降权，`match_reason` 标记 `"user_corrected"`
- corrections 表内记录原文不删除，证据链完整保留
- delete API 支持删除错误纠正记录

| column          | type | notes |
|-----------------|------|-------|
| id              | TEXT PK | |
| target_id       | TEXT NOT NULL | memory/entity being corrected |
| correction_text | TEXT NOT NULL | the correction |
| evidence_ref    | TEXT NOT NULL | default `''` |
| created_at      | TEXT NOT NULL | `CURRENT_TIMESTAMP` |

### id_sequences
Monotonic ID allocator. Replaces the fragile `COUNT(*) + 1` pattern with an
atomic `UPDATE ... RETURNING` for concurrency-safe ID generation.

| column    | type | notes |
|-----------|------|-------|
| name      | TEXT PK | table name (`memories`/`entities`/`edges`/...) |
| next_val  | INTEGER NOT NULL | next ID to allocate |

## ID convention

IDs are `prefix_N` strings allocated via `id_sequences` with atomic
`UPDATE ... RETURNING next_val - 1`. Legacy databases without `id_sequences` are
auto-seeded on first upgrade by scanning existing rows for the highest
watermark. This is safe under concurrent writers within a single SQLite
connection (WAL mode). Multi-writer scenarios would require an external sequence
service.

## Concurrency baseline

The store uses SQLite in WAL mode with `busy_timeout=5000` and atomic
`UPDATE ... RETURNING` for id allocation via `id_sequences`. This is safe
for single-process, multi-threaded access. Multi-process concurrent writers
to the same database file are **not yet supported** — they would need an
external locking or sequence service.

## Recall, briefly

`recall()` uses a two-phase pipeline: candidate selection → weighted ranking.

**Phase 1 — FTS5 trigram candidate selection** (see "FTS5 candidate path" below).

**Phase 2 — Multi-signal weighted ranking.** Candidates are scored by layer:

0. **Weighted term overlap** — ASCII/brand tokens, multi-char CJK, and single
   CJK chars are weighted by `RecallWeights` (injectable, defaults preserve
   historical behaviour).
1. **Entity hop** — boost memories sharing an extracted entity with the query
   (×`shared_entity_multiplier` per shared entity, default 1.2).
2. **Context-aware boosts** — `_asks_brand()` boosts product-bearing memories
   (×`brand_boost`, default 2.0); `_asks_identity()` boosts memories with
   person base attributes (×`identity_base_attr_boost`, default 2.0);
   `_asks_eval()` boosts sensory-tag memories (×`eval_sensory_boost`, default 1.5).
   All weights live in `RecallWeights` and can be injected at store construction.
3. **N-hop edge expansion** — BFS spreading activation over `edges` up to
   `max_recall_hops` (default 2). Activation = source × `recall_decay`(0.5) ×
   edge.confidence. CO_OCCURS_WITH(0.4) attenuates fast; CAUSED_BY(0.95)
   propagates farther. Below `activation_threshold` (0.03) stops spreading.
   Multi-path gains accumulate via re-propagation.
4. **Salience-based associative decay** — edges from a base_attr entity
   are dampened to ×`assoc_reach_base_attr` (0.15), so base attributes rarely
   pollute associative recall, while event/sensory entities propagate freely.
   All reach factors in `RecallWeights.assoc_reach_*`.

The result is the top-k `MemoryRecord` list (L3 context package), each carrying
`score`, `match_reason`, and `evidence_refs`. It is assembled per query and not
persisted.

### FTS5 candidate path

`save` mirrors each memory into a `memories_fts` FTS5 virtual table with the
`trigram` tokenizer; `recall` uses it to pre-filter candidates, then ranks those
candidates with the weighted scorer. The two concerns are decoupled by design:

- **FTS5 = candidate selection, not ranking.** Whatever rows survive the trigram
  match are still ordered by the weighted score. `recall` accepts a pluggable
  ranker via `store.ranker` (a callable `(query, terms, haystack, store) ->
  float`); the default ranker is the existing `_score` / `_term_weight`.
- **trigram needs ≥3 characters.** It matches ASCII brands (`HHKB`) and CJK
  words of ≥3 chars (`机械键盘`), as substrings with no whitespace-boundary
  requirement. It **cannot** match a 2-char CJK query (`键盘`, `张三` → 0 hits).
- **Chinese short-query fallback.** When the query has no trigram-eligible term
  (2-char CJK, single chars) `_fts_candidates` returns `None` meaning "scan
  everything", so the entity/topic/edge hops still reach answers trigram can't
  see. FTS5 only *prunes* the candidate set; it never *suppresses* the fallback,
  and a query that runs FTS but matches nothing still falls through to the full
  weighted scan.
- **Graceful degradation.** If the SQLite build lacks FTS5/trigram, `_init_fts`
  disables the path silently (`_fts_enabled = False`) and recall is correct via
  the full scan — only the pre-filter is skipped.

Probed on the dev box: SQLite 3.45.3, `fts5=true`, `trigram=true`,
ASCII match=1, 2-char CJK match=0, 3-char CJK match=1.

## Verification

- Schema + recall + gap-check + compaction + concurrent writes: `tests/test_store.py` (11 cases).
- Entity extraction + entity-hop recall + salience classification: `tests/test_entities.py` (5 cases).
- Typed edge writing + N-hop edge-driven recall expansion: `tests/test_edges.py` (8 cases).
- FTS5 trigram candidate path + pluggable ranker: `tests/test_fts.py` (10 cases, skipped on builds w/o FTS5/trigram).
- id_sequences monotonic allocator + legacy seed: `tests/test_id_sequences.py` (5 cases).
- Summary-baseline vs evidence-recall benchmark (34 hard gates): `tests/test_benchmark.py`.

Run all: `python -m pytest -q` (265 passed total across 9 test files).

## Indexes

Covering indexes for recall and graph-expansion hot paths:

| Index | Table | Columns | Purpose |
|-------|-------|---------|---------|
| `idx_memories_ns_created` | memories | (namespace, created_at) | namespace-scoped time-sorted recall |
| `idx_evidence_memory` | evidence | (memory_id) | reverse-lookup evidence → memory |
| `idx_distilled_memory` | distilled_memories | (memory_id) | reverse-lookup distilled → memory |
| `idx_edges_src_cover` | edges | (src_id, dst_id, relation, confidence) | BFS source-node covering index |
| `idx_memory_entities_entity` | memory_entities | (entity_id) | entity → memory lookup for edge expansion |
| `idx_memory_entities_memory` | memory_entities | (memory_id) | memory → entities batch lookup |
| memories_fts | memories (FTS5) | (summary) | trigram candidate filtering |

Each `save()` writes to all relevant tables and indexes in one transaction.

Benchmark CLI (proves evidence recall beats lossy summary):

```bash
PYTHONPATH=src python -m engram_router.cli \
  --db /tmp/engram-bench.db benchmark \
  --conversation examples/long_conversation_demo.md \
  --cases examples/benchmark_questions.jsonl --text
```

Observed: summary baseline answers 0/3, EngramRouter answers 3/3 with 3/3
evidence hits — verdict `engram_better`.
