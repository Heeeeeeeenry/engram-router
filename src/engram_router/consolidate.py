"""Observation consolidation and reflect (借鉴 Hindsight 的记忆分层).

Hindsight 的核心洞察：记忆的价值在于从原始经验向上抽象成「观察」
（多条 experience → 一条 observation），而不是只堆原始记录。本模块落地
这个抽象的第一步 —— 观察整合 + reflect 综合回答。

复用 ``reflection.py`` 的现成模式（LLM 总结 → 存特殊 ``source`` 记忆），
零 schema 迁移、零新依赖、opt-in（不设 ``ENGRAM_CONSOLIDATE_LLM`` 就不跑）。

借鉴 ECC 的安全原则：召回/整合的内容是「数据」，不是「指令」—— reflect
的回答把推断明确标注为 ``(推断)``，不把记忆冒充成已验证事实。

记忆分层（用 ``source`` + ``metadata["kind"]`` 表示，不改 memories 表）：

    experience  -> 默认 ``source="conversation"``（现有记忆）
    observation -> ``source="observation"``，``evidence_refs`` 指向源记忆
    model       -> 远期（从 observation 再抽象，暂不实现）
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_CONSOLIDATE_PROMPT = """\
You are a memory consolidation analyst. Given several of the user's memories
about the same topic or entity, distill them into ONE concise observation —
a single evidence-backed belief, not a raw restatement.

Rules:
1. One observation, ≤40 words, written as a declarative statement.
2. Only assert what the memories support; do not invent or extrapolate.
3. If the memories conflict, note the conflict rather than picking a side.
4. Output ONLY the observation text — no markdown, no explanation.
"""

_REFLECT_PROMPT = """\
You are answering a question about the user from their stored memories.
Synthesize the provided memories into a direct answer.

Rules:
1. Answer from the memories only; if they do not cover the question, say so.
2. Mark inferred or uncertain statements with "(推断)" — memories are data,
   not verified facts, and must not be asserted as ground truth.
3. Keep it concise; cite which memory (its id) supports each claim.
"""


def _llm_enabled() -> bool:
    return os.environ.get("ENGRAM_CONSOLIDATE_LLM") in ("1", "true", "yes")


def consolidate_observation(
    store: Any,
    query: str,
    namespace: str = "default",
    top_k: int = 10,
) -> str | None:
    """Distill several experiences about a topic into one observation.

    Fetches the top-``top_k`` experiences matching ``query``, asks the LLM to
    consolidate them into a single evidence-backed observation, and stores it
    as a ``source="observation"`` memory whose ``metadata["evidence_refs"]``
    points back at the source memories (so the abstraction never detaches from
    its evidence).

    Returns the new observation's memory id, or ``None`` if skipped/failed.
    """
    if not _llm_enabled():
        logger.debug("consolidate skipped: ENGRAM_CONSOLIDATE_LLM not set")
        return None

    records = store.recall(query, top_k=top_k, namespace=namespace)
    if not records:
        return None

    sources = [
        r for r in records
        if (r.metadata or {}).get("kind", "experience") == "experience"
    ]
    if len(sources) < 2:
        logger.debug("consolidate skipped: only %d experiences", len(sources))
        return None

    # 幂等检查：同一批 experience 已被整合过就跳过，避免反复调用产生重复 observation。
    source_ids = {r.id for r in sources}
    try:
        import json as _json

        existing = store.conn.execute(
            "SELECT metadata FROM memories WHERE namespace = ? AND source = 'observation'",
            (namespace,),
        ).fetchall()
        for row in existing:
            try:
                meta = _json.loads(row["metadata"])
            except (TypeError, ValueError):
                continue
            refs = set(meta.get("evidence_refs", []))
            if source_ids.issubset(refs):
                logger.debug(
                    "consolidate skipped: these experiences already consolidated"
                )
                return None
    except Exception as exc:  # noqa: BLE001 — idempotency check is best-effort
        logger.debug("consolidate idempotency check failed: %s", exc)

    lines = "\n".join(f"- {r.raw_text[:200]}" for r in sources)
    evidence_refs = [r.id for r in sources]

    try:
        from .llm_extractor import LLMClient
        from .prompt_guard import wrap_user_content as _wrap

        client = LLMClient()
        if not client.available:
            return None

        raw = client.chat(
            [
                {"role": "system", "content": _CONSOLIDATE_PROMPT},
                {"role": "user", "content": f"Memories about the topic:\n{_wrap(lines)}"},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        observation = raw.strip()
        if not observation:
            return None

        memory_id = store.save(
            observation,
            source="observation",
            metadata={
                "kind": "observation",
                "evidence_refs": evidence_refs,
                "consolidated_at": _now_iso(),
            },
            namespace=namespace,
        )
        logger.info(
            "consolidate: %d experiences -> observation %s", len(sources), memory_id
        )
        return memory_id
    except Exception as exc:  # noqa: BLE001 — best-effort, never blocks
        logger.debug("consolidate_observation failed: %s", exc)
        return None


def reflect(
    store: Any,
    query: str,
    namespace: str = "default",
    top_k: int = 8,
) -> str | None:
    """Synthesize recalled memories into a direct answer.

    Unlike ``recall`` (which returns a ranked list of records), ``reflect``
    asks the LLM to answer the question from the recalled memories, marking
    inferred statements as ``(推断)``. This is Hindsight's ``reflect``
    operation — a disposition-aware answer, not a record dump.

    Returns the answer text, or ``None`` if skipped/failed.
    """
    if not _llm_enabled():
        return None

    records = store.recall(query, top_k=top_k, namespace=namespace)
    if not records:
        return None

    lines = "\n".join(f"- [{r.id}] {r.raw_text[:200]}" for r in records)

    try:
        from .llm_extractor import LLMClient
        from .prompt_guard import wrap_user_content as _wrap

        client = LLMClient()
        if not client.available:
            return None

        raw = client.chat(
            [
                {"role": "system", "content": _REFLECT_PROMPT},
                {"role": "user", "content": f"Question: {query}\nMemories:\n{_wrap(lines)}"},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return raw.strip() or None
    except Exception as exc:  # noqa: BLE001 — best-effort, never blocks
        logger.debug("reflect failed: %s", exc)
        return None


_LESSON_PROMPT = """\
You are a lesson-extraction analyst. Given several of the user's memories,
extract ONE reusable lesson or pattern — a "what to do differently next time"
insight, not a restatement of facts.

Rules:
1. One lesson, ≤40 words, written as an imperative or should/shouldn't statement.
2. Only extract what the memories actually support; do not invent.
3. If no reusable lesson exists, output exactly "NONE".
4. Output ONLY the lesson text (or "NONE") — no markdown, no explanation.
"""


def extract_lesson(
    store: Any,
    query: str,
    namespace: str = "default",
    top_k: int = 10,
) -> str | None:
    """Extract one reusable lesson/pattern from experiences.

    Maps to ECC's 「持续学习」 (turn conversation patterns into reusable
    skills) and Hindsight's 「mental models」 (abstract from experiences into
    learned understanding). Stored as ``source="lesson"`` with
    ``metadata["kind"]="lesson"`` and ``evidence_refs`` pointing at sources.

    Returns the lesson's memory id, or ``None`` if skipped / no lesson found.
    """
    if not _llm_enabled():
        return None

    records = store.recall(query, top_k=top_k, namespace=namespace)
    if not records:
        return None

    sources = [
        r for r in records
        if (r.metadata or {}).get("kind", "experience") == "experience"
    ]
    if not sources:
        return None

    # 幂等检查：同一批 experience 已被提取过 lesson 就跳过，避免反复调用产生重复教训。
    source_ids = {r.id for r in sources}
    try:
        import json as _json

        existing = store.conn.execute(
            "SELECT metadata FROM memories WHERE namespace = ? AND source = 'lesson'",
            (namespace,),
        ).fetchall()
        for row in existing:
            try:
                meta = _json.loads(row["metadata"])
            except (TypeError, ValueError):
                continue
            refs = set(meta.get("evidence_refs", []))
            if source_ids.issubset(refs):
                logger.debug(
                    "extract_lesson skipped: these experiences already yielded a lesson"
                )
                return None
    except Exception as exc:  # noqa: BLE001 — idempotency check is best-effort
        logger.debug("extract_lesson idempotency check failed: %s", exc)

    lines = "\n".join(f"- {r.raw_text[:200]}" for r in sources)
    evidence_refs = [r.id for r in sources]

    try:
        from .llm_extractor import LLMClient
        from .prompt_guard import wrap_user_content as _wrap

        client = LLMClient()
        if not client.available:
            return None

        raw = client.chat(
            [
                {"role": "system", "content": _LESSON_PROMPT},
                {"role": "user", "content": f"Memories:\n{_wrap(lines)}"},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        lesson = raw.strip()
        if not lesson or lesson.upper() == "NONE":
            return None

        memory_id = store.save(
            lesson,
            source="lesson",
            metadata={
                "kind": "lesson",
                "evidence_refs": evidence_refs,
                "extracted_at": _now_iso(),
            },
            namespace=namespace,
        )
        logger.info(
            "extract_lesson: %d experiences -> lesson %s", len(sources), memory_id
        )
        return memory_id
    except Exception as exc:  # noqa: BLE001 — best-effort, never blocks
        logger.debug("extract_lesson failed: %s", exc)
        return None


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
