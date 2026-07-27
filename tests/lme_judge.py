"""LongMemEval-style LLM judge for answer correctness.

engram-router's local ``LLMClient`` (an OpenAI-compatible HTTP shim over
DeepSeek / Baidu OneAPI) is reused for both roles:

  1. **Answerer** — given the retrieved top-k memories as context, produce a
     free-text answer to the natural-language question. This is the
     "generation" step LongMemEval's paper measures downstream.
  2. **Judge** — compare the produced answer against the reference answer
     and emit a strict 0/1 correctness verdict + brief reason. Prompt shape
     mirrors the LongMemEval reference judge (task-specific, tolerates
     paraphrase / synonym / number-format shifts but rejects hallucinated
     content or refusals).

Both prompts are pinned as module constants so any future scoring drift is
attributable to a single diff.

Deliberate non-goals:
  - No caching. Retrieval-only latency is dominated by cross-encoder
    load; judge is called once per case and only 100-500 items are ever
    scored. Adding caching would introduce a source of stale answers when
    prompt evolves.
  - No streaming. Judge JSON is small (~50 tokens); non-streaming is
    simpler and equally fast for this scale.
  - No self-consistency (multi-vote). A single judgement per case matches
    what LongMemEval's own harness does; multiplying costs 5× for marginal
    stability improvement.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Prompts (pinned; version them here when you touch them)
# ─────────────────────────────────────────────────────────────────────────

_ANSWER_SYSTEM = """\
You are an assistant answering questions about a user's own recorded
memories. You are given a numbered set of memory snippets that were
retrieved for the question, together with the question itself.

Rules:
1. Answer using ONLY the information contained in the memory snippets.
2. If the snippets do not contain the answer, reply exactly with
   "I don't know based on my memories."
3. Do NOT invent details, dates, or names. Do NOT hedge with "possibly" if
   the memory is explicit.
4. Give a short, direct answer (one sentence when possible, at most 30
   words). Do not restate the question.
5. CONFLICT RESOLUTION: When two or more memory snippets provide
   conflicting information (e.g. different numbers, dates, or values for
   the same fact), prefer the MOST RECENT memory. The time-stamp (in UTC)
   shown after each snippet tells you when it was recorded — later dates
   are more reliable.  Give a single, definite answer — do NOT say
   "both X and Y" or hedge with "the user mentioned both".  Pick the
   latest value and state it as the answer.
"""

_ANSWER_USER_TEMPLATE = """\
Memory snippets:
{memories}

Question: {question}

Answer:"""


_JUDGE_SYSTEM = """\
You are grading whether a candidate answer is semantically correct
compared with a reference answer for a personal-memory retrieval task.

Rules:
1. Output ONLY a JSON object with two fields: {{\"correct\": 0 or 1,
   \"reason\": \"<one short sentence>\"}}. No prose outside the JSON.
2. Consider these EQUIVALENT to correct:
   - Paraphrases and synonyms conveying the same fact.
   - Number formatting shifts (e.g. \"14 days\" ↔ \"two weeks\",
     \"$350000\" ↔ \"350k\").
   - The candidate includes the reference and adds correct extra detail.
3. Consider these INCORRECT:
   - Missing the specific entity, number, or fact the question asked for.
   - Hallucinating information not in the reference.
   - \"I don't know\" or refusals, even when the reference is trivial.
4. When the reference explicitly names alternatives (\"14 days. 15 days
   also acceptable\"), accept any of them.
"""

_JUDGE_USER_TEMPLATE = """\
Question: {question}
Reference answer: {reference}
Candidate answer: {candidate}

Respond with JSON only."""


# ─────────────────────────────────────────────────────────────────────────
# Small dataclasses for structured output
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class AnswerResult:
    """Bundle for a single answer-generation call."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class JudgeResult:
    """Bundle for a single judge call."""

    correct: int  # 0 or 1
    reason: str
    raw: str = ""  # the judge's raw response, for post-hoc debugging


# ─────────────────────────────────────────────────────────────────────────
# Client construction
# ─────────────────────────────────────────────────────────────────────────

def _make_client() -> Any:
    """Build a shared LLMClient. Raises if no API key is configured — callers
    should catch and skip the judge branch gracefully."""
    from engram_router.llm_extractor import LLMClient

    client = LLMClient()
    if not client.available:
        raise RuntimeError(
            "LLMClient is not available: set DEEPSEEK_API_KEY or "
            "ENGRAM_LLM_API_KEY and configure ENGRAM_ALLOW_CLOUD_LLM=1"
        )
    return client


# ─────────────────────────────────────────────────────────────────────────
# Public API: generate_answer + judge_correctness
# ─────────────────────────────────────────────────────────────────────────

def generate_answer(
    client: Any,
    question: str,
    memory_texts: list[str],
    *,
    memory_timestamps: list[str] | None = None,
    max_tokens: int = 2000,
    temperature: float = 0.0,
) -> AnswerResult:
    """Ask the LLM to answer ``question`` given the retrieved memory texts.

    ``memory_texts`` is the ranked top-k (already truncated by the caller);
    we number them 1..N in the prompt so the model can cite by index if it
    wants (we don't parse citations back — the answer is free-text).

    When ``memory_timestamps`` is provided (same length as ``memory_texts``),
    each snippet is annotated with ``[recorded: ISO-8601]`` so the model can
    resolve conflicts by preferring the most recent fact.  This is the
    knowledge-update fix from OPTIMIZATION_ROADMAP.md step L.

    ``max_tokens=2000`` because DeepSeek-family models spend most of their
    budget on internal reasoning; the visible answer is short but caps below
    ~1000 truncate it. Lower ceilings silently produce empty answers.
    """
    if not memory_texts:
        return AnswerResult(text="I don't know based on my memories.")

    if memory_timestamps and len(memory_timestamps) == len(memory_texts):
        numbered = "\n".join(
            f"[{i + 1}] [recorded: {ts}] {t}"
            for i, (t, ts) in enumerate(zip(memory_texts, memory_timestamps))
        )
    else:
        numbered = "\n".join(
            f"[{i + 1}] {t}" for i, t in enumerate(memory_texts)
        )

    messages = [
        {"role": "system", "content": _ANSWER_SYSTEM},
        {"role": "user", "content": _ANSWER_USER_TEMPLATE.format(
            memories=numbered, question=question)},
    ]
    raw = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
    return AnswerResult(text=(raw or "").strip())


def judge_correctness(
    client: Any,
    question: str,
    reference: str,
    candidate: str,
    *,
    max_tokens: int = 2000,
    temperature: float = 0.0,
) -> JudgeResult:
    """Score whether ``candidate`` is semantically correct vs ``reference``.

    Failure modes are conservative: any unparseable output is scored 0 with
    the raw response preserved in ``reason`` so the caller can spot-check.

    ``max_tokens`` is generous (2000) because the DeepSeek-style backing
    model spends most of its budget on internal reasoning; the visible JSON
    is small but shrinking the ceiling truncates the answer mid-string and
    breaks the parser. A too-low ceiling silently poisons scores.
    """
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": _JUDGE_USER_TEMPLATE.format(
            question=question, reference=reference, candidate=candidate)},
    ]
    raw = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
    raw_str = (raw or "").strip()
    parsed = _parse_judge_json(raw_str)
    if parsed is None:
        return JudgeResult(correct=0, reason=f"unparseable: {raw_str[:80]}",
                           raw=raw_str)
    correct = 1 if int(parsed.get("correct", 0)) == 1 else 0
    reason = str(parsed.get("reason", ""))[:200]
    return JudgeResult(correct=correct, reason=reason, raw=raw_str)


# ─────────────────────────────────────────────────────────────────────────
# Faithfulness judge (L0.2 LLM-as-judge upgrade)
# ─────────────────────────────────────────────────────────────────────────

_FAITHFULNESS_SYSTEM = """\
You are a fact-checking judge for a personal memory retrieval system.
Your job is to determine whether a generated answer is faithful to the
retrieved memory snippets.

Rules:
1. Output ONLY a JSON object with fields: {"faithful": true or false,
   "hallucinated": true or false, "hallucination_rate": <float 0-1>,
   "unsupported_claims": [<string, ...>], "reason": "<one short sentence>"}.
   No prose outside the JSON.
2. For each factual claim in the answer, check if it can be confirmed by
   at least one of the provided memory snippets.
3. faithful=true means EVERY factual claim in the answer is supported by
   the memory snippets. faithful=false means at least one claim is unsupported.
4. hallucinated=true means the answer contains claims that directly contradict
   the memory snippets or invent facts not present anywhere.
5. hallucination_rate should be the fraction of total claims that are
   unsupported (0.0 = perfect, 1.0 = entirely fabricated).
6. For "I don't know" or refusal answers, treat them as faithful=true if
   indeed the memories lack the answer, and faithful=false if the memories
   DO contain the answer.
7. Be precise: paraphrase is okay, but changing numbers, dates, names,
   or adding new entities is not.
"""

_FAITHFULNESS_USER_TEMPLATE = """\
Question: {question}

Retrieved memory snippets:
{memories}

Generated answer: {answer}

Respond with JSON only."""


def faithfulness_judge(
    answer: str,
    memories: list[str],
    query: str,
    client: Any = None,
) -> dict:
    """Judge whether the answer is faithful to the retrieved memories.

    Detects:
    - Hallucination: answer contains facts not in memories
    - Faithful: all claims are supported by at least one memory
    - Partial: some claims supported, some not

    Args:
        answer: The generated answer text.
        memories: List of retrieved memory snippet texts.
        query: The original question.
        client: An LLMClient instance. If None, tries to build one from env.

    Returns:
        {
            "faithful": bool,
            "hallucinated": bool,
            "hallucination_rate": float,  # 0.0 = fully faithful
            "unsupported_claims": list[str],
            "reason": str,
        }
    """
    if client is None:
        try:
            client = _make_client()
        except RuntimeError:
            # Fallback: deterministic heuristic without LLM.
            return _heuristic_faithfulness(answer, memories, query)

    memories_block = "\n".join(
        f"[{i + 1}] {m}" for i, m in enumerate(memories)
    ) if memories else "(no memories retrieved)"

    messages = [
        {"role": "system", "content": _FAITHFULNESS_SYSTEM},
        {"role": "user", "content": _FAITHFULNESS_USER_TEMPLATE.format(
            question=query, memories=memories_block, answer=answer)},
    ]
    raw = client.chat(messages, temperature=0.0, max_tokens=2000)
    raw_str = (raw or "").strip()
    parsed = _parse_faithfulness_json(raw_str)

    if parsed is None:
        return {
            "faithful": False,
            "hallucinated": True,
            "hallucination_rate": 1.0,
            "unsupported_claims": ["[unparseable judge output]"],
            "reason": f"unparseable: {raw_str[:100]}",
        }

    return {
        "faithful": bool(parsed.get("faithful", False)),
        "hallucinated": bool(parsed.get("hallucinated", False)),
        "hallucination_rate": float(parsed.get("hallucination_rate", 0.0)),
        "unsupported_claims": list(parsed.get("unsupported_claims", [])),
        "reason": str(parsed.get("reason", ""))[:200],
    }


def _heuristic_faithfulness(
    answer: str,
    memories: list[str],
    query: str,
) -> dict:
    """Simple string-overlap heuristic when no LLM client is available.

    Checks whether at least some key content from the answer appears verbatim
    or near-verbatim in the memory snippets.  This is intentionally lenient
    — it errs toward faithful=true — to avoid false-positive hallucination
    flags when running offline.
    """
    combined = " ".join(memories).lower()
    answer_lower = answer.lower()

    # "I don't know" is always faithful.
    if "i don't know" in answer_lower or "don't know based" in answer_lower:
        return {
            "faithful": True,
            "hallucinated": False,
            "hallucination_rate": 0.0,
            "unsupported_claims": [],
            "reason": "Answer is a refusal / I-don't-know.",
        }

    # Split answer into short segments (rough claims).
    import re
    segments = re.split(r"[,.!?;:，。！？；：\n]", answer)
    segments = [s.strip() for s in segments if len(s.strip()) > 10]
    if not segments:
        return {
            "faithful": True,
            "hallucinated": False,
            "hallucination_rate": 0.0,
            "unsupported_claims": [],
            "reason": "Answer too short to evaluate heuristically.",
        }

    unsupported: list[str] = []
    for seg in segments:
        seg_lower = seg.lower()
        # Check if any keyword from the segment appears in memories.
        words = re.findall(r'\b\w{4,}\b', seg_lower)
        if not words:
            continue
        supported = any(w in combined for w in words)
        if not supported:
            unsupported.append(seg)

    rate = len(unsupported) / len(segments) if segments else 0.0
    fully_faithful = len(unsupported) == 0

    return {
        "faithful": fully_faithful,
        "hallucinated": not fully_faithful and rate >= 0.5,
        "hallucination_rate": round(rate, 4),
        "unsupported_claims": unsupported[:5],
        "reason": (
            f"Heuristic: {len(unsupported)}/{len(segments)} segments unsupported "
            f"(no LLM available)"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────
# Cross-encoder ground-truth (L0.2 automated relevance labeling)
# ─────────────────────────────────────────────────────────────────────────

def cross_encoder_ground_truth(
    question: str,
    memories: list[str],
    model_name: str = "BAAI/bge-reranker-v2-m3",
) -> list[float]:
    """Score each memory's relevance to the question using a cross-encoder.

    This is an automated relevance labeler — faster than human annotation
    and more objective than LLM-as-judge. Each memory gets a score in [0, 1].

    The cross-encoder evaluates each (query, memory) pair independently,
    producing a relevance score. Results are returned in the same order as
    the input ``memories`` list.

    Graceful degradation: if ``sentence-transformers`` is not installed,
    falls back to a simple word-overlap heuristic and logs a warning.

    Args:
        question: The query/question text.
        memories: List of memory texts to score.
        model_name: HuggingFace cross-encoder model name.

    Returns:
        List of float scores, one per memory, in the same order as input.
    """
    if not memories:
        return []

    # Try to use sentence-transformers CrossEncoder.
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        logging.warning(
            "sentence-transformers not installed; "
            "cross_encoder_ground_truth falling back to word-overlap heuristic."
        )
        return _fallback_relevance(question, memories)

    try:
        model = CrossEncoder(model_name)
        pairs = [(question, mem) for mem in memories]
        scores = model.predict(pairs)
        # Convert to float list; normalize if model returns logits.
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        return [float(s) for s in scores]
    except Exception as exc:
        logging.warning(
            "Cross-encoder failed (%s); falling back to word-overlap.", exc
        )
        return _fallback_relevance(question, memories)


def _fallback_relevance(question: str, memories: list[str]) -> list[float]:
    """Word-overlap relevance heuristic (no external deps)."""
    import re

    q_words = set(re.findall(r'\b\w{3,}\b', question.lower()))
    if not q_words:
        return [0.0] * len(memories)

    scores: list[float] = []
    for mem in memories:
        m_words = set(re.findall(r'\b\w{3,}\b', mem.lower()))
        if not m_words:
            scores.append(0.0)
            continue
        overlap = len(q_words & m_words)
        # Jaccard-like but scaled by question coverage.
        score = overlap / len(q_words)
        scores.append(round(score, 4))
    return scores

def _parse_faithfulness_json(raw: str) -> dict[str, Any] | None:
    """Parse the faithfulness judge's JSON verdict.

    Like _parse_judge_json but looks for ``faithful`` key instead of ``correct``.
    """
    if not raw:
        return None
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"\s*```", "", cleaned).strip()

    # Direct parse first.
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "faithful" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    # Balanced-brace scan.
    depth = 0
    start = -1
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(cleaned[start:i + 1])
                    if isinstance(obj, dict) and "faithful" in obj:
                        return obj
                except json.JSONDecodeError:
                    pass
                start = -1
    return None


def _parse_judge_json(raw: str) -> dict[str, Any] | None:
    """Parse the judge's JSON verdict.

    Robust to markdown fences and leading prose (some models like to prefix
    with "Here is my answer:" even when told not to). Returns None when no
    object with a ``correct`` field can be located — the caller treats that
    as a "score 0, keep the raw string for debugging" case.
    """
    if not raw:
        return None
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"\s*```", "", cleaned).strip()

    # Direct parse first.
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "correct" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    # Balanced-brace scan to find the first well-formed object.
    depth = 0
    start = -1
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(cleaned[start:i + 1])
                    if isinstance(obj, dict) and "correct" in obj:
                        return obj
                except json.JSONDecodeError:
                    pass
                start = -1
    return None
