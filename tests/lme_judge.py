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
# Internals
# ─────────────────────────────────────────────────────────────────────────

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
