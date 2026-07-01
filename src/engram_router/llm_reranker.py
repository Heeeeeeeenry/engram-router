"""LLM-based semantic re-ranker for EngramRouter.

Optional enhancement: after keyword + entity recall, use an LLM to
semantically re-rank the top candidates. This bridges the gap between
rule-based retrieval and true semantic understanding.

Usage:
    from engram_router.llm_reranker import LLMReranker
    store = MemoryStore(..., reranker=LLMReranker(api_key="..."))
    # recall now uses LLM re-ranking for better precision

Requires: OPENAI_API_KEY or compatible env var for the LLM endpoint.
Fallback: if LLM is unavailable, returns original ranking unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class LLMReranker:
    """Re-rank recall candidates using an LLM for semantic relevance.

    Sends the query and up to N candidate texts to an OpenAI-compatible
    LLM endpoint, asking it to score each candidate's relevance. Merges
    the LLM scores with the original rule-based scores.

    Config via env:
        ENGRAM_LLM_BASE_URL  — override API base URL
        ENGRAM_LLM_MODEL     — model name (default: gpt-4o-mini)
        ENGRAM_LLM_API_KEY   — API key (falls back to OPENAI_API_KEY)
        ENGRAM_LLM_MAX_CONCURRENT — max concurrent LLM calls (default: 3)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_candidates: int = 10,
        weight_llm: float = 0.4,
        weight_rule: float = 0.6,
        min_score_threshold: float = 0.2,
    ):
        self.api_key = api_key or os.environ.get("ENGRAM_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("ENGRAM_LLM_BASE_URL")
        self.model = model or os.environ.get("ENGRAM_LLM_MODEL", "gpt-4o-mini")
        self.max_candidates = max_candidates
        self.weight_llm = weight_llm
        self.weight_rule = weight_rule
        self.min_score_threshold = min_score_threshold
        self._available = bool(self.api_key)

    @property
    def available(self) -> bool:
        return self._available

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Re-rank candidates using LLM semantic scoring.

        Args:
            query: The original search query
            candidates: List of {text, score, id, ...} from rule-based recall

        Returns:
            Re-ranked candidates with updated 'llm_score' and merged 'score'
        """
        if not self.available or not candidates:
            return candidates

        top_n = candidates[: self.max_candidates]
        if len(top_n) <= 1:
            return candidates

        try:
            llm_scores = self._batch_score(query, top_n)
        except Exception as exc:
            logger.warning("LLM reranker failed, using rule-based scores: %s", exc)
            return candidates

        # Merge LLM scores with rule-based scores
        for cand, llm_score in zip(top_n, llm_scores):
            rule_score = cand.get("score", 0)
            # Normalize to 0-1 range for merging
            rule_norm = min(rule_score / 5.0, 1.0)  # rule scores typically 0-5
            merged = (
                self.weight_llm * float(llm_score)
                + self.weight_rule * rule_norm
            )
            cand["llm_score"] = round(float(llm_score), 3)
            cand["score"] = round(merged, 4)

        # Re-sort all candidates by merged score
        candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
        return candidates

    def _batch_score(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[float]:
        """Score candidates via LLM in a single batch call."""
        import urllib.request
        import urllib.error

        texts = []
        for i, c in enumerate(candidates):
            texts.append(f"[{i}] {c.get('text', '')[:200]}")

        prompt = f"""Score how relevant each candidate memory is to the query.

Query: {query[:200]}

Candidates:
{chr(10).join(texts)}

Return a JSON array of scores from 0.0 to 1.0, one per candidate.
0.0 = completely irrelevant, 1.0 = perfectly answers the query.
Only return the JSON array, no explanation.

Example: [0.9, 0.3, 0.0, 0.7]"""

        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a relevance scorer. Return only JSON arrays."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 200,
        }).encode("utf-8")

        url = (self.base_url or "https://api.openai.com/v1") + "/chat/completions"
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"].strip()
            # Extract JSON array from response
            if "[" in content and "]" in content:
                start = content.index("[")
                end = content.rindex("]") + 1
                scores = json.loads(content[start:end])
            else:
                scores = json.loads(content)
            # Ensure correct length
            while len(scores) < len(candidates):
                scores.append(0.0)
            return scores[: len(candidates)]
        except Exception as exc:
            logger.warning("LLM scoring failed: %s", exc)
            raise
