"""Tests for L0.2 LLM-as-judge upgrades in lme_judge.py."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


# ---------------------------------------------------------------------------
# faithfulness_judge tests
# ---------------------------------------------------------------------------

class TestFaithfulnessJudge:
    """Test faithfulness_judge function."""

    def test_faithful_answer(self):
        """A fully faithful answer should pass."""
        from lme_judge import faithfulness_judge

        query = "What is the user's cat's name?"
        memories = [
            "The user has a cat named Whiskers.",
            "Whiskers is a gray tabby.",
        ]
        answer = "The user's cat is named Whiskers."

        mock_client = _make_mock_client('{"faithful": true, "hallucinated": false, "hallucination_rate": 0.0, "unsupported_claims": [], "reason": "All claims match the memories."}')
        result = faithfulness_judge(answer, memories, query, client=mock_client)

        assert result["faithful"] is True
        assert result["hallucinated"] is False
        assert result["hallucination_rate"] == 0.0
        assert result["unsupported_claims"] == []

    def test_hallucinated_answer(self):
        """An answer with invented facts should be flagged."""
        from lme_judge import faithfulness_judge

        query = "What is the user's cat's name?"
        memories = [
            "The user has a cat named Whiskers.",
            "Whiskers is a gray tabby.",
        ]
        answer = "The user's cat is named Mittens, and it is 5 years old."

        mock_client = _make_mock_client(
            '{"faithful": false, "hallucinated": true, "hallucination_rate": 1.0, '
            '"unsupported_claims": ["Cat is named Mittens", "Cat is 5 years old"], '
            '"reason": "Answer invents a different cat name and age."}'
        )
        result = faithfulness_judge(answer, memories, query, client=mock_client)

        assert result["faithful"] is False
        assert result["hallucinated"] is True
        assert result["hallucination_rate"] == 1.0
        assert len(result["unsupported_claims"]) == 2

    def test_partial_faithfulness(self):
        """An answer with some supported and some unsupported claims."""
        from lme_judge import faithfulness_judge

        query = "What do we know about the user's cat?"
        memories = [
            "The user has a cat named Whiskers.",
        ]
        answer = "The cat is named Whiskers and the user also has a dog named Rex."

        mock_client = _make_mock_client(
            '{"faithful": false, "hallucinated": true, "hallucination_rate": 0.5, '
            '"unsupported_claims": ["User also has a dog named Rex"], '
            '"reason": "Partially faithful: cat name correct, dog is invented."}'
        )
        result = faithfulness_judge(answer, memories, query, client=mock_client)

        assert result["faithful"] is False
        assert result["hallucinated"] is True
        assert result["hallucination_rate"] == 0.5
        assert len(result["unsupported_claims"]) == 1

    def test_unparseable_output(self):
        """When judge returns unparseable output, should flag as hallucinated."""
        from lme_judge import faithfulness_judge

        query = "What is the cat's name?"
        memories = ["The cat is named Whiskers."]
        answer = "Whiskers."

        mock_client = _make_mock_client("nonsense output that is not JSON")
        result = faithfulness_judge(answer, memories, query, client=mock_client)

        assert result["faithful"] is False
        assert result["hallucinated"] is True
        assert result["hallucination_rate"] == 1.0

    def test_no_client_fallback(self):
        """When no LLM client is available, heuristic fallback should work."""
        from lme_judge import faithfulness_judge

        query = "What is the cat's name?"
        memories = ["The user has a cat named Whiskers."]
        # Answer matches memory.
        answer = "The cat is named Whiskers."

        # Patch _make_client to raise RuntimeError to force fallback path.
        with patch("lme_judge._make_client", side_effect=RuntimeError("no key")):
            result = faithfulness_judge(answer, memories, query, client=None)

        # Heuristic should detect the overlap with "Whiskers".
        assert result["faithful"] is True
        assert result["hallucinated"] is False
        assert result["hallucination_rate"] == 0.0

    def test_no_client_fallback_hallucinated(self):
        """Heuristic fallback should detect hallucination when words are absent."""
        from lme_judge import faithfulness_judge

        query = "What is the cat's name?"
        memories = ["The user has a cat named Whiskers."]
        answer = "The individual purchased an automobile in Constantinople."

        with patch("lme_judge._make_client", side_effect=RuntimeError("no key")):
            result = faithfulness_judge(answer, memories, query, client=None)

        # None of the 4+-char words (individual, purchased, automobile, Constantinople)
        # appear in "the user has a cat named whiskers."
        assert result["hallucinated"] is True
        assert result["hallucination_rate"] > 0.0


# ---------------------------------------------------------------------------
# cross_encoder_ground_truth tests
# ---------------------------------------------------------------------------

class TestCrossEncoderGroundTruth:
    """Test cross_encoder_ground_truth function."""

    def test_empty_memories(self):
        """Should return empty list for empty input."""
        from lme_judge import cross_encoder_ground_truth

        scores = cross_encoder_ground_truth("query", [])
        assert scores == []

    def test_fallback_relevance_ordering(self):
        """Fallback (word overlap) should score relevant memories higher."""
        from lme_judge import _fallback_relevance

        question = "What is the cat's name?"
        memories = [
            "The weather is nice today.",       # no overlap
            "The user went to the store.",       # no overlap
            "The cat is named Whiskers.",        # high overlap
            "I had lunch at noon.",              # no overlap
            "Whiskers is a gray tabby cat.",     # moderate overlap
        ]
        scores = _fallback_relevance(question, memories)
        assert len(scores) == 5
        # The high-overlap memory should score highest.
        idx3 = scores[2]  # "The cat is named Whiskers."
        idx5 = scores[4]  # "Whiskers is a gray tabby cat."
        assert max(scores) in (idx3, idx5), f"Expected high overlap to win, got {scores}"

    def test_fallback_no_overlap(self):
        """When no words overlap, all scores should be 0."""
        from lme_judge import _fallback_relevance

        # Use disjoint vocabulary to ensure zero overlap — avoid common
        # stopwords like "the" that accidentally bridge question and memory.
        question = "xylophone zephyr quantum rhombus"
        memories = [
            "The weather is nice today.",
            "The user went to the store.",
        ]
        scores = _fallback_relevance(question, memories)
        assert len(scores) == 2
        assert all(s == 0.0 for s in scores)

    def test_cross_encoder_import_fallback(self):
        """Verify graceful fallback when sentence_transformers is unavailable."""
        from lme_judge import _fallback_relevance

        # Directly test the fallback heuristic (used when sentence_transformers
        # is not installed or the CrossEncoder fails to load).
        question = "test question"
        memories = ["test memory with question"]
        scores = _fallback_relevance(question, memories)
        assert len(scores) == 1
        assert isinstance(scores[0], float)
        # "question" is a 4+-char word present in both.
        assert scores[0] > 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(json_response: str):
    """Return a mock LLMClient that returns *json_response* for any chat call."""
    client = MagicMock()
    client.chat.return_value = json_response
    return client
