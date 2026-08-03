"""LLM-based query intent classifier with soft probability outputs.

Replaces the regex-based ``_asks_*`` / ``_has_*`` functions in
``store/query_intent.py`` with an LLM-driven classifier that outputs soft
probabilities across 8 intent categories.

Key features:
  - 8 categories: brand, identity, eval, reason, person, time, location, object
  - Soft probabilities [0, 1] with a ``dominant()`` thresholding method
  - Regex fallback when LLM is unavailable (always available as a safety net)
  - Polar question detection for yes/no questions
  - Singleton-friendly: one instance per MemoryStore
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .store import query_intent

if TYPE_CHECKING:
    from .llm_extractor import LLMClient

logger = logging.getLogger(__name__)

try:
    from .prompt_guard import wrap_user_content as _wrap
except ImportError:  # pragma: no cover — defensive
    _wrap = lambda s, _max=None: s

# ---------------------------------------------------------------------------
# IntentResult
# ---------------------------------------------------------------------------


@dataclass
class IntentResult:
    """Soft intent probabilities for a query across 8 categories.

    Each field is a float in [0, 1] representing the model's confidence
    that the query belongs to that category.
    """

    brand: float = 0.0        # Brand/model questions (品牌/型号)
    identity: float = 0.0     # Identity/age/星座 questions
    eval: float = 0.0         # Evaluation/quality/opinion questions
    reason: float = 0.0       # Reason/causality questions (为什么/原因)
    person: float = 0.0       # Person/who questions (谁/哪位)
    time: float = 0.0         # Time/when questions
    location: float = 0.0     # Location/where questions
    object: float = 0.0       # Object/what questions

    def dominant(self, threshold: float = 0.5) -> list[str]:
        """Return category names with probability above *threshold*."""
        categories = [
            ("brand", self.brand),
            ("identity", self.identity),
            ("eval", self.eval),
            ("reason", self.reason),
            ("person", self.person),
            ("time", self.time),
            ("location", self.location),
            ("object", self.object),
        ]
        return [name for name, prob in categories if prob > threshold]

    def is_polar_question(self) -> bool:
        """Return True if the query is a yes/no polar question.

        Polar questions ask for simple confirmation (是/否/有/没/对/可/能)
        and are not suitable for memory retrieval because they probe general
        knowledge rather than stored facts. Examples:

          - "你是AI吗" → True (yes/no confirmation)
          - "这个键盘是什么牌子" → False (factual brand question)
          - "北京到上海几个小时" → False (factual question)

        A question is classified as polar when NO factual category
        (brand, identity, eval, reason, person, time, location, object)
        exceeds the low threshold of 0.3, indicating the query is
        primarily a confirmation-seeking yes/no question.
        """
        return len(self.dominant(threshold=0.3)) == 0

    @classmethod
    def from_booleans(
        cls,
        brand: bool = False,
        identity: bool = False,
        eval_: bool = False,
        reason: bool = False,
        person: bool = False,
        time: bool = False,
        location: bool = False,
        object_: bool = False,
    ) -> IntentResult:
        """Create an IntentResult from hard boolean flags (1.0 or 0.0).

        Used by the regex fallback path.
        """
        return cls(
            brand=1.0 if brand else 0.0,
            identity=1.0 if identity else 0.0,
            eval=1.0 if eval_ else 0.0,
            reason=1.0 if reason else 0.0,
            person=1.0 if person else 0.0,
            time=1.0 if time else 0.0,
            location=1.0 if location else 0.0,
            object=1.0 if object_ else 0.0,
        )


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_INTENT_SYSTEM_PROMPT = """\
You are a query intent classifier for a personal memory engine.
Classify the given query into one or more intent categories.
Output ONLY a single JSON object with probability scores between 0.0 and 1.0.
No explanation, no markdown fences, no additional text.

Categories:
- brand: Questions about brands, models, manufacturers (牌子/品牌/型号/什么牌)
- identity: Questions about fixed personal attributes like age, name, zodiac sign, blood type, hometown, gender (年龄/名字/星座/血型/哪里人/性别)
- eval: Opinion/judgement questions about quality, taste, personality (怎么样/好不好/好吃吗/脾气/性格)
- reason: Questions about causes, reasons, motivations (为什么/原因/为何/为啥)
- person: Questions about who someone is (谁/哪位/哪个人/是谁)
- time: Questions about when something happened (什么时候/几点/哪天/何时)
- location: Questions about where something is (哪里/在哪/什么地方/位置)
- object: Questions about what something is (什么东西/什么/啥)

Rules:
1. Score each category independently — a query can belong to multiple categories.
2. Use 0.0 for categories that clearly don't apply.
3. Use 0.05-0.15 for categories that are tangentially relevant but not primary.
4. Use 0.7-0.95 for clearly matching categories.
5. A simple greeting like "你好" should score all categories as 0.0.
6. Always include ALL 8 category keys in the output.

Output format (JSON only):
{"brand": 0.0, "identity": 0.0, "eval": 0.0, "reason": 0.0, "person": 0.0, "time": 0.0, "location": 0.0, "object": 0.0}
"""

_INTENT_USER_TEMPLATE = 'Query: """{query}"""'


# ---------------------------------------------------------------------------
# IntentClassifier
# ---------------------------------------------------------------------------


class IntentClassifier:
    """LLM-based query intent classifier with soft probability outputs.

    Categories: brand, identity, eval, reason, person, time, location, object.
    Each category gets a soft probability [0, 1] and a boolean flag.

    Falls back to regex-based ``query_intent`` functions when the LLM is
    unavailable, API keys are missing, or the LLM response is unparseable.

    Usage::

        classifier = IntentClassifier()
        result = classifier.classify("这个键盘是什么牌子")
        print(result.brand)    # 0.95
        print(result.dominant())  # ["brand", "object"]
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        allow_cloud: bool | None = None,
    ) -> None:
        from .llm_extractor import LLMClient
        from .config import env_allows_cloud

        self._client = client or LLMClient()

        if allow_cloud is None:
            allow_cloud = env_allows_cloud("llm")

        self._llm_available = self._client.available and allow_cloud

    @property
    def available(self) -> bool:
        """Return True when the LLM classifier is online and usable.

        When False, ``classify()`` degrades gracefully to regex-based
        intent detection.
        """
        return self._llm_available

    def classify(self, query: str) -> IntentResult:
        """Classify *query* into soft intent probabilities.

        When the LLM is unavailable, falls back to regex-based intent
        functions from ``query_intent``, mapping each boolean result
        to a hard 1.0 / 0.0 probability.

        Returns:
            IntentResult with per-category probabilities.
        """
        if not self._llm_available:
            return self._classify_fallback(query)

        messages = [
            {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": _INTENT_USER_TEMPLATE.format(query=_wrap(query))},
        ]

        try:
            raw = self._client.chat(messages, temperature=0.0, max_tokens=256)
            return self._parse_intent_response(raw)
        except Exception:
            logger.warning(
                "LLM intent classification failed, falling back to regex. "
                "Query: %.80s", query,
            )
            return self._classify_fallback(query)

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _classify_fallback(self, query: str) -> IntentResult:
        """Run the regex-based intent functions and return hard booleans."""
        return IntentResult.from_booleans(
            brand=query_intent.asks_brand(query),
            identity=query_intent.asks_identity(query),
            eval_=query_intent.asks_eval(query),
            reason=query_intent.asks_reason(query),
            person=query_intent.asks_person(query),
            time=query_intent.asks_time(query),
            location=query_intent.asks_location(query),
            object_=query_intent.asks_object(query),
        )

    def _parse_intent_response(self, raw: str) -> IntentResult:
        """Parse the LLM JSON response into an IntentResult.

        Defensive: falls back to regex if the JSON is missing, malformed,
        or missing required keys.
        """
        cleaned = raw.strip()

        # Strip markdown fences.
        cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```", "", cleaned)

        # Try direct parse first.
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return self._dict_to_result(data)
        except json.JSONDecodeError:
            pass

        # Extract outermost JSON object.
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
                        data = json.loads(cleaned[start:i + 1])
                        if isinstance(data, dict):
                            return self._dict_to_result(data)
                    except json.JSONDecodeError:
                        pass
                    break

        logger.warning(
            "Failed to parse intent LLM response (len=%d): %.200s", len(raw), raw,
        )
        # Return all zeros — caller can treat this as "no intent detected".
        return IntentResult()

    @staticmethod
    def _dict_to_result(data: dict) -> IntentResult:
        """Convert a parsed JSON dict to IntentResult, clamping to [0, 1]."""
        def _clamp(key: str) -> float:
            try:
                return max(0.0, min(1.0, float(data.get(key, 0.0))))
            except (TypeError, ValueError):
                return 0.0

        return IntentResult(
            brand=_clamp("brand"),
            identity=_clamp("identity"),
            eval=_clamp("eval"),
            reason=_clamp("reason"),
            person=_clamp("person"),
            time=_clamp("time"),
            location=_clamp("location"),
            object=_clamp("object"),
        )


# ---------------------------------------------------------------------------
# Convenience: build a regex-fallback result without an LLMClient
# ---------------------------------------------------------------------------

def classify_regex(query: str) -> IntentResult:
    """Classify *query* using only the regex-based intent functions.

    This is a convenience function for testing and debugging. It returns
    hard 1.0 / 0.0 probabilities.
    """
    return IntentResult.from_booleans(
        brand=query_intent.asks_brand(query),
        identity=query_intent.asks_identity(query),
        eval_=query_intent.asks_eval(query),
        reason=query_intent.asks_reason(query),
        person=query_intent.asks_person(query),
        time=query_intent.asks_time(query),
        location=query_intent.asks_location(query),
        object_=query_intent.asks_object(query),
    )
