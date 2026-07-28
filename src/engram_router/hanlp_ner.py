"""HanLP NER augmentation for EngramRouter entity extraction.

This module supplements the rule-based ``entities.extract_entities()`` with
HanLP's deep-learning NER (fine-grained perceptron). It runs lazily — the
first call triggers HanLP import + model warm-up (~1-3 seconds on CPU).
After that, ``hanlp_ner()`` is near-instant.

HanLP's NER tags relevant to engram:
  PERSON, ORGANIZATION, LOCATION, PRODUCT, EVENT, DATE, TIME, MONEY, PERCENT

Where it fits
-------------
``extract_entities()`` remains the single source of truth. This module provides
``enrich_entities()`` which takes rule-based entities + raw text, runs HanLP
NER, and merges detection that the rules missed.

Usage::

    from engram_router.hanlp_ner import enrich_entities
    entities = extract_entities(text)         # rule-based
    entities = enrich_entities(text, entities) # HanLP-augmented

Design notes
------------
- **NOT a replacement**: rules are still the primary extractor. HanLP fills
  gaps — especially pronoun resolution failures ("她" → "小李"), multi-word
  names ("北京协和医院"), and location/organisation detection.
- ENGRAM_SKIP_HANLP=1 disables at runtime.
- ``hanlp_ner()`` singleton ensures HanLP is loaded once across all calls.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_HANLP: Any = None  # singleton
_LOADED = False
_LOAD_ERROR: str | None = None

# HanLP NER tag → engram entity kind
_TAG_TO_KIND: dict[str, str] = {
    "PERSON": "person",
    "ORGANIZATION": "company",
    "LOCATION": "location",
    "PRODUCT": "object",
    "EVENT": "event",
    "DATE": "time",
    "TIME": "time",
    "MONEY": "attribute",
    "PERCENT": "attribute",
}


def _load_hanlp() -> Any | None:
    """Lazy-load HanLP once. Returns the NER pipeline or None."""
    global _HANLP, _LOADED, _LOAD_ERROR
    if _LOADED:
        return _HANLP
    _LOADED = True
    if os.environ.get("ENGRAM_SKIP_HANLP") == "1":
        logger.debug("HanLP skipped (ENGRAM_SKIP_HANLP=1)")
        return None
    try:
        import hanlp
        _HANLP = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)
        logger.info("HanLP NER loaded (MSRA_NER_ELECTRA_SMALL_ZH)")
        return _HANLP
    except ImportError:
        _LOAD_ERROR = "hanlp not installed. pip install hanlp"
    except Exception as exc:
        _LOAD_ERROR = f"HanLP load failed: {exc}"
    logger.debug("HanLP unavailable: %s", _LOAD_ERROR)
    return None


def hanlp_ner(text: str) -> list[dict[str, Any]]:
    """Run HanLP NER on *text*. Returns entity dicts with kind mapping.

    Returns ``[]`` if HanLP is unavailable or finds nothing.
    """
    nlp = _load_hanlp()
    if nlp is None:
        return []
    try:
        result = nlp(text)
    except Exception as exc:
        logger.debug("HanLP NER call failed: %s", exc)
        return []

    entities: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for token_info in result:
        word = token_info.get("word", token_info[0]) if isinstance(token_info, dict) else token_info[0]
        tag = token_info.get("ner", token_info[1]) if isinstance(token_info, dict) else token_info[1]
        kind = _TAG_TO_KIND.get(tag)
        if kind is None or not word or len(word) < 2:
            continue
        key = (word, kind)
        if key in seen:
            continue
        seen.add(key)
        entities.append({
            "name": word,
            "kind": kind,
            "evidence": word,
            "source": "hanlp",
        })
    return entities


def enrich_entities(text: str, existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge HanLP NER entities with existing rule-based entities.

    HanLP entities are appended only when they introduce a *new* (name, kind)
    pair not already present in *existing*. This avoids duplicate entries
    while letting HanLP fill gaps (especially person names, locations, and
    organisations that the regex patterns miss).
    """
    existing_keys: set[tuple[str, str]] = {
        (e.get("name", ""), e.get("kind", "")) for e in existing
    }
    hanlp_ents = hanlp_ner(text)
    for he in hanlp_ents:
        key = (he["name"], he["kind"])
        if key not in existing_keys:
            existing.append(he)
            existing_keys.add(key)
    return existing
