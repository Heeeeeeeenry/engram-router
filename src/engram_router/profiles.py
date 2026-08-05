"""Pre-defined RecallWeights profiles for different scenarios (L4).

Usage::

    from engram_router.profiles import PRODUCTION, DEV, EXPERIMENT_A
    from engram_router.store import RecallWeights
    store = MemoryStore(weights=RecallWeights(**DEV))

    # Or load from a YAML file:
    from engram_router.profiles import load_profile
    weights = load_profile("production")

Profiles can also be overridden by ``~/.engram/profiles.yaml``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# ── Built-in profiles ────────────────────────────────────────────────────

PRODUCTION: dict[str, Any] = {
    "ce_enabled": True,
    "ce_weight": 0.6,
    "ce_max_candidates": 20,
    "hyde_enabled": False,
    "multi_query_enabled": False,
    "intent_classifier_enabled": False,
    "recall_deadline_ms": 5000,
    "sampling_rate": 0.01,
}

DEV: dict[str, Any] = {
    "ce_enabled": False,
    "ce_weight": 0.4,
    "ce_max_candidates": 10,
    "hyde_enabled": True,
    "hyde_rrf_weight": 0.25,
    "multi_query_enabled": True,
    "multi_query_num_variants": 3,
    "intent_classifier_enabled": False,
    "recall_deadline_ms": 8000,
    "sampling_rate": 0.05,
}

EXPERIMENT_A: dict[str, Any] = {
    "ce_enabled": True,
    "ce_weight": 0.75,
    "ce_max_candidates": 30,
    "hyde_enabled": True,
    "hyde_rrf_weight": 0.35,
    "multi_query_enabled": True,
    "multi_query_num_variants": 5,
    "intent_classifier_enabled": True,
    "recall_deadline_ms": 10000,
    "sampling_rate": 0.10,
}

_BUILTIN: dict[str, dict[str, Any]] = {
    "production": PRODUCTION,
    "dev": DEV,
    "experiment_a": EXPERIMENT_A,
}


# ── Public API ────────────────────────────────────────────────────────────

def load_profile(name: str) -> dict[str, Any]:
    """Return RecallWeights-compatible kwargs for *name*.

    Priority: user YAML override > builtin preset.
    """
    builtin = dict(_BUILTIN.get(name, PRODUCTION))
    overrides = _load_user_overrides()
    if overrides and name in overrides:
        builtin.update(overrides[name])
    return builtin


def list_profiles() -> list[str]:
    """Return all available profile names."""
    names = list(_BUILTIN)
    overrides = _load_user_overrides()
    if overrides:
        for name in overrides:
            if name not in names:
                names.append(name)
    return names


# ── Internal ──────────────────────────────────────────────────────────────

def _profile_path() -> Path | None:
    env = os.environ.get("ENGRAM_PROFILES")
    if env:
        return Path(env)
    default = Path.home() / ".engram" / "profiles.yaml"
    return default if default.exists() else None


def _load_user_overrides() -> dict[str, dict[str, Any]] | None:
    path = _profile_path()
    if path is None:
        return None
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None
