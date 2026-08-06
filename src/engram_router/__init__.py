"""EngramRouter: lossless on-demand memory routing for AI agents."""

from .store import MemoryStore, MemoryRecord, RecallWeights
from .query_expansion import (
    QueryExpander,
    ExpandedQuery,
    SynonymTable,
    ExpansionCache,
    ExpansionStats,
)
from .causal import (
    CausalChain,
    CausalEdge,
    CausalPath,
    Timeline,
    TimedEvent,
)
from .persona import (
    PersonaStore,
    Persona,
    PersonaAttr,
    AttrEvidence,
)
from .forgetting import (
    ForgettingEngine,
    ForgettingConfig,
)
from .event_extractor import (
    extract_events,
    Event,
)
from .contradiction import (
    detect_contradictions,
    check_new_event,
)
from .reflection import (
    run_reflection,
    store_reflection,
)


# ── Exceptions (API audit) ───────────────────────────────────────────

class EngramError(Exception):
    """Base exception for all EngramRouter errors."""


class ConfigError(EngramError):
    """Configuration error — missing env var, invalid value, etc."""


class ValidationError(EngramError):
    """Input validation failed — bad namespace, oversized text, etc."""


class NamespaceError(EngramError):
    """Cross-namespace access denied."""


__version__ = "1.1.0"

__all__ = [
    "MemoryStore",
    "MemoryRecord",
    "RecallWeights",
    "QueryExpander",
    "ExpandedQuery",
    "SynonymTable",
    "ExpansionCache",
    "ExpansionStats",
    "CausalChain",
    "CausalEdge",
    "CausalPath",
    "Timeline",
    "TimedEvent",
    "PersonaStore",
    "Persona",
    "PersonaAttr",
    "AttrEvidence",
    "ForgettingEngine",
    "ForgettingConfig",
    # Exceptions
    "EngramError",
    "ConfigError",
    "ValidationError",
    "NamespaceError",
    # L2 data layer
    "Event",
    "extract_events",
    "detect_contradictions",
    "check_new_event",
    # L3 lifecycle
    "run_reflection",
    "store_reflection",
]