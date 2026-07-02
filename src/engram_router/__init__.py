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

__version__ = "0.1.0"

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
]