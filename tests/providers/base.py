"""MemoryProvider abstract base for multi-system evaluation.

This tiny protocol lets ``tests/eval_v2_matrix.py`` swap engram-router,
mem0, and a naive-vector baseline behind one interface so the same suite
of scenarios and metrics compare apples to apples.

Not a public runtime API — lives in ``tests/`` on purpose so it never
ships to end users.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProviderRecord:
    """Unified result row across providers.

    ``score`` semantics differ across systems (cosine sim vs weighted rank vs
    inverse distance). eval_v2's metrics only use ordering, never magnitude,
    so scores are informational only.
    """

    id: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryProvider(ABC):
    """Uniform interface for engram / mem0 / naive-vector / long-context.

    ``open`` and ``close`` bracket a run so an implementation can lazily
    materialise state on disk without leaking between scenarios. Each
    :class:`EvalScenario` gets a fresh ``open()`` and its own workspace
    directory.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-friendly identifier printed in the matrix table."""

    @abstractmethod
    def open(self, workspace: Path) -> None:
        """Prepare provider state under ``workspace``. Runs once per scenario."""

    @abstractmethod
    def save(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        """Ingest one memory. Returns a provider-scoped id string."""

    @abstractmethod
    def recall(self, query: str, top_k: int = 5) -> list[ProviderRecord]:
        """Return ranked matches, most relevant first."""

    @abstractmethod
    def close(self) -> None:
        """Release resources. Called even if a scenario raised."""

    # Optional hook — providers that support namespaced reset can override.
    def clear(self) -> None:
        """Best-effort content wipe within the current workspace."""
        return None
