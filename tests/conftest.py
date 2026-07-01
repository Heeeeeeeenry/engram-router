"""Shared pytest fixtures for EngramRouter tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from engram_router.store import MemoryStore

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cli_env() -> dict[str, str]:
    """Environment dict with ``src/`` on PYTHONPATH for CLI subprocess tests."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


@pytest.fixture
def temp_db(tmp_path: Path) -> MemoryStore:
    """MemoryStore backed by a temporary SQLite database."""
    return MemoryStore(path=tmp_path / "memory.db")


@pytest.fixture
def populated_store(temp_db: MemoryStore) -> MemoryStore:
    """A MemoryStore pre-filled with common demo data."""
    temp_db.save("张三前两天送我一把 HHKB，说是生日礼物")
    temp_db.save("张三是我的前同事，现在在腾讯。")
    return temp_db
