"""Tests for ENGRAM_LOCAL_LLM preset in config + LLMClient."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class TestLocalLLMPreset:
    """Verify ENGRAM_LOCAL_LLM=1 triggers correct default overrides."""

    def test_is_local_llm_true(self):
        from engram_router.llm_extractor import _is_local_llm

        # No env = False
        assert _is_local_llm() is False

        # Try setting the env
        os.environ["ENGRAM_LOCAL_LLM"] = "1"
        try:
            assert _is_local_llm() is True
        finally:
            del os.environ["ENGRAM_LOCAL_LLM"]

    def test_local_llm_returns_dummy_api_key(self):
        from engram_router.llm_extractor import _get_api_key

        os.environ["ENGRAM_LOCAL_LLM"] = "1"
        try:
            assert _get_api_key() == "local"
        finally:
            del os.environ["ENGRAM_LOCAL_LLM"]

    def test_local_llm_default_base_url(self):
        from engram_router.llm_extractor import _get_base_url

        os.environ["ENGRAM_LOCAL_LLM"] = "1"
        # Remove any explicit base_url so the default kicks in
        existing = os.environ.pop("ENGRA_LLM_BASE_URL", None)
        try:
            assert _get_base_url() == "http://localhost:8080/v1"
        finally:
            del os.environ["ENGRAM_LOCAL_LLM"]
            if existing is not None:
                os.environ["ENGRA_LLM_BASE_URL"] = existing

    def test_local_llm_explicit_base_url_wins(self):
        from engram_router.llm_extractor import _get_base_url

        os.environ["ENGRAM_LOCAL_LLM"] = "1"
        os.environ["ENGRA_LLM_BASE_URL"] = "http://mylocal:1234/v1"
        try:
            assert _get_base_url() == "http://mylocal:1234/v1"
        finally:
            del os.environ["ENGRAM_LOCAL_LLM"]
            del os.environ["ENGRA_LLM_BASE_URL"]

    def test_lm_client_available_with_local_llm(self):
        from engram_router.llm_extractor import LLMClient

        os.environ["ENGRAM_LOCAL_LLM"] = "1"
        try:
            client = LLMClient()
            assert client.available is True
            assert client.base_url == "http://localhost:8080/v1"
        finally:
            del os.environ["ENGRAM_LOCAL_LLM"]

    def test_local_llm_does_not_affect_cloud_privacy_gate(self):
        """ENGRAM_LOCAL_LLM=1 must NOT implicitly enable cloud access."""
        from engram_router.config import config, env_allows_cloud

        os.environ["ENGRAM_LOCAL_LLM"] = "1"
        try:
            # Cloud gates should still be off by default
            assert env_allows_cloud("llm") is False
            assert env_allows_cloud("embedding") is False
        finally:
            del os.environ["ENGRAM_LOCAL_LLM"]

    def test_local_llm_without_env_is_normal(self):
        """Without ENGRAM_LOCAL_LLM, everything works as before."""
        from engram_router.llm_extractor import _get_api_key, _get_base_url, _is_local_llm

        assert _is_local_llm() is False
        # base_url should be the normal default (internal proxy)
        base = _get_base_url()
        assert "localhost" not in base  # Not forced to localhost
