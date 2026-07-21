from engram_router.embedding import EmbeddingEngine
from engram_router.llm_extractor import LLMExtractor
from engram_router.llm_reranker import LLMReranker
from engram_router.query_expansion import QueryExpander


def test_embedding_engine_default_does_not_enable_remote(monkeypatch):
    monkeypatch.setenv("ENGRAM_EMBEDDING_API_KEY", "test-key")
    monkeypatch.delenv("ENGRAM_ALLOW_CLOUD", raising=False)
    monkeypatch.delenv("ENGRAM_ALLOW_CLOUD_EMBEDDING", raising=False)

    engine = EmbeddingEngine(backend="remote")

    assert not engine.available
    assert engine.backend_name == "remote"


def test_llm_reranker_default_does_not_enable_cloud(monkeypatch):
    monkeypatch.setenv("ENGRAM_LLM_API_KEY", "test-key")
    monkeypatch.delenv("ENGRAM_ALLOW_CLOUD", raising=False)
    monkeypatch.delenv("ENGRAM_ALLOW_CLOUD_RERANKER", raising=False)

    reranker = LLMReranker()

    assert not reranker.available


def test_llm_extractor_default_does_not_enable_cloud(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("ENGRAM_ALLOW_CLOUD", raising=False)
    monkeypatch.delenv("ENGRAM_ALLOW_CLOUD_LLM", raising=False)

    extractor = LLMExtractor()

    assert not extractor.available


def test_query_expander_default_does_not_enable_llm(monkeypatch):
    monkeypatch.delenv("ENGRAM_ALLOW_CLOUD", raising=False)
    monkeypatch.delenv("ENGRAM_ALLOW_CLOUD_LLM", raising=False)

    expander = QueryExpander()

    assert not expander._enable_llm


# ── Explicit-opt-in coverage ────────────────────────────────────────────────
#
# These tests ensure that the "default off" contract stays a *default*, not a
# hard lock: callers who explicitly pass allow_* = True (e.g., ones going
# through YAML config → PrivacyConfig.allow_cloud_* = True) get cloud enabled.
# Without this, a well-meaning refactor could silently break every user who
# opted in via config.


def test_llm_reranker_explicit_allow_cloud_enables(monkeypatch):
    monkeypatch.setenv("ENGRAM_LLM_API_KEY", "test-key")

    reranker = LLMReranker(allow_cloud=True)

    assert reranker.available


def test_llm_extractor_explicit_allow_cloud_enables(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    extractor = LLMExtractor(allow_cloud=True)

    assert extractor.available


def test_query_expander_explicit_allow_cloud_enables():
    expander = QueryExpander(allow_cloud_llm=True)

    assert expander._enable_llm


# ── Env-var opt-in coverage ─────────────────────────────────────────────────
#
# ENGRAM_ALLOW_CLOUD (global) and ENGRAM_ALLOW_CLOUD_{LLM,EMBEDDING,RERANKER}
# are the env-var route to opting into cloud. Without them, users running the
# CLI or a bare `EmbeddingEngine()` would have no way to enable cloud short of
# editing code — which is worse than the pre-fix state.


def test_engram_allow_cloud_all_enables_extractor(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("ENGRAM_ALLOW_CLOUD", "1")

    extractor = LLMExtractor()

    assert extractor.available


def test_engram_allow_cloud_llm_enables_extractor(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("ENGRAM_ALLOW_CLOUD", raising=False)
    monkeypatch.setenv("ENGRAM_ALLOW_CLOUD_LLM", "true")

    extractor = LLMExtractor()

    assert extractor.available


def test_engram_allow_cloud_reranker_enables(monkeypatch):
    monkeypatch.setenv("ENGRAM_LLM_API_KEY", "test-key")
    monkeypatch.delenv("ENGRAM_ALLOW_CLOUD", raising=False)
    monkeypatch.setenv("ENGRAM_ALLOW_CLOUD_RERANKER", "yes")

    reranker = LLMReranker()

    assert reranker.available


def test_engram_allow_cloud_llm_enables_query_expander(monkeypatch):
    monkeypatch.delenv("ENGRAM_ALLOW_CLOUD", raising=False)
    monkeypatch.setenv("ENGRAM_ALLOW_CLOUD_LLM", "on")

    expander = QueryExpander()

    assert expander._enable_llm


def test_engram_allow_cloud_ignored_when_value_is_falsy(monkeypatch):
    """Only 1/true/yes/on flip the switch. Anything else stays off."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("ENGRAM_ALLOW_CLOUD", "0")
    monkeypatch.setenv("ENGRAM_ALLOW_CLOUD_LLM", "false")

    extractor = LLMExtractor()

    assert not extractor.available
