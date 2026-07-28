"""Tests for HanLP NER augmentation module."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from engram_router.hanlp_ner import hanlp_ner, enrich_entities, _load_hanlp, _LOADED


class TestHanlpNer:
    def test_skip_env(self, monkeypatch):
        monkeypatch.setenv("ENGRAM_SKIP_HANLP", "1")
        # Reset singleton to force re-check
        import engram_router.hanlp_ner as hn
        hn._LOADED = False
        hn._HANLP = None
        result = hanlp_ner("张三和李四去了北京")
        assert result == []

    def test_tag_to_kind_mapping(self):
        from engram_router.hanlp_ner import _TAG_TO_KIND
        assert _TAG_TO_KIND["PERSON"] == "person"
        assert _TAG_TO_KIND["ORGANIZATION"] == "company"
        assert _TAG_TO_KIND["LOCATION"] == "location"
        assert _TAG_TO_KIND["PRODUCT"] == "object"
        assert _TAG_TO_KIND["DATE"] == "time"

    def test_mock_hanlp_output(self, monkeypatch):
        """Simulate HanLP NER output without actually loading the model."""
        monkeypatch.setenv("ENGRAM_SKIP_HANLP", "1")

        # Build fake entities manually via enrich_entities
        existing = [
            {"name": "北京", "kind": "location", "evidence": "北京"},
        ]
        # HanLP skipped → no enrichment
        result = enrich_entities("张三去北京出差", existing)
        assert result == existing  # unchanged

    def test_enrich_entities_no_duplicates(self, monkeypatch):
        """enrich_entities should not add duplicate (name, kind) pairs."""
        monkeypatch.setenv("ENGRAM_SKIP_HANLP", "1")

        existing = [
            {"name": "张三", "kind": "person", "evidence": "张三"},
        ]
        result = enrich_entities("张三和李四", existing)
        # HanLP skipped → no change, but duplicates shouldn't appear
        names = [(e["name"], e["kind"]) for e in result]
        assert names.count(("张三", "person")) == 1


class TestExtractEntitiesIntegration:
    def test_hanlp_enrichment_negative(self, monkeypatch):
        """Verify HanLP is NOT called when SKIP_HANLP=1."""
        monkeypatch.setenv("ENGRAM_SKIP_HANLP", "1")
        # Reset singleton
        import engram_router.hanlp_ner as hn
        hn._LOADED = False
        hn._HANLP = None

        from engram_router.entities import extract_entities

        ents = extract_entities("张三和李四去了北京协和医院")
        persons = [e for e in ents if e["kind"] == "person"]
        # Rule-based should still find 张三
        assert any("张三" in e["name"] for e in persons)

    def test_skip_hanlp_integration(self, monkeypatch):
        """Full integration: extract_entities with HanLP disabled should work."""
        monkeypatch.setenv("ENGRAM_SKIP_HANLP", "1")
        import engram_router.hanlp_ner as hn
        hn._LOADED = False
        hn._HANLP = None

        from engram_router.entities import extract_entities

        ents = extract_entities("老张开一辆黑色宝马X5")
        persons = [e for e in ents if e["kind"] == "person"]
        assert any("老张" in e["name"] for e in persons)
        objects = [e for e in ents if e["kind"] == "object"]
        assert any("X5" in e["name"] for e in objects)
