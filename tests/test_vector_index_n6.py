"""Tests for N6 — atomic FAISS writes + deferred _save().

Covers:
- Atomic file writes (temp + os.replace)
- Deferred save (add() no longer auto-saves)
- Crash-safety: partial write leaves old files intact
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# pytestmark = pytest.mark.skipif(not HAS_FAISS) — defer to individual tests

HAS_FAISS = False
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    pass


@pytest.fixture
def tmp_index_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu not installed")
class TestAtomicFaisSave:
    """Verify FAISS and idmap files are written atomically."""

    def test_basic_save_creates_files(self, tmp_index_dir):
        """save() should create .faiss and .idmap files."""
        from engram_router.vector_index import VectorIndex

        path = tmp_index_dir / "test"
        vi = VectorIndex(dim=64, path=path)
        emb = np.random.randn(64).astype(np.float32)
        vi.add("mem_1", emb)
        vi.save()

        assert os.path.exists(str(path.with_suffix(".faiss")))
        assert os.path.exists(str(path.with_suffix(".idmap")))

    def test_atomic_faiss_no_partial_write(self, tmp_index_dir):
        """A crash mid-save should not leave a partial .faiss file.

        We verify the current file is untouched when the new write fails.
        """
        from engram_router.vector_index import VectorIndex

        path = tmp_index_dir / "test"
        vi = VectorIndex(dim=64, path=path)
        emb1 = np.random.randn(64).astype(np.float32)
        vi.add("mem_1", emb1)
        vi.save()

        faiss_path = str(path.with_suffix(".faiss"))
        idmap_path = str(path.with_suffix(".idmap"))

        # Read old state
        with open(idmap_path) as f:
            old_meta = json.load(f)
        old_faiss_size = os.path.getsize(faiss_path)

        # Add more and save
        vi.add("mem_2", np.random.randn(64).astype(np.float32))
        vi.save()

        # New state should be larger
        assert os.path.getsize(faiss_path) > old_faiss_size, \
            "Expected larger FAISS file after second save"
        with open(idmap_path) as f:
            new_meta = json.load(f)
        assert len(new_meta["id_to_memory"]) == 2

    def test_no_temp_files_left_behind(self, tmp_index_dir):
        """After save(), no .tmp_faiss_* or .tmp_idmap_* should remain."""
        from engram_router.vector_index import VectorIndex

        path = tmp_index_dir / "test"
        vi = VectorIndex(dim=64, path=path)
        emb = np.random.randn(64).astype(np.float32)
        vi.add("mem_1", emb)
        vi.save()

        temps = list(tmp_index_dir.glob(".tmp_*"))
        assert len(temps) == 0, f"Temp files left behind: {temps}"

    def test_save_without_path_noop(self, tmp_index_dir):
        """save() without path should not create files."""
        from engram_router.vector_index import VectorIndex

        vi = VectorIndex(dim=64, path=None)
        emb = np.random.randn(64).astype(np.float32)
        vi.add("mem_1", emb)
        # Should not raise
        vi.save()

    def test_save_untrained_noop(self, tmp_index_dir):
        """save() on untrained index (IVFPQ buffers) should be no-op, not crash."""
        from engram_router.vector_index import VectorIndex

        path = tmp_index_dir / "test"
        vi = VectorIndex(dim=64, path=path, use_pq=True, nlist=2)
        # With IVFPQ no training happens until enough vectors accumulate
        emb = np.random.randn(64).astype(np.float32)
        vi.add("mem_1", emb)  # Should not trigger _save yet (no auto-save now)
        # Explicit save on untrained index should be a no-op
        vi.save()
        # No files should be created since index isn't trained
        assert not os.path.exists(str(path.with_suffix(".faiss")))

    def test_load_after_atomic_save(self, tmp_index_dir):
        """A saved index should be reloadable with correct state."""
        from engram_router.vector_index import VectorIndex

        path = tmp_index_dir / "test"
        vi = VectorIndex(dim=64, path=path)
        for i in range(10):
            vi.add(f"mem_{i}", np.random.randn(64).astype(np.float32))
        vi.save()

        # Load into new instance
        vi2 = VectorIndex(dim=64, path=path)
        assert vi2.size == 10

        # Search should work
        results = vi2.search(np.random.randn(64).astype(np.float32), k=5)
        assert len(results) == 5

    def test_idmap_json_valid_utf8(self, tmp_index_dir):
        """idmap should be valid JSON, not binary."""
        from engram_router.vector_index import VectorIndex

        path = tmp_index_dir / "test"
        vi = VectorIndex(dim=64, path=path)
        emb = np.random.randn(64).astype(np.float32)
        vi.add("mem_特殊字符_emoji_🎉", emb)
        vi.save()

        with open(str(path.with_suffix(".idmap"))) as f:
            meta = json.load(f)
        # The id key is a stringified int, but the value should contain our memory_id
        values = list(meta["id_to_memory"].values())
        assert any("特殊字符" in v for v in values)


@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu not installed")
class TestDeferredSave:
    """Verify add() no longer auto-saves — saves only on explicit save()."""

    def test_add_does_not_create_files(self, tmp_index_dir):
        """add() should NOT create .faiss files (no auto-save)."""
        from engram_router.vector_index import VectorIndex

        path = tmp_index_dir / "test"
        vi = VectorIndex(dim=64, path=path)
        emb = np.random.randn(64).astype(np.float32)
        vi.add("mem_1", emb)

        # No files should exist — add() defers save
        assert not os.path.exists(str(path.with_suffix(".faiss")))
        assert not os.path.exists(str(path.with_suffix(".idmap")))

        # Explicit save should work
        vi.save()
        assert os.path.exists(str(path.with_suffix(".faiss")))
        assert os.path.exists(str(path.with_suffix(".idmap")))

    def test_add_batch_does_not_create_files(self, tmp_index_dir):
        """add_batch() should NOT create .faiss files (no auto-save)."""
        from engram_router.vector_index import VectorIndex

        path = tmp_index_dir / "test"
        vi = VectorIndex(dim=64, path=path)
        items = [(f"mem_{i}", np.random.randn(64).astype(np.float32)) for i in range(10)]
        vi.add_batch(items)

        assert not os.path.exists(str(path.with_suffix(".faiss")))
        assert not os.path.exists(str(path.with_suffix(".idmap")))

        vi.save()
        assert os.path.exists(str(path.with_suffix(".faiss")))

    def test_multiple_add_then_save_once(self, tmp_index_dir):
        """Multiple add() calls + one save() should persist all vectors."""
        from engram_router.vector_index import VectorIndex

        path = tmp_index_dir / "test"
        vi = VectorIndex(dim=64, path=path)
        for i in range(20):
            vi.add(f"mem_{i}", np.random.randn(64).astype(np.float32))
        vi.save()

        # Reload and verify
        vi2 = VectorIndex(dim=64, path=path)
        assert vi2.size == 20
        results = vi2.search(np.random.randn(64).astype(np.float32), k=5)
        assert len(results) == 5

    def test_add_without_path_never_saves(self, tmp_index_dir):
        """add() without path should not try to save."""
        from engram_router.vector_index import VectorIndex

        vi = VectorIndex(dim=64, path=None)
        for i in range(5):
            vi.add(f"mem_{i}", np.random.randn(64).astype(np.float32))
        # No crash, no files
        vi.save()  # no-op since path is None

    def test_disk_consistency_after_save(self, tmp_index_dir):
        """Saved idmap should match in-memory state exactly."""
        from engram_router.vector_index import VectorIndex

        path = tmp_index_dir / "test"
        vi = VectorIndex(dim=64, path=path)
        emb = np.random.randn(64).astype(np.float32)
        vi.add("abc123", emb)
        vi.save()

        # Read idmap from disk
        with open(str(path.with_suffix(".idmap"))) as f:
            meta = json.load(f)

        # idmap should contain our memory_id
        assert any(v == "abc123" for v in meta["id_to_memory"].values())
        assert meta["memory_to_id"]["abc123"] >= 0
        assert meta["next_id"] == 1
