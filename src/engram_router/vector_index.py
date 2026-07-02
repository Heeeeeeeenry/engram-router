"""FAISS-backed vector index for EngramRouter Phase 2.

Provides ANN (Approximate Nearest Neighbor) search over memory embeddings.
Stored alongside the SQLite database as <db_path>.faiss and <db_path>.idmap.

Architecture:
  - FAISS IndexIVFFlat for sub-linear search (clusters → local search)
  - Cosine similarity via inner product on L2-normalized vectors
  - ID mapping between FAISS internal IDs and memory_ids
  - Incremental updates (add new memories without rebuild)
  - Persistent serialization (save/load on close/open)

Dependencies:
  - faiss-cpu (via pip install engram-router[llm])
  - numpy
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore

logger = logging.getLogger(__name__)


class VectorIndex:
    """FAISS index for memory embeddings with incremental updates."""

    def __init__(
        self,
        dim: int = 512,
        path: str | Path | None = None,
        nlist: int | None = None,
    ):
        self._dim = dim
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        self._id_to_memory: dict[int, str] = {}   # FAISS id → memory_id
        self._memory_to_id: dict[str, int] = {}    # memory_id → FAISS id
        self._next_id: int = 0
        self._index = None
        self._trained = False

        # Load existing index if available
        loaded = False
        try:
            loaded = self._load() if self._path else False
        except (ImportError, Exception) as exc:
            logger.debug("VectorIndex load skipped: %s", exc)

        if not loaded:
            self._nlist = nlist or 8
            self._index = None
            try:
                self._init_index()
            except ImportError as exc:
                logger.debug("VectorIndex init skipped (faiss not available): %s", exc)
                self._index = None

    # ── public ────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return len(self._id_to_memory)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def trained(self) -> bool:
        return self._trained

    def add(
        self, memory_id: str, embedding: np.ndarray, train_if_needed: bool = True
    ) -> None:
        """Add a single embedding to the index.

        If the index is new (< nlist*40 vectors), vectors are buffered.
        Training happens automatically when enough data is accumulated.
        """
        with self._lock:
            vec = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
            if vec.shape[1] != self._dim:
                raise ValueError(
                    f"Embedding dim {vec.shape[1]} != index dim {self._dim}"
                )

            faiss_id = self._next_id
            self._next_id += 1

            if self._trained:
                self._index.add(vec)
            else:
                # Buffer for later training
                if not hasattr(self, "_buffer"):
                    self._buffer_embs: list[np.ndarray] = []
                    self._buffer_ids: list[int] = []
                self._buffer_embs.append(vec)
                self._buffer_ids.append(faiss_id)

                min_for_train = 1  # Train immediately (flat index for small, IVF for large)
                if len(self._buffer_ids) >= min_for_train and train_if_needed:
                    self._train_from_buffer()

            self._id_to_memory[faiss_id] = memory_id
            self._memory_to_id[memory_id] = faiss_id

            if self._path:
                self._save()

    def add_batch(
        self, ids_and_embs: list[tuple[str, np.ndarray]]
    ) -> None:
        """Add multiple embeddings at once."""
        if not ids_and_embs:
            return
        with self._lock:
            ids, embs = zip(*ids_and_embs)
            vecs = np.asarray(embs, dtype=np.float32)
            if vecs.shape[1] != self._dim:
                raise ValueError(
                    f"Embedding dim {vecs.shape[1]} != index dim {self._dim}"
                )

            start = self._next_id
            count = len(ids)
            faiss_ids = list(range(start, start + count))
            self._next_id += count

            for fid, mid in zip(faiss_ids, ids):
                self._id_to_memory[fid] = mid
                self._memory_to_id[mid] = fid

            if self._trained:
                self._index.add(vecs)
            else:
                if not hasattr(self, "_buffer"):
                    self._buffer_embs = []
                    self._buffer_ids = []
                self._buffer_embs.append(vecs)
                self._buffer_ids.extend(faiss_ids)

                min_for_train = 1  # Train immediately (flat index for small, IVF for large)
                if len(self._buffer_ids) >= min_for_train:
                    self._train_from_buffer()

            if self._path:
                self._save()

    def search(
        self, query_vec: np.ndarray, k: int = 20
    ) -> list[tuple[str, float]]:
        """Search for k nearest neighbors.

        Returns:
            List of (memory_id, similarity_score) sorted by similarity DESC.
            Similarity is cosine (inner product on normalized vectors, range -1..1).
        """
        if not self._trained or self.size == 0:
            return []

        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self._dim:
            raise ValueError(
                f"Query dim {q.shape[1]} != index dim {self._dim}"
            )

        k = min(k, self.size)
        with self._lock:
            scores, indices = self._index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for not enough neighbors
                continue
            memory_id = self._id_to_memory.get(int(idx))
            if memory_id:
                results.append((memory_id, float(score)))
        return results

    def remove(self, memory_id: str) -> bool:
        """Remove a vector from the index.

        FAISS doesn't support efficient single-vector deletion.
        We mark it as removed in the ID map instead.
        """
        with self._lock:
            faiss_id = self._memory_to_id.pop(memory_id, None)
            if faiss_id is None:
                return False
            self._id_to_memory.pop(faiss_id, None)
            # Note: the vector remains in FAISS but its ID won't be returned
            # because our search method filters by _id_to_memory.
            # A full rebuild can be triggered via rebuild().
            return True

    def rebuild(self) -> None:
        """Rebuild the index from scratch (removes ghosts from remove() calls)."""
        with self._lock:
            # Collect all live vectors
            live = []
            id_map = {}
            for fid, mid in self._id_to_memory.items():
                if fid < self._index.ntotal:
                    vec = self._index.reconstruct(int(fid))
                    live.append(vec)
                    id_map[len(id_map)] = mid

            # Rebuild
            self._id_to_memory = id_map
            self._memory_to_id = {v: k for k, v in id_map.items()}
            self._next_id = len(id_map)
            self._init_index()
            if live:
                self._index.add(np.array(live, dtype=np.float32))
                self._trained = True

            if self._path:
                self._save()

    def save(self, path: str | Path | None = None) -> None:
        if path:
            self._path = Path(path)
        if not self._path:
            return
        if not loaded:
            self._nlist = nlist or 8
            self._index = None
            self._init_index()

    def _init_index(self) -> None:
        if faiss is None:
            raise ImportError("faiss-cpu is required for VectorIndex. Install with: pip install engram-router[llm]")
        quantizer = faiss.IndexFlatIP(self._dim)
        self._index = faiss.IndexIVFFlat(
            quantizer, self._dim, self._nlist, faiss.METRIC_INNER_PRODUCT
        )

    def _train_from_buffer(self) -> None:
        import faiss
        all_vecs = np.concatenate(self._buffer_embs, axis=0)
        n_vecs = len(all_vecs)
        if n_vecs < 1:
            return
        # For small datasets (< 16 vectors), use IndexFlatIP for exact search.
        # IVF requires nlist * 5 vectors (typically 40) to train properly.
        if n_vecs < max(self._nlist * 5, 16):
            logger.info(
                "Flat index (exact search): %d vectors, dim=%d",
                n_vecs, self._dim,
            )
            self._index = faiss.IndexFlatIP(self._dim)
            self._index.add(all_vecs)
            self._trained = True
        else:
            self._nlist = min(int(np.sqrt(n_vecs)), 256)
            self._init_index()
            self._index.train(all_vecs)
            self._index.add(all_vecs)
            self._trained = True
            logger.info(
                "FAISS IVF trained: %d vectors, nlist=%d, dim=%d",
                n_vecs, self._nlist, self._dim,
            )
        del self._buffer_embs
        del self._buffer_ids

    def _save(self) -> None:
        if not self._path or not self._trained:
            return
        import faiss
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss_path = str(self._path.with_suffix(".faiss"))
        faiss.write_index(self._index, faiss_path)

        # Save ID mapping (JSON, not pickle)
        idmap_path = str(self._path.with_suffix(".idmap"))
        meta = {
            "id_to_memory": {str(k): v for k, v in self._id_to_memory.items()},
            "memory_to_id": {str(k): v for k, v in self._memory_to_id.items()},
            "next_id": self._next_id,
            "dim": self._dim,
            "nlist": self._nlist,
        }
        with open(idmap_path, "w") as f:
            json.dump(meta, f)

    def _load(self) -> bool:
        faiss_path = str(self._path.with_suffix(".faiss"))
        idmap_path = str(self._path.with_suffix(".idmap"))
        if not os.path.exists(faiss_path) or not os.path.exists(idmap_path):
            return False
        try:
            import faiss
            self._index = faiss.read_index(faiss_path)
            self._trained = self._index.is_trained
            with open(idmap_path) as f:
                meta = json.load(f)
            self._id_to_memory = {int(k): v for k, v in meta["id_to_memory"].items()}
            self._memory_to_id = meta["memory_to_id"]
            self._next_id = meta["next_id"]
            self._dim = meta.get("dim", self._dim)
            self._nlist = meta.get("nlist", self._nlist)
            logger.info(
                "FAISS loaded: %d vectors, %dd, nlist=%d",
                len(self._id_to_memory), self._dim, self._nlist,
            )
            return True
        except Exception as exc:
            logger.warning("FAISS load failed: %s", exc)
            return False
