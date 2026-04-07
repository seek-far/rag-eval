"""
embedder.py — Encodes chunks to dense vectors with incremental disk cache.

Cache is a dict {chunk_id: vector} stored in vectors.pkl.
Adding new chunks only encodes the missing ones.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import faiss
import numpy as np

from indexing.encoder_runtime import SentenceEncoderRuntime

logger = logging.getLogger(__name__)

# BGE models need a query prefix; passage encoding has no prefix.
_BGE_QUERY_PREFIX = "Represent this sentence for searching: "
_E5_QUERY_PREFIX = "query: "
_E5_PASSAGE_PREFIX = "passage: "


class Embedder:
    def __init__(self, cfg):
        self.model_name = cfg.embed_model
        self.device = cfg.embed_device
        self.devices = cfg.embed_devices
        self.batch_size = cfg.embed_batch
        self.cache_dir = Path(cfg.embed_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Loading embedding model '%s' on %s ...",
            self.model_name,
            ",".join(self.devices),
        )
        self.runtime = SentenceEncoderRuntime(
            model_name=self.model_name,
            primary_device=self.devices[0],
            devices=self.devices,
            batch_size=self.batch_size,
            auto_batch=cfg.embed_batch_is_auto,
            batch_min=cfg.embed_batch_min,
            batch_max=cfg.embed_batch_max,
            batch_utilization=cfg.embed_batch_utilization,
            stage_name="embedding",
            keep_primary_model=True,
        )

        self._vectors: np.ndarray | None = None
        self._chunk_ids: list[str] = []
        self._index: faiss.IndexFlatIP | None = None

    def __del__(self) -> None:
        try:
            self.runtime.close()
        except Exception:
            pass

    # ── Encoding ──────────────────────────────────────────────────────────

    def encode_passages(self, chunks: list[dict]) -> np.ndarray:
        """Encode chunk texts with incremental caching. Returns (N, D) float32 array."""
        if not chunks:
            return np.empty((0, 0), dtype="float32")
        cache_path = self.cache_dir / "vectors.pkl"

        # Load existing cache: {chunk_id: vector}
        vec_cache: dict[str, np.ndarray] = {}
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                vec_cache = pickle.load(f)
            logger.info("Embedding cache loaded: %d vectors", len(vec_cache))

        # Find chunks that need encoding
        missing = [c for c in chunks if c["chunk_id"] not in vec_cache]

        if missing:
            texts = [self._passage_text(c["text"]) for c in missing]
            logger.info(
                "Encoding %d new passages (%d cached) with '%s' ...",
                len(missing), len(chunks) - len(missing), self.model_name,
            )
            new_vecs = self.runtime.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=True,
            ).astype("float32")

            for c, v in zip(missing, new_vecs):
                vec_cache[c["chunk_id"]] = v

            with open(cache_path, "wb") as f:
                pickle.dump(vec_cache, f)
            logger.info("Saved embedding cache: %d vectors", len(vec_cache))
        else:
            logger.info("All %d embeddings from cache.", len(chunks))

        # Return vectors in the same order as input chunks
        return np.stack([vec_cache[c["chunk_id"]] for c in chunks])

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string. Returns (D,) float32 array."""
        text = self._query_text(query)
        vec = self.runtime.encode_one(
            text,
            normalize_embeddings=True,
        ).astype("float32")
        return vec

    # ── FAISS index ───────────────────────────────────────────────────────

    def build_index(self, chunks: list[dict]) -> None:
        """Encode all chunks and build an in-memory FAISS IP index."""
        self._chunk_ids = [c["chunk_id"] for c in chunks]
        self._vectors = self.encode_passages(chunks)
        dim = self._vectors.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(self._vectors)
        logger.info(
            "FAISS index built: %d vectors, dim=%d", len(self._chunk_ids), dim
        )

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Returns [(chunk_id, score), ...] sorted by descending score."""
        if self._index is None:
            raise RuntimeError("Call build_index() before search().")
        q_vec = self.encode_query(query).reshape(1, -1)
        scores, indices = self._index.search(q_vec, min(k, len(self._chunk_ids)))
        return [
            (self._chunk_ids[idx], float(scores[0][rank]))
            for rank, idx in enumerate(indices[0])
            if idx >= 0
        ]

    # ── Helpers ───────────────────────────────────────────────────────────

    def _query_text(self, text: str) -> str:
        name = self.model_name.lower()
        if "bge" in name and "m3" not in name:
            return _BGE_QUERY_PREFIX + text
        if "e5" in name:
            return _E5_QUERY_PREFIX + text
        return text

    def _passage_text(self, text: str) -> str:
        name = self.model_name.lower()
        if "e5" in name:
            return _E5_PASSAGE_PREFIX + text
        return text
