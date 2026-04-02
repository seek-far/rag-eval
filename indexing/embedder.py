"""
embedder.py — Encodes chunks to dense vectors with a disk cache.

Cache key = (embed_model, chunk_id, text[:64])
Avoids re-encoding the same corpus when only retrieval/reranker params change.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# BGE models need a query prefix; passage encoding has no prefix.
_BGE_QUERY_PREFIX = "Represent this sentence for searching: "
_E5_QUERY_PREFIX = "query: "
_E5_PASSAGE_PREFIX = "passage: "


class Embedder:
    def __init__(self, cfg):
        self.model_name = cfg.embed_model
        self.device = cfg.embed_device
        self.batch_size = cfg.embed_batch
        self.cache_dir = Path(cfg.embed_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading embedding model '%s' on %s ...", self.model_name, self.device)
        self.model = SentenceTransformer(self.model_name, device=self.device)

        self._vectors: np.ndarray | None = None
        self._chunk_ids: list[str] = []
        self._index: faiss.IndexFlatIP | None = None

    # ── Encoding ──────────────────────────────────────────────────────────

    def encode_passages(self, chunks: list[dict]) -> np.ndarray:
        """Encode chunk texts. Returns (N, D) float32 array."""
        cache_path = self._cache_path(chunks)
        if cache_path.exists():
            logger.debug("Embedding cache hit: %s", cache_path.name)
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        texts = [self._passage_text(c["text"]) for c in chunks]
        logger.info(
            "Encoding %d passages with '%s' ...", len(texts), self.model_name
        )
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        with open(cache_path, "wb") as f:
            pickle.dump(vecs, f)
        logger.info("Saved embedding cache to %s", cache_path.name)
        return vecs

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string. Returns (D,) float32 array."""
        text = self._query_text(query)
        vec = self.model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype("float32")
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

    def _cache_path(self, chunks: list[dict]) -> Path:
        """Deterministic cache key based on chunk content (model is in dir path)."""
        key_parts = [c["chunk_id"] + c["text"][:64] for c in chunks]
        digest = hashlib.md5("".join(key_parts).encode()).hexdigest()[:12]
        return self.cache_dir / f"vectors_{digest}.pkl"
