"""
hybrid.py — Unified retriever supporting dense, sparse, and hybrid (RRF) modes.

Reciprocal Rank Fusion formula:
    RRF(d) = Σ weight_i / (k + rank_i(d))

Dense and sparse weights are configurable and should sum to 1.0.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from indexing.embedder import Embedder
    from indexing.bm25_index import BM25Index

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, embedder: "Embedder", bm25: "BM25Index", cfg):
        self.embedder = embedder
        self.bm25 = bm25
        self.mode = cfg.retrieval_mode
        self.k = cfg.retrieve_k
        self.rrf_k = cfg.rrf_k
        self.dense_w = cfg.dense_weight
        self.sparse_w = cfg.sparse_weight

    def retrieve(
        self,
        query: str,
        chunk_map: dict[str, dict],  # chunk_id → chunk dict
    ) -> list[dict]:
        """
        Returns up to self.k chunk dicts, sorted by relevance score (desc).
        Each returned chunk gets a 'retrieval_score' key added.
        """
        if self.mode == "dense":
            ranked = self._dense(query)
        elif self.mode == "sparse":
            ranked = self._sparse(query)
        else:
            ranked = self._hybrid(query)

        results = []
        for chunk_id, score in ranked[: self.k]:
            if chunk_id in chunk_map:
                chunk = dict(chunk_map[chunk_id])
                chunk["retrieval_score"] = score
                results.append(chunk)
        return results

    # ── Private ───────────────────────────────────────────────────────────

    def _dense(self, query: str) -> list[tuple[str, float]]:
        return self.embedder.search(query, self.k)

    def _sparse(self, query: str) -> list[tuple[str, float]]:
        return self.bm25.search(query, self.k)

    def _hybrid(self, query: str) -> list[tuple[str, float]]:
        dense_results = self.embedder.search(query, self.k)
        sparse_results = self.bm25.search(query, self.k)
        return _rrf(
            dense_results,
            sparse_results,
            k=self.rrf_k,
            w_dense=self.dense_w,
            w_sparse=self.sparse_w,
        )


def _rrf(
    dense: list[tuple[str, float]],
    sparse: list[tuple[str, float]],
    k: int = 60,
    w_dense: float = 0.6,
    w_sparse: float = 0.4,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank, (chunk_id, _) in enumerate(dense):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + w_dense / (k + rank + 1)
    for rank, (chunk_id, _) in enumerate(sparse):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + w_sparse / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
