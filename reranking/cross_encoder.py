"""
cross_encoder.py — Cross-encoder reranker.

Takes the top-K chunks from the retriever and reranks them using a
cross-encoder model (query + passage fed jointly → relevance score).
This is slower than bi-encoder retrieval but more accurate.
"""
from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, cfg):
        if cfg.reranker == "none":
            self._model = None
            return
        logger.info(
            "Loading reranker '%s' on %s ...", cfg.reranker_model, cfg.embed_device
        )
        self._model = CrossEncoder(
            cfg.reranker_model,
            device=cfg.embed_device,
            max_length=512,
        )

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        """
        Returns chunks sorted by cross-encoder relevance score (desc).
        Adds 'rerank_score' to each chunk dict.
        """
        if self._model is None or not chunks:
            return chunks

        pairs = [(query, c["text"]) for c in chunks]
        scores = self._model.predict(pairs, show_progress_bar=False)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
