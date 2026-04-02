"""
bm25_index.py — Sparse BM25 retrieval using rank_bm25.
"""
from __future__ import annotations

import logging
import re

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenise(text: str) -> list[str]:
    """Simple whitespace + punctuation tokeniser, lowercased."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Index:
    def __init__(self, chunks: list[dict]):
        self._chunk_ids = [c["chunk_id"] for c in chunks]
        tokenised = [_tokenise(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenised)
        logger.info("BM25 index built with %d documents.", len(chunks))

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Returns [(chunk_id, bm25_score), ...] sorted descending."""
        tokens = _tokenise(query)
        scores = self._bm25.get_scores(tokens)
        # argsort descending
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:k]
        return [(self._chunk_ids[idx], float(score)) for idx, score in ranked]
