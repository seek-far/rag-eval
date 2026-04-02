"""
retrieval_metrics.py — MRR, NDCG, Hit@K, Recall@K.
"""
from __future__ import annotations

import math


def mrr_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved[:k], start=1)
        if doc_id in relevant
    )
    ideal = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return float(any(d in relevant for d in retrieved[:k]))


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / len(relevant)


def answer_in_context_at_k(
    chunk_texts: list[str], answer_spans: list[str], k: int
) -> float:
    """1.0 if any answer span appears (case-insensitive) in any top-k chunk."""
    texts = [t.lower() for t in chunk_texts[:k]]
    for span in answer_spans:
        span_lower = span.strip().lower()
        if not span_lower:
            continue
        for t in texts:
            if span_lower in t:
                return 1.0
    return 0.0


def compute_retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k_values: list[int],
    active_metrics: list[str],
) -> dict[str, float]:
    rel = set(relevant_ids)
    result: dict[str, float] = {}
    for k in k_values:
        if "mrr" in active_metrics:
            result[f"mrr@{k}"] = mrr_at_k(retrieved_ids, rel, k)
        if "ndcg" in active_metrics:
            result[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, rel, k)
        if "hit" in active_metrics:
            result[f"hit@{k}"] = hit_at_k(retrieved_ids, rel, k)
        if "recall" in active_metrics:
            result[f"recall@{k}"] = recall_at_k(retrieved_ids, rel, k)
    return result


def compute_context_metrics(
    chunk_texts: list[str],
    answer_spans: list[str],
    k_values: list[int],
) -> dict[str, float]:
    return {
        f"answer_in_ctx@{k}": answer_in_context_at_k(chunk_texts, answer_spans, k)
        for k in k_values
    }
