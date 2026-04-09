"""
cross_encoder.py — Cross-encoder reranker.

Takes the top-K chunks from the retriever and reranks them using a
cross-encoder model (query + passage fed jointly → relevance score).
This is slower than bi-encoder retrieval but more accurate.
"""
from __future__ import annotations

import logging
import re

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\b\w+\b")
_NEGATION_TOKENS = {"not", "no", "never", "none", "without", "lack", "lacks"}
_COMPARE_UP = {
    "increase", "increases", "increased", "higher", "greater", "more",
    "promote", "promotes", "promoted",
}
_COMPARE_DOWN = {
    "decrease", "decreases", "decreased", "reduce", "reduces", "reduced",
    "lower", "less", "inhibit", "inhibits", "inhibited",
}


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?%?", text.lower()))


def _extract_caps_tokens(text: str) -> set[str]:
    return set(re.findall(r"\b[A-Z]{2,}\d*\b", text))


def _contains_any(tokens: set[str], vocab: set[str]) -> bool:
    return bool(tokens & vocab)


def _rule_score(query: str, passage: str) -> float:
    q_tokens = _tokenize(query)
    p_tokens = _tokenize(passage)
    q_set = set(q_tokens)
    p_set = set(p_tokens)

    score = len(q_set & p_set) * 0.02

    q_nums = _extract_numbers(query)
    p_nums = _extract_numbers(passage)
    score += len(q_nums & p_nums) * 0.45
    if q_nums and not (q_nums & p_nums):
        score -= 0.2

    score += len(_extract_caps_tokens(query) & _extract_caps_tokens(passage)) * 0.35

    if _contains_any(q_set, _NEGATION_TOKENS) == _contains_any(p_set, _NEGATION_TOKENS):
        score += 0.18
    else:
        score -= 0.18

    q_up = _contains_any(q_set, _COMPARE_UP)
    p_up = _contains_any(p_set, _COMPARE_UP)
    q_down = _contains_any(q_set, _COMPARE_DOWN)
    p_down = _contains_any(p_set, _COMPARE_DOWN)
    if q_up == p_up and q_down == p_down:
        score += 0.16
    elif (q_up and p_down) or (q_down and p_up):
        score -= 0.16

    if "(" in query and ")" in query and "(" in passage and ")" in passage:
        score += 0.08

    return score


def _rule_features(query: str, passage: str) -> dict[str, float | int | bool]:
    q_tokens = _tokenize(query)
    p_tokens = _tokenize(passage)
    q_set = set(q_tokens)
    p_set = set(p_tokens)
    q_nums = _extract_numbers(query)
    p_nums = _extract_numbers(passage)
    q_caps = _extract_caps_tokens(query)
    p_caps = _extract_caps_tokens(passage)
    q_has_neg = _contains_any(q_set, _NEGATION_TOKENS)
    p_has_neg = _contains_any(p_set, _NEGATION_TOKENS)
    q_up = _contains_any(q_set, _COMPARE_UP)
    p_up = _contains_any(p_set, _COMPARE_UP)
    q_down = _contains_any(q_set, _COMPARE_DOWN)
    p_down = _contains_any(p_set, _COMPARE_DOWN)
    return {
        "score": _rule_score(query, passage),
        "token_overlap": len(q_set & p_set),
        "number_matches": len(q_nums & p_nums),
        "caps_matches": len(q_caps & p_caps),
        "negation_match": q_has_neg == p_has_neg,
        "direction_match": q_up == p_up and q_down == p_down,
        "direction_conflict": (q_up and p_down) or (q_down and p_up),
    }


def _should_promote(challenger: dict, current: dict) -> bool:
    if challenger["score"] <= current["score"] + 0.35:
        return False
    strong_signal = (
        challenger["number_matches"] > current["number_matches"]
        or challenger["caps_matches"] > current["caps_matches"]
        or (challenger["negation_match"] and not current["negation_match"])
        or (challenger["direction_match"] and not current["direction_match"])
    )
    if not strong_signal:
        return False
    if challenger["direction_conflict"]:
        return False
    return challenger["token_overlap"] >= current["token_overlap"]


class Reranker:
    def __init__(self, cfg):
        self.mode = cfg.reranker
        if self.mode == "none":
            self._model = None
            return
        if self.mode == "rule-top3":
            self._model = None
            logger.info("Using lightweight rule-based top-3 reranker.")
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
        if not chunks:
            return chunks
        if self.mode == "none":
            return chunks
        if self.mode == "rule-top3":
            head = [dict(c) for c in chunks[:3]]
            tail = chunks[3:]
            features = [_rule_features(query, chunk["text"]) for chunk in head]
            for chunk, feat in zip(head, features):
                chunk["rerank_score"] = float(feat["score"])
            if len(head) >= 2:
                best_idx = 0
                for idx in range(1, len(head)):
                    if _should_promote(features[idx], features[best_idx]):
                        best_idx = idx
                if best_idx != 0:
                    head[0], head[best_idx] = head[best_idx], head[0]
                    features[0], features[best_idx] = features[best_idx], features[0]
            return head + tail

        pairs = [(query, c["text"]) for c in chunks]
        scores = self._model.predict(pairs, show_progress_bar=False)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
