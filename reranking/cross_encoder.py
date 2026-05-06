"""
cross_encoder.py — Cross-encoder reranker.

Takes the top-K chunks from the retriever and reranks them using a
cross-encoder model (query + passage fed jointly → relevance score).
This is slower than bi-encoder retrieval but more accurate.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request

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

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


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


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _truncate_passage(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def _extract_llm_scores(content: str) -> dict[int, float]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(content)
        if not match:
            raise ValueError(f"LLM reranker did not return JSON: {content!r}")
        payload = json.loads(match.group(0))

    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, list):
        raise ValueError(f"LLM reranker JSON missing 'scores' list: {payload!r}")

    scores: dict[int, float] = {}
    for item in raw_scores:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        score = item.get("score")
        try:
            scores[int(idx)] = float(score)
        except (TypeError, ValueError):
            continue
    if not scores:
        raise ValueError(f"LLM reranker returned no usable scores: {payload!r}")
    return scores


class LLMReranker:
    def __init__(self, cfg):
        if not cfg.llm_base_url:
            raise ValueError("LLM_BASE_URL is required when RERANKER=llm-cross-encoder")
        if not cfg.llm_model:
            raise ValueError("LLM_MODEL is required when RERANKER=llm-cross-encoder")
        self.url = _chat_completions_url(cfg.llm_base_url)
        self.api_key = cfg.llm_api_key
        self.model = cfg.llm_model
        self.temperature = cfg.llm_temperature
        self.max_tokens = cfg.llm_max_tokens
        self.timeout = cfg.llm_timeout
        self.max_chars = cfg.llm_rerank_max_chars
        logger.info(
            "Using OpenAI-compatible LLM reranker '%s' at %s.",
            self.model,
            self.url,
        )

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return chunks

        candidates = []
        for idx, chunk in enumerate(chunks, start=1):
            candidates.append(
                {
                    "index": idx,
                    "doc_id": chunk.get("doc_id", ""),
                    "text": _truncate_passage(chunk["text"], self.max_chars),
                }
            )

        content = self._score_candidates(query, candidates)
        scores = _extract_llm_scores(content)

        scored_chunks = []
        for idx, chunk in enumerate(chunks, start=1):
            row = dict(chunk)
            row["rerank_score"] = float(scores.get(idx, -1.0))
            scored_chunks.append(row)

        return sorted(scored_chunks, key=lambda c: c["rerank_score"], reverse=True)

    def _score_candidates(self, query: str, candidates: list[dict]) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict cross-encoder style reranker. "
                        "Score how well each candidate passage supports or answers the query. "
                        "Use 0 for unrelated, 50 for partially relevant, and 100 for directly relevant. "
                        "Return JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Query:\n"
                        f"{query}\n\n"
                        "Candidates:\n"
                        f"{json.dumps(candidates, ensure_ascii=False)}\n\n"
                        'Return exactly this JSON shape: {"scores":[{"index":1,"score":0.0}]}'
                    ),
                },
            ],
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM reranker request failed: HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM reranker request failed: {exc}") from exc

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM reranker response: {data!r}") from exc


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
        if self.mode in {"llm", "llm-cross-encoder"}:
            self._model = LLMReranker(cfg)
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
        if self.mode in {"llm", "llm-cross-encoder"}:
            return self._model.rerank(query, chunks)

        pairs = [(query, c["text"]) for c in chunks]
        scores = self._model.predict(pairs, show_progress_bar=False)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
