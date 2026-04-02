"""
answer_metrics.py — Token-level F1 and BERTScore for answer quality.

Both metrics compare a predicted answer string against a reference answer.
Used for datasets that supply reference answers (QASPER, QuALITY, NQ).
"""
from __future__ import annotations

import logging
import re
import string

logger = logging.getLogger(__name__)

# BERTScore is optional; gracefully degrade if not installed
try:
    from bert_score import score as bert_score_fn
    _BERTSCORE_AVAILABLE = True
except ImportError:
    _BERTSCORE_AVAILABLE = False
    logger.warning(
        "bert_score not installed. BERTScore will be skipped. "
        "Install with: pip install bert-score"
    )


def compute_answer_metrics(
    predicted: str,
    reference: str,
    active_metrics: list[str],
) -> dict[str, float]:
    result: dict[str, float] = {}

    if "f1" in active_metrics:
        result["f1"] = token_f1(predicted, reference)

    if "bertscore" in active_metrics:
        if _BERTSCORE_AVAILABLE:
            result["bertscore"] = bertscore(predicted, reference)
        else:
            result["bertscore"] = 0.0

    return result


# ── Token-level F1 ────────────────────────────────────────────────────────

def _normalise(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def token_f1(pred: str, ref: str) -> float:
    pred_tokens = _normalise(pred)
    ref_tokens = _normalise(ref)
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, ref: str) -> float:
    return float(_normalise(pred) == _normalise(ref))


# ── BERTScore ─────────────────────────────────────────────────────────────

def bertscore(pred: str, ref: str, lang: str = "en") -> float:
    if not _BERTSCORE_AVAILABLE:
        return 0.0
    try:
        _, _, f1 = bert_score_fn([pred], [ref], lang=lang, verbose=False)
        return float(f1[0])
    except Exception as e:
        logger.warning("BERTScore failed: %s", e)
        return 0.0
