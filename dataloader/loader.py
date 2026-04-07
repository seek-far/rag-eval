"""
loader.py — Unified loader for NQ, SciFact, QASPER, and QuALITY.

All loaders return a list of EvalSample with the same schema so the
rest of the pipeline doesn't need to know which dataset it's running on.
"""
from __future__ import annotations
import logging
from datasets import load_dataset
from .schema import EvalSample

logger = logging.getLogger(__name__)


def load_eval_samples(
    dataset: str, split: str, n: int = 0
) -> list[EvalSample]:
    """
    Args:
        dataset: one of "nq" | "scifact" | "qasper" | "quality"
        split:   "train" | "dev" | "test"
        n:       max samples to return; 0 = all
    """
    loaders = {
        "nq":      _load_nq,
        "scifact": _load_scifact,
        "qasper":  _load_qasper,
        "quality": _load_quality,
    }
    if dataset not in loaders:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from: {list(loaders)}"
        )
    logger.info("Loading dataset '%s' / split='%s' ...", dataset, split)
    samples = loaders[dataset](split)
    logger.info("Loaded %d samples.", len(samples))
    return samples[:n] if n > 0 else samples


# ─────────────────────────── NQ (BEIR) ────────────────────────────────────

def _load_nq(split: str) -> list[EvalSample]:
    """
    BEIR/NQ: large-scale open-domain QA.
    Corpus + queries are separate subsets; relevance via qrels.
    Answer spans are loaded from nq_open and matched by question text.
    """
    corpus_ds = load_dataset("BeIR/nq", "corpus", split="corpus")
    queries_ds = load_dataset("BeIR/nq", "queries", split="queries")

    # qrels split names differ from corpus/query splits
    qrel_split = {"train": "train", "dev": "validation", "test": "test"}.get(
        split, split
    )
    qrels_ds = load_dataset("BeIR/nq-qrels", split=qrel_split)

    # Load answer spans from nq_open (question → list of answer strings)
    answer_map = _load_nq_answers(split)

    doc_map = {str(row["_id"]): row["text"] for row in corpus_ds}

    qrel_map: dict[str, list[str]] = {}
    for row in qrels_ds:
        if int(row.get("score", 1)) > 0:
            qrel_map.setdefault(str(row["query-id"]), []).append(str(row["corpus-id"]))

    samples = []
    for row in queries_ds:
        qid = str(row["_id"])
        if qid not in qrel_map:
            continue
        relevant_ids = qrel_map[qid]
        # Build a focused document pool: relevant + a sample of irrelevant
        pool_ids = list(
            {*relevant_ids, *list(doc_map.keys())[:500]}
        )
        docs = [
            {"id": did, "text": doc_map[did]}
            for did in pool_ids
            if did in doc_map
        ]
        query_text = row["text"]
        spans = answer_map.get(query_text.strip().lower(), [])
        samples.append(
            EvalSample(
                id=qid,
                query=query_text,
                documents=docs,
                relevant_doc_ids=relevant_ids,
                reference_answer=spans[0] if spans else "",
                answer_spans=spans,
            )
        )
    if samples:
        matched = sum(1 for s in samples if s.answer_spans)
        logger.info(
            "NQ answer spans: matched %d/%d samples (%.0f%%)",
            matched, len(samples), 100 * matched / len(samples),
        )
    return samples


def _load_nq_answers(split: str) -> dict[str, list[str]]:
    """Load answer spans from nq_open, keyed by lowered question text."""
    nq_split = {"train": "train", "dev": "validation", "test": "validation"}.get(
        split, "validation"
    )
    try:
        ds = load_dataset("google-research-datasets/nq_open", split=nq_split)
    except Exception as e:
        logger.warning("Could not load nq_open for answer spans: %s", e)
        return {}
    answer_map: dict[str, list[str]] = {}
    for row in ds:
        key = row["question"].strip().lower()
        answer_map[key] = row["answer"]  # list of valid answer strings
    logger.info("Loaded %d answer entries from nq_open.", len(answer_map))
    return answer_map


# ─────────────────────────── SciFact (BEIR) ───────────────────────────────

def _load_scifact(split: str) -> list[EvalSample]:
    """
    BEIR/SciFact: scientific claim verification.
    Each claim maps to supporting/refuting abstracts.
    """
    corpus_ds = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries_ds = load_dataset("BeIR/scifact", "queries", split="queries")

    qrel_split = {"train": "train", "dev": "validation", "test": "test"}.get(
        split, split
    )
    try:
        qrels_ds = load_dataset("BeIR/scifact-qrels", split=qrel_split)
    except Exception:
        qrels_ds = load_dataset("BeIR/scifact-qrels", split="test")

    doc_map = {str(row["_id"]): row["text"] for row in corpus_ds}

    qrel_map: dict[str, list[str]] = {}
    for row in qrels_ds:
        if int(row.get("score", 1)) > 0:
            qrel_map.setdefault(str(row["query-id"]), []).append(str(row["corpus-id"]))

    samples = []
    for row in queries_ds:
        qid = str(row["_id"])
        if qid not in qrel_map:
            continue
        relevant_ids = qrel_map[qid]
        pool_ids = list({*relevant_ids, *list(doc_map.keys())[:300]})
        docs = [
            {"id": did, "text": doc_map[did]}
            for did in pool_ids
            if did in doc_map
        ]
        samples.append(
            EvalSample(
                id=qid,
                query=row["text"],
                documents=docs,
                relevant_doc_ids=relevant_ids,
                reference_answer="",
            )
        )
    return samples


# ──────────────────────────── QASPER ──────────────────────────────────────

def _load_qasper(split: str) -> list[EvalSample]:
    """
    QASPER: QA over ArXiv NLP papers.
    Documents are paper paragraphs; answers are extractive spans or free-form.
    relevant_doc_ids tracks which paragraph indices contain the answer.
    """
    hf_split = {"train": "train", "dev": "validation", "test": "test"}.get(
        split, split
    )
    ds = load_dataset("allenai/qasper", split=hf_split)

    samples = []
    for paper in ds:
        paper_id = paper["id"]

        # Flatten all paragraphs into indexed docs
        paragraphs: list[str] = []
        for section_paras in paper["full_text"]["paragraphs"]:
            paragraphs.extend(section_paras)

        docs = [
            {"id": f"{paper_id}_p{i}", "text": p}
            for i, p in enumerate(paragraphs)
            if p.strip()
        ]

        for qa in paper["qas"]:
            answers = qa["answers"]["answer"]
            if not answers:
                continue

            # Collect all annotated answer texts and evidence paragraphs
            ref_texts, relevant_ids = [], []
            for ans in answers:
                text = _extract_qasper_answer(ans)
                if text:
                    ref_texts.append(text)
                # evidence is a list of paragraph strings
                for ev in ans.get("evidence", []):
                    for i, p in enumerate(paragraphs):
                        if ev.strip() and ev.strip() in p:
                            relevant_ids.append(f"{paper_id}_p{i}")

            if not ref_texts:
                continue

            samples.append(
                EvalSample(
                    id=f"{paper_id}::{qa['question'][:40]}",
                    query=qa["question"],
                    documents=docs,
                    relevant_doc_ids=list(set(relevant_ids)),
                    reference_answer=ref_texts[0],
                    answer_spans=ref_texts,
                )
            )
    return samples


def _extract_qasper_answer(answer: dict) -> str:
    spans = answer.get("extractive_spans") or []
    if spans:
        return " ".join(spans)
    free = answer.get("free_form_answer", "")
    if free:
        return free
    yn = answer.get("yes_no")
    if yn is not None:
        return "yes" if yn else "no"
    return ""


# ──────────────────────────── QuALITY ─────────────────────────────────────

def _load_quality(split: str) -> list[EvalSample]:
    """
    QuALITY: long-document multiple-choice QA.
    Each sample has 4 choices; correct_choice is 1-indexed.
    relevant_doc_ids is empty (no passage-level relevance labels).
    """
    hf_split = {"train": "train", "dev": "validation", "test": "test"}.get(
        split, split
    )
    try:
        ds = load_dataset("QuALITY/QuALITY.v1.0.1", split=hf_split)
    except Exception:
        ds = load_dataset("emozilla/quality", split=hf_split)

    samples = []
    for row in ds:
        article = row.get("article", "")
        # Split article into paragraph-level docs
        paras = [p.strip() for p in article.split("\n\n") if len(p.strip()) > 50]
        docs = [{"id": f"{row['article_id']}_p{i}", "text": p}
                for i, p in enumerate(paras)]

        for q in row.get("questions", []):
            correct_idx = int(q.get("gold_label", 1)) - 1  # 0-indexed
            choices = q.get("options", [])
            correct_text = choices[correct_idx] if choices else ""

            samples.append(
                EvalSample(
                    id=f"{row['article_id']}::{q['question'][:40]}",
                    query=q["question"],
                    documents=docs,
                    relevant_doc_ids=[],
                    reference_answer=correct_text,
                    choices=choices,
                    correct_choice=correct_idx,
                )
            )
    return samples
