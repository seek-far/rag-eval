"""
chunker.py — Three chunking strategies controlled via Config.

  fixed               : split by token count with overlap
  semantic            : split at cosine-similarity drop points
  section_then_semantic : detect section headers first, then semantic-split
                         each section (best for ArXiv / structured docs)

Every chunk carries full metadata so context_builder can do parent-chunk
retrieval without an extra DB lookup.
"""
from __future__ import annotations

import re
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Section header patterns (covers ArXiv / academic papers) ──────────────
_SECTION_RE = re.compile(
    r"^(?:"
    r"#{1,3}\s+.+"                        # Markdown ## / ###
    r"|(?:\d+\.?)+\s+[A-Z][A-Za-z\s]+"   # 1. / 1.1 Introduction
    r"|[A-Z][A-Z\s]{3,40}$"              # ALL-CAPS HEADER
    r"|(?:Abstract|Introduction|Background|Related Work|"
    r"Methodology|Methods?|Experiments?|Results?|"
    r"Discussion|Conclusion|References|Appendix)"
    r"[\s\n:]"
    r")",
    re.MULTILINE,
)

# Sections we skip during indexing (noisy, no QA value)
_SKIP_SECTIONS = {"references", "appendix"}


# ─────────────────────────── Public entry point ───────────────────────────

def build_chunks(
    documents: list[dict],  # [{"id": str, "text": str}, ...]
    cfg,
) -> list[dict]:
    """
    Returns a flat list of chunk dicts.  Each chunk has:
      chunk_id        unique string
      doc_id          source document id
      doc_text        full source document text (for parent-chunk retrieval)
      section_type    normalised section name (semantic/section mode)
      section_text    full section text (for parent-chunk retrieval)
      text            the chunk text used for embedding & BM25
    """
    strategy = cfg.chunk_strategy

    if strategy == "fixed":
        chunker = _FixedChunker(cfg.chunk_size, cfg.chunk_overlap)
    elif strategy == "semantic":
        chunker = _SemanticChunker(
            cfg.embed_model,
            cfg.embed_device,
            cfg.semantic_threshold,
            cfg.chunk_min_chars,
            cfg.chunk_max_chars,
        )
    elif strategy == "section_then_semantic":
        chunker = _SectionSemanticChunker(
            cfg.embed_model,
            cfg.embed_device,
            cfg.semantic_threshold,
            cfg.chunk_min_chars,
            cfg.chunk_max_chars,
        )
    else:
        raise ValueError(f"Unknown chunk_strategy: {strategy!r}")

    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk(doc["text"], doc["id"])
        all_chunks.extend(chunks)

    logger.debug(
        "Built %d chunks from %d documents (strategy=%s)",
        len(all_chunks),
        len(documents),
        strategy,
    )
    return all_chunks


# ─────────────────────────── Fixed chunker ────────────────────────────────

class _FixedChunker:
    def __init__(self, size: int, overlap: int):
        self.size = size
        self.overlap = overlap

    def chunk(self, text: str, doc_id: str) -> list[dict]:
        words = text.split()
        step = max(1, self.size - self.overlap)
        chunks = []
        for i, start in enumerate(range(0, len(words), step)):
            chunk_text = " ".join(words[start : start + self.size])
            if len(chunk_text) < 20:
                continue
            chunks.append(
                _make_chunk(
                    doc_id=doc_id,
                    doc_text=text,
                    section_type="body",
                    section_text=text,
                    chunk_idx=i,
                    text=chunk_text,
                )
            )
        return chunks


# ─────────────────────────── Semantic chunker ─────────────────────────────

class _SemanticChunker:
    def __init__(
        self,
        model_name: str,
        device: str,
        threshold: float,
        min_chars: int,
        max_chars: int,
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.threshold = threshold
        self.min_chars = min_chars
        self.max_chars = max_chars

    def chunk(self, text: str, doc_id: str) -> list[dict]:
        sentences = _split_sentences(text)
        if len(sentences) <= 2:
            return [_make_chunk(doc_id, text, "body", text, 0, text)]

        embeddings = self.model.encode(
            sentences, show_progress_bar=False, convert_to_numpy=True
        )
        groups = self._group(sentences, embeddings)
        return [
            _make_chunk(doc_id, text, "body", text, i, " ".join(g))
            for i, g in enumerate(groups)
        ]

    def _group(
        self, sentences: list[str], embeddings: np.ndarray
    ) -> list[list[str]]:
        breakpoints: set[int] = set()
        for i in range(len(embeddings) - 1):
            sim = _cosine(embeddings[i], embeddings[i + 1])
            if sim < self.threshold:
                breakpoints.add(i)

        groups: list[list[str]] = []
        current: list[str] = []
        for i, sent in enumerate(sentences):
            current.append(sent)
            if i in breakpoints:
                merged = " ".join(current)
                if len(merged) < self.min_chars and groups:
                    groups[-1].extend(current)
                else:
                    groups.append(current)
                current = []
        if current:
            groups.append(current)

        # Hard-split groups that exceed max_chars
        final: list[list[str]] = []
        for g in groups:
            if len(" ".join(g)) > self.max_chars:
                final.extend(_hard_split(g, self.max_chars))
            else:
                final.append(g)
        return final


# ─────────────── Section-then-semantic chunker ────────────────────────────

class _SectionSemanticChunker:
    """
    Stage 1: split text by section headers → list of (section_type, text)
    Stage 2: apply SemanticChunker within each section
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        threshold: float,
        min_chars: int,
        max_chars: int,
    ):
        self._sem = _SemanticChunker(
            model_name, device, threshold, min_chars, max_chars
        )

    def chunk(self, text: str, doc_id: str) -> list[dict]:
        sections = _split_sections(text)
        all_chunks: list[dict] = []
        chunk_idx = 0
        for sec_type, sec_text in sections:
            if sec_type in _SKIP_SECTIONS:
                continue
            sentences = _split_sentences(sec_text)
            if not sentences:
                continue
            if len(sentences) <= 2:
                all_chunks.append(
                    _make_chunk(doc_id, text, sec_type, sec_text, chunk_idx, sec_text)
                )
                chunk_idx += 1
                continue
            embeddings = self._sem.model.encode(
                sentences, show_progress_bar=False, convert_to_numpy=True
            )
            groups = self._sem._group(sentences, embeddings)
            for g in groups:
                all_chunks.append(
                    _make_chunk(
                        doc_id, text, sec_type, sec_text, chunk_idx, " ".join(g)
                    )
                )
                chunk_idx += 1
        return all_chunks


# ─────────────────────────── Helpers ──────────────────────────────────────

def _make_chunk(
    doc_id: str,
    doc_text: str,
    section_type: str,
    section_text: str,
    chunk_idx: int,
    text: str,
) -> dict:
    return {
        "chunk_id": f"{doc_id}__c{chunk_idx}",
        "doc_id": doc_id,
        "doc_text": doc_text,          # full doc — for parent-chunk retrieval
        "section_type": section_type,
        "section_text": section_text,  # full section — for context enrichment
        "text": text,
    }


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [p.strip() for p in parts if len(p.strip()) > 20]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


def _hard_split(sentences: list[str], max_chars: int) -> list[list[str]]:
    result, current, length = [], [], 0
    for s in sentences:
        if length + len(s) > max_chars and current:
            result.append(current)
            current, length = [], 0
        current.append(s)
        length += len(s)
    if current:
        result.append(current)
    return result


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Return list of (normalised_section_type, section_text)."""
    boundaries: list[tuple[int, str]] = []
    for m in _SECTION_RE.finditer(text):
        title = m.group(0).strip().rstrip(":")
        boundaries.append((m.start(), title))

    if not boundaries:
        return [("body", text)]

    sections = []
    for i, (start, title) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        body = text[start:end].strip()
        sections.append((_normalise_section(title), body))
    return sections


_SECTION_MAP = {
    "abstract": "abstract",
    "introduction": "introduction",
    "related work": "related_work",
    "background": "background",
    "method": "methods",
    "experiment": "experiments",
    "result": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "references": "references",
    "appendix": "appendix",
}


def _normalise_section(title: str) -> str:
    t = title.lower()
    for key, val in _SECTION_MAP.items():
        if key in t:
            return val
    return "body"
