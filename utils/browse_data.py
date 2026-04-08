"""
browse_data.py — Browse dataset samples: documents, chunks, and queries.

Usage:
    python -m utils.browse_data                          # defaults: nq, test, 200 samples
    python -m utils.browse_data --dataset scifact --n 50
    python -m utils.browse_data --sample-id 42           # jump to a specific sample
    python -m utils.browse_data --show chunks            # auto-show chunks for each sample
"""
from __future__ import annotations

import argparse
import pickle
import sys
import textwrap
from pathlib import Path

# Allow running as `python -m utils.browse_data` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from dataloader.loader import load_eval_samples
from indexing.chunker import build_chunks, create_chunker


def _trunc(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _wrap(text: str, width: int = 100, indent: str = "    ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)


def _print_sample_summary(idx: int, sample, total: int) -> None:
    print(f"\n{'='*80}")
    print(f"  Sample [{idx+1}/{total}]  id={sample.id}")
    print(f"{'='*80}")
    print(f"  Query: {sample.query}")
    print(f"  Documents: {len(sample.documents)}")
    print(f"  Relevant doc IDs: {sample.relevant_doc_ids}")
    if sample.reference_answer:
        print(f"  Reference answer: {_trunc(sample.reference_answer, 150)}")
    if sample.answer_spans:
        print(f"  Answer spans: {sample.answer_spans[:3]}{'...' if len(sample.answer_spans) > 3 else ''}")
    if sample.choices:
        print(f"  Choices ({len(sample.choices)}): correct={sample.correct_choice}")
        for ci, ch in enumerate(sample.choices):
            marker = " *" if ci == sample.correct_choice else ""
            print(f"    [{ci}]{marker} {_trunc(ch, 120)}")


def _print_documents(sample) -> None:
    print(f"\n  --- Documents ({len(sample.documents)}) ---")
    for i, doc in enumerate(sample.documents):
        is_rel = doc["id"] in sample.relevant_doc_ids
        tag = " [RELEVANT]" if is_rel else ""
        print(f"\n  Doc {i+1}: id={doc['id']}{tag}")
        print(_wrap(_trunc(doc["text"], 500)))


def _print_chunks(chunks: list[dict]) -> None:
    print(f"\n  --- Chunks ({len(chunks)}) ---")
    for i, c in enumerate(chunks):
        print(f"\n  Chunk {i+1}: id={c['chunk_id']}  doc_id={c['doc_id']}  "
              f"section={c.get('section_type', 'N/A')}  len={len(c['text'])}")
        print(_wrap(_trunc(c["text"], 400)))


def _load_chunks_for_sample(sample, cfg, chunker) -> list[dict]:
    """Build chunks for a single sample's documents."""
    return build_chunks(sample.documents, cfg, chunker=chunker)


def _interactive_loop(samples, cfg) -> None:
    total = len(samples)
    chunker = None  # lazy init
    idx = 0

    print(f"\nLoaded {total} samples from {cfg.dataset}/{cfg.split}")
    print("Commands: [Enter]=next  [p]=prev  [d]=docs  [c]=chunks  [q]=query detail")
    print("          [g N]=goto N  [s TEXT]=search query  [h]=help  [x]=exit\n")

    while True:
        sample = samples[idx]
        _print_sample_summary(idx, sample, total)

        cmd = input("\n> ").strip().lower()

        if cmd in ("", "n"):
            idx = min(idx + 1, total - 1)
        elif cmd == "p":
            idx = max(idx - 1, 0)
        elif cmd == "d":
            _print_documents(sample)
        elif cmd == "c":
            if chunker is None:
                print("  (Initialising chunker...)")
                chunker = create_chunker(cfg)
            chunks = _load_chunks_for_sample(sample, cfg, chunker)
            _print_chunks(chunks)
        elif cmd == "q":
            print(f"\n  Query: {sample.query}")
            if sample.reference_answer:
                print(f"\n  Reference answer:\n{_wrap(sample.reference_answer)}")
            if sample.answer_spans:
                print(f"\n  Answer spans:")
                for sp in sample.answer_spans:
                    print(f"    - {sp}")
        elif cmd.startswith("g "):
            try:
                n = int(cmd[2:]) - 1
                idx = max(0, min(n, total - 1))
            except ValueError:
                print("  Usage: g <number>")
        elif cmd.startswith("s "):
            query_text = cmd[2:]
            matches = [(i, s) for i, s in enumerate(samples)
                       if query_text in s.query.lower()]
            if matches:
                print(f"  Found {len(matches)} match(es):")
                for mi, (si, s) in enumerate(matches[:10]):
                    print(f"    [{si+1}] {_trunc(s.query, 100)}")
                if len(matches) == 1:
                    idx = matches[0][0]
            else:
                print("  No matches found.")
        elif cmd == "h":
            print("\n  Commands:")
            print("    [Enter]/n  next sample")
            print("    p          previous sample")
            print("    d          show documents for this sample")
            print("    c          show chunks for this sample")
            print("    q          show full query + answer details")
            print("    g N        go to sample N")
            print("    s TEXT     search queries containing TEXT")
            print("    h          this help")
            print("    x          exit")
        elif cmd == "x":
            break
        else:
            print(f"  Unknown command: {cmd!r}. Type 'h' for help.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse RAG evaluation data")
    parser.add_argument("--dataset", default=None, help="Dataset name (nq, scifact, qasper, quality)")
    parser.add_argument("--split", default=None, help="Dataset split")
    parser.add_argument("--n", type=int, default=0, help="Number of samples to load (0=use env)")
    parser.add_argument("--sample-id", default=None, help="Jump to sample with this ID")
    parser.add_argument("--show", choices=["summary", "docs", "chunks"], default="summary",
                        help="What to show in non-interactive mode")
    parser.add_argument("--list", action="store_true", help="List all samples non-interactively and exit")
    args = parser.parse_args()

    # Build config from env, with CLI overrides
    cfg = Config()
    if args.dataset:
        cfg.dataset = args.dataset
    if args.split:
        cfg.split = args.split

    n = args.n if args.n > 0 else cfg.total_samples
    samples = load_eval_samples(cfg.dataset, cfg.split, n)[:n]

    if args.list:
        print(f"\n{'#':>4}  {'ID':>10}  {'Docs':>4}  {'Rel':>3}  Query")
        print("-" * 100)
        for i, s in enumerate(samples):
            print(f"{i+1:>4}  {s.id:>10}  {len(s.documents):>4}  "
                  f"{len(s.relevant_doc_ids):>3}  {_trunc(s.query, 70)}")
        print(f"\nTotal: {len(samples)} samples")
        return

    if args.sample_id:
        found = [i for i, s in enumerate(samples) if s.id == args.sample_id]
        if not found:
            print(f"Sample ID '{args.sample_id}' not found.")
            return

    _interactive_loop(samples, cfg)


if __name__ == "__main__":
    main()
