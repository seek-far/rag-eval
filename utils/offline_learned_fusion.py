"""Offline learned top-1 fusion over saved reranker artifacts.

This script does not rerun retrieval, cross-encoders, or LLM APIs. It reads
saved per-sample artifact files, builds one feature row per (query, doc)
candidate, trains lightweight classifiers with group-based cross-validation,
and evaluates the resulting document ranking.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_RUNS = {
    "medcpt": Path("results/artifacts/2026-05-06T21_39_02_exp_013_dense_ce_medcpt/per_sample_results.jsonl"),
    "biomedbert": Path("results/artifacts/2026-05-06T21_40_18_exp_014_dense_ce_biomedbert/per_sample_results.jsonl"),
    "bge_large": Path("results/artifacts/2026-05-06T21_43_43_exp_015_dense_ce_bge_large/per_sample_results.jsonl"),
    "llm_ce": Path(
        "results/artifacts/"
        "2026-05-06T22_56_43_exp_019_dense_llm_ce_deepseek_workers50_retry_parse/"
        "per_sample_results.jsonl"
    ),
}


def load_jsonl(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        return {row["sample_id"]: row for row in (json.loads(line) for line in f)}


def safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def minmax(values: list[float], value: float) -> float:
    if not values:
        return 0.0
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return 1.0
    return (value - lo) / (hi - lo)


def unique_docs(candidates: Iterable[dict]) -> list[str]:
    docs: list[str] = []
    for candidate in candidates:
        doc_id = str(candidate["doc_id"])
        if doc_id not in docs:
            docs.append(doc_id)
    return docs


def build_dataset(data: dict[str, dict[str, dict]], include_dense_candidates: bool = True):
    names = list(data)
    sample_ids = sorted(next(iter(data.values())).keys(), key=int)
    rows: list[dict] = []
    feature_names: list[str] | None = None

    for sample_id in sample_ids:
        base = data[names[0]][sample_id]
        relevant = set(map(str, base["relevant_doc_ids"]))
        docs: list[str] = []

        for name in names:
            docs.extend(unique_docs(data[name][sample_id]["final_candidates"]))
        if include_dense_candidates:
            docs.extend(unique_docs(base.get("pre_rerank_candidates", [])))
        docs = list(dict.fromkeys(docs))

        dense_candidates = base.get("pre_rerank_candidates", [])
        dense_scores = [safe_float(c.get("retrieval_score")) for c in dense_candidates]
        dense_by_doc = {}
        for rank, candidate in enumerate(dense_candidates, 1):
            doc_id = str(candidate["doc_id"])
            dense_by_doc.setdefault(
                doc_id,
                {
                    "rank": rank,
                    "score": safe_float(candidate.get("retrieval_score")),
                },
            )

        model_maps = {}
        for name in names:
            candidates = data[name][sample_id]["final_candidates"]
            scores = [safe_float(c.get("rerank_score")) for c in candidates]
            by_doc = {}
            for rank, candidate in enumerate(candidates, 1):
                doc_id = str(candidate["doc_id"])
                by_doc.setdefault(
                    doc_id,
                    {
                        "rank": rank,
                        "score": safe_float(candidate.get("rerank_score")),
                        "score_norm": minmax(scores, safe_float(candidate.get("rerank_score"))),
                    },
                )
            model_maps[name] = by_doc

        for doc_id in docs:
            features: dict[str, float] = {}
            dense = dense_by_doc.get(doc_id)
            features["dense_present"] = 1.0 if dense else 0.0
            features["dense_rank_inv"] = 1.0 / dense["rank"] if dense else 0.0
            features["dense_rank_missing"] = 0.0 if dense else 1.0
            features["dense_score"] = dense["score"] if dense else 0.0
            features["dense_score_norm"] = minmax(dense_scores, dense["score"]) if dense else 0.0

            agreement = 0.0
            top1_agreement = 0.0
            score_norms = []
            rank_invs = []
            for name in names:
                item = model_maps[name].get(doc_id)
                prefix = f"{name}_"
                features[prefix + "present"] = 1.0 if item else 0.0
                features[prefix + "rank_inv"] = 1.0 / item["rank"] if item else 0.0
                features[prefix + "rank_missing"] = 0.0 if item else 1.0
                features[prefix + "score"] = item["score"] if item else 0.0
                features[prefix + "score_norm"] = item["score_norm"] if item else 0.0
                if item:
                    agreement += 1.0
                    score_norms.append(item["score_norm"])
                    rank_invs.append(1.0 / item["rank"])
                    if item["rank"] == 1:
                        top1_agreement += 1.0

            features["model_agreement"] = agreement
            features["top1_agreement"] = top1_agreement
            features["max_score_norm"] = max(score_norms) if score_norms else 0.0
            features["mean_score_norm"] = mean(score_norms) if score_norms else 0.0
            features["max_rank_inv"] = max(rank_invs) if rank_invs else 0.0
            features["mean_rank_inv"] = mean(rank_invs) if rank_invs else 0.0

            if feature_names is None:
                feature_names = list(features)
            rows.append(
                {
                    "sample_id": sample_id,
                    "doc_id": doc_id,
                    "query": base["query"],
                    "label": 1 if doc_id in relevant else 0,
                    "features": [features[name] for name in feature_names],
                }
            )

    assert feature_names is not None
    return rows, feature_names


def metrics_for_rank(rank: list[str], relevant: Iterable[str]) -> dict[str, float]:
    rel = set(map(str, relevant))
    out = {}
    for k in (1, 3, 5, 10):
        top = rank[:k]
        rr = next((1.0 / i for i, doc_id in enumerate(top, 1) if doc_id in rel), 0.0)
        recall = len(set(top) & rel) / len(rel) if rel else 0.0
        dcg = sum(1.0 / math.log2(i + 2) for i, doc_id in enumerate(top) if doc_id in rel)
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(rel))))
        out[f"mrr@{k}"] = rr
        out[f"hit@{k}"] = 1.0 if rr else 0.0
        out[f"recall@{k}"] = recall
        out[f"ndcg@{k}"] = dcg / ideal if ideal else 0.0
    return out


def aggregate(per_sample: list[dict[str, float]]) -> dict[str, float]:
    return {key: round(mean(row[key] for row in per_sample), 4) for key in per_sample[0]}


def evaluate_predictions(rows: list[dict], scores: np.ndarray, data: dict[str, dict[str, dict]]) -> dict[str, float]:
    by_sample: dict[str, list[tuple[str, float]]] = {}
    for row, score in zip(rows, scores):
        by_sample.setdefault(row["sample_id"], []).append((row["doc_id"], float(score)))

    base = next(iter(data.values()))
    per_sample = []
    for sample_id, scored_docs in by_sample.items():
        rank = [doc_id for doc_id, _ in sorted(scored_docs, key=lambda item: (-item[1], item[0]))]
        per_sample.append(metrics_for_rank(rank, base[sample_id]["relevant_doc_ids"]))
    return aggregate(per_sample)


def evaluate_row_scores(rows: list[dict], scores: np.ndarray, indices: np.ndarray | None = None) -> dict[str, float]:
    selected = range(len(rows)) if indices is None else indices
    by_sample: dict[str, list[tuple[str, float, int]]] = {}
    for row_idx in selected:
        row = rows[int(row_idx)]
        by_sample.setdefault(row["sample_id"], []).append((row["doc_id"], float(scores[int(row_idx)]), row["label"]))

    per_sample = []
    for scored_docs in by_sample.values():
        ranked = sorted(scored_docs, key=lambda item: (-item[1], item[0]))
        labels = [label for _, _, label in ranked]
        relevant_count = max(1, sum(labels))
        out = {}
        for k in (1, 3, 5, 10):
            top = labels[:k]
            rr = next((1.0 / i for i, label in enumerate(top, 1) if label), 0.0)
            recall = sum(top) / relevant_count
            dcg = sum(1.0 / math.log2(i + 2) for i, label in enumerate(top) if label)
            ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, relevant_count)))
            out[f"mrr@{k}"] = rr
            out[f"hit@{k}"] = 1.0 if rr else 0.0
            out[f"recall@{k}"] = recall
            out[f"ndcg@{k}"] = dcg / ideal if ideal else 0.0
        per_sample.append(out)
    return aggregate(per_sample)


def build_fixed_weight_grid(step: float, dense_weights: list[float]) -> list[tuple[float, tuple[float, float, float, float]]]:
    total = round(1.0 / step)
    combos = []
    for medcpt in range(total + 1):
        for biomedbert in range(total + 1 - medcpt):
            for bge_large in range(total + 1 - medcpt - biomedbert):
                llm_ce = total - medcpt - biomedbert - bge_large
                weights = tuple(x / total for x in (medcpt, biomedbert, bge_large, llm_ce))
                if sum(weight > 0 for weight in weights) < 2:
                    continue
                for dense_weight in dense_weights:
                    combos.append((dense_weight, weights))
    return combos


def fixed_weight_scores(x: np.ndarray, feature_names: list[str], combo: tuple[float, tuple[float, float, float, float]]):
    dense_weight, weights = combo
    score = np.zeros(x.shape[0], dtype=np.float32)
    for name, weight in zip(DEFAULT_RUNS, weights):
        if weight:
            score += weight * x[:, feature_names.index(f"{name}_score_norm")]
    if dense_weight:
        score += dense_weight * x[:, feature_names.index("dense_score_norm")]
    return score


def _score_combo_on_indices(
    combo: tuple[float, tuple[float, float, float, float]],
    x: np.ndarray,
    rows: list[dict],
    feature_names: list[str],
    train_idx: np.ndarray,
):
    scores = fixed_weight_scores(x, feature_names, combo)
    metrics = evaluate_row_scores(rows, scores, train_idx)
    return metrics["mrr@1"], metrics["mrr@10"], metrics["recall@5"], combo, metrics


def run_fixed_weight_cv(
    rows: list[dict],
    feature_names: list[str],
    folds: int,
    workers: int,
    step: float,
    dense_weights: list[float],
):
    x = np.array([row["features"] for row in rows], dtype=np.float32)
    groups = np.array([row["sample_id"] for row in rows])
    combos = build_fixed_weight_grid(step, dense_weights)
    print(
        f"[fixed-fusion] grid: step={step} dense_weights={dense_weights} "
        f"combos={len(combos)} workers={workers}",
        flush=True,
    )

    cv_per_sample = []
    chosen = []
    splitter = GroupKFold(n_splits=folds)
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(x, groups=groups), 1):
        print(
            f"[fixed-fusion] fold {fold_id}/{folds}: optimizing on {len(set(groups[train_idx]))} "
            f"queries, validating on {len(set(groups[test_idx]))} queries",
            flush=True,
        )
        scored = Parallel(n_jobs=workers, prefer="processes", verbose=0)(
            delayed(_score_combo_on_indices)(combo, x, rows, feature_names, train_idx) for combo in combos
        )
        best_mrr1, best_mrr10, best_recall5, best_combo, train_metrics = max(scored)
        chosen.append(best_combo)
        test_scores = fixed_weight_scores(x, feature_names, best_combo)
        test_metrics = evaluate_row_scores(rows, test_scores, test_idx)
        cv_per_sample.append(test_metrics)
        dense_weight, weights = best_combo
        print(
            f"[fixed-fusion] fold {fold_id}/{folds}: chose dense={dense_weight} "
            f"weights={dict(zip(DEFAULT_RUNS, weights))} "
            f"train_mrr@1={best_mrr1:.4f} test_mrr@1={test_metrics['mrr@1']:.4f}",
            flush=True,
        )

    aggregate_metrics = {key: round(mean(fold[key] for fold in cv_per_sample), 4) for key in cv_per_sample[0]}
    return {
        "metrics": aggregate_metrics,
        "fold_metrics": cv_per_sample,
        "chosen_weights": [
            {"dense": dense_weight, **dict(zip(DEFAULT_RUNS, weights))} for dense_weight, weights in chosen
        ],
        "chosen_weight_counts": [
            {"count": count, "dense": combo[0], **dict(zip(DEFAULT_RUNS, combo[1]))}
            for combo, count in Counter(chosen).most_common()
        ],
        "grid_size": len(combos),
        "step": step,
        "dense_weights": dense_weights,
    }


def evaluate_original(name: str, data: dict[str, dict[str, dict]]) -> dict[str, float]:
    per_sample = []
    for sample_id, row in data[name].items():
        rank = unique_docs(row["final_candidates"])
        per_sample.append(metrics_for_rank(rank, row["relevant_doc_ids"]))
    return aggregate(per_sample)


def evaluate_oracle(data: dict[str, dict[str, dict]]) -> dict[str, float]:
    names = list(data)
    base = data[names[0]]
    per_sample = []
    for sample_id, row in base.items():
        relevant = set(map(str, row["relevant_doc_ids"]))
        picked = None
        rest = []
        for name in names:
            rank = unique_docs(data[name][sample_id]["final_candidates"])
            if rank and rank[0] in relevant and picked is None:
                picked = rank[0]
            rest.extend(rank)
        if picked is None:
            picked = unique_docs(data["llm_ce"][sample_id]["final_candidates"])[0]
        rank = [picked] + [doc_id for doc_id in dict.fromkeys(rest) if doc_id != picked]
        per_sample.append(metrics_for_rank(rank, relevant))
    return aggregate(per_sample)


def make_models(workers: int, seed: int):
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=600,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=workers,
            random_state=seed,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=800,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=workers,
            random_state=seed,
        ),
    }


def run_cv(rows: list[dict], data: dict[str, dict[str, dict]], folds: int, workers: int, seed: int):
    x = np.array([row["features"] for row in rows], dtype=np.float32)
    y = np.array([row["label"] for row in rows], dtype=np.int32)
    groups = np.array([row["sample_id"] for row in rows])

    models = make_models(workers, seed)
    results = {}
    predictions = {}

    for model_name, model in models.items():
        print(f"[learned-fusion] training {model_name} with {folds}-fold GroupKFold", flush=True)
        scores = np.zeros(len(rows), dtype=np.float32)
        splitter = GroupKFold(n_splits=folds)
        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), 1):
            print(
                f"[learned-fusion] {model_name} fold {fold_id}/{folds}: "
                f"train_rows={len(train_idx)} test_rows={len(test_idx)}",
                flush=True,
            )
            model.fit(x[train_idx], y[train_idx])
            if hasattr(model, "predict_proba"):
                scores[test_idx] = model.predict_proba(x[test_idx])[:, 1]
            else:
                scores[test_idx] = model.decision_function(x[test_idx])
        results[model_name] = evaluate_predictions(rows, scores, data)
        predictions[model_name] = scores
        print(f"[learned-fusion] {model_name} cv metrics: {results[model_name]}", flush=True)

    return results, predictions


def save_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    feature_names: list[str],
    rows: list[dict],
    results: dict[str, dict[str, float]],
    baselines: dict[str, dict[str, float]],
    oracle: dict[str, float],
    fixed_weight_fusion: dict | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "workers": args.workers,
        "folds": args.folds,
        "include_dense_candidates": args.include_dense_candidates,
        "n_rows": len(rows),
        "n_samples": len(set(row["sample_id"] for row in rows)),
        "positive_rows": sum(row["label"] for row in rows),
        "feature_names": feature_names,
        "baselines": baselines,
        "learned_fusion": results,
        "fixed_weight_fusion": fixed_weight_fusion,
        "oracle": oracle,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Offline Fusion Summary",
        "",
        f"- Created: `{summary['created_at']}`",
        f"- Workers: `{args.workers}`",
        f"- Folds: `{args.folds}`",
        f"- Candidate rows: `{summary['n_rows']}`",
        f"- Samples: `{summary['n_samples']}`",
        f"- Positive rows: `{summary['positive_rows']}`",
        "",
        "## Baselines",
        "",
        "| Method | MRR@1 | MRR@10 | Recall@5 |",
        "|---|---:|---:|---:|",
    ]
    for name, metric in baselines.items():
        lines.append(f"| {name} | {metric['mrr@1']:.4f} | {metric['mrr@10']:.4f} | {metric['recall@5']:.4f} |")
    if results:
        lines.extend(["", "## Learned Fusion", "", "| Model | MRR@1 | MRR@10 | Recall@5 |", "|---|---:|---:|---:|"])
        for name, metric in results.items():
            lines.append(f"| {name} | {metric['mrr@1']:.4f} | {metric['mrr@10']:.4f} | {metric['recall@5']:.4f} |")
    if fixed_weight_fusion:
        metric = fixed_weight_fusion["metrics"]
        lines.extend(
            [
                "",
                "## Fixed Weight Fusion",
                "",
                "| Method | MRR@1 | MRR@10 | Recall@5 |",
                "|---|---:|---:|---:|",
                (
                    f"| 5-fold selected fixed weights | {metric['mrr@1']:.4f} | "
                    f"{metric['mrr@10']:.4f} | {metric['recall@5']:.4f} |"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Oracle",
            "",
            f"- Top-1 oracle over the four reranker outputs: `MRR@1 = {oracle['mrr@1']:.4f}`, "
            f"`MRR@10 = {oracle['mrr@10']:.4f}`, `Recall@5 = {oracle['recall@5']:.4f}`.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=max(1, int((os.cpu_count() or 1) * 2 / 3)))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-learned", action="store_true", default=True)
    parser.add_argument("--no-run-learned", dest="run_learned", action="store_false")
    parser.add_argument("--run-fixed-cv", action="store_true")
    parser.add_argument("--fixed-step", type=float, default=0.05)
    parser.add_argument("--fixed-dense-weights", default="0,0.05,0.1,0.2,0.3,0.5")
    parser.add_argument("--include-dense-candidates", action="store_true", default=True)
    parser.add_argument("--no-include-dense-candidates", dest="include_dense_candidates", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir:
        output_dir = args.output_dir
    else:
        suffix = "fixed_weight_cv" if args.run_fixed_cv and not args.run_learned else "learned_top1_fusion"
        output_dir = Path("results/offline_fusion") / f"{datetime.now().strftime('%Y-%m-%dT%H_%M_%S')}_{suffix}"

    print(f"[learned-fusion] loading artifacts", flush=True)
    data = {name: load_jsonl(path) for name, path in DEFAULT_RUNS.items()}
    rows, feature_names = build_dataset(data, include_dense_candidates=args.include_dense_candidates)
    print(
        f"[learned-fusion] built dataset: samples={len(set(r['sample_id'] for r in rows))} "
        f"rows={len(rows)} positives={sum(r['label'] for r in rows)} "
        f"features={len(feature_names)} workers={args.workers}",
        flush=True,
    )

    baselines = {name: evaluate_original(name, data) for name in DEFAULT_RUNS}
    oracle = evaluate_oracle(data)
    print(f"[learned-fusion] baselines: {baselines}", flush=True)
    print(f"[learned-fusion] oracle: {oracle}", flush=True)

    results = {}
    if args.run_learned:
        results, _ = run_cv(rows, data, args.folds, args.workers, args.seed)
    fixed_weight_fusion = None
    if args.run_fixed_cv:
        dense_weights = [float(value) for value in args.fixed_dense_weights.split(",") if value.strip()]
        fixed_weight_fusion = run_fixed_weight_cv(
            rows,
            feature_names,
            args.folds,
            args.workers,
            args.fixed_step,
            dense_weights,
        )
        print(f"[fixed-fusion] cv metrics: {fixed_weight_fusion['metrics']}", flush=True)

    save_outputs(output_dir, args, feature_names, rows, results, baselines, oracle, fixed_weight_fusion)
    print(f"[learned-fusion] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
