#!/usr/bin/env python3
"""Run the final offline model evaluation for the FDA recall project."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate_predictions import (  # noqa: E402
    confusion_counts,
    pr_auc,
    precision_recall_f1,
    recall_at_top_fraction,
    roc_auc,
)
from models.linear_regression import LinearRegressionBaseline  # noqa: E402
from models.logistic_regression import LogisticRegressionClassifier  # noqa: E402
from models.random_forest import RandomForestClassifier  # noqa: E402


FEATURE_COLUMNS = [
    "faers_report_count_total",
    "unique_safetyreport_count",
    "faers_report_count_last_365d",
    "faers_report_count_prev_365d",
    "faers_report_growth_365d",
    "serious_count",
    "serious_rate",
    "death_count",
    "death_rate",
    "hospitalization_count",
    "hospitalization_rate",
    "lifethreatening_count",
    "lifethreatening_rate",
    "disabling_count",
    "disabling_rate",
    "other_serious_count",
    "other_serious_rate",
    "suspect_drug_count",
    "suspect_drug_rate",
    "reaction_count",
    "unique_reaction_count",
    "reaction_per_report_mean",
    "unique_indication_count",
    "unique_route_count",
    "indication_score_sum",
    "indication_score_mean",
    "indication_score_max",
    "high_risk_indication_count",
    "high_risk_indication_rate",
    "route_score_sum",
    "route_score_mean",
    "route_score_max",
    "high_risk_route_count",
    "high_risk_route_rate",
    "has_application_number",
    "has_brand_name",
    "has_generic_name",
    "has_substance_name",
    "has_manufacturer_name",
    "has_drugindication",
    "has_administrationroute",
    "active_days",
    "active_months",
    "report_count_per_active_month",
    "patient_age_mean",
    "patient_age_missing_rate",
    "female_rate",
    "male_rate",
]


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final FDA recall model evaluation.")
    parser.add_argument(
        "--db-path",
        default=str(REPO_ROOT / "data" / "improved_product_recall_features_2004_2008.db"),
        help="Product feature SQLite database.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "evaluation" / "results"),
        help="Directory for predictions, metrics, and protocol files.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--threshold-grid-size", type=int, default=101)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--rf-estimators", type=int, default=25)
    parser.add_argument("--rf-max-depth", type=int, default=8)
    parser.add_argument("--rf-min-samples-split", type=int, default=50)
    return parser.parse_args()


def load_dataset(db_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    if not db_path.is_file():
        raise FileNotFoundError(f"Feature database not found: {db_path}")
    query = (
        "SELECT entity_key, label_recalled, "
        + ", ".join(FEATURE_COLUMNS)
        + " FROM product_features ORDER BY entity_key"
    )
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(query).fetchall()
    if not rows:
        raise RuntimeError(f"No product_features rows found in {db_path}")
    entity_keys = [str(row[0]) for row in rows]
    labels = np.asarray([int(row[1]) for row in rows], dtype=int)
    features = np.asarray([row[2:] for row in rows], dtype=float)
    return entity_keys, labels, features


def stratified_split(
    y: np.ndarray,
    *,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> Split:
    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError("train/val/test fractions must sum to 1.0")

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for label in sorted(set(y.tolist())):
        idx = np.flatnonzero(y == label)
        idx = rng.permutation(idx)
        n_train = int(round(len(idx) * train_frac))
        n_val = int(round(len(idx) * val_frac))
        train_parts.append(idx[:n_train])
        val_parts.append(idx[n_train:n_train + n_val])
        test_parts.append(idx[n_train + n_val:])

    train = rng.permutation(np.concatenate(train_parts))
    val = rng.permutation(np.concatenate(val_parts))
    test = rng.permutation(np.concatenate(test_parts))
    return Split(train=train, val=val, test=test)


def rows_from_scores(y: np.ndarray, scores: np.ndarray, threshold: float) -> list[tuple[int, float, int]]:
    return [
        (int(label), float(score), 1 if float(score) >= threshold else 0)
        for label, score in zip(y, scores)
    ]


def choose_threshold(y_val: np.ndarray, scores: np.ndarray, grid_size: int) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.0, 1.0, grid_size):
        rows = rows_from_scores(y_val, scores, float(threshold))
        _precision, _recall, f1 = precision_recall_f1(rows)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold, best_f1


def evaluate_model(
    *,
    model_name: str,
    entity_keys: list[str],
    y_test: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    top_fraction: float,
    output_dir: Path,
) -> dict[str, float | int | str]:
    rows = rows_from_scores(y_test, scores, threshold)
    tp, fp, tn, fn = confusion_counts(rows)
    precision, recall, f1 = precision_recall_f1(rows)
    result = {
        "model": model_name,
        "rows": len(rows),
        "positive_rows": int(y_test.sum()),
        "threshold": threshold,
        "pr_auc": pr_auc(rows),
        "roc_auc": roc_auc(rows),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "recall_at_top_10pct": recall_at_top_fraction(rows, top_fraction),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }

    prediction_path = output_dir / f"{model_name}_test_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["entity_key", "label", "score", "pred"])
        for key, (label, score, pred) in zip(entity_keys, rows):
            writer.writerow([key, label, f"{score:.10f}", pred])
    result["prediction_file"] = prediction_path.name
    return result


def write_metrics(output_dir: Path, metrics: list[dict[str, float | int | str]]) -> None:
    fields = [
        "model",
        "rows",
        "positive_rows",
        "threshold",
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "recall_at_top_10pct",
        "tp",
        "fp",
        "tn",
        "fn",
        "prediction_file",
    ]
    with (output_dir / "final_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)


def write_protocol(
    *,
    output_dir: Path,
    db_path: Path,
    split: Split,
    y: np.ndarray,
    args: argparse.Namespace,
    metrics: list[dict[str, float | int | str]],
) -> None:
    payload = {
        "db_path": str(db_path),
        "seed": args.seed,
        "split": {
            "train_fraction": args.train_frac,
            "validation_fraction": args.val_frac,
            "test_fraction": args.test_frac,
            "train_rows": int(len(split.train)),
            "validation_rows": int(len(split.val)),
            "test_rows": int(len(split.test)),
            "train_positive_rows": int(y[split.train].sum()),
            "validation_positive_rows": int(y[split.val].sum()),
            "test_positive_rows": int(y[split.test].sum()),
        },
        "features": FEATURE_COLUMNS,
        "models": {
            "linear_regression_baseline": {"l2": 1.0, "standardize": True},
            "logistic_regression": {
                "lr": 0.05,
                "max_iter": 800,
                "batch_size": 512,
                "l2": 0.001,
                "class_weight": "balanced",
                "early_stopping_patience": 50,
                "random_state": args.seed,
            },
            "random_forest": {
                "n_estimators": args.rf_estimators,
                "max_depth": args.rf_max_depth,
                "min_samples_split": args.rf_min_samples_split,
                "max_features": "sqrt",
                "class_weight": "balanced",
                "random_state": args.seed,
            },
        },
    }
    with (output_dir / "evaluation_protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    lines = [
        "# Final Model Evaluation",
        "",
        f"- Feature database: `{db_path}`",
        f"- Split: stratified {args.train_frac:.0%} train / {args.val_frac:.0%} validation / {args.test_frac:.0%} test",
        f"- Random seed: `{args.seed}`",
        f"- Test rows: `{len(split.test)}` with `{int(y[split.test].sum())}` positive recall labels",
        "- Thresholds: selected on the validation split by maximizing F1-score",
        "- Final metrics: computed only on the held-out test split",
        "",
        "## Metrics",
        "",
        "| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Recall@Top-10% | TP | FP | TN | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            "| {model} | {pr_auc:.4f} | {roc_auc:.4f} | {precision:.4f} | "
            "{recall:.4f} | {f1:.4f} | {recall_at_top_10pct:.4f} | "
            "{tp} | {fp} | {tn} | {fn} |".format(**row)
        )
    lines.append("")
    lines.append("Prediction CSV files and `final_metrics.csv` are in this directory.")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    db_path = Path(args.db_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {db_path}")
    entity_keys, y, X = load_dataset(db_path)
    print(f"Rows={len(y):,} features={X.shape[1]} positives={int(y.sum()):,}")

    split = stratified_split(
        y,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )
    print(
        "Split sizes: "
        f"train={len(split.train):,}, val={len(split.val):,}, test={len(split.test):,}"
    )

    X_train, y_train = X[split.train], y[split.train]
    X_val, y_val = X[split.val], y[split.val]
    X_test, y_test = X[split.test], y[split.test]
    test_keys = [entity_keys[i] for i in split.test]

    model_specs = [
        (
            "linear_regression_baseline",
            LinearRegressionBaseline(l2=1.0, standardize=True),
        ),
        (
            "logistic_regression",
            LogisticRegressionClassifier(
                lr=0.05,
                max_iter=800,
                batch_size=512,
                l2=1e-3,
                class_weight="balanced",
                early_stopping_patience=50,
                verbose=False,
                random_state=args.seed,
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=args.rf_estimators,
                max_depth=args.rf_max_depth,
                min_samples_split=args.rf_min_samples_split,
                max_features="sqrt",
                class_weight="balanced",
                random_state=args.seed,
                verbose=True,
            ),
        ),
    ]

    metrics: list[dict[str, float | int | str]] = []
    for name, model in model_specs:
        print(f"\nTraining {name}...")
        model.train(X_train, y_train, X_val, y_val)
        val_scores = model.predict_prob(X_val)
        threshold, val_f1 = choose_threshold(y_val, val_scores, args.threshold_grid_size)
        print(f"Selected threshold={threshold:.4f} on validation F1={val_f1:.4f}")
        test_scores = model.predict_prob(X_test)
        result = evaluate_model(
            model_name=name,
            entity_keys=test_keys,
            y_test=y_test,
            scores=test_scores,
            threshold=threshold,
            top_fraction=args.top_fraction,
            output_dir=output_dir,
        )
        metrics.append(result)
        print(
            f"Test ROC-AUC={result['roc_auc']:.4f} "
            f"PR-AUC={result['pr_auc']:.4f} F1={result['f1']:.4f}"
        )

    write_metrics(output_dir, metrics)
    write_protocol(
        output_dir=output_dir,
        db_path=db_path,
        split=split,
        y=y,
        args=args,
        metrics=metrics,
    )
    print(f"\nWrote final evaluation outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
