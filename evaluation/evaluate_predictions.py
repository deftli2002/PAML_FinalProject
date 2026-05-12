#!/usr/bin/env python3
"""Score binary classification predictions from a CSV file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score model predictions from a CSV file."
    )
    parser.add_argument("--csv-path", required=True, help="Path to the predictions CSV.")
    parser.add_argument("--label-col", default="label", help="Ground-truth label column.")
    parser.add_argument("--score-col", default="score", help="Prediction score column.")
    parser.add_argument("--pred-col", default="pred", help="Optional predicted label column.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold to use when pred column is absent.",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.10,
        help="Top fraction for Recall@Top-K%%. Default: 0.10.",
    )
    return parser.parse_args()


def read_rows(
    csv_path: Path,
    *,
    label_col: str,
    score_col: str,
    pred_col: str,
    threshold: float,
) -> list[tuple[int, float, int]]:
    rows: list[tuple[int, float, int]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            label = int(float(raw[label_col]))
            score = float(raw[score_col])
            if pred_col in raw and raw[pred_col] not in (None, ""):
                pred = int(float(raw[pred_col]))
            else:
                pred = 1 if score >= threshold else 0
            rows.append((label, score, pred))
    return rows


def confusion_counts(rows: list[tuple[int, float, int]]) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for label, _score, pred in rows:
        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 0 and pred == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def precision_recall_f1(rows: list[tuple[int, float, int]]) -> tuple[float, float, float]:
    tp, fp, _tn, fn = confusion_counts(rows)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def recall_at_top_fraction(rows: list[tuple[int, float, int]], top_fraction: float) -> float:
    if not rows:
        return 0.0
    ranked = sorted(rows, key=lambda item: item[1], reverse=True)
    k = max(1, int(len(ranked) * top_fraction))
    top_rows = ranked[:k]
    total_positive = sum(label for label, _score, _pred in ranked)
    captured_positive = sum(label for label, _score, _pred in top_rows)
    return safe_div(captured_positive, total_positive)


def roc_auc(rows: list[tuple[int, float, int]]) -> float:
    positives = [(score) for label, score, _pred in rows if label == 1]
    negatives = [(score) for label, score, _pred in rows if label == 0]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    total = len(positives) * len(negatives)
    for p in positives:
        for n in negatives:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / total


def pr_auc(rows: list[tuple[int, float, int]]) -> float:
    ranked = sorted(rows, key=lambda item: item[1], reverse=True)
    total_positive = sum(label for label, _score, _pred in ranked)
    if total_positive == 0:
        return 0.0

    area = 0.0
    prev_recall = 0.0
    tp = 0
    fp = 0
    for label, _score, _pred in ranked:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, total_positive)
        area += precision * (recall - prev_recall)
        prev_recall = recall
    return area


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.is_file():
        print(f"Error: CSV not found: {csv_path}")
        return 1

    rows = read_rows(
        csv_path,
        label_col=args.label_col,
        score_col=args.score_col,
        pred_col=args.pred_col,
        threshold=args.threshold,
    )
    if not rows:
        print("Error: no prediction rows found.")
        return 1

    tp, fp, tn, fn = confusion_counts(rows)
    precision, recall, f1 = precision_recall_f1(rows)
    roc = roc_auc(rows)
    pr = pr_auc(rows)
    topk_recall = recall_at_top_fraction(rows, args.top_fraction)

    print(f"Rows: {len(rows):,}")
    print(f"Threshold: {args.threshold:.4f}")
    print(f"PR-AUC: {pr:.6f}")
    print(f"ROC-AUC: {roc:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1-score: {f1:.6f}")
    print(f"Recall@Top-{args.top_fraction * 100:.0f}%: {topk_recall:.6f}")
    print(f"Confusion matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
