# Evaluation

Scripts and saved outputs for model evaluation.

## Files

- `inspect_product_dataset.py`
  Read a product-level feature SQLite database and print dataset size, label
  balance, and a few feature summaries.
- `evaluate_predictions.py`
  Score a prediction CSV with PR-AUC, ROC-AUC, F1-score, and Recall@Top-K%.
- `run_final_evaluation.py`
  Run the train/validation/test experiment and save the result files.

## Midpoint usage

Inspect the feature database:

```bash
python3 evaluation/inspect_product_dataset.py \
  --db-path data/product_recall_features_2004_2005.db
```

## Final Evaluation

Run from the repository root:

```bash
python3 evaluation/run_final_evaluation.py
```

This uses `data/improved_product_recall_features_2004_2008.db`, a stratified
70%/15%/15% train/validation/test split, and `random_state=42`.
Thresholds are chosen on the validation set.

Outputs:

- `evaluation/results/final_metrics.csv`
- `evaluation/results/evaluation_protocol.json`
- `evaluation/results/linear_regression_baseline_test_predictions.csv`
- `evaluation/results/logistic_regression_test_predictions.csv`
- `evaluation/results/random_forest_test_predictions.csv`

Test metrics:

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Recall@Top-10% |
|---|---:|---:|---:|---:|---:|---:|
| Linear regression baseline | 0.1164 | 0.6777 | 0.1147 | 0.3102 | 0.1674 | 0.2481 |
| Logistic regression | 0.1119 | 0.6884 | 0.1008 | 0.4442 | 0.1643 | 0.2432 |
| Random forest | 0.2331 | 0.8325 | 0.2625 | 0.3375 | 0.2953 | 0.4194 |

## Standalone Prediction Scoring

Export a CSV with at least:

- `label`
- `score`

Then run:

```bash
python3 evaluation/evaluate_predictions.py \
  --csv-path path/to/predictions.csv \
  --label-col label \
  --score-col score
```

Optional column:

- `pred`
  If present, it is used for threshold-based metrics. Otherwise scores use
  `--threshold`.
