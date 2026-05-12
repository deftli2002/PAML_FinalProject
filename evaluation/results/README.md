# Final Model Evaluation

- Feature database: `/Users/alextang/Downloads/paml projct/FDA_Drug_Recalls_Predicting_System/data/improved_product_recall_features_2004_2008.db`
- Split: stratified 70% train / 15% validation / 15% test
- Random seed: `42`
- Test rows: `7790` with `403` positive recall labels
- Thresholds: chosen on the validation split
- Metrics: test split only

## Metrics

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Recall@Top-10% | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear_regression_baseline | 0.1164 | 0.6777 | 0.1147 | 0.3102 | 0.1674 | 0.2481 | 125 | 965 | 6422 | 278 |
| logistic_regression | 0.1119 | 0.6884 | 0.1008 | 0.4442 | 0.1643 | 0.2432 | 179 | 1597 | 5790 | 224 |
| random_forest | 0.2331 | 0.8325 | 0.2625 | 0.3375 | 0.2953 | 0.4194 | 136 | 382 | 7005 | 267 |

Prediction CSV files and `final_metrics.csv` are in this directory.
