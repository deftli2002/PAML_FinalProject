# Models

Machine learning models for the FDA Drug Recall Prediction project (RegLens), all implemented from scratch using NumPy.

## Files

| File | Description |
|------|-------------|
| `logistic_regression.py` | Logistic regression with class weighting, L2 regularization, mini-batch SGD, early stopping |
| `random_forest.py` | Random forest from CART decision trees using Gini impurity and bootstrap sampling |
| `linear_regression.py` | Ridge regression baseline, output used as a ranking score |
| `evaluate_predictions.py` | Computes PR-AUC, ROC-AUC, precision, recall, F1, Recall@Top-10% from a CSV |

No external ML libraries are used. All training is implemented in NumPy.

## How to Evaluate

```bash
python evaluate_predictions.py --csv-path predictions/logistic_regression_predictions.csv
python evaluate_predictions.py --csv-path predictions/random_forest_predictions.csv
python evaluate_predictions.py --csv-path predictions/linear_regression_predictions.csv
```

## Final Evaluation Results

- **Dataset**: `improved_product_recall_features_2004_2008.db`
- **Split**: stratified 70% train / 15% validation / 15% test, random seed 42
- **Test set**: 7,790 records, 403 positive recall labels
- **Threshold**: selected on validation set by maximizing F1-score

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Recall@Top-10% | TP | FP | TN | FN |
|-------|-------:|--------:|----------:|-------:|---:|---------------:|---:|---:|---:|---:|
| Linear Regression Baseline | 0.1164 | 0.6777 | 0.1147 | 0.3102 | 0.1674 | 0.2481 | 125 | 965 | 6422 | 278 |
| Logistic Regression | 0.1119 | 0.6884 | 0.1008 | 0.4442 | 0.1643 | 0.2432 | 179 | 1597 | 5790 | 224 |
| Random Forest | 0.2331 | 0.8325 | 0.2625 | 0.3375 | 0.2953 | 0.4194 | 136 | 382 | 7005 | 267 |

Random Forest achieved the highest ROC-AUC (0.8325) and F1-score (0.2953) and was selected as the deployed model. Logistic Regression had the highest Recall (0.4442) but more false positives. The Linear Regression Baseline is included for ranking comparison only.

## Model Details

### Logistic Regression
- Binary cross-entropy loss with balanced class weighting
- L2 regularization (lambda = 0.001), mini-batch gradient descent (batch size 256)
- Feature standardization fit on training set only
- Early stopping on validation loss (patience 50)

### Random Forest
- 100 trees, max depth 12, bootstrap sampling per tree
- Random feature subset at each split (sqrt of total features)
- Balanced class weighting via sample weights, Gini impurity for splits

### Linear Regression Baseline
- Ridge regression with closed-form solution
- Output clipped to [0, 1] and used as a continuous ranking score
- Included for AUC comparison only, not optimized for threshold-based classification
