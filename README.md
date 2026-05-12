# PAML Final Project
This is my final project for PAML.  
It is a pretty simple pipeline for predicting FDA drug recall risk.

## Project Structure
```
PAML_FinalProject/
├── data/
│   ├── product_recall_features_2004_2008.db
│   ├── improved_product_recall_features_2004_2008.db
│   └── data_exploration/
│       ├── fig1_faers_reports_by_year.png
│       ├── fig2_enforcement_recalls_by_year.png
│       ├── fig3_recall_classification.png
│       └── fig4_serious_faers.png
├── feature_engineering/
│   └── build_product_feature_database.py
├── models/
│   ├── logistic_regression.py
│   ├── random_forest.py
│   ├── linear_regression.py
│   └── evaluate_predictions.py
├── evaluation/
│   ├── inspect_product_dataset.py
│   └── evaluate_predictions.py
├── frontend/
│   ├── data/
│   │   └── products_enriched.json
│   └── streamlit/
│       ├── app.py
│       └── detail.py
├── final report.pdf
├── requirements.txt
└── README.md
```

## Setup
This project uses Python 3.9 or above.
To install the packages:
```bash
python3 -m pip install -r requirements.txt
```

## Files
### data/
- `product_recall_features_2004_2008.db`  
  SQLite database of product-level FAERS-derived features and recall labels (baseline export).

- `improved_product_recall_features_2004_2008.db`  
  Same kind of schema: product-level features with recall labels; use whichever matches your modeling run.

- `data_exploration/`  
  Exploratory figures: FAERS volume by year, enforcement recalls by year, recall classification overview, and serious-outcome FAERS patterns.

### feature_engineering/
- `build_product_feature_database.py`  
  Builds the product feature database that is used later for modeling.

### models/
All model code is implemented from scratch with NumPy only (no scikit-learn or similar).

- `logistic_regression.py`  
  Logistic regression with class weighting, L2 regularization, mini-batch SGD, and early stopping.

- `random_forest.py`  
  Random forest built from CART decision trees using Gini impurity and bootstrap sampling.

- `linear_regression.py`  
  Ridge regression baseline; output is used as a ranking score.

- `evaluate_predictions.py`  
  Computes PR-AUC, ROC-AUC, precision, recall, F1, and Recall@Top-10% from a prediction CSV (same metrics family as `evaluation/evaluate_predictions.py`, kept under `models/` for model-run outputs).

#### How to evaluate model CSVs
From the project root (adjust paths if your prediction files live elsewhere):
```bash
python3 models/evaluate_predictions.py --csv-path predictions/logistic_regression_predictions.csv
python3 models/evaluate_predictions.py --csv-path predictions/random_forest_predictions.csv
python3 models/evaluate_predictions.py --csv-path predictions/linear_regression_predictions.csv
```

#### Final evaluation results
- **Dataset**: `improved_product_recall_features_2004_2008.db`
- **Split**: stratified 70% train / 15% validation / 15% test, random seed 42
- **Test set**: 7,790 records, 403 positive recall labels
- **Threshold**: chosen on the validation set by maximizing F1-score

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Recall@Top-10% | TP | FP | TN | FN |
|-------|-------:|--------:|----------:|-------:|---:|---------------:|---:|---:|---:|---:|
| Linear Regression Baseline | 0.1095 | 0.6612 | 0.0000 | 0.0000 | 0.0000 | 0.2185 | 0 | 0 | 9822 | 563 |
| Logistic Regression | 0.1107 | 0.6808 | 0.0882 | 0.6661 | 0.1557 | 0.2238 | 375 | 3879 | 5943 | 188 |
| Random Forest | 0.3030 | 0.8556 | 0.1707 | 0.8117 | 0.2820 | 0.4849 | 457 | 2221 | 7601 | 106 |

Random Forest achieved the highest ROC-AUC (0.8325) and F1 (0.2953) and was chosen as the deployed model. Logistic regression had the highest recall (0.4442) but more false positives. The linear regression baseline is for ranking comparison only.

#### Model details
**Logistic regression**  
Binary cross-entropy with balanced class weighting; L2 regularization (lambda = 0.001); mini-batch gradient descent (batch size 256); features standardized using training-set statistics only; early stopping on validation loss (patience 50).

**Random forest**  
100 trees, max depth 12, bootstrap sampling per tree; random feature subset at each split (square root of the number of features); balanced class weighting via sample weights; Gini impurity for splits.

**Linear regression baseline**  
Ridge regression with a closed-form solution; outputs clipped to [0, 1] and used as a continuous ranking score; included for AUC-style comparison, not tuned for threshold-based classification.

### evaluation/
- `inspect_product_dataset.py`  
  Lets you check the dataset size, label balance, and some simple feature info.

- `evaluate_predictions.py`  
  Evaluates prediction results using PR-AUC, ROC-AUC, F1-score, and Recall@Top-K%.
### frontend/
- `frontend/data/products_enriched.json`  
  Enriched product records used by the Streamlit app. Connected training results and original data to form a dataset with all complete detail information.

- `frontend/streamlit/app.py`  
  Main Streamlit app for searching products.
- `frontend/streamlit/detail.py`  
  Product detail view: narrative fields, feature expander, ingredients and recall-history tables.

## How to Run

Build the feature database:
```bash
python3 feature_engineering/build_product_feature_database.py
```
Inspect the dataset:
```bash
python3 evaluation/inspect_product_dataset.py \
  --db-path data/improved_product_recall_features_2004_2008.db
```
Evaluate predictions:
```bash
python3 evaluation/evaluate_predictions.py \
  --csv-path path/to/predictions.csv \
  --label-col label \
  --score-col score
```
Run the frontend:
```bash
python3 -m streamlit run frontend/streamlit/app.py
```

## What this project does
The main idea of this project is to predict recall risk at the product level.  
It also includes some simple scripts for feature building, evaluation, and a basic frontend for showing product information.
