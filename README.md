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
│   └── logistic_regression.py
├── evaluation/
│   ├── inspect_product_dataset.py
│   └── evaluate_predictions.py
├── frontend/
│   ├── data/
│   │   └── products_enriched.json
│   └── streamlit/
│       ├── app.py
│       └── detail.py
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
- `logistic_regression.py`  
  A basic logistic regression model written with `numpy`.

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
