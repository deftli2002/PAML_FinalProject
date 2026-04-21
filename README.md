# PAML Final Project
This is my final project for PAML.  
It is a pretty simple pipeline for predicting FDA drug recall risk.

## Project Structure
```text
PAML_FinalProject/
├── data/
├── feature_engineering/
├── models/
├── evaluation/
├── frontend/
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
  SQLite database containing product recall features and labels for FDA drug recall prediction.

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
- `frontend/streamlit/app.py`  
  Main Streamlit app for searching products.
- `frontend/streamlit/detail.py`  
  Detail page for a selected product.
- `frontend/mock/products.json`  
  Mock data used by the frontend.

## How to Run

Build the feature database:
```bash
python3 feature_engineering/build_product_feature_database.py
```
Inspect the dataset:
```bash
python3 evaluation/inspect_product_dataset.py --db-path data/product_recall_features_2004_2008.db
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
