# Predictive Modeling Using Machine Learning

Portfolio-ready supervised learning project predicting customer churn.

## Models
- Logistic Regression
- Decision Tree
- Random Forest

## Workflow
Data cleaning → train/test split → imputation → scaling/one-hot encoding → model training → evaluation → ROC/confusion-matrix visualization → save best model.

## Metrics
Accuracy, precision, recall, F1-score, and ROC-AUC.

## Structure
```text
predictive-modeling-ml-project/
├── data/customer_churn_raw.csv
├── models/best_model.joblib
├── outputs/confusion_matrix.png
├── outputs/roc_curves.png
├── outputs/feature_importance.png
├── reports/model_metrics.csv
├── reports/best_model.json
├── src/train.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

The dataset is synthetic and intentionally contains missing values and duplicates for preprocessing practice.
