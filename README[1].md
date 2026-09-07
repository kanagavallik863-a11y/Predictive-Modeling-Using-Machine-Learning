# Predictive Modeling Using Machine Learning

Builds and compares three classification models — **Logistic Regression**,
**Decision Tree**, and **Random Forest** — to predict **customer churn**
(Yes/No) from a synthetic telecom-style customer dataset.

## Project Structure

```
ml_project/
├── data/
│   ├── generate_data.py             # Creates the synthetic dataset
│   └── customer_churn.csv           # Dataset (1000 rows, 9 columns)
├── notebooks/
│   └── train_models.py              # Preprocessing, training, evaluation
├── outputs/
│   ├── model_logistic_regression.joblib
│   ├── model_decision_tree.joblib
│   ├── model_random_forest.joblib
│   ├── model_comparison.csv         # Accuracy/Precision/Recall/F1 table
│   └── model_report.txt             # Full classification reports
├── visuals/
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_decision_tree.png
│   ├── confusion_matrix_random_forest.png
│   ├── roc_curves_comparison.png
│   ├── model_comparison.png
│   └── feature_importance.png
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib

python data/generate_data.py         # regenerate dataset (optional, already included)
python notebooks/train_models.py     # train, evaluate, and visualize
```

## Dataset

Synthetic customer records with a realistic churn relationship:

| Feature | Description |
|---|---|
| Age, TenureMonths, MonthlyCharges, SupportCalls | Numeric features |
| ContractType, PaymentMethod, InternetService | Categorical features |
| Churn | Target (Yes/No) |

Churn probability was built from a logistic combination of tenure, monthly
charges, support call frequency, contract type, and internet service —
month-to-month contracts and frequent support calls raise churn risk, longer
tenure lowers it — plus random noise, so the signal is realistic rather than
trivially separable.

## Methodology

1. **Train/test split** — 80/20, stratified on the target.
2. **Preprocessing pipeline** — `StandardScaler` for numeric features,
   `OneHotEncoder` for categorical features, wrapped in a single
   `sklearn.Pipeline` per model (no leakage between train/test).
3. **Models trained**:
   - Logistic Regression (linear baseline)
   - Decision Tree (max depth 5, to control overfitting)
   - Random Forest (200 trees, max depth 6)
4. **Evaluation** — Accuracy, Precision, Recall, F1, confusion matrices, and
   ROC/AUC for each model.

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.795 | 0.732 | 0.703 | 0.717 |
| Decision Tree | 0.715 | 0.623 | 0.581 | 0.601 |
| Random Forest | 0.760 | 0.717 | 0.581 | 0.642 |

**Logistic Regression performed best** on this dataset (AUC ≈ 0.86), likely
because the underlying churn signal was built from a roughly linear
combination of features. Random Forest was close behind (AUC ≈ 0.83) and the
feature importance chart shows `TenureMonths`, `MonthlyCharges`, and contract
type as the strongest churn predictors — consistent with real-world churn
drivers.

Full per-model classification reports are in `outputs/model_report.txt`.

## Expected Outcome (Task Goal)

Demonstrates the supervised learning workflow end-to-end: preprocessing mixed
numeric/categorical data, training multiple algorithms, and evaluating them
with the right metrics (not just accuracy — precision/recall/F1/ROC matter
more for an imbalanced target like churn).
