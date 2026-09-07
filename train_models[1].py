"""
train_models.py
-----------------
Predictive Modeling Using Machine Learning
Applies Logistic Regression, Decision Tree, and Random Forest to
predict customer churn. Trains/tests each model, evaluates accuracy,
and saves confusion matrices + ROC curves + a comparison report.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_curve, auc,
                              classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

DATA_PATH = "/home/claude/ml_project/data/customer_churn.csv"
VIS_DIR = "/home/claude/ml_project/visuals"
OUT_DIR = "/home/claude/ml_project/outputs"

sns.set_theme(style="whitegrid")

df = pd.read_csv(DATA_PATH)
df["ChurnLabel"] = (df["Churn"] == "Yes").astype(int)

feature_cols = ["Age", "TenureMonths", "MonthlyCharges", "SupportCalls",
                 "ContractType", "PaymentMethod", "InternetService"]
X = df[feature_cols]
y = df["ChurnLabel"]

numeric_features = ["Age", "TenureMonths", "MonthlyCharges", "SupportCalls"]
categorical_features = ["ContractType", "PaymentMethod", "InternetService"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop="first"), categorical_features),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
}

report_lines = []
results = {}
plt.figure(figsize=(7, 6))  # single combined ROC plot

for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

    report_lines.append(f"=== {name} ===")
    report_lines.append(f"Accuracy:  {acc:.3f}")
    report_lines.append(f"Precision: {prec:.3f}")
    report_lines.append(f"Recall:    {rec:.3f}")
    report_lines.append(f"F1 Score:  {f1:.3f}")
    report_lines.append(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    report_lines.append("")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"{VIS_DIR}/confusion_matrix_{safe_name}.png", dpi=150)
    plt.close()

    # ROC curve (added to combined plot)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    # Save trained pipeline
    joblib.dump(pipe, f"{OUT_DIR}/model_{safe_name}.joblib")

# Finalize combined ROC plot
plt.figure(1)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Model Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/roc_curves_comparison.png", dpi=150)
plt.close()

# Model comparison bar chart
results_df = pd.DataFrame(results).T
plt.figure(figsize=(9, 5))
results_df.plot(kind="bar", ax=plt.gca(), colormap="viridis")
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/model_comparison.png", dpi=150)
plt.close()

# Feature importance (Random Forest)
rf_pipe = Pipeline([("prep", preprocessor), ("model", models["Random Forest"])])
rf_pipe.fit(X_train, y_train)
feature_names = (numeric_features +
                  list(rf_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(categorical_features)))
importances = rf_pipe.named_steps["model"].feature_importances_
imp_df = pd.Series(importances, index=feature_names).sort_values(ascending=True)

plt.figure(figsize=(8, 6))
imp_df.plot(kind="barh", color="#0f766e")
plt.title("Random Forest - Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/feature_importance.png", dpi=150)
plt.close()

results_df.to_csv(f"{OUT_DIR}/model_comparison.csv")
with open(f"{OUT_DIR}/model_report.txt", "w") as f:
    f.write("\n".join(report_lines))

print("Model comparison:\n", results_df)
print("\nAll visuals, models, and reports saved.")
