"""
generate_data.py
-----------------
Generates a synthetic "Customer Churn" dataset for the Predictive
Modeling project. The target (Churn: Yes/No) has a realistic,
learnable relationship with the features so trained models achieve
sensible (not perfect) accuracy.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1000

age = np.random.randint(18, 70, N)
tenure_months = np.random.randint(1, 72, N)
monthly_charges = np.round(np.random.uniform(20, 120, N), 2)
support_calls = np.random.poisson(1.5, N)
contract_type = np.random.choice(["Month-to-Month", "One Year", "Two Year"], N, p=[0.55, 0.25, 0.20])
payment_method = np.random.choice(["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"], N)
internet_service = np.random.choice(["DSL", "Fiber Optic", "No"], N, p=[0.35, 0.45, 0.20])

# Build churn probability from a logical combination of features
churn_score = (
    -1.2
    - 0.035 * tenure_months
    + 0.015 * monthly_charges
    + 0.35 * support_calls
    + np.where(contract_type == "Month-to-Month", 1.0, 0)
    + np.where(contract_type == "One Year", 0.2, 0)
    + np.where(internet_service == "Fiber Optic", 0.35, 0)
    - 0.01 * age
    + np.random.normal(0, 1.0, N)  # noise
)

churn_prob = 1 / (1 + np.exp(-churn_score))
churn = (churn_prob > 0.5).astype(int)

df = pd.DataFrame({
    "CustomerID": [f"CUST{2000+i}" for i in range(N)],
    "Age": age,
    "TenureMonths": tenure_months,
    "MonthlyCharges": monthly_charges,
    "SupportCalls": support_calls,
    "ContractType": contract_type,
    "PaymentMethod": payment_method,
    "InternetService": internet_service,
    "Churn": np.where(churn == 1, "Yes", "No"),
})

df.to_csv("/home/claude/ml_project/data/customer_churn.csv", index=False)
print("Dataset generated:", df.shape)
print(df["Churn"].value_counts())
