"""
UPI Fraud Detection - ML Model Training
Uses synthetic data to simulate real UPI transaction patterns.
In production, replace with real labeled fraud datasets (e.g. from Kaggle).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import os

np.random.seed(42)
N = 5000
FRAUD_RATE = 0.15  # 15% fraud

def generate_dataset(n, fraud_rate):
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    def make_legit(n):
        return pd.DataFrame({
            "hour": np.random.choice(range(7, 23), n),
            "amount": np.random.exponential(scale=3000, size=n).clip(10, 50000),
            "freq_today": np.random.choice([1, 2, 3, 4, 5], n, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
            "device_match": np.random.choice([0, 1, 2, 3], n, p=[0.7, 0.15, 0.1, 0.05]),
            "receiver_type": np.random.choice([0, 1, 2, 3], n, p=[0.4, 0.35, 0.2, 0.05]),
            "sender_id_valid": np.random.choice([0, 1], n, p=[0.05, 0.95]),
            "receiver_id_valid": np.random.choice([0, 1], n, p=[0.05, 0.95]),
            "label": 0
        })

    def make_fraud(n):
        return pd.DataFrame({
            "hour": np.random.choice(list(range(0, 5)) + list(range(22, 24)), n),
            "amount": np.random.choice(
                [np.random.uniform(1, 500), np.random.uniform(50000, 200000)],
                n
            ),
            "freq_today": np.random.choice([6, 8, 11, 15, 20], n),
            "device_match": np.random.choice([0, 1, 2, 3], n, p=[0.05, 0.15, 0.25, 0.55]),
            "receiver_type": np.random.choice([0, 1, 2, 3], n, p=[0.05, 0.1, 0.35, 0.5]),
            "sender_id_valid": np.random.choice([0, 1], n, p=[0.4, 0.6]),
            "receiver_id_valid": np.random.choice([0, 1], n, p=[0.45, 0.55]),
            "label": 1
        })

    df = pd.concat([make_legit(n_legit), make_fraud(n_fraud)]).sample(frac=1).reset_index(drop=True)
    return df

print("Generating synthetic UPI transaction dataset...")
df = generate_dataset(N, FRAUD_RATE)

features = ["hour", "amount", "freq_today", "device_match", "receiver_type", "sender_id_valid", "receiver_id_valid"]
X = df[features]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples...")
model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

importances = dict(zip(features, model.feature_importances_.round(3)))
print("\nFeature Importances:")
for f, imp in sorted(importances.items(), key=lambda x: -x[1]):
    print(f"  {f}: {imp}")

model_path = os.path.join(os.path.dirname(__file__), "fraud_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump({"model": model, "features": features}, f)

print(f"\nModel saved to {model_path}")
