import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTEENN

# ======================================
# LOAD DATA
# ======================================
data = pd.read_csv("data/nhanes_clean.csv")

features = ["age", "sex", "height", "weight"]
target = "obstruction"

X = data[features].copy()
y = data[target].astype(int)

# Encode categorical feature
if X["sex"].dtype == "object":
    le = LabelEncoder()
    X["sex"] = le.fit_transform(X["sex"])

# ======================================
# GLOBAL TRAIN-TEST SPLIT
# ======================================
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ======================================
# SIMULATE 3 CLIENTS (Non-IID Split)
# ======================================
client_splits = np.array_split(X_train_full.index, 3)

clients = []

for idx in client_splits:
    X_client = X_train_full.loc[idx]
    y_client = y_train_full.loc[idx]
    clients.append((X_client, y_client))

# ======================================
# LOCAL TRAINING ON EACH CLIENT
# ======================================
local_models = []

for i, (X_c, y_c) in enumerate(clients):
    smote_enn = SMOTEENN(random_state=42)
    X_res, y_res = smote_enn.fit_resample(X_c, y_c)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_res, y_res)

    local_models.append(model)

    print(f"Client {i+1} trained.")

# ======================================
# FEDERATED AGGREGATION (Probability Averaging)
# ======================================
probs = []

for model in local_models:
    prob = model.predict_proba(X_test)[:, 1]
    probs.append(prob)

# Average probabilities
federated_prob = np.mean(probs, axis=0)

# Convert to predictions
federated_pred = (federated_prob >= 0.5).astype(int)

# ======================================
# EVALUATION
# ======================================
print("\n===== FEDERATED MODEL RESULTS =====")

print("Accuracy:", accuracy_score(y_test, federated_pred))
print("Precision:", precision_score(y_test, federated_pred))
print("Recall:", recall_score(y_test, federated_pred))
print("F1-Score:", f1_score(y_test, federated_pred))
print("ROC-AUC:", roc_auc_score(y_test, federated_prob))
