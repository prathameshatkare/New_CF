"""
Federated Learning for Rare Disease (Cystic Fibrosis)
Phase 2 (Corrected): TVAE-Based Augmentation
"""

import pandas as pd
import numpy as np
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==============================
# 1. LOAD CLEANED DATA
# ==============================

data = pd.read_csv("cleaned_cf_dataset.csv")

print("Loaded dataset shape:", data.shape)
print("Class distribution:")
print(data["target"].value_counts())

features = ["age", "sex", "height", "weight", "fev1", "BMI"]

# Separate real CF and Normal
real_cf = data[data["target"] == 1][features].copy()
normal_data = data[data["target"] == 0].copy()

print("\nReal CF samples:", real_cf.shape[0])

# ==============================
# 2. ADD SMALL NOISE (ANTI-MEMORIZATION)
# ==============================

real_cf_noisy = real_cf.copy()
for col in ["age", "height", "weight", "fev1", "BMI"]:
    real_cf_noisy[col] += np.random.normal(0, 0.01, size=len(real_cf_noisy))

# ==============================
# 3. TVAE METADATA
# ==============================

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_cf_noisy)

# ==============================
# 4. TRAIN TVAE
# ==============================

print("\nTraining TVAE on REAL CF only...")

synthesizer = TVAESynthesizer(
    metadata,
    epochs=1000,        # Higher epochs for stability
    batch_size=16
)

synthesizer.fit(real_cf_noisy)

# ==============================
# 5. GENERATE SYNTHETIC CF
# ==============================

print("\nGenerating synthetic CF samples...")
synthetic_cf = synthesizer.sample(num_rows=100)


import numpy as np

# Continuous features only
continuous_cols = ['age', 'height', 'weight', 'fev1', 'BMI']

for col in continuous_cols:
    if col in synthetic_cf.columns:
        noise = np.random.normal(
            loc=0,
            scale=synthetic_cf[col].std() * 0.02,  # 2% smoothing
            size=len(synthetic_cf)
        )
        synthetic_cf[col] += noise


synthetic_cf["target"] = 1

print("Synthetic samples generated:", synthetic_cf.shape)

# ==============================
# 6. VALIDATION — KS TEST
# ==============================

print("\n--- KS Test Between Real CF and Synthetic CF ---")

for feature in features:
    stat, p = ks_2samp(real_cf[feature], synthetic_cf[feature])
    print(f"{feature} → KS p-value: {p:.6f}")

# ==============================
# 7. CLASSIFIER TWO-SAMPLE TEST
# ==============================

print("\n--- Classifier Two-Sample Test ---")

real_cf_test = real_cf.copy()
real_cf_test["label"] = 0

synthetic_test = synthetic_cf.copy()
synthetic_test["label"] = 1

combined = pd.concat([real_cf_test, synthetic_test])

X_test = combined[features]
y_test = combined["label"]

X_train, X_val, y_train, y_val = train_test_split(
    X_test, y_test, test_size=0.3, random_state=42
)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

preds = clf.predict(X_val)
acc = accuracy_score(y_val, preds)

print("Classifier Accuracy (Real vs Synthetic):", round(acc, 4))

if acc < 0.70:
    print("Synthetic data quality: GOOD")
else:
    print("Synthetic data quality: Needs improvement")

# ==============================
# 8. MERGE FINAL DATASET
# ==============================

synthetic_cf = synthetic_cf[features + ["target"]]

final_data = pd.concat([
    normal_data,
    data[data["target"] == 1],
    synthetic_cf
], ignore_index=True)

print("\nFinal Dataset Shape:", final_data.shape)
print("Final Class Distribution:")
print(final_data["target"].value_counts())

# ==============================
# 9. SAVE FINAL DATASET
# ==============================

final_data.to_csv("augmented_cf_dataset.csv", index=False)

print("\nAugmented dataset saved as: augmented_cf_dataset.csv")
print("TVAE Augmentation Completed Successfully.")
