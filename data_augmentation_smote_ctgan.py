"""
Federated Learning for Rare Disease (Cystic Fibrosis)
Phase 2: SMOTE + CTGAN Augmentation
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from ctgan import CTGAN
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

X = data[features]
y = data["target"]

# ==============================
# 2. APPLY SMOTE (Minority Stabilization)
# ==============================

print("\nApplying SMOTE...")

smote = SMOTE(sampling_strategy=150 / len(y[y == 0]), random_state=42)
X_res, y_res = smote.fit_resample(X, y)

resampled_data = pd.DataFrame(X_res, columns=features)
resampled_data["target"] = y_res

print("After SMOTE class distribution:")
print(resampled_data["target"].value_counts())

# ==============================
# 3. TRAIN CTGAN ON MINORITY CLASS ONLY
# ==============================

cf_data = resampled_data[resampled_data["target"] == 1].drop(columns=["target"])

print("\nTraining CTGAN on minority class...")

ctgan = CTGAN(
    epochs=200,      # 300 is too high for 150 samples
    batch_size=30,
    pac=5,
    verbose=True
)


ctgan.fit(cf_data)

# ==============================
# 4. GENERATE SYNTHETIC CF DATA
# ==============================

print("\nGenerating synthetic CF samples...")

synthetic_samples = ctgan.sample(400)

synthetic_samples["target"] = 1

print("Synthetic samples generated:", synthetic_samples.shape)

# ==============================
# 5. VALIDATION — KS TEST
# ==============================

print("\n--- KS Test Between Real CF and Synthetic CF ---")

real_cf = cf_data

for feature in features:
    stat, p = ks_2samp(real_cf[feature], synthetic_samples[feature])
    print(f"{feature} → KS p-value: {p:.6f}")

# ==============================
# 6. CLASSIFIER TWO-SAMPLE TEST
# ==============================

print("\n--- Classifier Two-Sample Test ---")

real_cf["label"] = 0
synthetic_samples["label"] = 1

combined = pd.concat([real_cf, synthetic_samples])

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

if acc < 0.65:
    print("Synthetic data quality: GOOD (Hard to distinguish)")
else:
    print("Synthetic data quality: Needs improvement")

# ==============================
# 7. MERGE FINAL DATASET
# ==============================

normal_data = resampled_data[resampled_data["target"] == 0]

final_data = pd.concat([
    normal_data,
    resampled_data[resampled_data["target"] == 1],
    synthetic_samples.drop(columns=["label"])
], ignore_index=True)

print("\nFinal Dataset Shape:", final_data.shape)
print("Final Class Distribution:")
print(final_data["target"].value_counts())

# ==============================
# 8. SAVE FINAL AUGMENTED DATASET
# ==============================

final_data.to_csv("augmented_cf_dataset.csv", index=False)

print("\nAugmented dataset saved as: augmented_cf_dataset.csv")
print("Phase 2 Completed Successfully.")
