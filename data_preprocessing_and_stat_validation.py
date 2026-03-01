"""
Federated Learning for Rare Disease (Cystic Fibrosis)
Phase 1: Data Preprocessing and Statistical Validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# ==============================
# 1. LOAD DATA
# ==============================

CF_PATH = "data/cystfibr.csv"
NHANES_PATH = "data/nhanes_clean.csv"

cf = pd.read_csv(CF_PATH)
nhanes = pd.read_csv(NHANES_PATH)

print("CF Dataset Shape:", cf.shape)
print("NHANES Dataset Shape:", nhanes.shape)


# ==============================
# 2. FEATURE ALIGNMENT
# ==============================

selected_features = ["age", "sex", "height", "weight", "fev1"]

cf = cf[selected_features].copy()
nhanes = nhanes[selected_features].copy()


# ==============================
# 3. CLEAN SEX ENCODING
# ==============================

def encode_sex(x):
    if str(x).strip().lower() in ["male", "m", "1"]:
        return 1
    else:
        return 0

cf["sex"] = cf["sex"].apply(encode_sex)
nhanes["sex"] = nhanes["sex"].apply(encode_sex)


# ==============================
# 4. BMI FEATURE ENGINEERING
# ==============================

for df in [cf, nhanes]:
    df["BMI"] = df["weight"] / ((df["height"] / 100) ** 2)


# ==============================
# 5. CREATE TARGET COLUMN
# ==============================

cf["target"] = 1
nhanes["target"] = 0


# ==============================
# 6. REDUCE NHANES SIZE (OPTIONAL)
# ==============================

nhanes = nhanes.sample(n=3000, random_state=42)


# ==============================
# 7. MERGE DATASETS
# ==============================

data = pd.concat([cf, nhanes], ignore_index=True)

print("\nFinal Dataset Shape:", data.shape)
print("\nClass Distribution:")
print(data["target"].value_counts())


# ==============================
# 8. STATISTICAL VALIDATION
# ==============================

features = ["age", "height", "weight", "fev1", "BMI"]

cf_data = data[data["target"] == 1]
normal_data = data[data["target"] == 0]

print("\n--- Mann-Whitney U Test Results ---")

for feature in features:
    stat, p = mannwhitneyu(
        cf_data[feature],
        normal_data[feature],
        alternative="two-sided"
    )
    print(f"{feature} -> p-value: {p:.6f}")


# ==============================
# 9. EFFECT SIZE (COHEN'S D)
# ==============================

def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    pooled_std = np.sqrt(
        ((nx - 1) * np.var(x) + (ny - 1) * np.var(y)) / (nx + ny - 2)
    )
    return (np.mean(x) - np.mean(y)) / pooled_std

print("\n--- Effect Size (Cohen's d) ---")

for feature in features:
    d = cohens_d(cf_data[feature], normal_data[feature])
    print(f"{feature} -> Cohen's d: {d:.3f}")


# ==============================
# 10. CORRELATION ANALYSIS
# ==============================

corr_cf = cf_data[features].corr()
corr_normal = normal_data[features].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(corr_cf, annot=True, cmap="coolwarm")
plt.title("CF Correlation Matrix")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(corr_normal, annot=True, cmap="coolwarm")
plt.title("Normal Correlation Matrix")
plt.tight_layout()
plt.show()

# Structural difference
frobenius_norm = np.linalg.norm(corr_cf - corr_normal)
print("\nCorrelation Matrix Difference (Frobenius Norm):", round(frobenius_norm, 4))


# ==============================
# 11. SAVE CLEANED DATASET
# ==============================

data.to_csv("cleaned_cf_dataset.csv", index=False)
print("\nCleaned dataset saved as: cleaned_cf_dataset.csv")

print("\nPhase 1 Completed Successfully.")
