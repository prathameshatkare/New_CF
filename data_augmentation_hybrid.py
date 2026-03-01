"""
GAN/TVAE augmentation for cystic fibrosis (CF) federated learning.

Input: cleaned_cf_dataset.csv with columns
    age, sex, height, weight, fev1, BMI, target
Output:
    augmented_cf_dataset.csv
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
SOURCE_PATH = "cleaned_cf_dataset.csv"
OUTPUT_PATH = "augmented_cf_dataset.csv"
FEATURES = ["age", "sex", "height", "weight", "fev1", "BMI"]
TARGET = "target"
TARGET_CF = 1
SYNTHETIC_CF_SAMPLES = 300


def run() -> None:
    df = pd.read_csv(SOURCE_PATH)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].copy()
    df[TARGET] = df[TARGET].astype(int)
    df["sex"] = df["sex"].astype(int)

    real_cf = df[df[TARGET] == TARGET_CF][FEATURES].copy()
    normal = df[df[TARGET] != TARGET_CF].copy()

    if len(real_cf) < 10:
        raise ValueError("Not enough real CF samples for reliable synthesis.")

    print("Loaded dataset:", df.shape)
    print("Original class distribution:\n", df[TARGET].value_counts())
    print("Real CF samples:", len(real_cf))

    # Small perturbation to reduce exact memorization with very small minority class.
    noisy_cf = real_cf.copy()
    for col in ["age", "height", "weight", "fev1", "BMI"]:
        noisy_cf[col] = noisy_cf[col] + np.random.normal(0, 0.01, size=len(noisy_cf))

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(noisy_cf)

    synthesizer = TVAESynthesizer(
        metadata=metadata,
        epochs=1000,
        batch_size=16,
        enforce_min_max_values=True,
    )
    synthesizer.fit(noisy_cf)

    synthetic_cf = synthesizer.sample(num_rows=SYNTHETIC_CF_SAMPLES)
    synthetic_cf = synthetic_cf[FEATURES].copy()
    synthetic_cf["sex"] = synthetic_cf["sex"].round().clip(0, 1).astype(int)
    synthetic_cf[TARGET] = TARGET_CF

    print("\nSynthetic CF samples generated:", synthetic_cf.shape)

    print("\nKS-test (real CF vs synthetic CF):")
    for feature in FEATURES:
        _, p = ks_2samp(real_cf[feature], synthetic_cf[feature])
        print(f"  {feature}: p={p:.6f}")

    real_eval = real_cf.copy()
    real_eval["source"] = 0
    synth_eval = synthetic_cf[FEATURES].copy()
    synth_eval["source"] = 1
    eval_df = pd.concat([real_eval, synth_eval], ignore_index=True)

    X_train, X_val, y_train, y_val = train_test_split(
        eval_df[FEATURES],
        eval_df["source"],
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=eval_df["source"],
    )
    clf = RandomForestClassifier(random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    detect_acc = accuracy_score(y_val, clf.predict(X_val))
    print(f"\nReal-vs-synthetic detector accuracy: {detect_acc:.4f}")
    print("Lower is better (close to 0.5 means harder to distinguish).")

    final_df = pd.concat([normal, df[df[TARGET] == TARGET_CF], synthetic_cf], ignore_index=True)
    final_df = final_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    print("\nFinal dataset shape:", final_df.shape)
    print("Final class distribution:\n", final_df[TARGET].value_counts())

    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
