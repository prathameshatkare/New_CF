"""Strict medical-style validation for CF screening.

Goals:
- Avoid optimistic reporting from easy internal splits
- Evaluate with external-like split (train on augmented, test on cleaned)
- Tune threshold for a clinically useful operating point
- Report confidence intervals via bootstrap
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["age", "sex", "height", "weight", "fev1", "BMI"]
TARGET = "target"


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    brier: float


def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = FEATURES + [TARGET]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"{path} missing columns: {miss}")
    df = df[required].copy()
    df["sex"] = df["sex"].astype(int)
    df[TARGET] = df[TARGET].astype(int)
    return df


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Metrics:
    y_pred = (y_prob >= thr).astype(int)
    return Metrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_prob),
        brier=brier_score_loss(y_true, y_prob),
    )


def pick_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_accuracy: float = 0.92) -> float:
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thr = 0.5
    best_recall = -1.0
    best_f1 = -1.0
    for thr in thresholds:
        m = compute_metrics(y_true, y_prob, thr)
        if m.accuracy >= min_accuracy:
            if (m.recall > best_recall) or (m.recall == best_recall and m.f1 > best_f1):
                best_recall = m.recall
                best_f1 = m.f1
                best_thr = float(thr)
    return best_thr


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, thr: float, n_boot: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": [], "brier": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = compute_metrics(yt, yp, thr)
        metrics["accuracy"].append(m.accuracy)
        metrics["precision"].append(m.precision)
        metrics["recall"].append(m.recall)
        metrics["f1"].append(m.f1)
        metrics["roc_auc"].append(m.roc_auc)
        metrics["brier"].append(m.brier)
    out = {}
    for k, vals in metrics.items():
        arr = np.asarray(vals)
        out[k] = {
            "mean": float(np.mean(arr)),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
        }
    return out


def main() -> None:
    # External-like protocol:
    # - Train/validate on augmented data
    # - Final evaluation on cleaned data (real-only distribution)
    augmented = load_df("augmented_cf_dataset.csv")
    cleaned = load_df("cleaned_cf_dataset.csv")

    train_df, val_df = train_test_split(
        augmented, test_size=0.2, random_state=42, stratify=augmented[TARGET]
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(train_df[FEATURES], train_df[TARGET])

    val_prob = model.predict_proba(val_df[FEATURES])[:, 1]
    best_thr = pick_threshold(val_df[TARGET].values, val_prob, min_accuracy=0.92)

    test_prob = model.predict_proba(cleaned[FEATURES])[:, 1]
    m = compute_metrics(cleaned[TARGET].values, test_prob, best_thr)
    ci = bootstrap_ci(cleaned[TARGET].values, test_prob, best_thr, n_boot=500, seed=42)

    # Calibration curve (for plotting/inspection if needed later)
    frac_pos, mean_pred = calibration_curve(cleaned[TARGET].values, test_prob, n_bins=10, strategy="quantile")
    cal_df = pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac_pos})
    cal_df.to_csv("clinical_calibration_curve.csv", index=False)

    report = {
        "protocol": "train_on_augmented_validate_on_augmented_test_on_cleaned_real_only",
        "threshold_selected": best_thr,
        "target_min_accuracy": 0.92,
        "test_metrics": {
            "accuracy": m.accuracy,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "roc_auc": m.roc_auc,
            "brier": m.brier,
        },
        "bootstrap_ci95": ci,
    }

    with open("clinical_strict_validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    flat = {
        "threshold_selected": best_thr,
        "accuracy": m.accuracy,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "roc_auc": m.roc_auc,
        "brier": m.brier,
        "accuracy_ci95_low": ci["accuracy"]["ci95_low"],
        "accuracy_ci95_high": ci["accuracy"]["ci95_high"],
        "recall_ci95_low": ci["recall"]["ci95_low"],
        "recall_ci95_high": ci["recall"]["ci95_high"],
    }
    pd.DataFrame([flat]).to_csv("clinical_strict_validation_metrics.csv", index=False)

    print("=== Strict Clinical Validation ===")
    print(f"Threshold selected: {best_thr:.3f}")
    print(f"Accuracy : {m.accuracy:.4f}")
    print(f"Precision: {m.precision:.4f}")
    print(f"Recall   : {m.recall:.4f}")
    print(f"F1-score : {m.f1:.4f}")
    print(f"ROC-AUC  : {m.roc_auc:.4f}")
    print(f"Brier    : {m.brier:.4f}")
    print("Saved: clinical_strict_validation_report.json")
    print("Saved: clinical_strict_validation_metrics.csv")
    print("Saved: clinical_calibration_curve.csv")


if __name__ == "__main__":
    main()

