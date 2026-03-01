"""Research-style baseline and diagnostics for CF diagnosis.

Compares classical models on:
- cleaned_cf_dataset.csv
- augmented_cf_dataset.csv

Also adds:
- leakage diagnostics (duplicate checks, label-shuffle sanity test, single-feature signal)
- leave-one-feature-out ablation study
- threshold calibration for high-recall clinical screening

Outputs:
- model_comparison_results_updated.csv
- model_comparison_results_final.csv
- model_leakage_diagnostics.csv
- feature_ablation_results.csv
- threshold_calibration_results.csv
- robustness_stress_test_results.csv
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES = ["age", "sex", "height", "weight", "fev1", "BMI"]
TARGET = "target"


@dataclass
class DatasetSpec:
    name: str
    path: str


def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df = df[required].copy()
    df[TARGET] = df[TARGET].astype(int)
    df["sex"] = df["sex"].astype(int)
    return df


def get_models(seed: int):
    return {
        "LogReg (balanced)": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "SVM-RBF (balanced)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
            ]
        ),
        "RandomForest (balanced)": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=seed),
    }


def cv_metrics(X: pd.DataFrame, y: pd.Series, model, seed: int) -> dict[str, float]:
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
    res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    out = {}
    for key in scoring.keys():
        vals = res[f"test_{key}"]
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
    return out


def holdout_metrics(X: pd.DataFrame, y: pd.Series, model, seed: int) -> dict[str, float]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    return {
        "holdout_accuracy": accuracy_score(y_test, y_pred),
        "holdout_precision": precision_score(y_test, y_pred, zero_division=0),
        "holdout_recall": recall_score(y_test, y_pred, zero_division=0),
        "holdout_f1": f1_score(y_test, y_pred, zero_division=0),
        "holdout_roc_auc": roc_auc_score(y_test, y_prob),
        "holdout_pr_auc": average_precision_score(y_test, y_prob),
    }


def leakage_diagnostics(df: pd.DataFrame, dataset_name: str, seed: int) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    X = df[FEATURES]
    y = df[TARGET]

    # 1) Exact duplicate feature rows with conflicting labels indicate hard leakage/data issue.
    duplicated = df.duplicated(subset=FEATURES, keep=False)
    dup_df = df.loc[duplicated, FEATURES + [TARGET]].copy()

    conflicting = 0
    if not dup_df.empty:
        label_counts = dup_df.groupby(FEATURES, as_index=False)[TARGET].nunique()
        conflicting = int((label_counts[TARGET] > 1).sum())

    rows.append(
        {
            "dataset": dataset_name,
            "check": "conflicting_duplicate_feature_rows",
            "value": float(conflicting),
            "note": "Rows with same features but different labels",
        }
    )

    # 2) Shuffle-label sanity check (AUC should be around 0.5).
    shuffled_y = y.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=seed)),
        ]
    )
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    shuffled_scores = cross_validate(model, X, shuffled_y, cv=cv, scoring={"roc_auc": "roc_auc"}, n_jobs=-1)
    shuffled_auc = float(np.mean(shuffled_scores["test_roc_auc"]))

    rows.append(
        {
            "dataset": dataset_name,
            "check": "label_shuffle_roc_auc_mean",
            "value": shuffled_auc,
            "note": "Should be close to 0.5",
        }
    )

    # 3) Single-feature discriminatory power.
    for feature in FEATURES:
        feature_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=seed)),
            ]
        )
        feat_scores = cross_validate(
            feature_model,
            X[[feature]],
            y,
            cv=cv,
            scoring={"roc_auc": "roc_auc"},
            n_jobs=-1,
        )
        rows.append(
            {
                "dataset": dataset_name,
                "check": f"single_feature_auc::{feature}",
                "value": float(np.mean(feat_scores["test_roc_auc"])),
                "note": "Very high value can indicate strong separability",
            }
        )

    return rows


def feature_ablation(df: pd.DataFrame, dataset_name: str, seed: int) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    y = df[TARGET]

    base_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=seed)),
        ]
    )

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)

    # All features baseline
    baseline_scores = cross_validate(
        base_model,
        df[FEATURES],
        y,
        cv=cv,
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"},
        n_jobs=-1,
    )

    baseline_auc = float(np.mean(baseline_scores["test_roc_auc"]))
    baseline_pr = float(np.mean(baseline_scores["test_pr_auc"]))
    baseline_f1 = float(np.mean(baseline_scores["test_f1"]))

    rows.append(
        {
            "dataset": dataset_name,
            "setting": "all_features",
            "roc_auc_mean": baseline_auc,
            "pr_auc_mean": baseline_pr,
            "f1_mean": baseline_f1,
            "delta_roc_auc": 0.0,
        }
    )

    # Leave-one-feature-out
    for drop_feature in FEATURES:
        keep = [f for f in FEATURES if f != drop_feature]
        scores = cross_validate(
            base_model,
            df[keep],
            y,
            cv=cv,
            scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"},
            n_jobs=-1,
        )
        auc = float(np.mean(scores["test_roc_auc"]))
        rows.append(
            {
                "dataset": dataset_name,
                "setting": f"drop::{drop_feature}",
                "roc_auc_mean": auc,
                "pr_auc_mean": float(np.mean(scores["test_pr_auc"])),
                "f1_mean": float(np.mean(scores["test_f1"])),
                "delta_roc_auc": auc - baseline_auc,
            }
        )

    return rows


def calibrate_threshold(df: pd.DataFrame, dataset_name: str, seed: int, recall_target: float = 0.95) -> dict[str, float | str]:
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES],
        df[TARGET],
        test_size=0.2,
        stratify=df[TARGET],
        random_state=seed,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=seed)),
        ]
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.01, 0.99, 99)
    best = None

    y_true = y_test.values.astype(int)
    for thr in thresholds:
        pred = (probs >= thr).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        prec = precision_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)

        if rec >= recall_target:
            if best is None or prec > best["precision"] or (prec == best["precision"] and f1 > best["f1"]):
                best = {
                    "threshold": float(thr),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1),
                }

    # fallback to default threshold if target recall unattainable
    if best is None:
        pred = (probs >= 0.5).astype(int)
        best = {
            "threshold": 0.5,
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
        }

    best["dataset"] = dataset_name
    best["recall_target"] = float(recall_target)
    best["roc_auc"] = float(roc_auc_score(y_true, probs))
    best["pr_auc"] = float(average_precision_score(y_true, probs))
    return best


def stress_test_robustness(df: pd.DataFrame, dataset_name: str, seed: int) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES],
        df[TARGET],
        test_size=0.2,
        stratify=df[TARGET],
        random_state=seed,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=seed)),
        ]
    )
    model.fit(X_train, y_train)
    y_true = y_test.values.astype(int)

    scenarios = {
        "clean": X_test.copy(),
        "gaussian_noise_2pct": X_test.copy(),
        "gaussian_noise_5pct": X_test.copy(),
        "fev1_shift_minus_10pct": X_test.copy(),
        "weight_shift_plus_10pct": X_test.copy(),
    }

    for col in FEATURES:
        std = X_train[col].std()
        scenarios["gaussian_noise_2pct"][col] += np.random.normal(0, 0.02 * std, size=len(X_test))
        scenarios["gaussian_noise_5pct"][col] += np.random.normal(0, 0.05 * std, size=len(X_test))

    scenarios["fev1_shift_minus_10pct"]["fev1"] *= 0.9
    scenarios["weight_shift_plus_10pct"]["weight"] *= 1.1
    scenarios["weight_shift_plus_10pct"]["BMI"] = (
        scenarios["weight_shift_plus_10pct"]["weight"]
        / ((scenarios["weight_shift_plus_10pct"]["height"] / 100.0) ** 2)
    )

    for scenario_name, X_s in scenarios.items():
        probs = model.predict_proba(X_s)[:, 1]
        pred = (probs >= 0.5).astype(int)
        rows.append(
            {
                "dataset": dataset_name,
                "scenario": scenario_name,
                "accuracy": float(accuracy_score(y_true, pred)),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_true, probs)),
                "pr_auc": float(average_precision_score(y_true, probs)),
            }
        )

    return rows


def main() -> None:
    seed = 42
    datasets = [
        DatasetSpec("Cleaned", "cleaned_cf_dataset.csv"),
        DatasetSpec("Augmented", "augmented_cf_dataset.csv"),
    ]

    rows_cv: list[dict[str, float | str]] = []
    rows_holdout: list[dict[str, float | str]] = []
    rows_diag: list[dict[str, float | str]] = []
    rows_ablation: list[dict[str, float | str]] = []
    rows_threshold: list[dict[str, float | str]] = []
    rows_stress: list[dict[str, float | str]] = []

    for ds in datasets:
        df = load_df(ds.path)
        X = df[FEATURES]
        y = df[TARGET]

        print(f"\n=== Dataset: {ds.name} ({ds.path}) ===")
        print("Class distribution:", y.value_counts().to_dict())

        for model_name, model in get_models(seed).items():
            cv = cv_metrics(X, y, model, seed)
            hold = holdout_metrics(X, y, model, seed)

            row_base = {"dataset": ds.name, "model": model_name}
            rows_cv.append({**row_base, **cv})
            rows_holdout.append({**row_base, **hold})

            print(
                f"{model_name:24s} | CV ROC-AUC={cv['roc_auc_mean']:.4f} +- {cv['roc_auc_std']:.4f} "
                f"| CV PR-AUC={cv['pr_auc_mean']:.4f}"
            )

        rows_diag.extend(leakage_diagnostics(df, ds.name, seed))
        rows_ablation.extend(feature_ablation(df, ds.name, seed))
        rows_threshold.append(calibrate_threshold(df, ds.name, seed, recall_target=0.95))
        rows_stress.extend(stress_test_robustness(df, ds.name, seed))

    cv_df = pd.DataFrame(rows_cv).sort_values(["dataset", "roc_auc_mean", "pr_auc_mean"], ascending=[True, False, False])
    holdout_df = pd.DataFrame(rows_holdout).sort_values(["dataset", "holdout_roc_auc", "holdout_pr_auc"], ascending=[True, False, False])
    diag_df = pd.DataFrame(rows_diag).sort_values(["dataset", "check"])
    ablation_df = pd.DataFrame(rows_ablation).sort_values(["dataset", "roc_auc_mean"], ascending=[True, False])
    thr_df = pd.DataFrame(rows_threshold).sort_values(["dataset"])
    stress_df = pd.DataFrame(rows_stress).sort_values(["dataset", "scenario"])

    cv_df.to_csv("model_comparison_results_updated.csv", index=False)
    holdout_df.to_csv("model_comparison_results_final.csv", index=False)
    diag_df.to_csv("model_leakage_diagnostics.csv", index=False)
    ablation_df.to_csv("feature_ablation_results.csv", index=False)
    thr_df.to_csv("threshold_calibration_results.csv", index=False)
    stress_df.to_csv("robustness_stress_test_results.csv", index=False)

    print("\nSaved: model_comparison_results_updated.csv")
    print("Saved: model_comparison_results_final.csv")
    print("Saved: model_leakage_diagnostics.csv")
    print("Saved: feature_ablation_results.csv")
    print("Saved: threshold_calibration_results.csv")
    print("Saved: robustness_stress_test_results.csv")


if __name__ == "__main__":
    main()
