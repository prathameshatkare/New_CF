"""Run federated feature ablation experiments and aggregate results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

FEATURES = ["age", "sex", "height", "weight", "fev1", "BMI"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated ablation experiment runner")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out-dir", default="ablation_runs")
    parser.add_argument("--results-csv", default="federated_ablation_results.csv")
    parser.add_argument("--data-path", default="augmented_cf_dataset.csv")
    return parser.parse_args()


def experiment_specs() -> list[tuple[str, str]]:
    specs = [("all_features", "")]
    specs.extend((f"drop_{f}", f) for f in FEATURES)
    return specs


def run_experiment(args: argparse.Namespace, name: str, drop_feature: str, out_dir: Path) -> dict:
    metrics_path = out_dir / f"{name}_metrics.json"
    ckpt_path = out_dir / f"{name}.pt"

    if metrics_path.exists():
        print(f"\nSkipping {name} (existing metrics found)")
        with open(metrics_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        metrics = payload["metrics"]
        features = payload["features"]
        dropped = [f for f in FEATURES if f not in features]
        return {
            "experiment": name,
            "dropped_features": ",".join(dropped) if dropped else "none",
            "num_features": len(features),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "checkpoint_path": str(ckpt_path),
            "metrics_path": str(metrics_path),
        }

    cmd = [
        args.python,
        "federated_pytorch.py",
        "--data-path",
        args.data_path,
        "--rounds",
        str(args.rounds),
        "--local-epochs",
        str(args.local_epochs),
        "--batch-size",
        str(args.batch_size),
        "--checkpoint-path",
        str(ckpt_path),
        "--metrics-out",
        str(metrics_path),
    ]

    if drop_feature:
        cmd.extend(["--drop-features", drop_feature])

    print(f"\nRunning experiment: {name}")
    subprocess.run(cmd, check=True)

    with open(metrics_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    metrics = payload["metrics"]
    features = payload["features"]
    dropped = [f for f in FEATURES if f not in features]

    return {
        "experiment": name,
        "dropped_features": ",".join(dropped) if dropped else "none",
        "num_features": len(features),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "checkpoint_path": str(ckpt_path),
        "metrics_path": str(metrics_path),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, drop_feature in experiment_specs():
        row = run_experiment(args, name, drop_feature, out_dir)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["roc_auc", "pr_auc", "f1"], ascending=[False, False, False]).reset_index(drop=True)
    df.to_csv(args.results_csv, index=False)

    print(f"\nSaved ablation table: {args.results_csv}")
    print(df[["experiment", "dropped_features", "f1", "roc_auc", "pr_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
