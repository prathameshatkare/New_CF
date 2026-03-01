"""One-command runner for the full CF federated learning project pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def run_step(cmd: list[str], title: str) -> None:
    print(f"\n=== {title} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CF federated-learning project pipeline")
    parser.add_argument("--full-regeneration", action="store_true")
    parser.add_argument("--ablation-rounds", type=int, default=8)
    parser.add_argument("--ablation-local-epochs", type=int, default=2)
    parser.add_argument("--federated-rounds", type=int, default=20)
    parser.add_argument("--federated-local-epochs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable

    cleaned_exists = Path("cleaned_cf_dataset.csv").exists()
    augmented_exists = Path("augmented_cf_dataset.csv").exists()

    if args.full_regeneration or not cleaned_exists:
        run_step([py, "data_preprocessing_and_stat_validation.py"], "1) Preprocessing + Statistical Validation")
    else:
        print("\n=== 1) Preprocessing + Statistical Validation ===")
        print("Skipped (cleaned_cf_dataset.csv already exists)")

    if args.full_regeneration or not augmented_exists:
        run_step([py, "data_augmentation_hybrid.py"], "2) Minority Augmentation (TVAE)")
    else:
        print("\n=== 2) Minority Augmentation (TVAE) ===")
        print("Skipped (augmented_cf_dataset.csv already exists)")

    run_step([py, "federated_data_partition.py"], "3) Client Partitioning")
    run_step(
        [
            py,
            "federated_pytorch.py",
            "--rounds",
            str(args.federated_rounds),
            "--local-epochs",
            str(args.federated_local_epochs),
            "--checkpoint-path",
            "best_federated_cf_model.pt",
            "--metrics-out",
            "best_federated_cf_metrics.json",
        ],
        "4) Federated Training (Research-grade)",
    )
    run_step(
        [
            py,
            "federated_ablation_runner.py",
            "--python",
            py,
            "--rounds",
            str(args.ablation_rounds),
            "--local-epochs",
            str(args.ablation_local_epochs),
        ],
        "5) Federated Feature Ablation",
    )
    run_step([py, "model_comparison.py"], "6) Baselines + Leakage + Stress Tests")
    run_step([py, "plot_federated_ablation.py"], "7) Plot Generation")

    print("\nPipeline completed successfully.")
    print("You can now run edge inference with:")
    print(f"{py} edge_device_app.py --checkpoint best_federated_cf_model.pt")


if __name__ == "__main__":
    main()
