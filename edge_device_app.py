"""Edge-device inference using trained federated checkpoint."""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


class CFNet(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edge CF risk inference from federated model")
    parser.add_argument("--checkpoint", default="best_federated_cf_model.pt")
    return parser.parse_args()


def read_inputs(feature_order: Sequence[str]) -> np.ndarray:
    prompts = {
        "age": "Age: ",
        "sex": "Sex (Male/Female): ",
        "height": "Height (cm): ",
        "weight": "Weight (kg): ",
        "fev1": "FEV1: ",
        "BMI": "BMI (press Enter to auto-compute): ",
    }
    values: dict[str, float] = {}

    for feature in feature_order:
        raw = input(prompts.get(feature, f"{feature}: ")).strip()
        if feature == "sex":
            values["sex"] = 1.0 if raw.lower() in {"male", "m", "1"} else 0.0
        elif feature == "BMI" and raw == "":
            # Compute BMI later after height/weight are available.
            continue
        else:
            values[feature] = float(raw)

    if "BMI" in feature_order and "BMI" not in values:
        if "height" not in values or "weight" not in values:
            raise ValueError("BMI requires height and weight when auto-computing.")
        values["BMI"] = values["weight"] / ((values["height"] / 100.0) ** 2)

    ordered = np.array([[values[f] for f in feature_order]], dtype=np.float32)
    return ordered


def apply_scaler(x: np.ndarray, mean: list[float], scale: list[float]) -> np.ndarray:
    mu = np.asarray(mean, dtype=np.float32)
    sigma = np.asarray(scale, dtype=np.float32)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (x - mu) / sigma


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    features = ckpt["features"]
    scaler_mean = ckpt.get("scaler_mean")
    scaler_scale = ckpt.get("scaler_scale")

    if scaler_mean is None or scaler_scale is None:
        raise ValueError(
            "Checkpoint missing scaler stats. Retrain with the updated federated_pytorch.py."
        )

    model = CFNet(in_dim=len(features)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("\n=== Edge CF Risk Screening (Federated Model) ===")
    print("Model features:", ", ".join(features))

    sample = read_inputs(features)
    sample_scaled = apply_scaler(sample, scaler_mean, scaler_scale)

    with torch.no_grad():
        logits = model(torch.tensor(sample_scaled, dtype=torch.float32, device=device)).squeeze(1)
        prob = float(torch.sigmoid(logits).cpu().item())

    print(f"\nPredicted CF risk probability: {prob:.3f}")
    if prob >= 0.5:
        print("High CF risk (screen positive)")
    else:
        print("Lower CF risk (screen negative)")


if __name__ == "__main__":
    main()

