"""FastAPI backend for CF risk prediction web UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


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


class PredictRequest(BaseModel):
    age: float = Field(ge=0, le=120)
    sex: str | int
    height: float = Field(gt=30, le=250)
    weight: float = Field(gt=5, le=350)
    fev1_l: float = Field(gt=0, le=12)
    bmi: float | None = Field(default=None, gt=5, le=90)


class PredictResponse(BaseModel):
    risk_probability: float
    risk_label: str
    used_features: dict[str, float]


app = FastAPI(title="CF Federated Web API", version="1.0.0")
WEB_DIR = Path("web")
CKPT_PATH = Path("best_federated_cf_model.pt")


class ModelBundle:
    def __init__(self) -> None:
        self.model: nn.Module | None = None
        self.features: list[str] = []
        self.scaler_mean: np.ndarray | None = None
        self.scaler_scale: np.ndarray | None = None

    def load(self) -> None:
        if not CKPT_PATH.exists():
            raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        self.features = ckpt["features"]
        self.scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
        self.scaler_scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
        self.scaler_scale = np.where(self.scaler_scale == 0, 1.0, self.scaler_scale)

        model = CFNet(in_dim=len(self.features))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        self.model = model

    def predict(self, values: dict[str, float]) -> float:
        if self.model is None or self.scaler_mean is None or self.scaler_scale is None:
            raise RuntimeError("Model bundle not initialized.")

        x = np.array([[values[f] for f in self.features]], dtype=np.float32)
        x = (x - self.scaler_mean) / self.scaler_scale
        with torch.no_grad():
            logits = self.model(torch.tensor(x, dtype=torch.float32)).squeeze(1)
            prob = float(torch.sigmoid(logits).item())
        return prob


bundle = ModelBundle()


@app.on_event("startup")
def _startup() -> None:
    bundle.load()


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "checkpoint": str(CKPT_PATH),
        "features": bundle.features,
    }


@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        sex_val = payload.sex
        if isinstance(sex_val, str):
            sx = sex_val.strip().lower()
            sex_num = 1.0 if sx in {"male", "m", "1"} else 0.0
        else:
            sex_num = 1.0 if int(sex_val) == 1 else 0.0

        bmi_val = payload.bmi
        if bmi_val is None:
            bmi_val = payload.weight / ((payload.height / 100.0) ** 2)

        feature_values = {
            "age": float(payload.age),
            "sex": float(sex_num),
            "height": float(payload.height),
            "weight": float(payload.weight),
            "fev1": float(payload.fev1_l),
            "BMI": float(bmi_val),
        }
        prob = bundle.predict(feature_values)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    return PredictResponse(
        risk_probability=prob,
        risk_label="High CF risk (screen positive)" if prob >= 0.5 else "Lower CF risk (screen negative)",
        used_features=feature_values,
    )


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")

