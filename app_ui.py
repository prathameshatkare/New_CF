"""Enhanced Streamlit UI for CF risk prediction using federated checkpoint."""

from __future__ import annotations

import numpy as np
import streamlit as st
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


@st.cache_resource
def load_checkpoint_model(path: str):
    device = torch.device("cpu")
    ckpt = torch.load(path, map_location=device)
    features = ckpt["features"]
    scaler_mean = ckpt["scaler_mean"]
    scaler_scale = ckpt["scaler_scale"]

    model = CFNet(in_dim=len(features))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, features, scaler_mean, scaler_scale


def scale_input(x: np.ndarray, mean: list[float], scale: list[float]) -> np.ndarray:
    mu = np.asarray(mean, dtype=np.float32)
    sigma = np.asarray(scale, dtype=np.float32)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (x - mu) / sigma


def main() -> None:
    st.set_page_config(page_title="CF Federated Risk UI", page_icon="🫁", layout="centered")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 1.2rem;}
        .risk-card {
            border: 1px solid #d7dce2;
            border-radius: 14px;
            padding: 14px 16px;
            background: linear-gradient(180deg,#fbfcfe 0%,#f6f8fb 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Cystic Fibrosis Risk Screening")
    st.caption("Federated model inference from `best_federated_cf_model.pt`")

    ckpt_path = st.text_input("Checkpoint path", value="best_federated_cf_model.pt")
    try:
        model, features, scaler_mean, scaler_scale = load_checkpoint_model(ckpt_path)
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        return

    st.info(f"Model features: {', '.join(features)}")

    with st.expander("How to get FEV1 (important)", expanded=False):
        st.markdown(
            """
            - FEV1 is measured using a **spirometry test** (hospital, lab, or clinic).
            - In most reports, FEV1 may appear as:
              - **FEV1 in liters (L)**, or
              - **FEV1 in mL**, or
              - **FEV1 % predicted** with a predicted/reference FEV1 value.
            - If your report gives `% predicted`, compute:
              - `FEV1 (L) = Predicted FEV1 (L) * (% predicted / 100)`
            - Use recent spirometry values. This tool supports screening, not final diagnosis.
            """
        )

    with st.form("cf_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=0.0, max_value=120.0, value=18.0, step=1.0)
            sex = st.selectbox("Sex", options=["Female", "Male"])
            height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=165.0, step=0.1)
        with c2:
            weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=60.0, step=0.1)
            fev1_mode = st.selectbox(
                "FEV1 input mode",
                options=[
                    "Direct FEV1 (L)",
                    "Direct FEV1 (mL)",
                    "From report: % predicted + predicted FEV1 (L)",
                ],
            )
            if fev1_mode == "Direct FEV1 (L)":
                fev1_l = st.number_input("FEV1 (L)", min_value=0.0, max_value=10.0, value=2.5, step=0.01)
            elif fev1_mode == "Direct FEV1 (mL)":
                fev1_ml = st.number_input("FEV1 (mL)", min_value=0.0, max_value=10000.0, value=2500.0, step=10.0)
                fev1_l = fev1_ml / 1000.0
            else:
                fev1_pct = st.number_input("FEV1 % predicted", min_value=0.0, max_value=300.0, value=80.0, step=0.1)
                fev1_pred_l = st.number_input(
                    "Predicted FEV1 from report (L)", min_value=0.0, max_value=10.0, value=3.0, step=0.01
                )
                fev1_l = fev1_pred_l * (fev1_pct / 100.0)

        auto_bmi = st.checkbox("Auto-compute BMI from height/weight", value=True)
        bmi_manual = st.number_input("BMI (used only if auto-compute is off)", min_value=5.0, max_value=80.0, value=22.0, step=0.1)
        submitted = st.form_submit_button("Predict CF Risk")

    if not submitted:
        return

    bmi = weight / ((height / 100.0) ** 2) if auto_bmi else bmi_manual
    sex_val = 1.0 if sex == "Male" else 0.0

    values = {
        "age": age,
        "sex": sex_val,
        "height": height,
        "weight": weight,
        "fev1": fev1_l,
        "BMI": bmi,
    }

    x = np.array([[values[f] for f in features]], dtype=np.float32)
    x = scale_input(x, scaler_mean, scaler_scale)

    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32)).squeeze(1)
        prob = float(torch.sigmoid(logits).item())

    st.subheader("Prediction")
    st.markdown('<div class="risk-card">', unsafe_allow_html=True)
    st.metric("CF risk probability", f"{prob:.3f}")
    if prob >= 0.5:
        st.error("High CF risk (screen positive)")
    else:
        st.success("Lower CF risk (screen negative)")
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(f"Computed BMI: {bmi:.2f} | Effective FEV1 used: {fev1_l:.3f} L")

    st.progress(min(max(prob, 0.0), 1.0))


if __name__ == "__main__":
    main()
