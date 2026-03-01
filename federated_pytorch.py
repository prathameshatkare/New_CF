"""Research-grade federated learning pipeline for cystic fibrosis diagnosis.

Key features:
- Non-IID client simulation via Dirichlet partitioning
- Partial client participation each round
- FedProx local objective for heterogeneity robustness
- Weighted FedAvg aggregation by client sample size
- Round-wise validation and best-checkpoint tracking
- Rich test metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC)
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

FEATURES = ["age", "sex", "height", "weight", "fev1", "BMI"]
TARGET = "target"


@dataclass
class Config:
    data_path: str = "augmented_cf_dataset.csv"
    rounds: int = 40
    num_clients: int = 5
    client_fraction: float = 0.8
    local_epochs: int = 6
    batch_size: int = 64
    lr: float = 8e-4
    weight_decay: float = 1e-4
    fedprox_mu: float = 0.01
    dirichlet_alpha: float = 0.6
    test_size: float = 0.2
    val_size: float = 0.2
    seed: int = 42
    early_stopping_patience: int = 10
    checkpoint_path: str = "best_federated_cf_model.pt"
    features: list[str] | None = None
    metrics_out: str | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataframe(path: str, features: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = features + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].copy()
    df[TARGET] = df[TARGET].astype(int)
    if "sex" in features:
        df["sex"] = df["sex"].astype(int)
    return df


def prepare_splits(df: pd.DataFrame, cfg: Config):
    train_df, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        stratify=df[TARGET],
        random_state=cfg.seed,
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=cfg.val_size,
        stratify=train_df[TARGET],
        random_state=cfg.seed,
    )

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_df[cfg.features].values)
    val_x = scaler.transform(val_df[cfg.features].values)
    test_x = scaler.transform(test_df[cfg.features].values)

    train_y = train_df[TARGET].values.astype(np.float32)
    val_y = val_df[TARGET].values.astype(np.float32)
    test_y = test_df[TARGET].values.astype(np.float32)

    return train_x, train_y, val_x, val_y, test_x, test_y, scaler


def dirichlet_partition(y: np.ndarray, num_clients: int, alpha: float, seed: int) -> list[np.ndarray]:
    """Create non-IID partitions with class-wise Dirichlet allocation."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y.astype(int))

    # Retry allocation until all clients have at least some data.
    for _ in range(50):
        client_indices: list[list[int]] = [[] for _ in range(num_clients)]

        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)

            proportions = rng.dirichlet(np.full(num_clients, alpha))
            splits = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
            chunks = np.split(cls_idx, splits)

            for cid, chunk in enumerate(chunks):
                client_indices[cid].extend(chunk.tolist())

        sizes = [len(v) for v in client_indices]
        if min(sizes) >= 8:
            return [np.array(v, dtype=int) for v in client_indices]

    # Safe fallback: near-IID split when Dirichlet is too sparse.
    all_idx = np.arange(len(y))
    rng.shuffle(all_idx)
    return [arr.astype(int) for arr in np.array_split(all_idx, num_clients)]


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


def to_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def local_train_fedprox(
    global_model: nn.Module,
    client_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    fedprox_mu: float,
) -> tuple[dict[str, torch.Tensor], int, float]:
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()

    global_params = {n: p.detach().clone() for n, p in global_model.named_parameters()}
    optimizer = optim.AdamW(local_model.parameters(), lr=lr, weight_decay=weight_decay)

    running_loss = 0.0
    n_samples = 0

    for _ in range(epochs):
        for xb, yb in client_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = local_model(xb).squeeze(1)
            base_loss = criterion(logits, yb)

            prox_penalty = torch.tensor(0.0, device=device)
            for name, p_local in local_model.named_parameters():
                prox_penalty += torch.sum((p_local - global_params[name]) ** 2)

            loss = base_loss + 0.5 * fedprox_mu * prox_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=2.0)
            optimizer.step()

            bs = yb.size(0)
            running_loss += float(base_loss.item()) * bs
            n_samples += bs

    avg_loss = running_loss / max(n_samples, 1)
    return local_model.state_dict(), n_samples, avg_loss


def weighted_fedavg(state_dicts: list[dict[str, torch.Tensor]], sizes: list[int]) -> dict[str, torch.Tensor]:
    total = float(sum(sizes))
    out = copy.deepcopy(state_dicts[0])

    for k in out.keys():
        out[k] = state_dicts[0][k] * (sizes[0] / total)
        for i in range(1, len(state_dicts)):
            out[k] += state_dicts[i][k] * (sizes[i] / total)

    return out


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32, device=device)).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()

    y_true = y.astype(int)
    y_pred = (probs >= 0.5).astype(int)

    # Handle degenerate cases gracefully.
    roc_auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan")
    pr_auc = average_precision_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def sample_clients(num_clients: int, fraction: float, rng: np.random.Generator) -> list[int]:
    k = max(1, int(np.ceil(num_clients * fraction)))
    return rng.choice(num_clients, size=k, replace=False).tolist()


def run(cfg: Config) -> dict[str, float]:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.features is None:
        cfg.features = FEATURES.copy()
    if len(cfg.features) == 0:
        raise ValueError("At least one feature is required.")

    df = load_dataframe(cfg.data_path, cfg.features)
    train_x, train_y, val_x, val_y, test_x, test_y, scaler = prepare_splits(df, cfg)

    # Class imbalance handling at global-train level.
    n_pos = int((train_y == 1).sum())
    n_neg = int((train_y == 0).sum())
    pos_weight = max(n_neg / max(n_pos, 1), 1.0)

    print("Dataset summary")
    print(f"  Train: {len(train_y)} (pos={n_pos}, neg={n_neg})")
    print(f"  Val  : {len(val_y)}")
    print(f"  Test : {len(test_y)}")

    client_idx = dirichlet_partition(
        train_y,
        num_clients=cfg.num_clients,
        alpha=cfg.dirichlet_alpha,
        seed=cfg.seed,
    )

    client_loaders: list[DataLoader] = []
    for cid, idx in enumerate(client_idx, start=1):
        cx, cy = train_x[idx], train_y[idx]
        loader = to_loader(cx, cy, batch_size=cfg.batch_size, shuffle=True)
        client_loaders.append(loader)
        print(f"Client {cid}: n={len(cy)}, pos={int((cy == 1).sum())}, neg={int((cy == 0).sum())}")

    print(f"Using features: {cfg.features}")

    global_model = CFNet(in_dim=len(cfg.features)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    rng = np.random.default_rng(cfg.seed)
    best_val_auc = -np.inf
    patience = 0

    for r in range(1, cfg.rounds + 1):
        selected = sample_clients(cfg.num_clients, cfg.client_fraction, rng)

        states: list[dict[str, torch.Tensor]] = []
        sizes: list[int] = []
        local_losses: list[float] = []

        for cid in selected:
            state, n, loss = local_train_fedprox(
                global_model=global_model,
                client_loader=client_loaders[cid],
                criterion=criterion,
                device=device,
                epochs=cfg.local_epochs,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                fedprox_mu=cfg.fedprox_mu,
            )
            states.append(state)
            sizes.append(n)
            local_losses.append(loss)

        new_global = weighted_fedavg(states, sizes)
        global_model.load_state_dict(new_global)

        val_metrics = evaluate(global_model, val_x, val_y, device)
        mean_local_loss = float(np.mean(local_losses)) if local_losses else float("nan")

        print(
            f"Round {r:02d} | clients={selected} | "
            f"local_loss={mean_local_loss:.4f} | val_auc={val_metrics['roc_auc']:.4f} | val_f1={val_metrics['f1']:.4f}"
        )

        current_auc = val_metrics["roc_auc"]
        if np.isfinite(current_auc) and current_auc > best_val_auc:
            best_val_auc = current_auc
            patience = 0
            torch.save(
                {
                    "model_state_dict": global_model.state_dict(),
                    "features": cfg.features,
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "config": cfg.__dict__,
                },
                cfg.checkpoint_path,
            )
        else:
            patience += 1

        if patience >= cfg.early_stopping_patience:
            print(f"Early stopping triggered at round {r}.")
            break

    if Path(cfg.checkpoint_path).exists():
        ckpt = torch.load(cfg.checkpoint_path, map_location=device)
        global_model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(global_model, test_x, test_y, device)
    print("\n=== Final Test Metrics (Best Checkpoint) ===")
    print(f"Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall   : {test_metrics['recall']:.4f}")
    print(f"F1-score : {test_metrics['f1']:.4f}")
    print(f"ROC-AUC  : {test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC   : {test_metrics['pr_auc']:.4f}")
    print(f"Checkpoint saved at: {cfg.checkpoint_path}")
    print(f"Features : {cfg.features}")

    if cfg.metrics_out:
        payload = {
            "features": cfg.features,
            "checkpoint_path": cfg.checkpoint_path,
            "metrics": test_metrics,
            "config": cfg.__dict__,
        }
        with open(cfg.metrics_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Metrics written to: {cfg.metrics_out}")

    return test_metrics


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Research-grade FL for CF diagnosis")
    parser.add_argument("--data-path", default="augmented_cf_dataset.csv")
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--client-fraction", type=float, default=0.8)
    parser.add_argument("--local-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.6)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--checkpoint-path", default="best_federated_cf_model.pt")
    parser.add_argument("--metrics-out", default="")
    parser.add_argument(
        "--drop-features",
        default="",
        help="Comma-separated feature names to drop, e.g. fev1,BMI",
    )

    args = parser.parse_args()
    drop_features = [f.strip() for f in args.drop_features.split(",") if f.strip()]
    unknown = [f for f in drop_features if f not in FEATURES]
    if unknown:
        raise ValueError(f"Unknown features in --drop-features: {unknown}")
    active_features = [f for f in FEATURES if f not in drop_features]

    return Config(
        data_path=args.data_path,
        rounds=args.rounds,
        num_clients=args.num_clients,
        client_fraction=args.client_fraction,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        fedprox_mu=args.fedprox_mu,
        dirichlet_alpha=args.dirichlet_alpha,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_path=args.checkpoint_path,
        features=active_features,
        metrics_out=args.metrics_out if args.metrics_out else None,
    )


if __name__ == "__main__":
    run(parse_args())
