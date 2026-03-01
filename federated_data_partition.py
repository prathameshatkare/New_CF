"""Create client-specific federated datasets from augmented CF data."""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SOURCE_PATH = "augmented_cf_dataset.csv"
OUT_DIR = "federated_clients"
NUM_CLIENTS = 5
RANDOM_STATE = 42
TARGET = "target"


def run() -> None:
    df = pd.read_csv(SOURCE_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Expected '{TARGET}' column in {SOURCE_PATH}")

    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    os.makedirs(OUT_DIR, exist_ok=True)

    splitter = StratifiedKFold(
        n_splits=NUM_CLIENTS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    y = df[TARGET].astype(int).values
    X_dummy = np.zeros(len(df))

    for i, (_, client_idx) in enumerate(splitter.split(X_dummy, y), start=1):
        client_df = df.iloc[client_idx].reset_index(drop=True)
        out_path = os.path.join(OUT_DIR, f"client_{i}.csv")
        client_df.to_csv(out_path, index=False)
        print(f"Saved {out_path}: shape={client_df.shape}")
        print(client_df[TARGET].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    run()
