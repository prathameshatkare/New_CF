"""Create publication-style plots for federated feature ablation results."""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

IN_CSV = "federated_ablation_results.csv"
OUT_PNG = "federated_ablation_plot.png"
OUT_PAPER_PNG = "federated_ablation_plot_paper_bw.png"
OUT_PAPER_PDF = "federated_ablation_plot_paper_bw.pdf"
OUT_PAPER_SVG = "federated_ablation_plot_paper_bw.svg"


def main() -> None:
    df = pd.read_csv(IN_CSV)
    df = df.copy()
    df["label"] = df["experiment"].str.replace("_", " ", regex=False)

    # Keep a stable order by F1 to emphasize practical classification impact.
    df = df.sort_values("f1", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=220)
    bars = ax.barh(df["label"], df["f1"], color="#2C7FB8", alpha=0.9, label="F1")
    ax.plot(df["roc_auc"], df["label"], "o-", color="#D95F0E", linewidth=2.0, label="ROC-AUC")
    ax.plot(df["pr_auc"], df["label"], "s--", color="#238B45", linewidth=2.0, label="PR-AUC")

    for bar, val in zip(bars, df["f1"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8)

    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("Score")
    ax.set_ylabel("Ablation Experiment")
    ax.set_title("Federated Feature Ablation (Cystic Fibrosis Diagnosis)")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(OUT_PNG, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")

    # Paper-ready version: serif font and print-safe grayscale styling.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
    )

    fig2, ax2 = plt.subplots(figsize=(7.2, 4.6), dpi=300)
    bars2 = ax2.barh(
        df["label"],
        df["f1"],
        color="#5C5C5C",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.95,
        label="F1",
    )
    ax2.plot(df["roc_auc"], df["label"], "o-", color="#000000", linewidth=1.4, markersize=4, label="ROC-AUC")
    ax2.plot(
        df["pr_auc"],
        df["label"],
        "s--",
        color="#8A8A8A",
        linewidth=1.4,
        markersize=4,
        label="PR-AUC",
    )

    for bar, val in zip(bars2, df["f1"]):
        ax2.text(val + 0.004, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=7.5)

    ax2.set_xlim(0.0, 1.02)
    ax2.set_xlabel("Score")
    ax2.set_ylabel("Ablation Experiment")
    ax2.set_title("Federated Feature Ablation (CF Diagnosis)")
    ax2.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.6)
    ax2.legend(loc="lower right", frameon=True, edgecolor="black")
    plt.tight_layout()
    plt.savefig(OUT_PAPER_PNG, bbox_inches="tight", dpi=300)
    plt.savefig(OUT_PAPER_PDF, bbox_inches="tight")
    plt.savefig(OUT_PAPER_SVG, bbox_inches="tight")
    print(f"Saved: {OUT_PAPER_PNG}")
    print(f"Saved: {OUT_PAPER_PDF}")
    print(f"Saved: {OUT_PAPER_SVG}")


if __name__ == "__main__":
    main()
