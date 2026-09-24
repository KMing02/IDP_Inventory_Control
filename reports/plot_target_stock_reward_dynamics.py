"""Plot the post-protected validation-only target-stock reward dynamics.

This script reads only completed validation CSV artifacts.  It does not load
models, generate products, or enter a training/evaluation code path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DIRECT_SOURCE = (
    REPOSITORY_ROOT
    / "reports/data/direct_q_checkpoint_seed_metrics.csv"
)
TARGET_SOURCE = (
    REPOSITORY_ROOT
    / "reports/data/target_stock_extended_checkpoint_seed_metrics.csv"
)
OUTPUT = Path(__file__).resolve().parent / "figures/target_stock_reward_dynamics.pdf"
CHECKPOINTS = (0, 1, 5, 15, 30, 60, 90, 120)


def main() -> None:
    direct = pd.read_csv(DIRECT_SOURCE)[
        ["policy_name", "training_seed", "reward_update", "mean_reward_per_period"]
    ]
    target = pd.read_csv(TARGET_SOURCE)[
        ["policy_name", "training_seed", "reward_update", "mean_reward_per_period"]
    ]
    data = pd.concat([direct, target], ignore_index=True)
    expected = {
        "PaperFCNNDirectQ120",
        "TargetStockFCNNStandard",
        "TargetStockFCNNInformed",
    }
    if set(data["policy_name"]) != expected:
        raise RuntimeError("Unexpected policy set in validation dynamics artifacts")
    if set(data["training_seed"].astype(int)) != {0, 1, 2}:
        raise RuntimeError("Expected exactly validation training seeds 0, 1, and 2")
    if set(data["reward_update"].astype(int)) != set(CHECKPOINTS):
        raise RuntimeError("Unexpected validation checkpoint set")

    summary = (
        data.groupby(["policy_name", "reward_update"], as_index=False)
        ["mean_reward_per_period"]
        .agg(["mean", "std"])
        .reset_index()
    )
    styles = {
        "PaperFCNNDirectQ120": {
            "label": "Direct-q FCNN",
            "color": "#4C78A8",
            "marker": "o",
            "linestyle": "-",
        },
        "TargetStockFCNNStandard": {
            "label": "TargetStockFCNN (standard init.)",
            "color": "#F28E2B",
            "marker": "s",
            "linestyle": "--",
        },
        "TargetStockFCNNInformed": {
            "label": "TargetStockFCNN (informed init.)",
            "color": "#8A6D1D",
            "marker": "^",
            "linestyle": "-.",
        },
    }

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.15, 4.15), constrained_layout=True)
    for policy, style in styles.items():
        group = summary.loc[summary["policy_name"] == policy].sort_values("reward_update")
        x = group["reward_update"].to_numpy(dtype=float)
        mean = group["mean"].to_numpy(dtype=float)
        sd = group["std"].to_numpy(dtype=float)
        ax.fill_between(x, mean - sd, mean + sd, color=style["color"], alpha=0.13, linewidth=0)
        ax.plot(
            x,
            mean,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=4.8,
            markerfacecolor="white",
            markeredgewidth=1.2,
        )
    ax.axvline(15, color="#666666", linewidth=0.9, linestyle=":")
    ax.text(16.5, 1260, "original 15-update budget", color="#555555", fontsize=8)
    ax.set_title("Reward-only training dynamics: direct order versus target stock")
    ax.set_xlabel("Reward optimizer updates")
    ax.set_ylabel("Validation reward per scored period")
    # Update 1 remains visible as a marker but is not separately labelled,
    # because labels 0 and 1 overlap on an honest linear update axis.
    ax.set_xticks((0, 5, 15, 30, 60, 90, 120))
    ax.set_xlim(-2, 123)
    ax.set_ylim(1180, 2205)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    ax.text(
        0.0,
        -0.19,
        "Post-protected exploratory validation analysis; mean and sample-SD band over seeds 0--2.",
        transform=ax.transAxes,
        fontsize=8,
        color="#4A4A4A",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
