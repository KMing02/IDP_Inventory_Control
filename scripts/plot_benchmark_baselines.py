"""Plot benchmark baseline results."""

from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs/.cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    """Create scenario-level benchmark plots."""
    results_path = Path("outputs/results/benchmark_baselines.csv")
    if not results_path.exists():
        raise FileNotFoundError("run scripts/run_benchmark_baselines.py first")

    results = pd.read_csv(results_path)
    output_dir = Path("outputs/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(results["scenario_name"], results["mean_total_cost"])
    ax.set_ylabel("Mean Total Cost")
    ax.set_title("Baseline Performance by Scenario")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "benchmark_baseline_performance.png", dpi=150)

    breakdown_columns = ["mean_holding_cost", "mean_lost_sales_cost", "mean_order_cost"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = None
    for column in breakdown_columns:
        values = results[column]
        ax.bar(results["scenario_name"], values, bottom=bottom, label=column)
        bottom = values if bottom is None else bottom + values
    ax.set_ylabel("Mean Episode Cost")
    ax.set_title("Benchmark Cost Breakdown")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "benchmark_cost_breakdown.png", dpi=150)
    plt.close("all")


if __name__ == "__main__":
    main()
