"""Create contour-style policy diagnostics from saved benchmark predictions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from inventory_control.policies.neural.registry import RESIDUAL_COMPARISON_POLICY_NAMES

DEFAULT_PLOT_POLICIES = [
    *RESIDUAL_COMPARISON_POLICY_NAMES,
    "HistoricalDemandBaseStock",
    "NonOmniscientOptimal",
    "OmniscientOptimal",
]


def _require_prediction_columns(predictions: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions.csv is missing required columns: {sorted(missing)}")


def _plot_inventory_demand_contours(predictions: pd.DataFrame, policies: list[str], output_path: Path) -> None:
    """Plot deployed order q_theta against inventory and historical mean demand."""
    _require_prediction_columns(
        predictions,
        {"policy_name", "historical_mean_demand", "inventory_normalized", "deployed_order", "state_index"},
    )
    available = [policy for policy in policies if policy in set(predictions["policy_name"])]
    if not available:
        raise ValueError(f"None of the requested policies are available: {policies}")

    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5), squeeze=False)
    for ax, policy_name in zip(axes[0], available):
        rows = predictions.loc[predictions["policy_name"] == policy_name].copy()
        pivot = rows.pivot_table(
            index="historical_mean_demand",
            columns="inventory_normalized",
            values="deployed_order",
            aggfunc="mean",
        ).sort_index()
        x_values = pivot.columns.to_numpy(dtype=float)
        y_values = pivot.index.to_numpy(dtype=float)
        z_values = pivot.to_numpy(dtype=float)
        levels = min(14, max(4, len(pd.unique(rows["state_index"])) // 2))
        contour = ax.contourf(x_values, y_values, z_values, levels=levels, cmap="viridis")
        line = ax.contour(x_values, y_values, z_values, levels=levels, colors="black", linewidths=0.45, alpha=0.65)
        ax.clabel(line, inline=True, fontsize=7, fmt="%.0f")
        fig.colorbar(contour, ax=ax, label="deployed order")
        ax.set_title(policy_name)
        ax.set_xlabel("inventory / historical mean demand")
        ax.set_ylabel("historical mean demand")
        ax.grid(alpha=0.2)
    fig.suptitle("Policy surface: deployed order q_theta by inventory and demand scale")
    fig.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def _plot_raw_vs_deployed_curves(predictions: pd.DataFrame, policies: list[str], output_path: Path) -> None:
    """Plot raw proposal, deployed order q_theta, and oracle order for one product."""
    _require_prediction_columns(predictions, {"product_id", "policy_name", "inventory", "raw_order", "deployed_order", "oracle_order"})
    product_id = str(predictions["product_id"].iloc[0])
    rows = predictions.loc[predictions["product_id"] == product_id].copy()
    available = [policy for policy in policies if policy in set(rows["policy_name"])]
    if not available:
        raise ValueError(f"None of the requested policies are available for product {product_id}: {policies}")
    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 4), squeeze=False)
    for ax, policy_name in zip(axes[0], available):
        policy_rows = rows.loc[rows["policy_name"] == policy_name].sort_values("inventory")
        ax.plot(policy_rows["inventory"], policy_rows["raw_order"], label="raw proposal before guard", linewidth=2)
        ax.plot(policy_rows["inventory"], policy_rows["deployed_order"], label="deployed order q_theta", linewidth=2)
        ax.plot(policy_rows["inventory"], policy_rows["oracle_order"], label="oracle order", linewidth=1.5, linestyle="--")
        ax.set_title(f"{policy_name} on {product_id}")
        ax.set_xlabel("inventory")
        ax.set_ylabel("order")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Raw proposal vs deployed policy curve")
    fig.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_dir", type=Path, help="Saved benchmark directory containing predictions.csv")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=DEFAULT_PLOT_POLICIES,
        help="Policy names to plot if present.",
    )
    args = parser.parse_args()

    predictions_path = args.benchmark_dir / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"missing predictions file: {predictions_path}")
    try:
        predictions = pd.read_csv(predictions_path)
    except pd.errors.EmptyDataError as error:
        raise ValueError(
            f"predictions file is empty: {predictions_path}; rerun benchmark without --fast-diagnostics"
        ) from error
    if predictions.empty:
        raise ValueError(f"predictions file has no rows: {predictions_path}; rerun benchmark without --fast-diagnostics")
    plots_dir = args.benchmark_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    contour_path = plots_dir / "policy_inventory_demand_contours.png"
    curves_path = plots_dir / "policy_raw_vs_deployed_curves.png"
    _plot_inventory_demand_contours(predictions, args.policies, contour_path)
    _plot_raw_vs_deployed_curves(predictions, args.policies, curves_path)
    print(f"saved_contours={contour_path}")
    print(f"saved_curves={curves_path}")


if __name__ == "__main__":
    main()
