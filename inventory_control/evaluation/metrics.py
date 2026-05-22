"""Metrics for lost-sales inventory experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_summary_metrics(episodes: list[pd.DataFrame]) -> dict[str, float]:
    """Compute aggregate metrics from per-period episode traces."""
    if not episodes:
        raise ValueError("episodes must not be empty")

    total_costs = np.array([episode["total_cost"].sum() for episode in episodes], dtype=float)
    holding_costs = np.array([episode["holding_cost"].sum() for episode in episodes], dtype=float)
    lost_sales_costs = np.array([episode["lost_sales_cost"].sum() for episode in episodes], dtype=float)
    order_costs = np.array([episode["order_cost"].sum() for episode in episodes], dtype=float)
    total_periods = sum(len(episode) for episode in episodes)
    all_rows = pd.concat(episodes, ignore_index=True)
    total_demand = float(all_rows["demand"].sum())
    total_sales = float(all_rows["sales"].sum())
    total_lost_sales = float(all_rows["lost_sales"].sum())

    service_level = total_sales / total_demand if total_demand > 0 else 1.0
    lost_sales_rate = total_lost_sales / total_demand if total_demand > 0 else 0.0

    metrics = {
        "mean_total_cost": float(total_costs.mean()),
        "std_total_cost": float(total_costs.std(ddof=0)),
        "mean_cost_per_period": float(total_costs.sum() / total_periods),
        "mean_holding_cost": float(holding_costs.mean()),
        "mean_lost_sales_cost": float(lost_sales_costs.mean()),
        "mean_order_cost": float(order_costs.mean()),
        "mean_service_level": float(service_level),
        "mean_lost_sales_rate": float(lost_sales_rate),
        "mean_inventory": float(all_rows["ending_inventory"].mean()),
        "mean_order_quantity": float(all_rows["order_quantity"].mean()),
    }
    if "regular_order_quantity" in all_rows:
        regular_order_costs = np.array(
            [episode["regular_order_cost"].sum() for episode in episodes],
            dtype=float,
        )
        expedited_order_costs = np.array(
            [episode["expedited_order_cost"].sum() for episode in episodes],
            dtype=float,
        )
        metrics.update(
            {
                "mean_regular_order_quantity": float(all_rows["regular_order_quantity"].mean()),
                "mean_expedited_order_quantity": float(all_rows["expedited_order_quantity"].mean()),
                "mean_regular_order_cost": float(regular_order_costs.mean()),
                "mean_expedited_order_cost": float(expedited_order_costs.mean()),
            }
        )
    return metrics


def episode_total(trace: pd.DataFrame) -> dict[str, Any]:
    """Return simple totals for one episode trace."""
    return {
        "total_cost": float(trace["total_cost"].sum()),
        "total_demand": int(trace["demand"].sum()),
        "total_sales": int(trace["sales"].sum()),
        "total_lost_sales": int(trace["lost_sales"].sum()),
    }
