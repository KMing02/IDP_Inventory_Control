"""Scientific-validity diagnostics for monotonic architecture benchmarks."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


def paired_policy_monotonicity_rates(
    raw_policy: Callable[[float], float],
    deployed_policy: Callable[[float], float],
    inventory_low: Sequence[float],
    inventory_high: Sequence[float],
    tolerance: float = 1e-6,
) -> tuple[float, float]:
    """Return raw and deployed violation rates for paired inventory states."""
    low = np.asarray(inventory_low, dtype=float)
    high = np.asarray(inventory_high, dtype=float)
    if low.shape != high.shape or np.any(high <= low):
        raise ValueError("inventory pairs must have matching shapes and satisfy y_hi > y_lo")
    raw_low = np.asarray([raw_policy(float(value)) for value in low], dtype=float)
    raw_high = np.asarray([raw_policy(float(value)) for value in high], dtype=float)
    deployed_low = np.asarray([deployed_policy(float(value)) for value in low], dtype=float)
    deployed_high = np.asarray([deployed_policy(float(value)) for value in high], dtype=float)
    return (
        float(np.mean(raw_high > raw_low + tolerance)),
        float(np.mean(deployed_high > deployed_low + tolerance)),
    )


def monotonicity_summary_from_predictions(
    predictions: pd.DataFrame,
    policy_names: Sequence[str],
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Summarize adjacent-state raw and deployed monotonicity violations."""
    rows: list[dict[str, float | int | str]] = []
    for policy_name in policy_names:
        policy_rows = predictions.loc[predictions["policy_name"] == policy_name]
        raw_differences: list[float] = []
        deployed_differences: list[float] = []
        for _, product_rows in policy_rows.groupby("product_id", sort=False):
            ordered = product_rows.sort_values("inventory")
            raw_differences.extend(np.diff(ordered["raw_order"].to_numpy(dtype=float)).tolist())
            deployed_differences.extend(np.diff(ordered["deployed_order"].to_numpy(dtype=float)).tolist())
        raw = np.asarray(raw_differences, dtype=float)
        deployed = np.asarray(deployed_differences, dtype=float)
        rows.append(
            {
                "policy_name": policy_name,
                "paired_state_count": int(raw.size),
                "raw_violation_rate": float(np.mean(raw > tolerance)) if raw.size else np.nan,
                "deployed_violation_rate": float(np.mean(deployed > tolerance)) if deployed.size else np.nan,
                "raw_max_positive_difference": float(np.maximum(raw, 0.0).max(initial=0.0)),
                "deployed_max_positive_difference": float(np.maximum(deployed, 0.0).max(initial=0.0)),
            }
        )
    return pd.DataFrame(rows)


def order_up_to_diagnostic_rows(
    predictions: pd.DataFrame,
    policy_names: Sequence[str],
) -> pd.DataFrame:
    """Return per-product active/inactive slope and q(0)-oracle-S diagnostics."""
    rows: list[dict[str, float | int | str]] = []
    selected = predictions.loc[predictions["policy_name"].isin(policy_names)]
    for (policy_name, product_id), product_rows in selected.groupby(["policy_name", "product_id"], sort=False):
        ordered = product_rows.sort_values("inventory")
        inventory = ordered["inventory"].to_numpy(dtype=float)
        order = ordered["deployed_order"].to_numpy(dtype=float)
        oracle_order = ordered["oracle_order"].to_numpy(dtype=float)
        slopes = np.diff(order) / np.maximum(np.diff(inventory), 1e-12)
        active = oracle_order[:-1] > 1e-6
        inactive = ~active
        q_zero = float(order[0])
        oracle_s = float(ordered["omniscient_order_up_to_level"].iloc[0])
        rows.append(
            {
                "policy_name": policy_name,
                "product_id": product_id,
                "active_interval_count": int(active.sum()),
                "inactive_interval_count": int(inactive.sum()),
                "active_slope_deviation_from_minus_one": (
                    float(np.mean(np.abs(slopes[active] + 1.0))) if np.any(active) else np.nan
                ),
                "inactive_slope_deviation_from_zero": (
                    float(np.mean(np.abs(slopes[inactive]))) if np.any(inactive) else np.nan
                ),
                "q_at_zero": q_zero,
                "oracle_S": oracle_s,
                "q_at_zero_minus_oracle_S": q_zero - oracle_s,
            }
        )
    return pd.DataFrame(rows)


def monokan_spline_diagnostics(
    spline_values: pd.DataFrame,
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Check each sampled first-layer inventory spline for positive increments."""
    rows: list[dict[str, float | int | str]] = []
    for hidden_unit, unit_rows in spline_values.groupby("hidden_unit", sort=True):
        ordered = unit_rows.sort_values("inventory_feature")
        differences = np.diff(ordered["edge_value"].to_numpy(dtype=float))
        positive = differences > tolerance
        rows.append(
            {
                "hidden_unit": int(hidden_unit),
                "segment_count": int(differences.size),
                "violating_segment_count": int(positive.sum()),
                "fraction_violating_segments": float(positive.mean()) if differences.size else 0.0,
                "max_positive_finite_difference": float(np.maximum(differences, 0.0).max(initial=0.0)),
                "status": "PASS" if not np.any(positive) else "FAIL",
            }
        )
    return pd.DataFrame(rows)
