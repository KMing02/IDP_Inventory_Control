"""Validation-only policy behavior and inventory-state extrapolation analysis.

No optimization is performed. Existing imitation-assisted checkpoints are
evaluated on frozen product contexts while inventory is swept from 0 to 8
times historical mean demand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from inventory_control.idp_benchmark import (  # noqa: E402
    _model_config,
    _model_orders,
    _profile_settings,
)
from inventory_control.paper_configs import load_paper_lost_sales_config  # noqa: E402
from inventory_control.paper_lost_sales import (  # noqa: E402
    non_omniscient_gamma_base_stock_policy,
    omniscient_gamma_base_stock_policy,
)
from inventory_control.policies.neural.fcnn import (  # noqa: E402
    compute_paper_fcnn_feature_stats,
)
from inventory_control.products import (  # noqa: E402
    PaperLostSalesProduct,
    PaperLostSalesProductGenerator,
)
from inventory_control.training.direct_backprop import (  # noqa: E402
    make_paper_lost_sales_training_batch,
    paper_lost_sales_fcnn_order,
)
from scripts.run_residual_order_up_to_validation import (  # noqa: E402
    _make_model as _make_residual_model,
    _resolved_config as _resolved_residual_config,
)
from scripts.tune_primary_neural_policies import (  # noqa: E402
    PRIMARY_POLICIES,
    _atomic_write_csv,
    _atomic_write_json,
    _environment_record,
    _fingerprint,
    _make_model,
    _repository_relative_path,
    _safe_generate_products,
    _write_or_validate_json,
    data_fingerprints,
    validate_split_and_output,
)


PRIMARY_SEEDS = tuple(range(5))
RESIDUAL_SEEDS = (0, 1, 2)
GRID_POINTS = 1025
SWEEP_MAX_FACTOR = 8.0
OBSERVED_COVERAGE_QUANTILE = 0.995
VIOLATION_TOLERANCE = 1e-6
DEFAULT_OUTPUT_DIR = Path("outputs/idp_followup/policy_behavior_extrapolation_validation")
MATCHED_ROOT = Path("outputs/idp_followup/equal_capacity_validation")
MATCHED_10SEED_ROOT = Path("outputs/idp_followup/matched_width_depth_10seed_validation")
NO_IMITATION_ROOT = Path("outputs/idp_followup/no_imitation_matched_width_validation")
RESIDUAL_ROOT = Path("outputs/idp_followup/residual_order_up_to_validation")

COLORS = {
    "PaperFCNN": "#2563EB",
    "PenaltyMLP": "#D97706",
    "PositiveWeightMLP": "#708238",
    "ConstrainedMonoMLP": "#C24174",
    "ConstrainedMLPResidualOrderUpTo": "#B7791F",
    "HistoricalGammaBaseStock": "#20242B",
    "OmniscientGammaBaseStock": "#6B7280",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstructed_observed_upper_factor(normalized_states: np.ndarray) -> float:
    """Return the preregistered central 99.5% reconstructed-state boundary."""
    values = np.asarray(normalized_states, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("At least one finite reconstructed inventory state is required")
    return float(np.quantile(finite, OBSERVED_COVERAGE_QUANTILE))


def _source_table(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "seed_results.csv")
    required = {"policy_name", "training_seed", "source_run_dir"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Source table is missing columns: {required - set(frame.columns)}")
    return frame


def _load_primary_models(
    base_config: dict[str, Any],
    settings: dict[str, Any],
    source_table: pd.DataFrame,
) -> tuple[dict[tuple[str, int], torch.nn.Module], pd.DataFrame]:
    models: dict[tuple[str, int], torch.nn.Module] = {}
    manifest_rows = []
    for policy in PRIMARY_POLICIES:
        for seed in PRIMARY_SEEDS:
            row = source_table.loc[
                (source_table["policy_name"] == policy)
                & (source_table["training_seed"] == seed)
            ]
            if len(row) != 1:
                raise RuntimeError(f"Expected one matched checkpoint for {policy}, seed {seed}")
            source = REPOSITORY_ROOT / str(row.iloc[0]["source_run_dir"])
            checkpoint = source / "model.pt"
            model = _make_model(policy, _model_config(base_config, settings), seed)
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            model.eval()
            models[(policy, seed)] = model
            manifest_rows.append(
                {
                    "policy_name": policy,
                    "training_seed": seed,
                    "source_run_dir": source.relative_to(REPOSITORY_ROOT).as_posix(),
                    "checkpoint_sha256": _sha256(checkpoint),
                    "imitation_epochs_run": int(
                        (pd.read_csv(source / "training_log.csv")["phase"] == 0).sum()
                    ),
                }
            )
    return models, pd.DataFrame(manifest_rows)


def _load_residual_cmnn_models(
    base_config: dict[str, Any], profile: str, root: Path
) -> dict[int, torch.nn.Module]:
    models = {}
    for seed in RESIDUAL_SEEDS:
        config, _ = _resolved_residual_config(base_config, profile, seed)
        settings = _profile_settings(config, profile)
        model = _make_residual_model(
            "ConstrainedMLPResidualOrderUpTo", _model_config(config, settings), seed
        )
        checkpoint = root / "runs" / "ConstrainedMLPResidualOrderUpTo" / f"seed_{seed}" / "model.pt"
        if not checkpoint.exists():
            raise RuntimeError(f"Missing residual CMNN checkpoint: {checkpoint}")
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        models[seed] = model
    return models


def _load_extended_paper_cmnn_models(
    base_config: dict[str, Any],
    settings: dict[str, Any],
    source_table: pd.DataFrame,
) -> tuple[dict[tuple[str, int], torch.nn.Module], pd.DataFrame]:
    models = {}
    rows = []
    for policy in ("PaperFCNN", "ConstrainedMonoMLP"):
        for seed in range(10):
            source_row = source_table.loc[
                (source_table["policy_name"] == policy)
                & (source_table["training_seed"] == seed)
            ]
            if len(source_row) != 1:
                raise RuntimeError(f"Expected one ten-seed checkpoint for {policy}, seed {seed}")
            source = REPOSITORY_ROOT / str(source_row.iloc[0]["source_run_dir"])
            checkpoint = source / "model.pt"
            model = _make_model(policy, _model_config(base_config, settings), seed)
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            model.eval()
            models[(policy, seed)] = model
            rows.append(
                {
                    "policy_name": policy,
                    "training_seed": seed,
                    "source_run_dir": source.relative_to(REPOSITORY_ROOT).as_posix(),
                    "checkpoint_sha256": _sha256(checkpoint),
                    "analysis_scope": "paper_cmnn_matched10",
                }
            )
    return models, pd.DataFrame(rows)


@torch.no_grad()
def reconstruct_frozen_training_states(
    model: torch.nn.Module,
    policy: str,
    seed: int,
    imitation_epochs: int,
    train_products: list[PaperLostSalesProduct],
    feature_stats: Any,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Replay training RNG/scenarios with a frozen final checkpoint.

    Main-generator draws consumed by imitation batches and their four sampled
    inventory grids are replayed before the 15 reward batches. States after
    the randomized initial condition (t >= 1) define the endogenous pool.
    """
    del policy
    generator = torch.Generator().manual_seed(int(seed))
    batch_size = min(int(settings["batch_size"]), len(train_products))
    horizon = int(settings["training_horizon"])
    for _ in range(int(imitation_epochs)):
        indices = torch.randint(
            0, len(train_products), (batch_size,), generator=generator
        ).tolist()
        batch_products = [train_products[index] for index in indices]
        make_paper_lost_sales_training_batch(
            batch_products, horizon, generator, feature_stats=feature_stats
        )
        for _ in range(4):
            torch.rand(batch_size, generator=generator)

    endogenous: list[np.ndarray] = []
    initial: list[np.ndarray] = []
    for _ in range(15):
        indices = torch.randint(
            0, len(train_products), (batch_size,), generator=generator
        ).tolist()
        batch_products = [train_products[index] for index in indices]
        batch = make_paper_lost_sales_training_batch(
            batch_products, horizon, generator, feature_stats=feature_stats
        )
        inventory = batch["initial_inventory"]
        means = torch.clamp(batch["historical_mean_demand"], min=1.0)
        initial.append((inventory / means).cpu().numpy())
        for period in range(horizon):
            order = paper_lost_sales_fcnn_order(
                model,
                batch,
                inventory,
                max_order_factor=float(settings["max_order_factor"]),
                scale_action_by_history_mean=bool(settings["scale_action_by_history_mean"]),
            )
            inventory = torch.relu(inventory + order - batch["demand_paths"][:, period])
            if period + 1 < horizon:
                endogenous.append((inventory / means).cpu().numpy())
    return np.concatenate(endogenous), np.concatenate(initial)


def _state_distribution_rows(
    models: dict[tuple[str, int], torch.nn.Module],
    manifest: pd.DataFrame,
    train_products: list[PaperLostSalesProduct],
    feature_stats: Any,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    pooled = []
    quantiles = (0.5, 0.9, 0.95, 0.99, 0.995, 0.999)
    for record in manifest.itertuples(index=False):
        endogenous, initial = reconstruct_frozen_training_states(
            models[(record.policy_name, int(record.training_seed))],
            record.policy_name,
            int(record.training_seed),
            int(record.imitation_epochs_run),
            train_products,
            feature_stats,
            settings,
        )
        pooled.append(endogenous)
        row: dict[str, Any] = {
            "policy_name": record.policy_name,
            "training_seed": int(record.training_seed),
            "endogenous_state_count": len(endogenous),
            "initial_state_count": len(initial),
            "endogenous_minimum": float(endogenous.min()),
            "endogenous_maximum": float(endogenous.max()),
            "initial_maximum": float(initial.max()),
        }
        for quantile in quantiles:
            label = str(quantile).replace(".", "p")
            row[f"endogenous_q{label}"] = float(np.quantile(endogenous, quantile))
            row[f"initial_q{label}"] = float(np.quantile(initial, quantile))
        rows.append(row)
    return pd.DataFrame(rows), np.concatenate(pooled)


def region_diagnostics(
    inventory: np.ndarray,
    raw_order: np.ndarray,
    deployed_order: np.ndarray,
    reference_level: float,
) -> dict[str, float | int]:
    inventory = np.asarray(inventory, dtype=float)
    raw = np.asarray(raw_order, dtype=float)
    deployed = np.asarray(deployed_order, dtype=float)
    if len(inventory) < 2:
        raise ValueError("Region diagnostics require at least two inventory points")
    dx = np.diff(inventory)
    raw_delta = np.diff(raw)
    deployed_delta = np.diff(deployed)
    finite_raw = np.isfinite(raw_delta)
    finite_deployed = np.isfinite(deployed_delta)
    raw_positive = np.maximum(raw_delta[finite_raw], 0.0)
    deployed_positive = np.maximum(deployed_delta[finite_deployed], 0.0)
    raw_strict = raw_positive[raw_positive > VIOLATION_TOLERANCE]
    deployed_strict = deployed_positive[deployed_positive > VIOLATION_TOLERANCE]
    raw_slopes = raw_delta / np.maximum(dx, 1e-12)
    deployed_slopes = deployed_delta / np.maximum(dx, 1e-12)
    reference_active = np.maximum(reference_level - inventory[:-1], 0.0) > VIOLATION_TOLERANCE
    active_raw = reference_active & np.isfinite(raw_slopes)
    active_deployed = reference_active & np.isfinite(deployed_slopes)
    raw_active_deviation = np.abs(raw_slopes[active_raw] + 1.0)
    deployed_active_deviation = np.abs(deployed_slopes[active_deployed] + 1.0)
    implied_level_error = inventory + raw - reference_level
    meaningful = np.isfinite(implied_level_error) & (raw > VIOLATION_TOLERANCE)
    return {
        "grid_point_count": len(inventory),
        "raw_interval_count": int(finite_raw.sum()),
        "raw_violating_interval_count": len(raw_strict),
        "raw_violation_rate": len(raw_strict) / max(int(finite_raw.sum()), 1),
        "raw_maximum_positive_violation": float(raw_strict.max()) if len(raw_strict) else 0.0,
        "deployed_interval_count": int(finite_deployed.sum()),
        "deployed_violating_interval_count": len(deployed_strict),
        "deployed_violation_rate": len(deployed_strict) / max(int(finite_deployed.sum()), 1),
        "deployed_maximum_positive_violation": (
            float(deployed_strict.max()) if len(deployed_strict) else 0.0
        ),
        "raw_active_slope_deviation_sum": float(raw_active_deviation.sum()),
        "raw_active_slope_count": len(raw_active_deviation),
        "deployed_active_slope_deviation_sum": float(deployed_active_deviation.sum()),
        "deployed_active_slope_count": len(deployed_active_deviation),
        "raw_zero_order_count": int(np.sum(np.abs(raw) <= VIOLATION_TOLERANCE)),
        "deployed_zero_order_count": int(np.sum(np.abs(deployed) <= VIOLATION_TOLERANCE)),
        "finite_raw_point_count": int(np.isfinite(raw).sum()),
        "finite_deployed_point_count": int(np.isfinite(deployed).sum()),
        "order_up_to_absolute_error_sum": float(np.abs(implied_level_error[meaningful]).sum()),
        "order_up_to_error_count": int(meaningful.sum()),
        "non_finite_action_count": int(np.sum(~(np.isfinite(raw) & np.isfinite(deployed)))),
    }


def _region_slices(inventory_factor: np.ndarray, observed_upper: float) -> dict[str, np.ndarray]:
    return {
        "observed_training_state": inventory_factor <= observed_upper + 1e-12,
        "inventory_state_extrapolation": inventory_factor > observed_upper + 1e-12,
    }


def _diagnostic_row(
    policy: str,
    seed: int,
    product: PaperLostSalesProduct,
    region: str,
    inventory_factor: np.ndarray,
    inventory: np.ndarray,
    raw: np.ndarray,
    deployed: np.ndarray,
    mask: np.ndarray,
    reference_level: float,
) -> dict[str, Any]:
    metrics = region_diagnostics(inventory[mask], raw[mask], deployed[mask], reference_level)
    mean = max(float(np.mean(product.historical_demand)), 1.0)
    zero_raw = float(raw[0])
    return {
        "policy_name": policy,
        "training_seed": seed,
        "product_id": product.product_id,
        "region": region,
        "historical_mean_demand": mean,
        "reference_base_stock_level": reference_level,
        "zero_inventory_raw_order": zero_raw,
        "zero_inventory_target_stock_absolute_error": abs(zero_raw - reference_level),
        "raw_order_at_8d": float(raw[-1]),
        "deployed_order_at_8d": float(deployed[-1]),
        "raw_order_at_8d_divided_by_mean": float(raw[-1] / mean),
        "raw_zero_at_8d": int(abs(float(raw[-1])) <= VIOLATION_TOLERANCE),
        "region_lower_factor": float(inventory_factor[mask][0]),
        "region_upper_factor": float(inventory_factor[mask][-1]),
        **metrics,
    }


def aggregate_diagnostics(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for (policy, region), group in rows.groupby(["policy_name", "region"], sort=False):
        raw_intervals = int(group["raw_interval_count"].sum())
        dep_intervals = int(group["deployed_interval_count"].sum())
        raw_slope_count = int(group["raw_active_slope_count"].sum())
        dep_slope_count = int(group["deployed_active_slope_count"].sum())
        raw_points = int(group["finite_raw_point_count"].sum())
        dep_points = int(group["finite_deployed_point_count"].sum())
        out_count = int(group["order_up_to_error_count"].sum())
        output.append(
            {
                "policy_name": policy,
                "region": region,
                "product_seed_count": len(group),
                "raw_violation_rate": float(group["raw_violating_interval_count"].sum())
                / max(raw_intervals, 1),
                "deployed_violation_rate": float(
                    group["deployed_violating_interval_count"].sum()
                )
                / max(dep_intervals, 1),
                "raw_maximum_positive_violation": float(
                    group["raw_maximum_positive_violation"].max()
                ),
                "deployed_maximum_positive_violation": float(
                    group["deployed_maximum_positive_violation"].max()
                ),
                "raw_active_slope_deviation_from_minus_one": float(
                    group["raw_active_slope_deviation_sum"].sum()
                )
                / max(raw_slope_count, 1),
                "raw_active_slope_interval_count": raw_slope_count,
                "deployed_active_slope_deviation_from_minus_one": float(
                    group["deployed_active_slope_deviation_sum"].sum()
                )
                / max(dep_slope_count, 1),
                "deployed_active_slope_interval_count": dep_slope_count,
                "raw_zero_order_rate": float(group["raw_zero_order_count"].sum())
                / max(raw_points, 1),
                "deployed_zero_order_rate": float(group["deployed_zero_order_count"].sum())
                / max(dep_points, 1),
                "raw_order_at_8d_mean": float(group["raw_order_at_8d"].mean()),
                "raw_order_at_8d_maximum": float(group["raw_order_at_8d"].max()),
                "raw_order_at_8d_divided_by_mean_mean": float(
                    group["raw_order_at_8d_divided_by_mean"].mean()
                ),
                "raw_zero_at_8d_rate": float(group["raw_zero_at_8d"].mean()),
                "zero_inventory_target_stock_mae": float(
                    group["zero_inventory_target_stock_absolute_error"].mean()
                ),
                "active_order_up_to_mae": float(
                    group["order_up_to_absolute_error_sum"].sum()
                )
                / max(out_count, 1),
                "non_finite_action_count": int(group["non_finite_action_count"].sum()),
            }
        )
    return pd.DataFrame(output)


def _robust_typical_score(products: Sequence[PaperLostSalesProduct]) -> pd.DataFrame:
    records = []
    for product in products:
        history = np.asarray(product.historical_demand, dtype=float)
        mean = max(float(history.mean()), 1.0)
        records.append(
            {
                "product_id": product.product_id,
                "mean_demand": mean,
                "demand_cv": float(history.std(ddof=0) / mean),
                "price": product.price,
                "purchase_cost_ratio": product.purchase_cost / max(product.price, 1.0),
                "holding_cost_ratio": product.holding_cost / max(product.price, 1.0),
                "lost_sale_penalty_ratio": product.lost_sale_penalty / max(product.price, 1.0),
            }
        )
    frame = pd.DataFrame(records)
    features = frame.drop(columns="product_id")
    median = features.median()
    mad = (features - median).abs().median().replace(0.0, 1.0)
    frame["typical_score"] = np.sqrt((((features - median) / mad) ** 2).sum(axis=1))
    return frame


def choose_representative_products(
    products: Sequence[PaperLostSalesProduct],
    diagnostic_rows: pd.DataFrame,
    similarity_scores: dict[str, float],
) -> pd.DataFrame:
    """Select products by deterministic, mutually exclusive objective rules."""
    typical_scores = _robust_typical_score(products)
    chosen: set[str] = set()
    rows = []

    typical = typical_scores.sort_values(["typical_score", "product_id"]).iloc[0]
    chosen.add(str(typical.product_id))
    rows.append(
        {
            "role": "typical",
            "product_id": typical.product_id,
            "selection_score": typical.typical_score,
            "selection_rule": "minimum robust multivariate distance to validation-product medians",
        }
    )

    outside = diagnostic_rows.loc[diagnostic_rows["policy_name"] == "PaperFCNN"]
    violation = (
        outside.groupby("product_id", as_index=False)["raw_maximum_positive_violation"]
        .mean()
        .sort_values(["raw_maximum_positive_violation", "product_id"], ascending=[False, True])
    )
    violation = violation.loc[~violation["product_id"].isin(chosen)].iloc[0]
    chosen.add(str(violation.product_id))
    rows.append(
        {
            "role": "fcnn_violation",
            "product_id": violation.product_id,
            "selection_score": violation.raw_maximum_positive_violation,
            "selection_rule": "largest mean FCNN positive raw-order increase over the full 0-8d sweep",
        }
    )

    cmnn = diagnostic_rows.loc[
        (diagnostic_rows["policy_name"] == "ConstrainedMonoMLP")
        & (diagnostic_rows["region"] == "observed_training_state")
    ]
    calibration = (
        cmnn.groupby("product_id", as_index=False)[
            "zero_inventory_target_stock_absolute_error"
        ]
        .mean()
        .sort_values(
            ["zero_inventory_target_stock_absolute_error", "product_id"],
            ascending=[False, True],
        )
    )
    calibration = calibration.loc[~calibration["product_id"].isin(chosen)].iloc[0]
    chosen.add(str(calibration.product_id))
    rows.append(
        {
            "role": "cmnn_calibration_error",
            "product_id": calibration.product_id,
            "selection_score": calibration.zero_inventory_target_stock_absolute_error,
            "selection_rule": "largest mean CMNN zero-inventory target-stock absolute error",
        }
    )

    similarity = pd.DataFrame(
        [{"product_id": key, "selection_score": value} for key, value in similarity_scores.items()]
    ).sort_values(["selection_score", "product_id"])
    similarity = similarity.loc[~similarity["product_id"].isin(chosen)].iloc[0]
    rows.append(
        {
            "role": "similar_behavior",
            "product_id": similarity.product_id,
            "selection_score": similarity.selection_score,
            "selection_rule": "smallest mean cross-policy dispersion of normalized raw actions",
        }
    )
    return pd.DataFrame(rows)


def _curve_quantiles(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.stack(curves)
    return (
        np.quantile(values, 0.2, axis=0),
        np.median(values, axis=0),
        np.quantile(values, 0.8, axis=0),
    )


def _style_axes(ax: plt.Axes, observed_upper: float) -> None:
    ax.axvspan(0.0, observed_upper, color="#DCEAF7", alpha=0.55, zorder=0)
    ax.axvline(observed_upper, color="#64748B", linewidth=1.0, linestyle="--")
    ax.grid(axis="both", color="#E5E7EB", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0.0, SWEEP_MAX_FACTOR)
    ax.set_xlabel("Inventory / historical mean demand")
    ax.set_ylabel("Order quantity")


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_representative_product(
    product: PaperLostSalesProduct,
    role: str,
    models: dict[tuple[str, int], torch.nn.Module],
    feature_stats: Any,
    settings: dict[str, Any],
    observed_upper: float,
    output_dir: Path,
    seeds_by_policy: dict[str, Sequence[int]],
    highlight_fcnn_seed: int | None = None,
    reconstructed_state_percentiles: dict[str, float] | None = None,
) -> pd.DataFrame:
    mean = max(float(np.mean(product.historical_demand)), 1.0)
    factors = np.linspace(0.0, SWEEP_MAX_FACTOR, GRID_POINTS)
    inventory = factors * mean
    historical_level = non_omniscient_gamma_base_stock_policy(product).S
    historical_raw = np.maximum(historical_level - inventory, 0.0)
    cap = np.maximum(4.0 * mean - inventory, 0.0)
    historical_deployed = np.minimum(historical_raw, cap)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=True, sharey=False)
    for action_kind, ax in zip(("raw", "deployed"), axes):
        for policy in PRIMARY_POLICIES:
            curves = []
            seed_curves: dict[int, np.ndarray] = {}
            for seed in seeds_by_policy[policy]:
                raw, deployed = _model_orders(
                    models[(policy, seed)],
                    product,
                    feature_stats,
                    inventory,
                    float(settings["max_order_factor"]),
                    bool(settings["scale_action_by_history_mean"]),
                )
                values = raw if action_kind == "raw" else deployed
                curves.append(values)
                seed_curves[int(seed)] = values
                if action_kind == "raw":
                    rows.extend(
                        {
                            "role": role,
                            "product_id": product.product_id,
                            "policy_name": policy,
                            "training_seed": seed,
                            "inventory_factor": factor,
                            "inventory": inv,
                            "raw_order": raw_value,
                            "deployed_order": deployed_value,
                        }
                        for factor, inv, raw_value, deployed_value in zip(
                            factors, inventory, raw, deployed
                        )
                    )
            lower, median, upper = _curve_quantiles(curves)
            color = COLORS[policy]
            ax.fill_between(factors, lower, upper, color=color, alpha=0.10, linewidth=0)
            ax.plot(factors, median, color=color, linewidth=1.8, label=policy)
            if policy == "PaperFCNN" and highlight_fcnn_seed is not None:
                ax.plot(
                    factors,
                    seed_curves[highlight_fcnn_seed],
                    color="#1E3A8A",
                    linewidth=1.3,
                    linestyle=":",
                    label=f"PaperFCNN seed {highlight_fcnn_seed} (largest violation)",
                )
        reference = historical_raw if action_kind == "raw" else historical_deployed
        ax.plot(
            factors,
            reference,
            color=COLORS["HistoricalGammaBaseStock"],
            linewidth=2.0,
            linestyle="--",
            label="Historical Gamma base-stock",
        )
        _style_axes(ax, observed_upper)
        if reconstructed_state_percentiles:
            percentile_positions = list(reconstructed_state_percentiles.values())
            ax.vlines(
                percentile_positions,
                0.01,
                0.055,
                transform=ax.get_xaxis_transform(),
                color="#6B7280",
                linewidth=1.2,
                alpha=0.8,
                zorder=4,
            )
        ax.set_title(f"{action_kind.capitalize()} action")
    fig.suptitle(f"Policy behavior: {role.replace('_', ' ')} product {product.product_id}")
    shared_seed_set = len({tuple(seeds_by_policy[policy]) for policy in PRIMARY_POLICIES}) == 1
    if shared_seed_set:
        shared_seeds = tuple(seeds_by_policy[PRIMARY_POLICIES[0]])
        seed_note = f"All neural policies use common seeds {shared_seeds[0]}–{shared_seeds[-1]}."
    else:
        seed_note = "PaperFCNN/CMNN use seeds 0–9; Penalty/Positive use seeds 0–4."
    state_note = ""
    if reconstructed_state_percentiles:
        formatted = ", ".join(
            f"{label}={value:.2f}d"
            for label, value in reconstructed_state_percentiles.items()
        )
        state_note = f"\nGray baseline ticks show reconstructed endogenous-state percentiles: {formatted}."
    fig.text(
        0.5,
        0.01,
        f"{seed_note} Bands show P20–P80. "
        f"Blue shading is the reconstructed 99.5% training-state range (0–{observed_upper:.2f}d)."
        f"{state_note}",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.84))
    _save_figure(fig, output_dir / "figures" / f"representative_{role}_{product.product_id}")
    return pd.DataFrame(rows)


def plot_fcnn_violation_zoom(
    product: PaperLostSalesProduct,
    model: torch.nn.Module,
    feature_stats: Any,
    settings: dict[str, Any],
    observed_upper: float,
    output_dir: Path,
    seed: int,
) -> None:
    """Show the objectively largest adjacent-grid FCNN increase at readable scale."""
    mean = max(float(np.mean(product.historical_demand)), 1.0)
    factors = np.linspace(0.0, SWEEP_MAX_FACTOR, GRID_POINTS)
    inventory = factors * mean
    raw, deployed = _model_orders(
        model,
        product,
        feature_stats,
        inventory,
        float(settings["max_order_factor"]),
        bool(settings["scale_action_by_history_mean"]),
    )
    differences = np.diff(raw)
    violation_index = int(np.argmax(differences))
    maximum_increase = float(max(differences[violation_index], 0.0))
    window_radius = 32
    left = max(0, violation_index - window_radius)
    right = min(len(factors) - 1, violation_index + window_radius + 1)
    historical_level = non_omniscient_gamma_base_stock_policy(product).S
    historical_raw = np.maximum(historical_level - inventory, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
    axes[0].plot(
        factors,
        raw,
        color=COLORS["PaperFCNN"],
        linewidth=2.0,
        label="PaperFCNN raw action",
    )
    axes[0].plot(
        factors,
        deployed,
        color="#60A5FA",
        linewidth=1.5,
        linestyle=":",
        label="PaperFCNN deployed action",
    )
    axes[0].plot(
        factors,
        historical_raw,
        color=COLORS["HistoricalGammaBaseStock"],
        linewidth=1.8,
        linestyle="--",
        label="Historical Gamma base-stock",
    )
    axes[0].axvspan(
        factors[left],
        factors[right],
        color="#FDE68A",
        alpha=0.45,
        label="Zoomed interval",
    )
    _style_axes(axes[0], observed_upper)
    axes[0].set_title("Full inventory sweep")

    zoom_slice = slice(left, right + 1)
    axes[1].plot(
        factors[zoom_slice],
        raw[zoom_slice],
        color=COLORS["PaperFCNN"],
        linewidth=2.2,
    )
    axes[1].scatter(
        factors[violation_index : violation_index + 2],
        raw[violation_index : violation_index + 2],
        color="#991B1B",
        s=32,
        zorder=3,
    )
    axes[1].plot(
        factors[violation_index : violation_index + 2],
        raw[violation_index : violation_index + 2],
        color="#991B1B",
        linewidth=2.5,
    )
    axes[1].annotate(
        f"largest adjacent increase = {maximum_increase:.3f}",
        xy=(factors[violation_index + 1], raw[violation_index + 1]),
        xytext=(0.04, 0.92),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#991B1B"},
        color="#7F1D1D",
        fontsize=9,
    )
    axes[1].axvspan(0.0, observed_upper, color="#DCEAF7", alpha=0.55, zorder=0)
    axes[1].grid(axis="both", color="#E5E7EB", linewidth=0.7)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].set_xlim(factors[left], factors[right])
    axes[1].set_xlabel("Inventory / historical mean demand")
    axes[1].set_ylabel("Raw order quantity")
    axes[1].set_title("Objective maximum-increase interval")
    fig.suptitle(
        f"PaperFCNN monotonicity violation: {product.product_id}, seed {seed}"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
        frameon=False,
    )
    fig.text(
        0.5,
        0.01,
        "Product and seed were selected by the largest measured positive adjacent-grid action increase; no manual selection was used.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.84))
    _save_figure(fig, output_dir / "figures" / "fcnn_violation_zoom")


def policy_active_slope_error(inventory: np.ndarray, raw_order: np.ndarray) -> float:
    """Mean |dq/dI + 1| on intervals where the evaluated policy is active."""
    inventory_values = np.asarray(inventory, dtype=float)
    order_values = np.asarray(raw_order, dtype=float)
    slopes = np.diff(order_values) / np.diff(inventory_values)
    active = (
        (order_values[:-1] > VIOLATION_TOLERANCE)
        & (order_values[1:] > VIOLATION_TOLERANCE)
        & np.isfinite(slopes)
    )
    if not np.any(active):
        return float("nan")
    return float(np.mean(np.abs(slopes[active] + 1.0)))


def plot_direct_vs_residual_cmnn(
    product: PaperLostSalesProduct,
    direct_models: dict[tuple[str, int], torch.nn.Module],
    residual_models: dict[int, torch.nn.Module],
    feature_stats: Any,
    settings: dict[str, Any],
    observed_upper: float,
    output_dir: Path,
    figure_stem: str,
) -> dict[str, float | str]:
    mean = max(float(np.mean(product.historical_demand)), 1.0)
    factors = np.linspace(0.0, SWEEP_MAX_FACTOR, GRID_POINTS)
    inventory = factors * mean
    oracle_level = omniscient_gamma_base_stock_policy(product).S
    historical_level = non_omniscient_gamma_base_stock_policy(product).S
    ideal = np.maximum(oracle_level - inventory, 0.0)
    anchor = np.maximum(historical_level - inventory, 0.0)
    direct = []
    residual = []
    for seed in RESIDUAL_SEEDS:
        direct.append(
            _model_orders(
                direct_models[("ConstrainedMonoMLP", seed)],
                product,
                feature_stats,
                inventory,
                4.0,
                True,
            )[0]
        )
        residual.append(
            _model_orders(
                residual_models[seed], product, feature_stats, inventory, 4.0, True
            )[0]
        )
    direct_zero_inventory_errors = [
        abs(float(curve[0]) - historical_level) for curve in direct
    ]
    direct_slope_errors = [
        policy_active_slope_error(inventory, curve) for curve in direct
    ]
    residual_slope_errors = [
        policy_active_slope_error(inventory, curve) for curve in residual
    ]
    direct_median = np.median(np.stack(direct), axis=0)
    residual_median = np.median(np.stack(residual), axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    axes[0].plot(factors, ideal, color="#20242B", linestyle="--", linewidth=2.2, label="Ideal omniscient order-up-to")
    axes[0].plot(factors, anchor, color="#6B7280", linestyle=":", linewidth=1.8, label="Historical Gamma anchor")
    axes[0].plot(factors, direct_median, color=COLORS["ConstrainedMonoMLP"], linewidth=2.0, label="Direct CMNN")
    axes[0].plot(factors, residual_median, color=COLORS["ConstrainedMLPResidualOrderUpTo"], linewidth=2.0, label="Residual order-up-to CMNN")
    _style_axes(axes[0], observed_upper)
    axes[0].set_title("Raw order quantity")

    active_direct = direct_median > VIOLATION_TOLERANCE
    active_residual = residual_median > VIOLATION_TOLERANCE
    axes[1].hlines(
        oracle_level,
        0.0,
        min(float(oracle_level / mean), SWEEP_MAX_FACTOR),
        color="#20242B",
        linestyle="--",
        linewidth=2.2,
        label="Ideal target stock",
    )
    axes[1].hlines(
        historical_level,
        0.0,
        min(float(historical_level / mean), SWEEP_MAX_FACTOR),
        color="#6B7280",
        linestyle=":",
        linewidth=1.8,
        label="Historical anchor",
    )
    axes[1].plot(factors[active_direct], (inventory + direct_median)[active_direct], color=COLORS["ConstrainedMonoMLP"], linewidth=2.0, label="Direct CMNN implied I+q")
    axes[1].plot(factors[active_residual], (inventory + residual_median)[active_residual], color=COLORS["ConstrainedMLPResidualOrderUpTo"], linewidth=2.0, label="Residual CMNN implied I+q")
    _style_axes(axes[1], observed_upper)
    axes[1].set_title("Active-region implied target-stock level")
    axes[1].set_ylabel("Inventory + raw order (active region only)")
    fig.suptitle(f"Monotonicity versus explicit order-up-to structure: {product.product_id}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)
    fig.text(
        0.5,
        0.01,
        "Curves are shown only where each policy orders (q(I) > 0). For q(I) = max(S - I, 0), I + q(I) = S only when I < S; once q(I) = 0, I + q(I) = I.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.84))
    _save_figure(fig, output_dir / "figures" / figure_stem)
    return {
        "product_id": product.product_id,
        "direct_cmnn_zero_inventory_target_stock_absolute_error": float(
            np.mean(direct_zero_inventory_errors)
        ),
        "direct_cmnn_active_region_slope_error": float(
            np.mean(direct_slope_errors)
        ),
        "residual_cmnn_active_region_slope_error": float(
            np.mean(residual_slope_errors)
        ),
        "figure_png": f"figures/{figure_stem}.png",
        "figure_pdf": f"figures/{figure_stem}.pdf",
    }


def plot_aggregate_diagnostics(
    aggregate: pd.DataFrame,
    output_dir: Path,
    figure_stem: str,
) -> None:
    neural = aggregate.loc[aggregate["policy_name"].isin(PRIMARY_POLICIES)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    policies = list(PRIMARY_POLICIES)
    x = np.arange(len(policies))
    width = 0.34
    for offset, region, label, hatch in (
        (-width / 2, "observed_training_state", "Observed-range", ""),
        (width / 2, "inventory_state_extrapolation", "Extrapolation", "//"),
    ):
        frame = neural.loc[neural["region"] == region].set_index("policy_name")
        values = [
            (
                0.0
                if frame.loc[policy, "raw_maximum_positive_violation"]
                <= VIOLATION_TOLERANCE
                else frame.loc[policy, "raw_violation_rate"]
            )
            for policy in policies
        ]
        axes[0].bar(
            x + offset,
            values,
            width,
            label=label,
            color=[COLORS[policy] for policy in policies],
            alpha=0.85 if not hatch else 0.45,
            hatch=hatch,
            edgecolor="#374151",
            linewidth=0.6,
        )
        slope = [frame.loc[policy, "raw_active_slope_deviation_from_minus_one"] for policy in policies]
        axes[1].bar(
            x + offset,
            slope,
            width,
            label=label,
            color=[COLORS[policy] for policy in policies],
            alpha=0.85 if not hatch else 0.45,
            hatch=hatch,
            edgecolor="#374151",
            linewidth=0.6,
        )
    for ax, title, ylabel in (
        (axes[0], "Raw monotonicity violations", "Violation rate"),
        (
            axes[1],
            "Active-region order-up-to slope error",
            "Mean |dq/dI + 1| (0 = ideal; lower is better)",
        ),
    ):
        ax.set_xticks(x, ["FCNN", "Penalty", "Positive", "CMNN"], rotation=15)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    positive_extrapolation = neural.loc[
        (neural["policy_name"] == "PositiveWeightMLP")
        & (neural["region"] == "inventory_state_extrapolation")
    ].iloc[0]
    positive_rate = float(positive_extrapolation["raw_violation_rate"])
    positive_maximum = float(
        positive_extrapolation["raw_maximum_positive_violation"]
    )
    if positive_maximum > VIOLATION_TOLERANCE:
        axes[0].annotate(
            "numerical-scale only",
            xy=(x[2] + width / 2, positive_rate),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#4B5563",
        )
        axes[0].set_ylim(0.0, max(positive_rate * 1.24, 1e-6))
    fig.suptitle("Aggregate structural diagnostics — equal seeds 0–4")
    fig.text(
        0.5,
        0.01,
        "All four direct policies use seeds 0–4. Observed range is 0–4d; extrapolation is 4–8d.\n"
        f"Positive extrapolation: max +Δq={positive_maximum:.2e} > tolerance={VIOLATION_TOLERANCE:.0e} (numerical-scale only). "
        "FCNN is violation-free for seeds 0–4; its genuine seed-9 violation is shown in the separate FCNN figure.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    _save_figure(fig, output_dir / "figures" / figure_stem)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", default="full")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--matched-root", type=Path, default=MATCHED_ROOT)
    parser.add_argument("--matched-10seed-root", type=Path, default=MATCHED_10SEED_ROOT)
    parser.add_argument("--no-imitation-root", type=Path, default=NO_IMITATION_ROOT)
    parser.add_argument("--residual-root", type=Path, default=RESIDUAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    validate_split_and_output(args.split, args.output_dir)
    config_path, config_relative = _repository_relative_path(args.config, "--config")
    output_dir, _ = _repository_relative_path(args.output_dir, "--output-dir")
    matched_root, _ = _repository_relative_path(args.matched_root, "--matched-root")
    matched_10seed_root, _ = _repository_relative_path(
        args.matched_10seed_root, "--matched-10seed-root"
    )
    no_imitation_root, _ = _repository_relative_path(args.no_imitation_root, "--no-imitation-root")
    residual_root, _ = _repository_relative_path(args.residual_root, "--residual-root")

    started = time.perf_counter()
    base_config = load_paper_lost_sales_config(config_path)
    settings = _profile_settings(base_config, args.profile)
    generator = PaperLostSalesProductGenerator(base_config)
    train_products = _safe_generate_products(generator, 128, "train")
    validation_products = _safe_generate_products(generator, 32, "validation")
    feature_stats = compute_paper_fcnn_feature_stats(train_products)
    fingerprints = data_fingerprints(
        base_config, settings, train_products, validation_products
    )
    shared = _fingerprint(fingerprints)
    fingerprints["shared_data_fingerprint"] = shared
    matched_fingerprints = json.loads(
        (matched_root / "data_fingerprints.json").read_text(encoding="utf-8")
    )["128"]
    if any(matched_fingerprints.get(key) != value for key, value in fingerprints.items()):
        raise RuntimeError("Matched checkpoint data fingerprints do not match analysis data")
    matched_10seed_fingerprints = json.loads(
        (matched_10seed_root / "data_fingerprints.json").read_text(encoding="utf-8")
    )["128"]
    if any(
        matched_10seed_fingerprints.get(key) != value
        for key, value in fingerprints.items()
    ):
        raise RuntimeError("Ten-seed checkpoint data fingerprints do not match analysis data")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    _write_or_validate_json(output_dir / "environment.json", _environment_record(config_relative, base_config))
    _write_or_validate_json(output_dir / "data_fingerprints.json", fingerprints)
    models, manifest = _load_primary_models(
        base_config, settings, _source_table(matched_root)
    )
    extended_models, extended_manifest = _load_extended_paper_cmnn_models(
        base_config, settings, _source_table(matched_10seed_root)
    )
    manifest["analysis_scope"] = "all_primary_matched5"
    _atomic_write_csv(
        output_dir / "checkpoint_manifest.csv",
        pd.concat([manifest, extended_manifest], ignore_index=True).drop_duplicates(
            ["policy_name", "training_seed", "checkpoint_sha256"]
        ),
    )

    state_distribution, pooled_states = _state_distribution_rows(
        models, manifest, train_products, feature_stats, settings
    )
    observed_upper = reconstructed_observed_upper_factor(pooled_states)
    reconstructed_state_percentiles = {
        "P50": float(np.quantile(pooled_states, 0.50)),
        "P90": float(np.quantile(pooled_states, 0.90)),
        "P95": float(np.quantile(pooled_states, 0.95)),
        "P99": float(np.quantile(pooled_states, 0.99)),
        "P99.5": float(np.quantile(pooled_states, 0.995)),
    }
    _atomic_write_csv(output_dir / "reconstructed_training_state_distribution.csv", state_distribution)
    _atomic_write_json(
        output_dir / "observed_training_state_rule.json",
        {
            "method": "frozen-final-checkpoint replay of original reward-training batches",
            "randomized_initial_state_included_in_boundary": False,
            "endogenous_state_definition": "pre-order inventories after at least one demand transition",
            "coverage_quantile": OBSERVED_COVERAGE_QUANTILE,
            "observed_upper_factor": observed_upper,
            "pooled_endogenous_state_count": len(pooled_states),
            "pooled_endogenous_maximum_factor": float(np.max(pooled_states)),
            "fraction_above_observed_upper": float(np.mean(pooled_states > observed_upper)),
            "exact_original_evolving_policy_states_available": False,
            "protected_test_accessed": False,
        },
    )

    factors = np.linspace(0.0, SWEEP_MAX_FACTOR, GRID_POINTS)
    diagnostic_rows = []
    similarity_scores: dict[str, float] = {}
    for product in validation_products:
        mean = max(float(np.mean(product.historical_demand)), 1.0)
        inventory = factors * mean
        cap = np.maximum(4.0 * mean - inventory, 0.0)
        reference_level = non_omniscient_gamma_base_stock_policy(product).S
        reference_raw = np.maximum(reference_level - inventory, 0.0)
        reference_deployed = np.minimum(reference_raw, cap)
        policy_means = []
        for policy in PRIMARY_POLICIES:
            policy_curves = []
            for seed in PRIMARY_SEEDS:
                raw, deployed = _model_orders(
                    models[(policy, seed)], product, feature_stats, inventory, 4.0, True
                )
                policy_curves.append(raw / mean)
                for region, mask in _region_slices(factors, observed_upper).items():
                    diagnostic_rows.append(
                        _diagnostic_row(
                            policy,
                            seed,
                            product,
                            region,
                            factors,
                            inventory,
                            raw,
                            deployed,
                            mask,
                            reference_level,
                        )
                    )
            policy_means.append(np.mean(np.stack(policy_curves), axis=0))
        inside = factors <= observed_upper + 1e-12
        similarity_scores[product.product_id] = float(
            np.mean(np.std(np.stack(policy_means)[:, inside], axis=0))
        )
        for region, mask in _region_slices(factors, observed_upper).items():
            diagnostic_rows.append(
                _diagnostic_row(
                    "HistoricalGammaBaseStock",
                    -1,
                    product,
                    region,
                    factors,
                    inventory,
                    reference_raw,
                    reference_deployed,
                    mask,
                    reference_level,
                )
            )

    diagnostics = pd.DataFrame(diagnostic_rows)
    aggregate = aggregate_diagnostics(diagnostics)

    extended_rows = []
    for product in validation_products:
        mean = max(float(np.mean(product.historical_demand)), 1.0)
        inventory = factors * mean
        reference_level = non_omniscient_gamma_base_stock_policy(product).S
        for policy in ("PaperFCNN", "ConstrainedMonoMLP"):
            for seed in range(10):
                raw, deployed = _model_orders(
                    extended_models[(policy, seed)],
                    product,
                    feature_stats,
                    inventory,
                    4.0,
                    True,
                )
                for region, mask in _region_slices(factors, observed_upper).items():
                    extended_rows.append(
                        _diagnostic_row(
                            policy,
                            seed,
                            product,
                            region,
                            factors,
                            inventory,
                            raw,
                            deployed,
                            mask,
                            reference_level,
                        )
                    )
    extended_diagnostics = pd.DataFrame(extended_rows)
    extended_aggregate = aggregate_diagnostics(extended_diagnostics)
    selection = choose_representative_products(
        validation_products, extended_diagnostics, similarity_scores
    )
    _atomic_write_csv(output_dir / "diagnostics_product_seed.csv", diagnostics)
    _atomic_write_csv(output_dir / "diagnostics_aggregate.csv", aggregate)
    _atomic_write_csv(
        output_dir / "diagnostics_paper_cmnn_10seed_product_seed.csv",
        extended_diagnostics,
    )
    _atomic_write_csv(
        output_dir / "diagnostics_paper_cmnn_10seed_aggregate.csv",
        extended_aggregate,
    )
    available_seed_aggregate = pd.concat(
        [
            extended_aggregate.loc[
                extended_aggregate["policy_name"].isin(
                    ["PaperFCNN", "ConstrainedMonoMLP"]
                )
            ].assign(seed_scope="seeds_0_to_9"),
            aggregate.loc[
                aggregate["policy_name"].isin(
                    ["PenaltyMLP", "PositiveWeightMLP"]
                )
            ].assign(seed_scope="seeds_0_to_4"),
        ],
        ignore_index=True,
    )
    _atomic_write_csv(
        output_dir / "diagnostics_available_seed_aggregate.csv",
        available_seed_aggregate,
    )
    _atomic_write_csv(output_dir / "representative_product_selection.csv", selection)

    product_map = {product.product_id: product for product in validation_products}
    selected_curve_frames = []
    plot_models = {**models, **extended_models}
    seeds_by_policy = {
        "PaperFCNN": tuple(range(10)),
        "PenaltyMLP": PRIMARY_SEEDS,
        "PositiveWeightMLP": PRIMARY_SEEDS,
        "ConstrainedMonoMLP": tuple(range(10)),
    }
    common_seed_plot = {policy: PRIMARY_SEEDS for policy in PRIMARY_POLICIES}
    violation_product = str(
        selection.loc[selection["role"] == "fcnn_violation", "product_id"].iloc[0]
    )
    violation_seed = int(
        extended_diagnostics.loc[
            (extended_diagnostics["policy_name"] == "PaperFCNN")
            & (extended_diagnostics["product_id"] == violation_product)
        ]
        .sort_values(
            ["raw_maximum_positive_violation", "training_seed"],
            ascending=[False, True],
        )
        .iloc[0]["training_seed"]
    )
    for row in selection.itertuples(index=False):
        representative_seeds = (
            common_seed_plot if row.role == "typical" else seeds_by_policy
        )
        selected_curve_frames.append(
            plot_representative_product(
                product_map[row.product_id],
                row.role,
                plot_models,
                feature_stats,
                settings,
                observed_upper,
                output_dir,
                representative_seeds,
                violation_seed if row.role == "fcnn_violation" else None,
                (
                    reconstructed_state_percentiles
                    if row.role == "typical"
                    else None
                ),
            )
        )
    plot_fcnn_violation_zoom(
        product_map[violation_product],
        plot_models[("PaperFCNN", violation_seed)],
        feature_stats,
        settings,
        observed_upper,
        output_dir,
        violation_seed,
    )
    _atomic_write_csv(
        output_dir / "representative_product_curves.csv",
        pd.concat(selected_curve_frames, ignore_index=True),
    )
    residual_models = _load_residual_cmnn_models(
        base_config, args.profile, residual_root
    )
    aligned_cmnn = extended_diagnostics.loc[
        (extended_diagnostics["policy_name"] == "ConstrainedMonoMLP")
        & (extended_diagnostics["region"] == "observed_training_state")
        & (extended_diagnostics["training_seed"].isin(RESIDUAL_SEEDS))
    ]
    per_product_zero_error = (
        aligned_cmnn.groupby("product_id", as_index=False)[
            "zero_inventory_target_stock_absolute_error"
        ]
        .mean()
        .rename(
            columns={
                "zero_inventory_target_stock_absolute_error": "aligned_seed_mean_zero_inventory_error"
            }
        )
    )
    median_zero_error = float(
        per_product_zero_error["aligned_seed_mean_zero_inventory_error"].median()
    )
    per_product_zero_error["distance_to_cross_product_median"] = (
        per_product_zero_error["aligned_seed_mean_zero_inventory_error"]
        - median_zero_error
    ).abs()
    median_product = per_product_zero_error.sort_values(
        ["distance_to_cross_product_median", "product_id"]
    ).iloc[0]
    typical_product_id = str(
        selection.loc[selection["role"] == "typical", "product_id"].iloc[0]
    )
    similar_product_id = str(
        selection.loc[selection["role"] == "similar_behavior", "product_id"].iloc[0]
    )
    figure_specs = (
        ("multivariate_typical", typical_product_id),
        ("minimum_cross_policy_dispersion", similar_product_id),
        ("median_cmnn_zero_inventory_error", str(median_product.product_id)),
    )
    figure_metric_rows = []
    for selection_rule, product_id in figure_specs:
        metrics = plot_direct_vs_residual_cmnn(
            product_map[product_id],
            plot_models,
            residual_models,
            feature_stats,
            settings,
            observed_upper,
            output_dir,
            f"direct_vs_residual_cmnn_{selection_rule}_{product_id}",
        )
        metrics.update(
            {
                "selection_rule": selection_rule,
                "aligned_training_seeds": "0,1,2",
                "slope_error_definition": "mean_abs_dq_dI_plus_one_on_policy_active_intervals",
            }
        )
        figure_metric_rows.append(metrics)
    plot_direct_vs_residual_cmnn(
        product_map["validation_000017"],
        plot_models,
        residual_models,
        feature_stats,
        settings,
        observed_upper,
        output_dir,
        "direct_vs_residual_cmnn",
    )
    figure_selection = pd.DataFrame(
        [
            {
                "selection_rule": selection_rule,
                "product_id": product_id,
                "cross_product_median_zero_inventory_error": median_zero_error,
                "distance_to_cross_product_median": (
                    float(median_product.distance_to_cross_product_median)
                    if selection_rule == "median_cmnn_zero_inventory_error"
                    else np.nan
                ),
                "status": "regenerated",
            }
            for selection_rule, product_id in figure_specs
        ]
        + [
            {
                "selection_rule": "extreme_calibration_error_appendix",
                "product_id": "validation_000017",
                "cross_product_median_zero_inventory_error": median_zero_error,
                "distance_to_cross_product_median": np.nan,
                "status": "regenerated_semantic_correction_only",
            }
        ]
    )
    _atomic_write_csv(
        output_dir / "direct_vs_residual_cmnn_three_product_metrics.csv",
        pd.DataFrame(figure_metric_rows),
    )
    _atomic_write_csv(
        output_dir / "direct_vs_residual_cmnn_product_selection.csv",
        figure_selection,
    )
    plot_aggregate_diagnostics(
        aggregate,
        output_dir,
        "aggregate_structural_diagnostics_equal_seeds",
    )
    plot_aggregate_diagnostics(
        aggregate,
        output_dir,
        "aggregate_structural_diagnostics",
    )

    no_imitation = pd.read_csv(no_imitation_root / "aggregate_results.csv")
    _atomic_write_csv(
        output_dir / "secondary_no_imitation_comparison.csv",
        no_imitation.loc[
            no_imitation["policy_name"].isin(PRIMARY_POLICIES),
            [
                "policy_name",
                "mean_reward_per_period",
                "reward_sample_sd",
                "raw_violation_rate",
                "deployed_violation_rate",
                "raw_max_violation",
                "deployed_max_violation",
            ],
        ],
    )
    _atomic_write_json(
        output_dir / "analysis_completion.json",
        {
            "primary_checkpoint_count": len(models),
            "paper_cmnn_ten_seed_checkpoint_count": len(extended_models),
            "residual_checkpoint_count": len(residual_models),
            "new_optimization_runs": 0,
            "validation_product_count": len(validation_products),
            "grid_points_per_product": GRID_POINTS,
            "observed_upper_factor": observed_upper,
            "wall_seconds": time.perf_counter() - started,
            "protected_test_accessed": False,
        },
    )
    print(
        f"completed checkpoints={len(models)} products={len(validation_products)} "
        f"observed_upper={observed_upper:.6f} new_optimization_runs=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
