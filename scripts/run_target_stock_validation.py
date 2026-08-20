"""Run the validation-only reward-only TargetStockFCNN follow-up.

The script first verifies exact reuse compatibility of the completed direct-q
PaperFCNN no-imitation baseline. It generates only train and validation
products and refuses test-labelled splits or output paths.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from inventory_control.idp_benchmark import (  # noqa: E402
    _model_config,
    _oracle_levels,
    _parameter_count,
    _prediction_summary,
    _prediction_table,
    _profile_settings,
    _train_model,
)
from inventory_control.paper_configs import load_paper_lost_sales_config  # noqa: E402
from inventory_control.paper_lost_sales import (  # noqa: E402
    non_omniscient_gamma_base_stock_policy,
)
from inventory_control.policies.neural.fcnn import (  # noqa: E402
    PaperCausalCNNPolicy,
    PaperFCNNFeatureStats,
    compute_paper_fcnn_feature_stats,
    make_paper_fcnn_model,
    make_paper_target_stock_fcnn_model,
)
from inventory_control.products import (  # noqa: E402
    PaperLostSalesProduct,
    PaperLostSalesProductGenerator,
)
from inventory_control.training.direct_backprop import (  # noqa: E402
    paper_lost_sales_context_features,
    paper_lost_sales_target_stock_level,
)
from scripts.run_no_imitation_ablation import (  # noqa: E402
    FIXED_SEEDS,
    _operational_metrics,
    resolve_no_imitation_config,
)
from scripts.tune_primary_neural_policies import (  # noqa: E402
    _atomic_torch_save,
    _atomic_write_csv,
    _atomic_write_json,
    _environment_record,
    _fingerprint,
    _repository_relative_path,
    _safe_generate_products,
    data_fingerprints,
    validate_split_and_output,
)


INITIALIZATION_MODES = ("standard", "target_stock_informed")
POLICY_NAMES = {
    "standard": "TargetStockFCNNStandard",
    "target_stock_informed": "TargetStockFCNNInformed",
}
DIRECT_POLICY = "PaperFCNN"
BASELINE_ROOT = Path("outputs/idp_followup/no_imitation_matched_width_validation")
DEFAULT_OUTPUT_DIR = Path("outputs/idp_followup/target_stock_reward_only_validation")
HISTORICAL_ANCHOR_AUDIT = Path(
    "outputs/idp_followup/residual_order_up_to_anchor_audit/audit_summary.json"
)
RESIDUAL_AGGREGATE = Path(
    "outputs/idp_followup/residual_order_up_to_validation/aggregate_results.csv"
)
EXTREME_TARGET_MULTIPLE = 8.0
CAP_TOLERANCE = 1e-6


def resolve_target_stock_config(
    base_config: dict[str, Any],
    profile: str,
    initialization_mode: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve the target-stock fit while preserving the direct-q budget."""
    if initialization_mode not in INITIALIZATION_MODES:
        raise ValueError(f"Unsupported initialization mode: {initialization_mode}")
    config, direct = resolve_no_imitation_config(base_config, profile, DIRECT_POLICY, seed)
    resolved = {
        **direct,
        "policy_name": POLICY_NAMES[initialization_mode],
        "action_parameterization": "target_stock",
        "inventory_network_input": False,
        "normalized_target_activation": "softplus",
        "target_stock_scale": "historical_mean_demand",
        "action_decoder": "relu(target_stock-current_inventory)",
        "initialization_mode": initialization_mode,
        "training_semantics": "reward_only_direct_backprop_target_stock",
    }
    return config, resolved


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _context_batch(
    products: Sequence[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
) -> dict[str, torch.Tensor]:
    """Build context-only inputs using the shared training normalization."""
    return {
        "history": torch.tensor(
            [
                [float(value) / feature_stats.demand_scale for value in product.historical_demand]
                for product in products
            ],
            dtype=torch.float32,
        ),
        "feature_price_scale": torch.full(
            (len(products),), float(feature_stats.price_scale), dtype=torch.float32
        ),
        "historical_mean_demand": torch.tensor(
            [max(float(np.mean(product.historical_demand)), 1.0) for product in products],
            dtype=torch.float32,
        ),
        "price": torch.tensor([product.price for product in products], dtype=torch.float32),
        "purchase_cost": torch.tensor(
            [product.purchase_cost for product in products], dtype=torch.float32
        ),
        "holding_cost": torch.tensor(
            [product.holding_cost for product in products], dtype=torch.float32
        ),
        "lost_sale_penalty": torch.tensor(
            [product.lost_sale_penalty for product in products], dtype=torch.float32
        ),
    }


def verify_direct_baseline_reuse(
    *,
    baseline_root: Path,
    base_config: dict[str, Any],
    profile: str,
    train_products: list[PaperLostSalesProduct],
    validation_products: list[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
    fingerprints: dict[str, Any],
) -> dict[str, Any]:
    """Fail before training unless the existing direct-q baseline is exact."""
    expected_fingerprints = dict(fingerprints)
    expected_fingerprints["shared_data_fingerprint"] = _fingerprint(fingerprints)
    recorded_fingerprints = _read_json(baseline_root / "data_fingerprints.json")
    if recorded_fingerprints != expected_fingerprints:
        raise RuntimeError("Direct-q baseline data/evaluation fingerprints do not match")

    expected_product_ids = [product.product_id for product in validation_products]
    expected_evaluation_seeds = list(range(40000, 40000 + len(validation_products)))
    seed_checks: list[dict[str, Any]] = []
    reference_shapes: dict[str, tuple[int, ...]] | None = None
    for seed in FIXED_SEEDS:
        config, expected_resolved = resolve_no_imitation_config(
            base_config, profile, DIRECT_POLICY, seed
        )
        expected_resolved["shared_data_fingerprint"] = expected_fingerprints[
            "shared_data_fingerprint"
        ]
        expected_resolved["resolved_config_fingerprint"] = _fingerprint(expected_resolved)
        fit_dir = baseline_root / "runs" / DIRECT_POLICY / f"seed_{seed}"
        required = (
            "completion.json",
            "resolved_config.json",
            "training_log.csv",
            "operational_metrics.csv",
            "predictions.csv",
            "model.pt",
        )
        missing = [name for name in required if not (fit_dir / name).exists()]
        if missing:
            raise RuntimeError(f"Direct-q baseline seed {seed} is incomplete: {missing}")
        recorded_resolved = _read_json(fit_dir / "resolved_config.json")
        if recorded_resolved != expected_resolved:
            raise RuntimeError(f"Direct-q baseline resolved settings differ for seed {seed}")
        completion = _read_json(fit_dir / "completion.json")
        if (
            completion.get("resolved_config_fingerprint")
            != expected_resolved["resolved_config_fingerprint"]
            or completion.get("imitation_optimizer_updates") != 0
            or completion.get("reward_optimizer_updates") != 15
        ):
            raise RuntimeError(f"Direct-q baseline completion metadata differs for seed {seed}")

        training_log = pd.read_csv(fit_dir / "training_log.csv")
        if len(training_log) != 15 or not (training_log["phase"] == 1).all():
            raise RuntimeError(f"Direct-q seed {seed} is not exactly 15 reward-only updates")
        operational = pd.read_csv(fit_dir / "operational_metrics.csv")
        if (
            operational["product_id"].tolist() != expected_product_ids
            or operational["evaluation_seed"].astype(int).tolist() != expected_evaluation_seeds
            or set(operational["evaluation_split"]) != {"validation"}
        ):
            raise RuntimeError(f"Direct-q evaluation scenarios differ for seed {seed}")

        model_config = _model_config(config, _profile_settings(config, profile))
        direct_model = make_paper_fcnn_model(model_config)
        if not isinstance(direct_model, PaperCausalCNNPolicy):
            raise RuntimeError("Direct-q baseline does not resolve to the causal-CNN encoder")
        state = torch.load(fit_dir / "model.pt", map_location="cpu", weights_only=True)
        direct_model.load_state_dict(state, strict=True)
        shapes = {name: tuple(tensor.shape) for name, tensor in state.items()}
        if reference_shapes is None:
            reference_shapes = shapes
        elif shapes != reference_shapes:
            raise RuntimeError(f"Direct-q checkpoint architecture differs for seed {seed}")
        seed_checks.append(
            {
                "training_seed": seed,
                "resolved_config_fingerprint": expected_resolved[
                    "resolved_config_fingerprint"
                ],
                "reward_optimizer_updates": 15,
                "imitation_optimizer_updates": 0,
                "validation_product_count": len(operational),
                "evaluation_seed_min": int(operational["evaluation_seed"].min()),
                "evaluation_seed_max": int(operational["evaluation_seed"].max()),
                "checkpoint_parameter_count": sum(tensor.numel() for tensor in state.values()),
            }
        )

    expected_stats = compute_paper_fcnn_feature_stats(train_products)
    if expected_stats != feature_stats:
        raise RuntimeError("Feature-normalization statistics are not reproducible")
    return {
        "reusable": True,
        "policy_name": DIRECT_POLICY,
        "seed_count": len(seed_checks),
        "history_encoder": "causal_cnn",
        "hidden_layers": 2,
        "hidden_units": 32,
        "learning_rate": 0.001,
        "reward_optimizer_updates": 15,
        "imitation_optimizer_updates": 0,
        "training_product_count": len(train_products),
        "validation_product_count": len(validation_products),
        "feature_demand_scale": feature_stats.demand_scale,
        "feature_price_scale": feature_stats.price_scale,
        "shared_data_fingerprint": expected_fingerprints["shared_data_fingerprint"],
        "evaluation_seeds": expected_evaluation_seeds,
        "seed_checks": seed_checks,
    }


def _fit_dir(output_dir: Path, initialization_mode: str, seed: int) -> Path:
    return output_dir / "runs" / POLICY_NAMES[initialization_mode] / f"seed_{seed}"


def _load_completed(fit_dir: Path, expected_fingerprint: str) -> dict[str, Any] | None:
    required = (
        "completion.json",
        "resolved_config.json",
        "training_log.csv",
        "operational_metrics.csv",
        "predictions.csv",
        "target_stock_by_product.csv",
        "model.pt",
    )
    if any(not (fit_dir / name).exists() for name in required):
        return None
    record = _read_json(fit_dir / "completion.json")
    if record.get("resolved_config_fingerprint") != expected_fingerprint:
        raise RuntimeError(f"Completed target-stock fit has incompatible settings: {fit_dir}")
    return record


def _target_stock_rows(
    model: torch.nn.Module,
    products: Sequence[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
) -> pd.DataFrame:
    batch = _context_batch(products, feature_stats)
    model.eval()
    with torch.no_grad():
        context = paper_lost_sales_context_features(batch)
        normalized = model(context).detach().cpu().numpy().astype(float)
        target = paper_lost_sales_target_stock_level(model, batch).detach().cpu().numpy().astype(float)
    means = batch["historical_mean_demand"].numpy().astype(float)
    return pd.DataFrame(
        {
            "product_id": [product.product_id for product in products],
            "historical_mean_demand": means,
            "normalized_target_stock": normalized,
            "target_stock_level": target,
            "extreme_target_stock": (~np.isfinite(normalized))
            | (normalized > EXTREME_TARGET_MULTIPLE),
        }
    )


def _historical_gamma_rmse(
    predictions: pd.DataFrame,
    products: Sequence[PaperLostSalesProduct],
    max_order_factor: float,
) -> float:
    product_map = {product.product_id: product for product in products}
    squared_errors: list[float] = []
    for product_id, rows in predictions.groupby("product_id", sort=False):
        product = product_map[str(product_id)]
        inventory = rows["inventory"].to_numpy(dtype=float)
        mean = max(float(np.mean(product.historical_demand)), 1.0)
        historical_level = float(non_omniscient_gamma_base_stock_policy(product).S)
        raw_reference = np.maximum(historical_level - inventory, 0.0)
        cap = np.maximum(float(max_order_factor) * mean - inventory, 0.0)
        deployed_reference = np.minimum(raw_reference, cap)
        error = rows["deployed_order"].to_numpy(dtype=float) - deployed_reference
        squared_errors.extend(np.square(error).tolist())
    return float(np.sqrt(np.mean(squared_errors)))


def _run_fit(
    *,
    base_config: dict[str, Any],
    profile: str,
    initialization_mode: str,
    seed: int,
    output_dir: Path,
    train_products: list[PaperLostSalesProduct],
    validation_products: list[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
    oracle_levels: dict[str, float],
    shared_data_fingerprint: str,
) -> dict[str, Any]:
    config, resolved = resolve_target_stock_config(
        base_config, profile, initialization_mode, seed
    )
    settings = _profile_settings(config, profile)
    resolved["shared_data_fingerprint"] = shared_data_fingerprint
    resolved["extreme_target_multiple_threshold"] = EXTREME_TARGET_MULTIPLE
    resolved["resolved_config_fingerprint"] = _fingerprint(resolved)
    fit_dir = _fit_dir(output_dir, initialization_mode, seed)
    fit_dir.mkdir(parents=True, exist_ok=True)
    (fit_dir / "completion.json").unlink(missing_ok=True)
    _atomic_write_json(fit_dir / "resolved_config.json", resolved)

    torch.manual_seed(int(seed))
    model = make_paper_target_stock_fcnn_model(
        _model_config(config, settings),
        initialization_mode=initialization_mode,
        training_products=train_products if initialization_mode == "target_stock_informed" else None,
    )
    initial_targets = _target_stock_rows(model, validation_products, feature_stats)
    started = time.perf_counter()
    result = _train_model(
        config,
        settings,
        train_products,
        validation_products,
        feature_stats,
        model,
        POLICY_NAMES[initialization_mode],
        "validation",
    )
    wall_seconds = time.perf_counter() - started
    training_log = pd.DataFrame(result.training_log or [])
    if (
        result.imitation_epochs_completed != 0
        or result.best_imitation_state_dict is not None
        or len(training_log) != 15
        or not (training_log["phase"] == 1).all()
    ):
        raise RuntimeError("Target-stock fit was not exactly 15 reward-only updates")

    policy_name = POLICY_NAMES[initialization_mode]
    predictions = _prediction_table(
        validation_products, {policy_name: model}, feature_stats, settings, oracle_levels
    )
    predictions = predictions.loc[predictions["policy_name"] == policy_name].reset_index(drop=True)
    prediction_summary = _prediction_summary(predictions).iloc[0].to_dict()
    operational = _operational_metrics(
        validation_products, policy_name, model, feature_stats, settings
    )
    final_targets = _target_stock_rows(model, validation_products, feature_stats)

    cap_hits = int(operational["cap_hit_count"].sum())
    scored_actions = int(operational["scored_action_count"].sum())
    grid_clipping = (
        predictions["raw_order"].to_numpy(dtype=float)
        - predictions["deployed_order"].to_numpy(dtype=float)
    )
    aggregate_columns = (
        "mean_total_reward",
        "mean_reward_per_period",
        "service_level",
        "lost_sales_rate",
        "mean_order_quantity",
        "mean_inventory",
        "mean_holding_cost",
    )
    aggregate = {column: float(operational[column].mean()) for column in aggregate_columns}
    normalized = final_targets["normalized_target_stock"].to_numpy(dtype=float)
    finite = normalized[np.isfinite(normalized)]
    record = {
        "policy_name": policy_name,
        "initialization_mode": initialization_mode,
        "training_seed": int(seed),
        "resolved_config_fingerprint": resolved["resolved_config_fingerprint"],
        "shared_data_fingerprint": shared_data_fingerprint,
        "parameter_count": _parameter_count(model),
        "training_seconds": float(result.elapsed_seconds),
        "fit_wall_seconds": float(wall_seconds),
        "imitation_optimizer_updates": 0,
        "reward_optimizer_updates": int(len(training_log)),
        "failed_fit": False,
        **aggregate,
        "deployed_rmse": float(prediction_summary["deployed_rmse"]),
        "deployed_rmse_vs_historical_gamma": _historical_gamma_rmse(
            predictions, validation_products, float(settings["max_order_factor"])
        ),
        "cap_hit_count": cap_hits,
        "scored_action_count": scored_actions,
        "cap_hit_rate": cap_hits / max(scored_actions, 1),
        "mean_clipping_amount": float(
            (operational["mean_clipping_amount"] * operational["scored_action_count"]).sum()
            / max(scored_actions, 1)
        ),
        "maximum_clipping_amount": float(operational["maximum_clipping_amount"].max()),
        "grid_cap_hit_rate": float(np.mean(grid_clipping > CAP_TOLERANCE)),
        "grid_mean_clipping_amount": float(np.mean(np.maximum(grid_clipping, 0.0))),
        "grid_maximum_clipping_amount": float(np.max(np.maximum(grid_clipping, 0.0))),
        "normalized_target_stock_minimum": float(np.min(finite)) if finite.size else np.nan,
        "normalized_target_stock_median": float(np.median(finite)) if finite.size else np.nan,
        "normalized_target_stock_mean": float(np.mean(finite)) if finite.size else np.nan,
        "normalized_target_stock_maximum": float(np.max(finite)) if finite.size else np.nan,
        "extreme_target_stock_count": int(final_targets["extreme_target_stock"].sum()),
        "extreme_target_stock_rate": float(final_targets["extreme_target_stock"].mean()),
        "non_finite_target_stock_count": int((~np.isfinite(normalized)).sum()),
        "initial_normalized_target_stock_mean": float(
            initial_targets["normalized_target_stock"].mean()
        ),
        "informed_initial_scalar": model.target_stock_initialization_scalar,
    }
    _atomic_write_csv(fit_dir / "training_log.csv", training_log)
    _atomic_write_csv(fit_dir / "operational_metrics.csv", operational)
    _atomic_write_csv(fit_dir / "predictions.csv", predictions)
    _atomic_write_csv(
        fit_dir / "prediction_diagnostics.csv", pd.DataFrame([prediction_summary])
    )
    _atomic_write_csv(fit_dir / "initial_target_stock_by_product.csv", initial_targets)
    _atomic_write_csv(fit_dir / "target_stock_by_product.csv", final_targets)
    _atomic_torch_save(fit_dir / "model.pt", model.state_dict())
    _atomic_write_json(fit_dir / "fit_result.json", record)
    _atomic_write_json(fit_dir / "completion.json", record)
    return record


def _scan_target_results(output_dir: Path) -> pd.DataFrame:
    records = [
        _read_json(marker)
        for marker in sorted((output_dir / "runs").glob("*/seed_*/completion.json"))
    ]
    return pd.DataFrame(records)


def _direct_seed_results(
    baseline_root: Path,
    validation_products: Sequence[PaperLostSalesProduct],
    max_order_factor: float,
) -> pd.DataFrame:
    seed_results = pd.read_csv(baseline_root / "seed_results.csv")
    direct = seed_results.loc[seed_results["policy_name"] == DIRECT_POLICY].copy()
    historical_rmse: list[float] = []
    for seed in direct["training_seed"].astype(int):
        predictions = pd.read_csv(
            baseline_root / "runs" / DIRECT_POLICY / f"seed_{seed}" / "predictions.csv"
        )
        historical_rmse.append(
            _historical_gamma_rmse(predictions, validation_products, max_order_factor)
        )
    direct["deployed_rmse_vs_historical_gamma"] = historical_rmse
    direct["initialization_mode"] = "standard_direct_q"
    for column in (
        "normalized_target_stock_minimum",
        "normalized_target_stock_median",
        "normalized_target_stock_mean",
        "normalized_target_stock_maximum",
        "extreme_target_stock_count",
        "extreme_target_stock_rate",
        "non_finite_target_stock_count",
        "grid_cap_hit_rate",
        "grid_mean_clipping_amount",
        "grid_maximum_clipping_amount",
    ):
        direct[column] = np.nan
    return direct


def _aggregate_results(seed_results: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "mean_reward_per_period",
        "service_level",
        "lost_sales_rate",
        "mean_inventory",
        "mean_order_quantity",
        "mean_holding_cost",
        "deployed_rmse",
        "deployed_rmse_vs_historical_gamma",
        "cap_hit_rate",
        "mean_clipping_amount",
        "maximum_clipping_amount",
        "grid_cap_hit_rate",
        "grid_mean_clipping_amount",
        "grid_maximum_clipping_amount",
        "normalized_target_stock_minimum",
        "normalized_target_stock_median",
        "normalized_target_stock_mean",
        "normalized_target_stock_maximum",
        "extreme_target_stock_rate",
        "non_finite_target_stock_count",
    )
    rows: list[dict[str, Any]] = []
    for policy, group in seed_results.groupby("policy_name", sort=False):
        row: dict[str, Any] = {
            "policy_name": policy,
            "seed_count": len(group),
            "reward_mean": float(group["mean_reward_per_period"].mean()),
            "reward_sample_sd": float(group["mean_reward_per_period"].std(ddof=1)),
            "parameter_count": int(group["parameter_count"].iloc[0]),
            "failed_fit_count": int(group.get("failed_fit", pd.Series(False, index=group.index)).sum()),
        }
        for metric in metrics:
            if metric in group:
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_sample_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_statistics(
    seed_results: pd.DataFrame,
    left_policy: str,
    right_policy: str,
    label: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return frozen paired reward summaries for left minus right."""
    rewards = seed_results.pivot(
        index="training_seed", columns="policy_name", values="mean_reward_per_period"
    ).sort_index()
    paired = rewards[[left_policy, right_policy]].dropna()
    differences = paired[left_policy] - paired[right_policy]
    count = len(differences)
    if count != len(FIXED_SEEDS):
        raise RuntimeError(f"Paired comparison {label} does not contain all ten seeds")
    mean = float(differences.mean())
    sample_sd = float(differences.std(ddof=1))
    half_width = float(stats.t.ppf(0.975, count - 1) * sample_sd / np.sqrt(count))
    t_result = stats.ttest_rel(paired[left_policy], paired[right_policy])
    if np.allclose(differences.to_numpy(dtype=float), 0.0):
        wilcoxon_statistic = 0.0
        wilcoxon_pvalue = 1.0
    else:
        wilcoxon = stats.wilcoxon(
            differences, alternative="two-sided", zero_method="wilcox"
        )
        wilcoxon_statistic = float(wilcoxon.statistic)
        wilcoxon_pvalue = float(wilcoxon.pvalue)
    detail = paired.reset_index().rename(
        columns={left_policy: "left_reward", right_policy: "right_reward"}
    )
    detail.insert(0, "comparison", label)
    detail.insert(1, "left_policy", left_policy)
    detail.insert(2, "right_policy", right_policy)
    detail["paired_difference_left_minus_right"] = differences.to_numpy(dtype=float)
    summary = {
        "comparison": label,
        "left_policy": left_policy,
        "right_policy": right_policy,
        "seed_count": count,
        "paired_mean_difference": mean,
        "paired_difference_sample_sd": sample_sd,
        "student_t_confidence_interval_95_lower": mean - half_width,
        "student_t_confidence_interval_95_upper": mean + half_width,
        "paired_t_statistic": float(t_result.statistic),
        "paired_t_pvalue": float(t_result.pvalue),
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_pvalue": wilcoxon_pvalue,
        "left_wins": int((differences > 0.0).sum()),
        "right_wins": int((differences < 0.0).sum()),
        "ties": int((differences == 0.0).sum()),
    }
    return summary, detail


def _analytical_context() -> pd.DataFrame:
    anchor_audit = _read_json(REPOSITORY_ROOT / HISTORICAL_ANCHOR_AUDIT)
    anchor = next(
        row
        for row in anchor_audit["baseline_metrics"]
        if row["policy_name"] == "NonOmniscientOptimal"
    )
    residual = pd.read_csv(REPOSITORY_ROOT / RESIDUAL_AGGREGATE)
    residual_row = residual.loc[
        residual["policy_name"] == "PaperFCNNResidualOrderUpTo"
    ].iloc[0]
    return pd.DataFrame(
        [
            {
                "policy_name": "HistoricalGammaOrderUpTo",
                "validation_reward_mean": float(anchor["mean_reward_per_period"]),
                "validation_reward_sample_sd": np.nan,
                "source": HISTORICAL_ANCHOR_AUDIT.as_posix(),
                "context_only_not_retrained": True,
            },
            {
                "policy_name": "PaperFCNNResidualOrderUpTo",
                "validation_reward_mean": float(residual_row["mean_reward_per_period_mean"]),
                "validation_reward_sample_sd": float(
                    residual_row["mean_reward_per_period_sample_sd"]
                ),
                "source": RESIDUAL_AGGREGATE.as_posix(),
                "context_only_not_retrained": True,
            },
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", default="full")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(FIXED_SEEDS))
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if tuple(args.seeds) != FIXED_SEEDS:
        raise ValueError("Target-stock follow-up requires exactly seeds 0 through 9")
    validate_split_and_output(args.split, args.output_dir)
    config_path, config_relative = _repository_relative_path(args.config, "--config")
    baseline_root, baseline_relative = _repository_relative_path(
        args.baseline_root, "--baseline-root"
    )
    output_dir, _ = _repository_relative_path(args.output_dir, "--output-dir")
    if baseline_relative != BASELINE_ROOT:
        raise ValueError(f"Direct baseline must be the frozen validation source {BASELINE_ROOT}")

    base_config = load_paper_lost_sales_config(config_path)
    audit_config, _ = resolve_no_imitation_config(base_config, args.profile, DIRECT_POLICY, 0)
    settings = _profile_settings(audit_config, args.profile)
    if (
        int(settings["training_products"]) != 128
        or int(settings["validation_products"]) != 32
        or float(settings["max_order_factor"]) != 4.0
    ):
        raise RuntimeError("Target-stock follow-up requires 128/32 products and F=4")

    generator = PaperLostSalesProductGenerator(audit_config)
    train_products = _safe_generate_products(generator, 128, "train")
    validation_products = _safe_generate_products(generator, 32, "validation")
    feature_stats = compute_paper_fcnn_feature_stats(train_products)
    fingerprints = data_fingerprints(
        audit_config, settings, train_products, validation_products
    )
    shared_data_fingerprint = _fingerprint(fingerprints)
    baseline_audit = verify_direct_baseline_reuse(
        baseline_root=baseline_root,
        base_config=base_config,
        profile=args.profile,
        train_products=train_products,
        validation_products=validation_products,
        feature_stats=feature_stats,
        fingerprints=fingerprints,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_dir / "baseline_reuse_audit.json", baseline_audit)
    _atomic_write_json(
        output_dir / "environment.json", _environment_record(config_relative, audit_config)
    )
    fingerprint_record = dict(fingerprints)
    fingerprint_record["shared_data_fingerprint"] = shared_data_fingerprint
    _atomic_write_json(output_dir / "data_fingerprints.json", fingerprint_record)
    manifest = pd.DataFrame(
        [
            {
                "policy_name": POLICY_NAMES[mode],
                "initialization_mode": mode,
                "training_seed": seed,
                "hidden_layers": 2,
                "hidden_units": 32,
                "learning_rate": 0.001,
                "reward_optimizer_updates": 15,
                "imitation_optimizer_updates": 0,
                "training_products": 128,
                "validation_products": 32,
                "deployment_action_cap_factor": 4.0,
            }
            for mode in INITIALIZATION_MODES
            for seed in FIXED_SEEDS
        ]
    )
    _atomic_write_csv(output_dir / "manifest.csv", manifest)
    if not args.run:
        print(
            f"baseline_reusable=true target_fits={len(manifest)} "
            f"shared_data_fingerprint={shared_data_fingerprint}"
        )
        return 0

    oracle_levels = _oracle_levels(
        validation_products, int(settings["oracle_mc_samples"]), 10_000
    )
    started = time.perf_counter()
    completed = 0
    failures: list[dict[str, Any]] = []
    for mode in INITIALIZATION_MODES:
        for seed in FIXED_SEEDS:
            _, resolved = resolve_target_stock_config(base_config, args.profile, mode, seed)
            resolved["shared_data_fingerprint"] = shared_data_fingerprint
            resolved["extreme_target_multiple_threshold"] = EXTREME_TARGET_MULTIPLE
            expected = _fingerprint(resolved)
            fit_dir = _fit_dir(output_dir, mode, seed)
            existing = _load_completed(fit_dir, expected) if fit_dir.exists() else None
            if existing is not None and args.resume:
                completed += 1
                print(
                    f"completed={completed}/20 policy={POLICY_NAMES[mode]} seed={seed} "
                    f"reused=true reward={existing['mean_reward_per_period']:.6f}"
                )
                continue
            if existing is not None:
                raise RuntimeError("Completed target-stock fits exist; pass --resume")
            try:
                record = _run_fit(
                    base_config=base_config,
                    profile=args.profile,
                    initialization_mode=mode,
                    seed=seed,
                    output_dir=output_dir,
                    train_products=train_products,
                    validation_products=validation_products,
                    feature_stats=feature_stats,
                    oracle_levels=oracle_levels,
                    shared_data_fingerprint=shared_data_fingerprint,
                )
                completed += 1
                print(
                    f"completed={completed}/20 policy={POLICY_NAMES[mode]} seed={seed} "
                    f"reused=false reward={record['mean_reward_per_period']:.6f}"
                )
            except Exception as exc:
                failures.append(
                    {
                        "policy_name": POLICY_NAMES[mode],
                        "initialization_mode": mode,
                        "training_seed": seed,
                        "error": repr(exc),
                    }
                )
                fit_dir.mkdir(parents=True, exist_ok=True)
                (fit_dir / "failure.log").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
                print(
                    f"failed policy={POLICY_NAMES[mode]} seed={seed} "
                    f"error={type(exc).__name__}"
                )

    target_results = _scan_target_results(output_dir)
    direct_results = _direct_seed_results(
        baseline_root, validation_products, float(settings["max_order_factor"])
    )
    comparison = pd.concat([direct_results, target_results], ignore_index=True, sort=False)
    _atomic_write_csv(output_dir / "target_seed_results.csv", target_results)
    _atomic_write_csv(output_dir / "comparison_seed_results.csv", comparison)
    _atomic_write_csv(output_dir / "aggregate_results.csv", _aggregate_results(comparison))

    paired_summaries: list[dict[str, Any]] = []
    paired_details: list[pd.DataFrame] = []
    for left, right, label in (
        (
            "TargetStockFCNNStandard",
            "PaperFCNN",
            "target_stock_standard_minus_direct_q",
        ),
        (
            "TargetStockFCNNInformed",
            "TargetStockFCNNStandard",
            "target_stock_informed_minus_standard",
        ),
    ):
        summary, detail = paired_statistics(comparison, left, right, label)
        paired_summaries.append(summary)
        paired_details.append(detail)
    _atomic_write_csv(output_dir / "paired_statistics.csv", pd.DataFrame(paired_summaries))
    _atomic_write_csv(
        output_dir / "paired_reward_differences.csv",
        pd.concat(paired_details, ignore_index=True),
    )
    _atomic_write_csv(output_dir / "analytical_context.csv", _analytical_context())
    _atomic_write_csv(
        output_dir / "failures.csv",
        pd.DataFrame(
            failures,
            columns=["policy_name", "initialization_mode", "training_seed", "error"],
        ),
    )
    _atomic_write_json(
        output_dir / "analysis_completion.json",
        {
            "baseline_reused": True,
            "completed_target_fits": len(target_results),
            "expected_target_fits": 20,
            "failed_target_fits": len(failures),
            "wall_seconds": time.perf_counter() - started,
            "protected_test_accessed": False,
            "imitation_enabled": False,
        },
    )
    return 0 if len(target_results) == 20 and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
