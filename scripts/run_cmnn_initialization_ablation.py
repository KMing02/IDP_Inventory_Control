"""Run the validation-only CMNN base-stock-informed initialization ablation."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from inventory_control.idp_benchmark import (  # noqa: E402
    _model_config,
    _model_orders,
    _oracle_levels,
    _parameter_count,
    _prediction_summary,
    _prediction_table,
    _profile_settings,
    _train_model,
)
from inventory_control.paper_configs import load_paper_lost_sales_config  # noqa: E402
from inventory_control.paper_lost_sales import historical_demand_base_stock_policy  # noqa: E402
from inventory_control.policies.neural.constrained_mlp import (  # noqa: E402
    ConstrainedMonotoneMLP,
    NonnegativeLinear,
)
from inventory_control.policies.neural.fcnn import (  # noqa: E402
    PaperFCNNFeatureStats,
    compute_paper_fcnn_feature_stats,
)
from inventory_control.products import (  # noqa: E402
    PaperLostSalesProduct,
    PaperLostSalesProductGenerator,
)
from scripts.run_no_imitation_ablation import (  # noqa: E402
    FIXED_SEEDS,
    _operational_metrics,
    output_diagnostics,
    resolve_no_imitation_config,
)
from scripts.tune_primary_neural_policies import (  # noqa: E402
    _atomic_torch_save,
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


POLICY = "ConstrainedMonoMLP"
INITIALIZATIONS = ("standard_random", "base_stock_informed")
DEFAULT_OUTPUT_DIR = Path("outputs/idp_followup/cmnn_initialization_ablation_validation")
STANDARD_RESULTS_ROOT = Path("outputs/idp_followup/no_imitation_matched_width_validation")
EFFECTIVE_ZERO_DTYPE = torch.float32
EFFECTIVE_ZERO_WEIGHT = float(torch.finfo(EFFECTIVE_ZERO_DTYPE).eps)


def empirical_base_stock_normalized_order(
    products: Sequence[PaperLostSalesProduct],
) -> float:
    """Return the zero-inventory normalized order minimizing training MSE.

    With zero final effective weights, the direct-order model predicts
    ``q_i(0) = historical_mean_i * softplus(b)``.  The returned value is the
    least-squares solution against empirical historical-demand quantiles.
    """
    means = np.asarray(
        [max(float(np.mean(product.historical_demand)), 1.0) for product in products],
        dtype=float,
    )
    targets = np.asarray(
        [historical_demand_base_stock_policy(product).S for product in products],
        dtype=float,
    )
    denominator = float(np.dot(means, means))
    if denominator <= 0.0:  # pragma: no cover - historical means are clamped above
        raise ValueError("Cannot calibrate base-stock bias from zero demand scale")
    return float(np.dot(means, targets) / denominator)


def inverse_softplus_scalar(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("inverse softplus requires a finite positive value")
    return float(value + math.log(-math.expm1(-value)))


def initialize_base_stock_informed_cmnn(
    model: ConstrainedMonotoneMLP,
    training_products: Sequence[PaperLostSalesProduct],
) -> dict[str, float]:
    """Numerically zero the final effective weights and calibrate its bias.

    ``NonnegativeLinear`` represents effective weights as ``softplus(raw)``.
    Exact zero requires ``raw=-inf`` and would permanently eliminate its
    gradient.  Float32 machine epsilon is therefore used as a numerical zero;
    this is an implementation constant, not a tuned hyperparameter.
    """
    final_layer = model.network[-1]
    if not isinstance(final_layer, NonnegativeLinear):
        raise TypeError("ConstrainedMonoMLP final layer must be NonnegativeLinear")
    normalized_order = empirical_base_stock_normalized_order(training_products)
    output_bias = inverse_softplus_scalar(normalized_order)
    raw_zero = inverse_softplus_scalar(EFFECTIVE_ZERO_WEIGHT)
    with torch.no_grad():
        final_layer.raw_weight.fill_(raw_zero)
        final_layer.bias.fill_(output_bias)
    return {
        "empirical_base_stock_normalized_order": normalized_order,
        "output_bias": output_bias,
        "requested_effective_final_weight": 0.0,
        "implemented_effective_final_weight": EFFECTIVE_ZERO_WEIGHT,
        "raw_final_weight": raw_zero,
    }


def _final_layer_diagnostics(model: ConstrainedMonotoneMLP) -> dict[str, float]:
    final_layer = model.network[-1]
    if not isinstance(final_layer, NonnegativeLinear):  # pragma: no cover
        raise TypeError("ConstrainedMonoMLP final layer must be NonnegativeLinear")
    effective = final_layer.effective_weight().detach()
    return {
        "final_effective_weight_l2": float(torch.linalg.vector_norm(effective).item()),
        "final_effective_weight_max": float(effective.max().item()),
        "final_raw_weight_l2": float(torch.linalg.vector_norm(final_layer.raw_weight.detach()).item()),
        "final_bias": float(final_layer.bias.detach().item()),
    }


def _initial_metrics(
    model: ConstrainedMonotoneMLP,
    products: list[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
    settings: dict[str, Any],
    oracle_levels: dict[str, float],
) -> dict[str, float | int]:
    predictions = _prediction_table(
        products, {POLICY: model}, feature_stats, settings, oracle_levels
    )
    predictions = predictions.loc[predictions["policy_name"] == POLICY].reset_index(drop=True)
    raw_zero = []
    deployed_zero = []
    raw_targets = []
    deployed_targets = []
    for product in products:
        raw, deployed = _model_orders(
            model,
            product,
            feature_stats,
            np.asarray([0.0]),
            float(settings["max_order_factor"]),
            bool(settings["scale_action_by_history_mean"]),
        )
        historical_mean = max(float(np.mean(product.historical_demand)), 1.0)
        target = float(historical_demand_base_stock_policy(product).S)
        raw_zero.append(float(raw[0]))
        deployed_zero.append(float(deployed[0]))
        raw_targets.append(target)
        deployed_targets.append(min(target, float(settings["max_order_factor"]) * historical_mean))
    raw_error = np.asarray(raw_zero) - np.asarray(raw_targets)
    deployed_error = np.asarray(deployed_zero) - np.asarray(deployed_targets)
    operational = _operational_metrics(products, POLICY, model, feature_stats, settings)
    return {
        "initial_grid_mean_raw_order": float(predictions["raw_order"].mean()),
        "initial_grid_mean_deployed_order": float(predictions["deployed_order"].mean()),
        "initial_zero_inventory_mean_raw_order": float(np.mean(raw_zero)),
        "initial_zero_inventory_empirical_base_stock_mae": float(np.mean(np.abs(raw_error))),
        "initial_zero_inventory_empirical_base_stock_rmse": float(
            np.sqrt(np.mean(np.square(raw_error)))
        ),
        "initial_zero_inventory_deployed_empirical_base_stock_mae": float(
            np.mean(np.abs(deployed_error))
        ),
        "initial_zero_inventory_deployed_empirical_base_stock_rmse": float(
            np.sqrt(np.mean(np.square(deployed_error)))
        ),
        "initial_operational_reward_per_period": float(operational["mean_reward_per_period"].mean()),
        "initial_operational_service_level": float(operational["service_level"].mean()),
        "initial_operational_mean_inventory": float(operational["mean_inventory"].mean()),
        "initial_operational_mean_order": float(operational["mean_order_quantity"].mean()),
        "initial_operational_cap_hit_rate": float(
            operational["cap_hit_count"].sum() / max(operational["scored_action_count"].sum(), 1)
        ),
        **{
            f"initial_{key}": value
            for key, value in _final_layer_diagnostics(model).items()
        },
    }


def _aggregate(seed_results: pd.DataFrame) -> pd.DataFrame:
    mean_metrics = (
        "mean_reward_per_period",
        "mean_inventory",
        "service_level",
        "lost_sales_rate",
        "mean_order_quantity",
        "mean_holding_cost",
        "deployed_rmse",
        "raw_violation_rate",
        "deployed_violation_rate",
        "cap_hit_rate",
        "non_finite_output_rate",
        "initial_grid_mean_raw_order",
        "initial_grid_mean_deployed_order",
        "initial_zero_inventory_mean_raw_order",
        "initial_zero_inventory_empirical_base_stock_mae",
        "initial_zero_inventory_empirical_base_stock_rmse",
        "initial_zero_inventory_deployed_empirical_base_stock_mae",
        "initial_zero_inventory_deployed_empirical_base_stock_rmse",
        "initial_operational_reward_per_period",
        "initial_operational_service_level",
        "initial_operational_mean_inventory",
        "initial_operational_mean_order",
        "initial_operational_cap_hit_rate",
    )
    rows = []
    for initialization, group in seed_results.groupby("initialization", sort=False):
        row: dict[str, Any] = {
            "initialization": initialization,
            "completed_seed_count": len(group),
            "failed_fit_count": len(FIXED_SEEDS) - len(group),
            "reward_sample_sd": float(group["mean_reward_per_period"].std(ddof=1)),
            "parameter_count": int(group["parameter_count"].iloc[0]),
            "non_finite_run_count": int((group["non_finite_output_count"] > 0).sum()),
        }
        for metric in mean_metrics:
            row[metric] = float(group[metric].mean())
        row["raw_max_violation"] = float(group["raw_max_violation"].max())
        row["deployed_max_violation"] = float(group["deployed_max_violation"].max())
        rows.append(row)
    return pd.DataFrame(rows)


def _trajectory_rows(
    log: pd.DataFrame, initialization: str, seed: int
) -> pd.DataFrame:
    selected = [
        column
        for column in (
            "epoch",
            "training_loss",
            "primary_loss",
            "training_mean_total_reward",
            "validation_mean_total_reward",
            "raw_order_mean",
            "raw_order_min",
            "raw_order_max",
            "deployed_order_mean",
            "cap_hit_rate",
        )
        if column in log.columns
    ]
    frame = log[selected].copy()
    frame.insert(0, "reward_update", np.arange(1, len(frame) + 1))
    frame.insert(0, "training_seed", int(seed))
    frame.insert(0, "initialization", initialization)
    return frame


def _run_informed_fit(
    *,
    base_config: dict[str, Any],
    profile: str,
    seed: int,
    output_dir: Path,
    train_products: list[PaperLostSalesProduct],
    validation_products: list[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
    oracle_levels: dict[str, float],
    shared_data_fingerprint: str,
) -> dict[str, Any]:
    config, resolved = resolve_no_imitation_config(base_config, profile, POLICY, seed)
    settings = _profile_settings(config, profile)
    model = _make_model(POLICY, _model_config(config, settings), seed)
    if not isinstance(model, ConstrainedMonotoneMLP):  # pragma: no cover
        raise TypeError("Factory did not return ConstrainedMonotoneMLP")
    initialization_metadata = initialize_base_stock_informed_cmnn(model, train_products)
    initial = _initial_metrics(model, validation_products, feature_stats, settings, oracle_levels)
    resolved.update(
        {
            "initialization": "base_stock_informed",
            "initialization_target": "empirical_historical_demand_quantile_at_zero_inventory",
            "initialization_fit_split": "train",
            "initialization_mapping": (
                "r=sum(m_i*S_emp_i)/sum(m_i^2); bias=softplus_inverse(r); "
                "effective_final_weight=float32_epsilon"
            ),
            **initialization_metadata,
            "shared_data_fingerprint": shared_data_fingerprint,
        }
    )
    resolved["resolved_config_fingerprint"] = _fingerprint(resolved)
    fit_dir = output_dir / "runs" / "base_stock_informed" / f"seed_{seed}"
    fit_dir.mkdir(parents=True, exist_ok=True)
    (fit_dir / "completion.json").unlink(missing_ok=True)
    _atomic_write_json(fit_dir / "resolved_config.json", resolved)
    _atomic_write_json(fit_dir / "initialization_metrics.json", initial)

    started = time.perf_counter()
    result = _train_model(
        config,
        settings,
        train_products,
        validation_products,
        feature_stats,
        model,
        POLICY,
        "validation",
        penalty_weight=0.0,
        clip_imitation_predictions=True,
    )
    wall_seconds = time.perf_counter() - started
    log = pd.DataFrame(result.training_log or [])
    if result.imitation_epochs_completed != 0 or len(log) != 15 or not (log["phase"] == 1).all():
        raise RuntimeError("Initialization fit was not exactly 15 reward-only updates")
    predictions = _prediction_table(
        validation_products, {POLICY: model}, feature_stats, settings, oracle_levels
    )
    predictions = predictions.loc[predictions["policy_name"] == POLICY].reset_index(drop=True)
    diagnostics = _prediction_summary(predictions).iloc[0].to_dict()
    operational = _operational_metrics(
        validation_products, POLICY, model, feature_stats, settings
    )
    cap_hits = int(operational["cap_hit_count"].sum())
    scored = int(operational["scored_action_count"].sum())
    final_layer = _final_layer_diagnostics(model)
    record = {
        "policy_name": POLICY,
        "initialization": "base_stock_informed",
        "training_seed": int(seed),
        "resolved_config_fingerprint": resolved["resolved_config_fingerprint"],
        "shared_data_fingerprint": shared_data_fingerprint,
        "parameter_count": _parameter_count(model),
        "training_seconds": float(result.elapsed_seconds),
        "fit_wall_seconds": float(wall_seconds),
        "imitation_optimizer_updates": 0,
        "reward_optimizer_updates": len(log),
        "failed_fit": False,
        "mean_reward_per_period": float(operational["mean_reward_per_period"].mean()),
        "mean_inventory": float(operational["mean_inventory"].mean()),
        "service_level": float(operational["service_level"].mean()),
        "lost_sales_rate": float(operational["lost_sales_rate"].mean()),
        "mean_order_quantity": float(operational["mean_order_quantity"].mean()),
        "mean_holding_cost": float(operational["mean_holding_cost"].mean()),
        "cap_hit_count": cap_hits,
        "scored_action_count": scored,
        "cap_hit_rate": cap_hits / max(scored, 1),
        **initial,
        **{f"trained_{key}": value for key, value in final_layer.items()},
        **diagnostics,
        **output_diagnostics(predictions),
    }
    _atomic_write_csv(fit_dir / "training_log.csv", log)
    _atomic_write_csv(fit_dir / "operational_metrics.csv", operational)
    _atomic_write_csv(fit_dir / "predictions.csv", predictions)
    _atomic_torch_save(fit_dir / "model.pt", model.state_dict())
    _atomic_write_json(fit_dir / "fit_result.json", record)
    _atomic_write_json(fit_dir / "completion.json", record)
    return record


def _existing_standard_record(
    root: Path,
    seed: int,
    model: ConstrainedMonotoneMLP,
    initial: dict[str, float | int],
) -> dict[str, Any]:
    fit_dir = root / "runs" / POLICY / f"seed_{seed}"
    completion = fit_dir / "completion.json"
    if not completion.exists():
        raise RuntimeError(f"Missing completed standard-random fit: {fit_dir}")
    record = json.loads(completion.read_text(encoding="utf-8"))
    return {
        **record,
        "initialization": "standard_random",
        "source_run_dir": fit_dir.relative_to(REPOSITORY_ROOT).as_posix(),
        **initial,
        **{f"initial_{key}": value for key, value in _final_layer_diagnostics(model).items()},
    }


def _scan_informed(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for marker in sorted((output_dir / "runs" / "base_stock_informed").glob("seed_*/completion.json")):
        rows.append(json.loads(marker.read_text(encoding="utf-8")))
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", default="full")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--standard-results-root", type=Path, default=STANDARD_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_split_and_output(args.split, args.output_dir)
    if tuple(args.seeds) != FIXED_SEEDS:
        raise ValueError("Initialization ablation requires exactly seeds 0 through 9")
    config_path, config_relative = _repository_relative_path(args.config, "--config")
    output_dir, _ = _repository_relative_path(args.output_dir, "--output-dir")
    standard_root, _ = _repository_relative_path(
        args.standard_results_root, "--standard-results-root"
    )
    base_config = load_paper_lost_sales_config(config_path)
    settings = _profile_settings(base_config, args.profile)
    if (
        int(settings["training_products"]) != 128
        or int(settings["validation_products"]) != 32
        or float(settings["max_order_factor"]) != 4.0
    ):
        raise ValueError("Initialization ablation requires 128/32 products and F=4")
    generator = PaperLostSalesProductGenerator(base_config)
    train_products = _safe_generate_products(generator, 128, "train")
    validation_products = _safe_generate_products(generator, 32, "validation")
    feature_stats = compute_paper_fcnn_feature_stats(train_products)
    oracle_levels = _oracle_levels(validation_products, int(settings["oracle_mc_samples"]), 10_000)
    fingerprints = data_fingerprints(base_config, settings, train_products, validation_products)
    shared = _fingerprint(fingerprints)
    fingerprints["shared_data_fingerprint"] = shared
    standard_fingerprints = json.loads(
        (standard_root / "data_fingerprints.json").read_text(encoding="utf-8")
    )
    if standard_fingerprints != fingerprints:
        raise RuntimeError("Standard and initialization-ablation data fingerprints differ")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_or_validate_json(output_dir / "environment.json", _environment_record(config_relative, base_config))
    _write_or_validate_json(output_dir / "data_fingerprints.json", fingerprints)
    normalized_order = empirical_base_stock_normalized_order(train_products)
    design = {
        "policy": POLICY,
        "initializations": list(INITIALIZATIONS),
        "new_fit_count": 10,
        "reused_standard_fit_count": 10,
        "empirical_base_stock_normalized_order": normalized_order,
        "output_bias": inverse_softplus_scalar(normalized_order),
        "effective_zero_weight": EFFECTIVE_ZERO_WEIGHT,
        "exact_zero_not_used_reason": (
            "softplus(raw_weight)=0 requires raw_weight=-infinity and would permanently block gradients"
        ),
        "protected_test_accessed": False,
    }
    _write_or_validate_json(output_dir / "initialization_design.json", design)
    if args.dry_run:
        print(json.dumps(design, indent=2))
        return 0

    standard_rows = []
    trajectory_frames = []
    for seed in args.seeds:
        config, _ = resolve_no_imitation_config(base_config, args.profile, POLICY, seed)
        local_settings = _profile_settings(config, args.profile)
        model = _make_model(POLICY, _model_config(config, local_settings), seed)
        if not isinstance(model, ConstrainedMonotoneMLP):  # pragma: no cover
            raise TypeError("Factory did not return ConstrainedMonotoneMLP")
        initial = _initial_metrics(model, validation_products, feature_stats, local_settings, oracle_levels)
        record = _existing_standard_record(standard_root, seed, model, initial)
        standard_rows.append(record)
        standard_log = pd.read_csv(REPOSITORY_ROOT / record["source_run_dir"] / "training_log.csv")
        trajectory_frames.append(_trajectory_rows(standard_log, "standard_random", seed))

    failures = []
    started = time.perf_counter()
    completed = 0
    for seed in args.seeds:
        fit_dir = output_dir / "runs" / "base_stock_informed" / f"seed_{seed}"
        marker = fit_dir / "completion.json"
        if marker.exists() and args.resume:
            completed += 1
            record = json.loads(marker.read_text(encoding="utf-8"))
            print(f"completed={completed}/10 seed={seed} reused=true reward={record['mean_reward_per_period']:.6f}")
            continue
        if marker.exists():
            raise RuntimeError("Completed informed fits exist; pass --resume")
        try:
            record = _run_informed_fit(
                base_config=base_config,
                profile=args.profile,
                seed=seed,
                output_dir=output_dir,
                train_products=train_products,
                validation_products=validation_products,
                feature_stats=feature_stats,
                oracle_levels=oracle_levels,
                shared_data_fingerprint=shared,
            )
            completed += 1
            print(f"completed={completed}/10 seed={seed} reused=false reward={record['mean_reward_per_period']:.6f}")
        except Exception as exc:
            failures.append({"training_seed": seed, "error": repr(exc)})
            fit_dir.mkdir(parents=True, exist_ok=True)
            (fit_dir / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
            _atomic_write_csv(output_dir / "failures.csv", pd.DataFrame(failures))
            print(f"failed seed={seed} error={type(exc).__name__}")

    informed_rows = _scan_informed(output_dir)
    for record in informed_rows:
        log = pd.read_csv(
            output_dir / "runs" / "base_stock_informed" / f"seed_{record['training_seed']}" / "training_log.csv"
        )
        trajectory_frames.append(
            _trajectory_rows(log, "base_stock_informed", int(record["training_seed"]))
        )
    seed_results = pd.DataFrame(standard_rows + informed_rows).sort_values(
        ["initialization", "training_seed"]
    )
    trajectories = pd.concat(trajectory_frames, ignore_index=True)
    numeric = trajectories.select_dtypes(include=[np.number]).columns.tolist()
    trajectory_metrics = [
        column for column in numeric if column not in {"training_seed", "reward_update", "epoch"}
    ]
    trajectory_aggregate = trajectories.groupby(
        ["initialization", "reward_update"], as_index=False
    )[trajectory_metrics].agg(["mean", "std"])
    trajectory_aggregate.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in trajectory_aggregate.columns
    ]
    _atomic_write_csv(output_dir / "seed_results.csv", seed_results)
    _atomic_write_csv(output_dir / "aggregate_results.csv", _aggregate(seed_results))
    _atomic_write_csv(output_dir / "training_trajectory_seed_results.csv", trajectories)
    _atomic_write_csv(output_dir / "training_trajectory_aggregate.csv", trajectory_aggregate)
    _atomic_write_csv(
        output_dir / "failures.csv",
        pd.DataFrame(failures, columns=["training_seed", "error"]),
    )
    _atomic_write_json(
        output_dir / "analysis_completion.json",
        {
            "new_completed_fits": len(informed_rows),
            "expected_new_fits": 10,
            "reused_standard_fits": len(standard_rows),
            "failed_fits": len(failures),
            "wall_seconds": time.perf_counter() - started,
            "protected_test_accessed": False,
        },
    )
    return 0 if not failures and len(informed_rows) == 10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
