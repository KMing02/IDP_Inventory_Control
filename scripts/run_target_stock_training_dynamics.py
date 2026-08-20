"""Validation-only optimization diagnostic for TargetStockFCNN.

This exploratory runner never generates a protected split. It first audits
the existing 15-update artifacts and the fixed informed initial policy, then
optionally trains six fixed 120-update diagnostic runs. Optimizer-step hooks
observe gradients and parameters without altering the optimized objective.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Iterator, Sequence
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import stats
import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from inventory_control.idp_benchmark import (  # noqa: E402
    _model_config,
    _parameter_count,
    _profile_settings,
)
from inventory_control.paper_configs import load_paper_lost_sales_config  # noqa: E402
from inventory_control.paper_lost_sales import (  # noqa: E402
    non_omniscient_gamma_base_stock_policy,
)
from inventory_control.policies.neural.fcnn import (  # noqa: E402
    PaperFCNNFeatureStats,
    compute_paper_fcnn_feature_stats,
    make_paper_target_stock_fcnn_model,
)
from inventory_control.products import (  # noqa: E402
    PaperLostSalesProduct,
    PaperLostSalesProductGenerator,
)
from inventory_control.training.direct_backprop import (  # noqa: E402
    train_paper_lost_sales_fcnn,
)
from scripts.run_no_imitation_ablation import (  # noqa: E402
    _operational_metrics,
    resolve_no_imitation_config,
)
from scripts.run_target_stock_validation import (  # noqa: E402
    INITIALIZATION_MODES,
    POLICY_NAMES,
    _context_batch,
    _read_json,
    _target_stock_rows,
    verify_direct_baseline_reuse,
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


SEEDS = (0, 1, 2)
EVALUATION_UPDATES = (0, 1, 5, 15, 30, 60, 90, 120)
GRADIENT_UPDATES = (1, 2, 5, 15, 30, 60, 120)
EXISTING_ROOT = Path("outputs/idp_followup/target_stock_reward_only_validation")
DIRECT_BASELINE_ROOT = Path("outputs/idp_followup/no_imitation_matched_width_validation")
DEFAULT_OUTPUT_DIR = Path("outputs/idp_followup/target_stock_training_dynamics")
CORRELATION_TOLERANCE = 1e-12


def _historical_target_multiples(
    products: Sequence[PaperLostSalesProduct],
) -> np.ndarray:
    return np.asarray(
        [
            float(non_omniscient_gamma_base_stock_policy(product).S)
            / max(float(np.mean(product.historical_demand)), 1.0)
            for product in products
        ],
        dtype=float,
    )


def _target_summary(
    target_rows: pd.DataFrame,
    historical_multiples: np.ndarray,
    initial_multiples: np.ndarray,
) -> dict[str, float]:
    values = target_rows["normalized_target_stock"].to_numpy(dtype=float)
    if np.std(values) > CORRELATION_TOLERANCE and np.std(historical_multiples) > CORRELATION_TOLERANCE:
        correlation = float(stats.pearsonr(values, historical_multiples).statistic)
    else:
        correlation = np.nan
    return {
        "normalized_target_stock_mean": float(np.mean(values)),
        "normalized_target_stock_median": float(np.median(values)),
        "normalized_target_stock_across_product_sd": float(np.std(values, ddof=1)),
        "normalized_target_stock_minimum": float(np.min(values)),
        "normalized_target_stock_maximum": float(np.max(values)),
        "historical_gamma_multiple_rmse": float(
            np.sqrt(np.mean(np.square(values - historical_multiples)))
        ),
        "historical_gamma_multiple_pearson": correlation,
        "distance_from_initial_target_rmse": float(
            np.sqrt(np.mean(np.square(values - initial_multiples)))
        ),
        "distance_from_initial_target_mean_absolute": float(
            np.mean(np.abs(values - initial_multiples))
        ),
        "distance_from_initial_target_maximum_absolute": float(
            np.max(np.abs(values - initial_multiples))
        ),
        "non_finite_target_count": int((~np.isfinite(values)).sum()),
    }


def audit_existing_fits(
    *,
    existing_root: Path,
    output_dir: Path,
    validation_products: list[PaperLostSalesProduct],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Record only training dynamics already present in the 20 fit artifacts."""
    historical = _historical_target_multiples(validation_products)
    log_frames: list[pd.DataFrame] = []
    target_rows: list[dict[str, Any]] = []
    for mode in INITIALIZATION_MODES:
        policy = POLICY_NAMES[mode]
        for seed in range(10):
            fit_dir = existing_root / "runs" / policy / f"seed_{seed}"
            required = (
                "completion.json",
                "training_log.csv",
                "initial_target_stock_by_product.csv",
                "target_stock_by_product.csv",
            )
            if any(not (fit_dir / name).exists() for name in required):
                raise RuntimeError(f"Existing target-stock fit is incomplete: {fit_dir}")
            completion = _read_json(fit_dir / "completion.json")
            if (
                completion.get("reward_optimizer_updates") != 15
                or completion.get("imitation_optimizer_updates") != 0
            ):
                raise RuntimeError(f"Existing fit is not 15-update reward-only: {fit_dir}")
            log = pd.read_csv(fit_dir / "training_log.csv")
            if len(log) != 15 or not (log["phase"] == 1).all():
                raise RuntimeError(f"Existing training log is incompatible: {fit_dir}")
            keep = log[
                [
                    "epoch",
                    "training_loss",
                    "primary_loss",
                    "training_mean_total_reward",
                    "validation_mean_total_reward",
                    "raw_order_mean",
                    "deployed_order_mean",
                    "cap_hit_rate",
                ]
            ].copy()
            keep.insert(0, "training_seed", seed)
            keep.insert(0, "initialization_mode", mode)
            keep.insert(0, "policy_name", policy)
            keep["validation_reward_per_scored_period"] = (
                keep["validation_mean_total_reward"] / 100.0
            )
            keep["target_distribution_available"] = keep["epoch"].eq(15)
            log_frames.append(keep)

            initial = pd.read_csv(fit_dir / "initial_target_stock_by_product.csv")
            final = pd.read_csv(fit_dir / "target_stock_by_product.csv")
            initial_values = initial["normalized_target_stock"].to_numpy(dtype=float)
            for update, frame in ((0, initial), (15, final)):
                target_rows.append(
                    {
                        "policy_name": policy,
                        "initialization_mode": mode,
                        "training_seed": seed,
                        "reward_update": update,
                        "source_artifact_only": True,
                        **_target_summary(frame, historical, initial_values),
                    }
                )
    logs = pd.concat(log_frames, ignore_index=True)
    targets = pd.DataFrame(target_rows)
    _atomic_write_csv(output_dir / "existing_training_logs.csv", logs)
    _atomic_write_csv(output_dir / "existing_initial_final_target_dynamics.csv", targets)
    log_aggregate = (
        logs.groupby(["policy_name", "initialization_mode", "epoch"], as_index=False)
        .agg(
            seed_count=("training_seed", "count"),
            training_loss_mean=("training_loss", "mean"),
            training_loss_sample_sd=("training_loss", "std"),
            training_mean_total_reward_mean=("training_mean_total_reward", "mean"),
            training_mean_total_reward_sample_sd=("training_mean_total_reward", "std"),
            validation_reward_per_scored_period_mean=(
                "validation_reward_per_scored_period",
                "mean",
            ),
            validation_reward_per_scored_period_sample_sd=(
                "validation_reward_per_scored_period",
                "std",
            ),
        )
    )
    _atomic_write_csv(output_dir / "existing_training_log_aggregate.csv", log_aggregate)
    target_metrics = [
        column
        for column in targets.columns
        if column
        not in {
            "policy_name",
            "initialization_mode",
            "training_seed",
            "reward_update",
            "source_artifact_only",
        }
        and pd.api.types.is_numeric_dtype(targets[column])
    ]
    target_aggregate_rows: list[dict[str, Any]] = []
    for (policy, mode, update), group in targets.groupby(
        ["policy_name", "initialization_mode", "reward_update"], sort=False
    ):
        row: dict[str, Any] = {
            "policy_name": policy,
            "initialization_mode": mode,
            "reward_update": update,
            "seed_count": len(group),
        }
        for metric in target_metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sample_sd"] = float(group[metric].std(ddof=1))
        target_aggregate_rows.append(row)
    _atomic_write_csv(
        output_dir / "existing_initial_final_target_aggregate.csv",
        pd.DataFrame(target_aggregate_rows),
    )
    availability = pd.DataFrame(
        [
            {
                "metric": "training_loss_and_batch_reward",
                "available_updates": "1-15",
                "regenerated": False,
            },
            {
                "metric": "validation_reward",
                "available_updates": "5,10,15",
                "regenerated": False,
            },
            {
                "metric": "target_stock_distribution_and_historical_gamma_rmse",
                "available_updates": "0,15",
                "regenerated": False,
            },
            {
                "metric": "gradient_and_parameter_update_norms",
                "available_updates": "none",
                "regenerated": False,
            },
        ]
    )
    _atomic_write_csv(output_dir / "existing_artifact_metric_availability.csv", availability)
    return logs, targets


def evaluate_initial_informed_policy(
    *,
    base_config: dict[str, Any],
    profile: str,
    train_products: list[PaperLostSalesProduct],
    validation_products: list[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate the one frozen global informed scalar before optimization."""
    summary_path = output_dir / "initial_informed_policy_summary.json"
    metrics_path = output_dir / "initial_informed_policy_operational_metrics.csv"
    targets_path = output_dir / "initial_informed_policy_targets.csv"
    if summary_path.exists() and metrics_path.exists() and targets_path.exists():
        return _read_json(summary_path)
    config, _ = resolve_no_imitation_config(base_config, profile, "PaperFCNN", 0)
    settings = _profile_settings(config, profile)
    torch.manual_seed(0)
    model = make_paper_target_stock_fcnn_model(
        _model_config(config, settings),
        initialization_mode="target_stock_informed",
        training_products=train_products,
    )
    targets = _target_stock_rows(model, validation_products, feature_stats)
    operational = _operational_metrics(
        validation_products,
        "TargetStockInitialInformedHeuristic",
        model,
        feature_stats,
        settings,
    )
    cap_hits = int(operational["cap_hit_count"].sum())
    scored_actions = int(operational["scored_action_count"].sum())
    summary = {
        "policy_name": "TargetStockInitialInformedHeuristic",
        "reward_optimizer_updates": 0,
        "normalized_target_stock_scalar": float(
            targets["normalized_target_stock"].iloc[0]
        ),
        "mean_reward_per_period": float(operational["mean_reward_per_period"].mean()),
        "service_level": float(operational["service_level"].mean()),
        "lost_sales_rate": float(operational["lost_sales_rate"].mean()),
        "mean_inventory": float(operational["mean_inventory"].mean()),
        "mean_order_quantity": float(operational["mean_order_quantity"].mean()),
        "mean_holding_cost": float(operational["mean_holding_cost"].mean()),
        "cap_hit_rate": cap_hits / max(scored_actions, 1),
        "validation_product_count": len(validation_products),
        "evaluation_seed_start": int(operational["evaluation_seed"].min()),
        "evaluation_seed_end": int(operational["evaluation_seed"].max()),
    }
    _atomic_write_csv(metrics_path, operational)
    _atomic_write_csv(targets_path, targets)
    _atomic_write_json(summary_path, summary)
    return summary


def _parameter_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    final = model.mlp[-1]
    if not isinstance(final, nn.Linear):
        raise TypeError("TargetStockFCNN final readout must be linear")
    hidden_parameters = [
        parameter
        for module in list(model.mlp.children())[:-1]
        for parameter in module.parameters(recurse=True)
    ]
    return {
        "final_readout_weight": [final.weight],
        "final_readout_bias": [final.bias],
        "hidden_mlp": hidden_parameters,
        "history_encoder": list(model.history_encoder.parameters()),
    }


def _gradient_norm(parameters: Sequence[nn.Parameter]) -> float:
    squared = sum(
        float(torch.sum(parameter.grad.detach() ** 2).item())
        for parameter in parameters
        if parameter.grad is not None
    )
    return float(np.sqrt(squared))


def _update_norm(
    parameters: Sequence[nn.Parameter], before: Sequence[torch.Tensor]
) -> float:
    squared = sum(
        float(torch.sum((parameter.detach() - prior) ** 2).item())
        for parameter, prior in zip(parameters, before)
    )
    return float(np.sqrt(squared))


def _distance_norm(
    parameters: Sequence[nn.Parameter], initial: Sequence[torch.Tensor]
) -> float:
    return _update_norm(parameters, initial)


def _parameter_norm(parameters: Sequence[nn.Parameter]) -> float:
    squared = sum(float(torch.sum(parameter.detach() ** 2).item()) for parameter in parameters)
    return float(np.sqrt(squared))


@contextmanager
def optimizer_diagnostic_hook(
    *,
    model: nn.Module,
    evaluate_checkpoint: Any,
) -> Iterator[list[dict[str, Any]]]:
    """Observe Adam steps without modifying gradients, updates, or RNG."""
    groups = _parameter_groups(model)
    initial = {
        name: [parameter.detach().clone() for parameter in parameters]
        for name, parameters in groups.items()
    }
    original_step = torch.optim.Adam.step
    records: list[dict[str, Any]] = []
    update = 0

    def observed_step(optimizer: torch.optim.Adam, closure=None):
        nonlocal update
        update += 1
        before = {
            name: [parameter.detach().clone() for parameter in parameters]
            for name, parameters in groups.items()
        }
        gradients = {name: _gradient_norm(parameters) for name, parameters in groups.items()}
        result = original_step(optimizer, closure=closure)
        if update in GRADIENT_UPDATES:
            for name, parameters in groups.items():
                records.append(
                    {
                        "reward_update": update,
                        "parameter_group": name,
                        "gradient_norm": gradients[name],
                        "parameter_update_norm": _update_norm(parameters, before[name]),
                        "cumulative_parameter_change_norm": _distance_norm(
                            parameters, initial[name]
                        ),
                        "parameter_norm": _parameter_norm(parameters),
                    }
                )
        if update in EVALUATION_UPDATES:
            evaluate_checkpoint(update)
        return result

    with patch.object(torch.optim.Adam, "step", observed_step):
        yield records


def _checkpoint_metrics(
    *,
    model: nn.Module,
    policy_name: str,
    initialization_mode: str,
    seed: int,
    update: int,
    validation_products: list[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
    settings: dict[str, Any],
    historical_multiples: np.ndarray,
    initial_multiples: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    was_training = model.training
    targets = _target_stock_rows(model, validation_products, feature_stats)
    operational = _operational_metrics(
        validation_products, policy_name, model, feature_stats, settings
    )
    if was_training:
        model.train()
    cap_hits = int(operational["cap_hit_count"].sum())
    scored_actions = int(operational["scored_action_count"].sum())
    row = {
        "policy_name": policy_name,
        "initialization_mode": initialization_mode,
        "training_seed": seed,
        "reward_update": update,
        "mean_reward_per_period": float(operational["mean_reward_per_period"].mean()),
        "service_level": float(operational["service_level"].mean()),
        "lost_sales_rate": float(operational["lost_sales_rate"].mean()),
        "mean_inventory": float(operational["mean_inventory"].mean()),
        "mean_order_quantity": float(operational["mean_order_quantity"].mean()),
        "mean_holding_cost": float(operational["mean_holding_cost"].mean()),
        "cap_hit_rate": cap_hits / max(scored_actions, 1),
        **_target_summary(targets, historical_multiples, initial_multiples),
    }
    targets = targets.copy()
    targets.insert(0, "reward_update", update)
    targets.insert(0, "training_seed", seed)
    targets.insert(0, "initialization_mode", initialization_mode)
    targets.insert(0, "policy_name", policy_name)
    return row, targets


def _extended_fit_dir(output_dir: Path, mode: str, seed: int) -> Path:
    return output_dir / "runs" / POLICY_NAMES[mode] / f"seed_{seed}"


def run_extended_fit(
    *,
    base_config: dict[str, Any],
    profile: str,
    mode: str,
    seed: int,
    output_dir: Path,
    train_products: list[PaperLostSalesProduct],
    validation_products: list[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
    shared_data_fingerprint: str,
) -> dict[str, Any]:
    config, _ = resolve_no_imitation_config(base_config, profile, "PaperFCNN", seed)
    config["idp_benchmark"][profile]["reward_epochs"] = 120
    settings = _profile_settings(config, profile)
    policy_name = POLICY_NAMES[mode]
    resolved = {
        "policy_name": policy_name,
        "initialization_mode": mode,
        "training_seed": seed,
        "action_parameterization": "target_stock",
        "hidden_layers": 2,
        "hidden_units": 32,
        "history_encoder": "causal_cnn",
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "reward_optimizer_updates": 120,
        "imitation_optimizer_updates": 0,
        "training_products": 128,
        "validation_products": 32,
        "training_horizon": 50,
        "evaluation_horizon": 120,
        "burn_in": 20,
        "deployment_action_cap_factor": 4.0,
        "normalize_reward_loss": True,
        "checkpoint_updates": list(EVALUATION_UPDATES),
        "gradient_diagnostic_updates": list(GRADIENT_UPDATES),
        "checkpoint_selection": None,
        "early_stopping": False,
        "shared_data_fingerprint": shared_data_fingerprint,
    }
    resolved["resolved_config_fingerprint"] = _fingerprint(resolved)
    fit_dir = _extended_fit_dir(output_dir, mode, seed)
    fit_dir.mkdir(parents=True, exist_ok=True)
    (fit_dir / "completion.json").unlink(missing_ok=True)
    _atomic_write_json(fit_dir / "resolved_config.json", resolved)

    torch.manual_seed(seed)
    model = make_paper_target_stock_fcnn_model(
        _model_config(config, settings),
        initialization_mode=mode,
        training_products=train_products if mode == "target_stock_informed" else None,
    )
    initial_target_rows = _target_stock_rows(model, validation_products, feature_stats)
    initial_multiples = initial_target_rows["normalized_target_stock"].to_numpy(dtype=float)
    historical_multiples = _historical_target_multiples(validation_products)
    checkpoint_rows: list[dict[str, Any]] = []
    target_frames: list[pd.DataFrame] = []

    def evaluate(update: int) -> None:
        row, targets = _checkpoint_metrics(
            model=model,
            policy_name=policy_name,
            initialization_mode=mode,
            seed=seed,
            update=update,
            validation_products=validation_products,
            feature_stats=feature_stats,
            settings=settings,
            historical_multiples=historical_multiples,
            initial_multiples=initial_multiples,
        )
        checkpoint_rows.append(row)
        target_frames.append(targets)

    evaluate(0)
    started = time.perf_counter()
    with optimizer_diagnostic_hook(model=model, evaluate_checkpoint=evaluate) as gradient_rows:
        result = train_paper_lost_sales_fcnn(
            train_products,
            model,
            horizon=int(settings["training_horizon"]),
            batch_size=min(int(settings["batch_size"]), len(train_products)),
            epochs=120,
            learning_rate=float(config["learning_rate"]),
            seed=int(settings["seed"]),
            imitation_epochs=0,
            validation_products=None,
            validation_interval=0,
            validation_callback=None,
            feature_stats=feature_stats,
            max_order_factor=float(settings["max_order_factor"]),
            scale_action_by_history_mean=bool(settings["scale_action_by_history_mean"]),
            normalize_reward_loss=bool(settings["normalize_reward_loss"]),
            monotonicity_penalty_weight=0.0,
            monotonicity_samples_per_product=int(
                settings["monotonicity_samples_per_product"]
            ),
            include_terminal_salvage=True,
            clip_imitation_predictions=True,
            imitation_early_stopping=False,
            imitation_max_epochs=None,
            track_imitation_validation=False,
            reset_optimizer_after_imitation=False,
            adam_kwargs=dict(config.get("adam", {})),
        )
    wall_seconds = time.perf_counter() - started
    training_log = pd.DataFrame(result.training_log or [])
    if (
        len(training_log) != 120
        or not (training_log["phase"] == 1).all()
        or result.imitation_epochs_completed != 0
        or result.best_state_dict is not None
    ):
        raise RuntimeError("Extended fit did not remain a 120-update reward-only run")
    checkpoints = pd.DataFrame(checkpoint_rows).sort_values("reward_update")
    checkpoint_training = training_log.loc[
        training_log["epoch"].isin(EVALUATION_UPDATES),
        ["epoch", "training_loss", "training_mean_total_reward"],
    ].rename(columns={"epoch": "reward_update"})
    checkpoints = checkpoints.merge(checkpoint_training, on="reward_update", how="left")
    gradients = pd.DataFrame(gradient_rows)
    gradients.insert(0, "training_seed", seed)
    gradients.insert(0, "initialization_mode", mode)
    gradients.insert(0, "policy_name", policy_name)
    target_predictions = pd.concat(target_frames, ignore_index=True)
    _atomic_write_csv(fit_dir / "training_log.csv", training_log)
    _atomic_write_csv(fit_dir / "checkpoint_metrics.csv", checkpoints)
    _atomic_write_csv(fit_dir / "gradient_diagnostics.csv", gradients)
    _atomic_write_csv(fit_dir / "checkpoint_target_stocks.csv", target_predictions)
    _atomic_torch_save(fit_dir / "model_update_120.pt", model.state_dict())
    completion = {
        "policy_name": policy_name,
        "initialization_mode": mode,
        "training_seed": seed,
        "resolved_config_fingerprint": resolved["resolved_config_fingerprint"],
        "shared_data_fingerprint": shared_data_fingerprint,
        "parameter_count": _parameter_count(model),
        "reward_optimizer_updates": 120,
        "imitation_optimizer_updates": 0,
        "checkpoint_selection_used": False,
        "early_stopping_used": False,
        "checkpoint_count": len(checkpoints),
        "gradient_record_count": len(gradients),
        "training_seconds": float(result.elapsed_seconds),
        "fit_wall_seconds": float(wall_seconds),
        "failed_fit": False,
    }
    _atomic_write_json(fit_dir / "completion.json", completion)
    return completion


def _load_completed(fit_dir: Path, expected: str) -> dict[str, Any] | None:
    required = (
        "completion.json",
        "training_log.csv",
        "checkpoint_metrics.csv",
        "gradient_diagnostics.csv",
        "checkpoint_target_stocks.csv",
        "model_update_120.pt",
    )
    if any(not (fit_dir / name).exists() for name in required):
        return None
    completion = _read_json(fit_dir / "completion.json")
    if completion.get("resolved_config_fingerprint") != expected:
        raise RuntimeError(f"Incompatible completed diagnostic fit: {fit_dir}")
    return completion


def aggregate_extended_results(output_dir: Path) -> None:
    checkpoint_frames = [
        pd.read_csv(path)
        for path in sorted((output_dir / "runs").glob("*/seed_*/checkpoint_metrics.csv"))
    ]
    gradient_frames = [
        pd.read_csv(path)
        for path in sorted((output_dir / "runs").glob("*/seed_*/gradient_diagnostics.csv"))
    ]
    checkpoints = pd.concat(checkpoint_frames, ignore_index=True)
    gradients = pd.concat(gradient_frames, ignore_index=True)
    _atomic_write_csv(output_dir / "extended_checkpoint_seed_metrics.csv", checkpoints)
    _atomic_write_csv(output_dir / "extended_gradient_seed_metrics.csv", gradients)

    checkpoint_metrics = [
        column
        for column in checkpoints.columns
        if column
        not in {
            "policy_name",
            "initialization_mode",
            "training_seed",
            "reward_update",
        }
        and pd.api.types.is_numeric_dtype(checkpoints[column])
    ]
    aggregate_rows: list[dict[str, Any]] = []
    for (policy, mode, update), group in checkpoints.groupby(
        ["policy_name", "initialization_mode", "reward_update"], sort=False
    ):
        row: dict[str, Any] = {
            "policy_name": policy,
            "initialization_mode": mode,
            "reward_update": update,
            "seed_count": len(group),
        }
        for metric in checkpoint_metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sample_sd"] = float(group[metric].std(ddof=1))
        aggregate_rows.append(row)
    _atomic_write_csv(
        output_dir / "extended_checkpoint_aggregate.csv", pd.DataFrame(aggregate_rows)
    )

    gradient_metrics = (
        "gradient_norm",
        "parameter_update_norm",
        "cumulative_parameter_change_norm",
        "parameter_norm",
    )
    gradient_rows: list[dict[str, Any]] = []
    for (policy, mode, update, parameter_group), group in gradients.groupby(
        ["policy_name", "initialization_mode", "reward_update", "parameter_group"],
        sort=False,
    ):
        row = {
            "policy_name": policy,
            "initialization_mode": mode,
            "reward_update": update,
            "parameter_group": parameter_group,
            "seed_count": len(group),
        }
        for metric in gradient_metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sample_sd"] = float(group[metric].std(ddof=1))
        gradient_rows.append(row)
    _atomic_write_csv(
        output_dir / "extended_gradient_aggregate.csv", pd.DataFrame(gradient_rows)
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "extended"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", default="full")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--existing-root", type=Path, default=EXISTING_ROOT)
    parser.add_argument("--direct-baseline-root", type=Path, default=DIRECT_BASELINE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if tuple(args.seeds) != SEEDS:
        raise ValueError("Extended target-stock diagnostic requires exactly seeds 0 1 2")
    validate_split_and_output(args.split, args.output_dir)
    config_path, config_relative = _repository_relative_path(args.config, "--config")
    existing_root, existing_relative = _repository_relative_path(
        args.existing_root, "--existing-root"
    )
    direct_root, direct_relative = _repository_relative_path(
        args.direct_baseline_root, "--direct-baseline-root"
    )
    output_dir, _ = _repository_relative_path(args.output_dir, "--output-dir")
    if existing_relative != EXISTING_ROOT or direct_relative != DIRECT_BASELINE_ROOT:
        raise ValueError("Diagnostic source roots must be the frozen validation-only outputs")

    base_config = load_paper_lost_sales_config(config_path)
    config, _ = resolve_no_imitation_config(base_config, args.profile, "PaperFCNN", 0)
    settings = _profile_settings(config, args.profile)
    generator = PaperLostSalesProductGenerator(config)
    train_products = _safe_generate_products(generator, 128, "train")
    validation_products = _safe_generate_products(generator, 32, "validation")
    feature_stats = compute_paper_fcnn_feature_stats(train_products)
    fingerprints = data_fingerprints(config, settings, train_products, validation_products)
    shared_data_fingerprint = _fingerprint(fingerprints)
    verify_direct_baseline_reuse(
        baseline_root=direct_root,
        base_config=base_config,
        profile=args.profile,
        train_products=train_products,
        validation_products=validation_products,
        feature_stats=feature_stats,
        fingerprints=fingerprints,
    )
    existing_fingerprints = _read_json(existing_root / "data_fingerprints.json")
    if existing_fingerprints.get("shared_data_fingerprint") != shared_data_fingerprint:
        raise RuntimeError("Existing target-stock fit data fingerprint differs")

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_dir / "environment.json", _environment_record(config_relative, config))
    fingerprint_record = dict(fingerprints)
    fingerprint_record["shared_data_fingerprint"] = shared_data_fingerprint
    _atomic_write_json(output_dir / "data_fingerprints.json", fingerprint_record)
    audit_existing_fits(
        existing_root=existing_root,
        output_dir=output_dir,
        validation_products=validation_products,
    )
    initial_summary = evaluate_initial_informed_policy(
        base_config=base_config,
        profile=args.profile,
        train_products=train_products,
        validation_products=validation_products,
        feature_stats=feature_stats,
        output_dir=output_dir,
    )
    _atomic_write_json(
        output_dir / "audit_completion.json",
        {
            "existing_fit_count": 20,
            "initial_policy_evaluated": True,
            "initial_policy_reward": initial_summary["mean_reward_per_period"],
            "protected_test_accessed": False,
            "new_optimizer_updates": 0 if args.stage == "audit" else None,
        },
    )
    if args.stage == "audit":
        print(
            f"existing_fits_audited=20 initial_policy_reward="
            f"{initial_summary['mean_reward_per_period']:.6f} optimizer_updates=0"
        )
        return 0

    manifest = pd.DataFrame(
        [
            {
                "policy_name": POLICY_NAMES[mode],
                "initialization_mode": mode,
                "training_seed": seed,
                "reward_optimizer_updates": 120,
                "evaluation_updates": ",".join(map(str, EVALUATION_UPDATES)),
                "gradient_updates": ",".join(map(str, GRADIENT_UPDATES)),
                "checkpoint_selection": False,
                "early_stopping": False,
            }
            for mode in INITIALIZATION_MODES
            for seed in SEEDS
        ]
    )
    _atomic_write_csv(output_dir / "extended_manifest.csv", manifest)
    started = time.perf_counter()
    completed = 0
    failures: list[dict[str, Any]] = []
    for mode in INITIALIZATION_MODES:
        for seed in SEEDS:
            resolved = {
                "policy_name": POLICY_NAMES[mode],
                "initialization_mode": mode,
                "training_seed": seed,
                "action_parameterization": "target_stock",
                "hidden_layers": 2,
                "hidden_units": 32,
                "history_encoder": "causal_cnn",
                "learning_rate": 0.001,
                "optimizer": "Adam",
                "reward_optimizer_updates": 120,
                "imitation_optimizer_updates": 0,
                "training_products": 128,
                "validation_products": 32,
                "training_horizon": 50,
                "evaluation_horizon": 120,
                "burn_in": 20,
                "deployment_action_cap_factor": 4.0,
                "normalize_reward_loss": True,
                "checkpoint_updates": list(EVALUATION_UPDATES),
                "gradient_diagnostic_updates": list(GRADIENT_UPDATES),
                "checkpoint_selection": None,
                "early_stopping": False,
                "shared_data_fingerprint": shared_data_fingerprint,
            }
            expected = _fingerprint(resolved)
            fit_dir = _extended_fit_dir(output_dir, mode, seed)
            existing = _load_completed(fit_dir, expected) if fit_dir.exists() else None
            if existing is not None and args.resume:
                completed += 1
                print(
                    f"completed={completed}/6 policy={POLICY_NAMES[mode]} seed={seed} reused=true"
                )
                continue
            if existing is not None:
                raise RuntimeError("Completed diagnostic fits exist; pass --resume")
            try:
                run_extended_fit(
                    base_config=base_config,
                    profile=args.profile,
                    mode=mode,
                    seed=seed,
                    output_dir=output_dir,
                    train_products=train_products,
                    validation_products=validation_products,
                    feature_stats=feature_stats,
                    shared_data_fingerprint=shared_data_fingerprint,
                )
                completed += 1
                print(
                    f"completed={completed}/6 policy={POLICY_NAMES[mode]} seed={seed} reused=false"
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
    markers = list((output_dir / "runs").glob("*/seed_*/completion.json"))
    if markers:
        aggregate_extended_results(output_dir)
    _atomic_write_csv(
        output_dir / "extended_failures.csv",
        pd.DataFrame(
            failures,
            columns=["policy_name", "initialization_mode", "training_seed", "error"],
        ),
    )
    _atomic_write_json(
        output_dir / "extended_completion.json",
        {
            "completed_fits": len(markers),
            "expected_fits": 6,
            "failed_fits": len(failures),
            "reward_optimizer_updates_per_fit": 120,
            "imitation_optimizer_updates": 0,
            "checkpoint_selection_used": False,
            "early_stopping_used": False,
            "wall_seconds": time.perf_counter() - started,
            "protected_test_accessed": False,
        },
    )
    return 0 if len(markers) == 6 and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
