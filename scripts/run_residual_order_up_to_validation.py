"""Validation-only direct versus residual order-up-to parameterization diagnostic."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from inventory_control.idp_benchmark import (  # noqa: E402
    _model_config,
    _operational_table,
    _oracle_levels,
    _parameter_count,
    _prediction_summary,
    _prediction_table,
    _profile_settings,
    _train_model,
)
from inventory_control.paper_configs import load_paper_lost_sales_config  # noqa: E402
from inventory_control.paper_lost_sales import non_omniscient_gamma_base_stock_policy  # noqa: E402
from inventory_control.policies.neural.constrained_mlp import (  # noqa: E402
    make_paper_residual_order_up_to_constrained_mlp_model,
)
from inventory_control.policies.neural.fcnn import (  # noqa: E402
    compute_paper_fcnn_feature_stats,
    make_paper_residual_order_up_to_fcnn_model,
)
from inventory_control.products import PaperLostSalesProduct, PaperLostSalesProductGenerator  # noqa: E402
from inventory_control.training.direct_backprop import (  # noqa: E402
    paper_lost_sales_context_features,
    paper_lost_sales_target_level,
)
from scripts.tune_primary_neural_policies import (  # noqa: E402
    VALIDATION_GRID_SEED,
    _atomic_torch_save,
    _atomic_write_csv,
    _atomic_write_json,
    _completion_record,
    _environment_record,
    _fingerprint,
    _fixed_validation_grid_seed,
    _repository_relative_path,
    _safe_generate_products,
    data_fingerprints,
    validate_split_and_output,
)


RESIDUAL_POLICIES = (
    "PaperFCNNResidualOrderUpTo",
    "ConstrainedMLPResidualOrderUpTo",
)
DIRECT_SOURCE_POLICIES = ("PaperFCNN", "ConstrainedMonoMLP")
SEEDS = (0, 1, 2)
FCNN_COMPARISON_CENTER_TOLERANCE = 0.10
ZERO_RESIDUAL_FLOAT32_TOLERANCE = 1e-6
EXTREME_ACTION_FACTOR = 8.0


def _resolved_config(base: dict[str, Any], profile: str, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    config = copy.deepcopy(base)
    config["learning_rate"] = 0.001
    config["fcnn"]["hidden_layers"] = 2
    config["fcnn"]["hidden_units"] = 32
    benchmark = config["idp_benchmark"]
    benchmark["seed"] = int(seed)
    benchmark["hidden_layers"] = 2
    benchmark["hidden_units"] = 32
    benchmark["imitation_early_stopping"] = True
    benchmark["imitation_max_epochs"] = 100
    benchmark["imitation_patience"] = 10
    benchmark["imitation_min_delta"] = 0.5
    benchmark["reset_optimizer_after_imitation"] = False
    benchmark[profile]["training_products"] = 128
    benchmark[profile]["validation_products"] = 32
    benchmark[profile]["reward_epochs"] = 15
    config["constrained_monotone_mlp"]["activation_weights"] = [1.0, 1.0, 1.0]
    settings = _profile_settings(config, profile)
    if float(settings["max_order_factor"]) != 4.0:
        raise RuntimeError("Residual validation diagnostic requires deployment cap F=4")
    return config, settings


def _make_model(policy: str, model_config: dict[str, Any], seed: int) -> torch.nn.Module:
    torch.manual_seed(int(seed))
    if policy == "PaperFCNNResidualOrderUpTo":
        return make_paper_residual_order_up_to_fcnn_model(model_config)
    if policy == "ConstrainedMLPResidualOrderUpTo":
        return make_paper_residual_order_up_to_constrained_mlp_model(model_config)
    raise ValueError(policy)


def _context_batch(products: Sequence[PaperLostSalesProduct], feature_stats: Any) -> dict[str, torch.Tensor]:
    return {
        "history": torch.tensor(
            [[float(value) / feature_stats.demand_scale for value in product.historical_demand] for product in products],
            dtype=torch.float32,
        ),
        "feature_price_scale": torch.full((len(products),), float(feature_stats.price_scale)),
        "historical_mean_demand": torch.tensor(
            [max(float(np.mean(product.historical_demand)), 1.0) for product in products], dtype=torch.float32
        ),
        "historical_base_stock_level": torch.tensor(
            [non_omniscient_gamma_base_stock_policy(product).S for product in products], dtype=torch.float32
        ),
        "price": torch.tensor([product.price for product in products], dtype=torch.float32),
        "purchase_cost": torch.tensor([product.purchase_cost for product in products], dtype=torch.float32),
        "holding_cost": torch.tensor([product.holding_cost for product in products], dtype=torch.float32),
        "lost_sale_penalty": torch.tensor([product.lost_sale_penalty for product in products], dtype=torch.float32),
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "standard_deviation": float(np.std(values, ddof=0)),
        "p5": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def initialization_audit(
    base_config: dict[str, Any],
    profile: str,
    train_products: list[PaperLostSalesProduct],
    validation_products: list[PaperLostSalesProduct],
    feature_stats: Any,
) -> tuple[pd.DataFrame, bool]:
    rows: list[dict[str, Any]] = []
    passed = True
    for seed in SEEDS:
        config, settings = _resolved_config(base_config, profile, seed)
        del settings
        model_config = _model_config(config, _profile_settings(config, profile))
        for policy in RESIDUAL_POLICIES:
            model = _make_model(policy, model_config, seed)
            model.eval()
            for split, products in (("train", train_products), ("validation", validation_products)):
                batch = _context_batch(products, feature_stats)
                with torch.no_grad():
                    context = paper_lost_sales_context_features(batch)
                    preactivation = model.preactivation(context)
                    output = model(context)
                    target = paper_lost_sales_target_level(model, batch)
                correction = target - batch["historical_base_stock_level"]
                normalized = correction / batch["historical_mean_demand"]
                output_values = output.numpy()
                correction_values = correction.numpy()
                normalized_values = normalized.numpy()
                preactivation_values = preactivation.numpy()
                if policy == "ConstrainedMLPResidualOrderUpTo":
                    centered = bool(
                        np.max(np.abs(preactivation_values)) <= ZERO_RESIDUAL_FLOAT32_TOLERANCE
                        and np.max(np.abs(output_values - 1.0)) <= ZERO_RESIDUAL_FLOAT32_TOLERANCE
                        and np.max(np.abs(correction_values)) <= ZERO_RESIDUAL_FLOAT32_TOLERANCE
                        and np.max(np.abs(normalized_values)) <= ZERO_RESIDUAL_FLOAT32_TOLERANCE
                    )
                    passed = passed and centered
                    gate_applicable = True
                else:
                    centered = bool(
                        abs(float(np.mean(output_values)) - 1.0) <= FCNN_COMPARISON_CENTER_TOLERANCE
                        and abs(float(np.median(output_values)) - 1.0) <= FCNN_COMPARISON_CENTER_TOLERANCE
                    )
                    gate_applicable = False
                for metric, values in (
                    ("z_theta", preactivation_values),
                    ("u_theta", output_values),
                    ("target_stock_minus_anchor", correction_values),
                    ("normalized_residual", normalized_values),
                ):
                    rows.append(
                        {
                            "policy_name": policy,
                            "training_seed": seed,
                            "split": split,
                            "metric": metric,
                            "observation_count": len(values),
                            "zero_residual_gate_applicable": gate_applicable,
                            "center_gate_passed": centered,
                            **_distribution(values),
                        }
                    )
    return pd.DataFrame(rows), passed


def _fit_dir(output_dir: Path, policy: str, seed: int) -> Path:
    return output_dir / "runs" / policy / f"seed_{seed}"


def _fit_complete(path: Path) -> dict[str, Any] | None:
    return _completion_record(path)


def _run_residual_fit(
    *,
    base_config: dict[str, Any],
    profile: str,
    policy: str,
    seed: int,
    output_dir: Path,
    output_relative: Path,
    train_products: list[PaperLostSalesProduct],
    validation_products: list[PaperLostSalesProduct],
    feature_stats: Any,
    oracle_levels: dict[str, float],
    shared_fingerprint: str,
) -> dict[str, Any]:
    config, settings = _resolved_config(base_config, profile, seed)
    model_config = _model_config(config, settings)
    model = _make_model(policy, model_config, seed)
    fit_dir = _fit_dir(output_dir, policy, seed)
    fit_dir.mkdir(parents=True, exist_ok=True)
    (fit_dir / "completion.json").unlink(missing_ok=True)
    resolved = {
        "policy_name": policy,
        "training_seed": seed,
        "split": "validation",
        "hidden_layers": 2,
        "hidden_units": 32,
        "learning_rate": 0.001,
        "reward_epochs": 15,
        "activation_weights": [1, 1, 1] if policy == "ConstrainedMLPResidualOrderUpTo" else None,
        "action_parameterization": "residual_order_up_to",
        "residual_anchor": "historical_gamma_fit",
        "residual_output_bounds": [0.0, 2.0],
        "residual_center": 1.0,
        "residual_scale": 0.5,
        "inventory_network_input": False,
        "deployment_action_cap_factor": 4.0,
        "imitation_semantics": "deployed_prediction_vs_capped_target",
        "imitation_checkpoint_metric": "deployed_imitation_rmse",
        "validation_grid_seed": VALIDATION_GRID_SEED,
        "shared_data_fingerprint": shared_fingerprint,
    }
    resolved["resolved_config_fingerprint"] = _fingerprint(resolved)
    _atomic_write_json(fit_dir / "resolved_config.json", resolved)

    started = time.perf_counter()
    with _fixed_validation_grid_seed(validation_products):
        result = _train_model(
            config,
            settings,
            train_products,
            validation_products,
            feature_stats,
            model,
            policy,
            "validation",
            clip_imitation_predictions=True,
        )
    elapsed = time.perf_counter() - started
    if result.imitation_selection_metric != "deployed_imitation_rmse":
        raise RuntimeError("residual fit did not use deployed imitation RMSE checkpoint selection")
    if result.best_imitation_state_dict is None or result.best_imitation_epoch is None:
        raise RuntimeError("residual fit did not save a validation-best imitation checkpoint")

    predictions = _prediction_table(validation_products, {policy: model}, feature_stats, settings, oracle_levels)
    predictions = predictions.loc[predictions["policy_name"] == policy].reset_index(drop=True)
    prediction_diagnostics = _prediction_summary(predictions)
    operational = _operational_table(
        validation_products, {policy: model}, feature_stats, settings, oracle_levels, "validation"
    )
    operational = operational.loc[operational["policy_name"] == policy].reset_index(drop=True)
    operational_summary = operational.groupby("policy_name", as_index=False).mean(numeric_only=True)
    training_log = pd.DataFrame(result.training_log or [])
    final_batch = _context_batch(validation_products, feature_stats)
    model.eval()
    with torch.no_grad():
        final_context = paper_lost_sales_context_features(final_batch)
        final_output = model(final_context).detach().cpu().numpy()
    final_normalized_residual = 0.5 * (final_output - 1.0)
    final_distribution = pd.DataFrame(
        [
            {"metric": "u_theta", **_distribution(final_output)},
            {"metric": "normalized_residual", **_distribution(final_normalized_residual)},
        ]
    )
    gradient_evidence: dict[str, Any] = {
        "tracked": policy == "ConstrainedMLPResidualOrderUpTo",
        "first_step_backbone_gradient_norm": np.nan,
        "first_step_backbone_update_norm": np.nan,
        "first_step_readout_gradient_norm": np.nan,
        "first_step_readout_update_norm": np.nan,
        "first_step_readout_weight_norm": np.nan,
        "maximum_later_backbone_gradient_norm": np.nan,
        "maximum_later_backbone_update_norm": np.nan,
        "first_nonzero_backbone_gradient_epoch": None,
        "backbone_became_trainable_after_first_step": None,
        "final_readout_weight_norm": np.nan,
    }
    if policy == "ConstrainedMLPResidualOrderUpTo":
        first = training_log.iloc[0]
        later = training_log.iloc[1:]
        nonzero = later.loc[later["residual_backbone_gradient_norm"] > 1e-12]
        gradient_evidence.update(
            {
                "first_step_backbone_gradient_norm": float(first["residual_backbone_gradient_norm"]),
                "first_step_backbone_update_norm": float(first["residual_backbone_update_norm"]),
                "first_step_readout_gradient_norm": float(first["residual_readout_gradient_norm"]),
                "first_step_readout_update_norm": float(first["residual_readout_update_norm"]),
                "first_step_readout_weight_norm": float(first["residual_readout_weight_norm"]),
                "maximum_later_backbone_gradient_norm": float(later["residual_backbone_gradient_norm"].max()),
                "maximum_later_backbone_update_norm": float(later["residual_backbone_update_norm"].max()),
                "first_nonzero_backbone_gradient_epoch": (
                    int(nonzero.iloc[0]["epoch"]) if not nonzero.empty else None
                ),
                "backbone_became_trainable_after_first_step": bool(
                    (later["residual_backbone_gradient_norm"] > 1e-12).any()
                    and (later["residual_backbone_update_norm"] > 1e-12).any()
                ),
                "final_readout_weight_norm": float(
                    torch.linalg.vector_norm(model.residual_readout.weight.detach()).item()
                ),
            }
        )
    fit_result = {
        "policy_name": policy,
        "training_seed": seed,
        "source_run_dir": (output_relative / "runs" / policy / f"seed_{seed}").as_posix(),
        "shared_data_fingerprint": shared_fingerprint,
        "resolved_config_fingerprint": resolved["resolved_config_fingerprint"],
        "parameter_count": _parameter_count(model),
        "training_seconds": float(result.elapsed_seconds),
        "fit_wall_seconds": float(elapsed),
        "imitation_epochs_run": int(result.imitation_epochs_completed),
        "selected_best_imitation_epoch": int(result.best_imitation_epoch),
        "best_validation_imitation_mae": result.best_imitation_validation_mae,
        "best_validation_imitation_rmse": result.best_imitation_validation_rmse,
        "imitation_checkpoint_metric": result.imitation_selection_metric,
        "best_imitation_checkpoint_restored_before_reward_training": True,
        "final_u_theta_mean": float(np.mean(final_output)),
        "final_u_theta_standard_deviation": float(np.std(final_output, ddof=0)),
        "final_normalized_residual_mean": float(np.mean(final_normalized_residual)),
        "final_normalized_residual_standard_deviation": float(np.std(final_normalized_residual, ddof=0)),
        **gradient_evidence,
    }
    _atomic_write_csv(fit_dir / "training_log.csv", training_log)
    _atomic_write_json(
        fit_dir / "best_imitation_checkpoint.json",
        {
            "selected_epoch": result.best_imitation_epoch,
            "validation_mae": result.best_imitation_validation_mae,
            "validation_rmse": result.best_imitation_validation_rmse,
            "selection_metric": result.imitation_selection_metric,
            "epochs_run": result.imitation_epochs_completed,
            "restored_before_reward_training": True,
        },
    )
    _atomic_write_csv(fit_dir / "operational_metrics.csv", operational)
    _atomic_write_csv(fit_dir / "operational_summary.csv", operational_summary)
    _atomic_write_csv(fit_dir / "predictions.csv", predictions)
    _atomic_write_csv(fit_dir / "prediction_diagnostics.csv", prediction_diagnostics)
    _atomic_write_csv(fit_dir / "final_residual_distribution.csv", final_distribution)
    _atomic_write_json(fit_dir / "residual_gradient_evidence.json", gradient_evidence)
    _atomic_torch_save(fit_dir / "model.pt", model.state_dict())
    _atomic_torch_save(fit_dir / "best_imitation.pt", result.best_imitation_state_dict)
    _atomic_write_json(fit_dir / "fit_result.json", fit_result)
    _atomic_write_json(fit_dir / "completion.json", fit_result)
    return fit_result


def _direct_source(matched_root: Path, policy: str, seed: int) -> Path:
    record_path = matched_root / "records" / policy / "matched_width_depth" / f"seed_{seed}" / "record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    source = REPOSITORY_ROOT / record["source_run_dir"]
    if _completion_record(source) is None:
        raise RuntimeError(f"incomplete direct-order source fit: {source}")
    return source


def _seed_summary(policy: str, seed: int, source: Path, parameter_count: int | None = None) -> dict[str, Any]:
    operational = pd.read_csv(source / "operational_metrics.csv")
    predictions = pd.read_csv(source / "predictions.csv")
    diagnostics = pd.read_csv(source / "prediction_diagnostics.csv").iloc[0]
    completion = json.loads((source / "completion.json").read_text(encoding="utf-8"))
    clipping = predictions["raw_order"].to_numpy(dtype=float) - predictions["deployed_order"].to_numpy(dtype=float)
    raw = predictions["raw_order"].to_numpy(dtype=float)
    historical_mean = predictions["historical_mean_demand"].to_numpy(dtype=float)
    extreme = (~np.isfinite(raw)) | (raw < -1e-6) | (raw > EXTREME_ACTION_FACTOR * historical_mean)
    zero = predictions.loc[predictions["state_index"] == 0]
    target_errors = zero["raw_order_up_to_level"].to_numpy(dtype=float) - zero[
        "raw_omniscient_order_up_to_level"
    ].to_numpy(dtype=float)
    means = operational.mean(numeric_only=True)
    return {
        "policy_name": policy,
        "training_seed": seed,
        "source_run_dir": source.resolve().relative_to(REPOSITORY_ROOT).as_posix(),
        "mean_reward_per_period": float(means["mean_reward_per_period"]),
        "mean_inventory": float(means["mean_inventory"]),
        "mean_holding_cost": float(means["mean_holding_cost"]),
        "mean_order_quantity": float(means["mean_order_quantity"]),
        "service_level": float(means["service_level"]),
        "lost_sales_rate": float(means["lost_sales_rate"]),
        "mean_purchase_cost": float(means["mean_purchase_cost"]),
        "mean_lost_sales_penalty": float(means["mean_lost_sales_penalty"]),
        "deployed_rmse": float(diagnostics["deployed_rmse"]),
        "target_stock_mae": float(np.mean(np.abs(target_errors))),
        "target_stock_rmse": float(np.sqrt(np.mean(np.square(target_errors)))),
        "cap_hit_rate": float(np.mean(clipping > 1e-6)),
        "mean_clipping": float(np.mean(clipping)),
        "max_clipping": float(np.max(clipping)),
        "raw_violation_rate": float(diagnostics["raw_violation_rate"]),
        "deployed_violation_rate": float(diagnostics["deployed_violation_rate"]),
        "extreme_raw_action_rate": float(np.mean(extreme)),
        "parameter_count": int(parameter_count if parameter_count is not None else completion["parameter_count"]),
        "training_seconds": float(completion.get("training_seconds", np.nan)),
        "selected_best_imitation_epoch": completion.get("selected_best_imitation_epoch"),
        "final_u_theta_mean": completion.get("final_u_theta_mean"),
        "final_u_theta_standard_deviation": completion.get("final_u_theta_standard_deviation"),
        "final_normalized_residual_mean": completion.get("final_normalized_residual_mean"),
        "final_normalized_residual_standard_deviation": completion.get("final_normalized_residual_standard_deviation"),
        "final_readout_weight_norm": completion.get("final_readout_weight_norm"),
        "first_step_backbone_gradient_norm": completion.get("first_step_backbone_gradient_norm"),
        "maximum_later_backbone_gradient_norm": completion.get("maximum_later_backbone_gradient_norm"),
        "backbone_became_trainable_after_first_step": completion.get("backbone_became_trainable_after_first_step"),
    }


def _write_reports(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    seed_results = pd.DataFrame(rows).sort_values(["policy_name", "training_seed"])
    _atomic_write_csv(output_dir / "seed_results.csv", seed_results)
    metrics = [
        "mean_reward_per_period", "mean_inventory", "mean_holding_cost", "mean_order_quantity",
        "service_level", "lost_sales_rate", "mean_purchase_cost", "mean_lost_sales_penalty",
        "deployed_rmse", "target_stock_mae", "target_stock_rmse", "cap_hit_rate", "mean_clipping",
        "max_clipping", "raw_violation_rate", "deployed_violation_rate", "extreme_raw_action_rate",
    ]
    aggregate_rows = []
    for policy, group in seed_results.groupby("policy_name", sort=False):
        row: dict[str, Any] = {"policy_name": policy, "seed_count": len(group), "parameter_count": int(group["parameter_count"].iloc[0])}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sample_sd"] = float(group[metric].std(ddof=1))
        aggregate_rows.append(row)
    _atomic_write_csv(output_dir / "aggregate_results.csv", pd.DataFrame(aggregate_rows))

    pairs = {
        "A_FCNN_residual_minus_direct": ("PaperFCNNResidualOrderUpTo", "PaperFCNN"),
        "B_CMNN_residual_minus_direct": ("ConstrainedMLPResidualOrderUpTo", "ConstrainedMonoMLP"),
        "C_direct_CMNN_minus_FCNN": ("ConstrainedMonoMLP", "PaperFCNN"),
        "D_residual_CMNN_minus_FCNN": ("ConstrainedMLPResidualOrderUpTo", "PaperFCNNResidualOrderUpTo"),
    }
    effect_rows = []
    indexed = seed_results.set_index(["policy_name", "training_seed"])
    for effect, (left, right) in pairs.items():
        for seed in SEEDS:
            row = {"effect": effect, "left_policy": left, "right_policy": right, "training_seed": seed}
            for metric in metrics:
                left_value = float(indexed.loc[(left, seed), metric])
                right_value = float(indexed.loc[(right, seed), metric])
                row[f"{metric}_difference"] = left_value - right_value
                if metric == "mean_reward_per_period":
                    row["reward_percentage_difference"] = 100.0 * (left_value - right_value) / max(abs(right_value), 1e-12)
            effect_rows.append(row)
    effects = pd.DataFrame(effect_rows)
    _atomic_write_csv(output_dir / "paired_effects_seed.csv", effects)
    effect_aggregate = effects.groupby("effect", as_index=False).mean(numeric_only=True)
    _atomic_write_csv(output_dir / "paired_effects_aggregate.csv", effect_aggregate)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", default="full")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--direct-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if tuple(args.seeds) != SEEDS:
        raise ValueError("the approved diagnostic requires exactly seeds 0 1 2")
    output_dir, output_relative = _repository_relative_path(args.output_dir, "output-dir")
    direct_root, _ = _repository_relative_path(args.direct_root, "direct-root")
    validate_split_and_output(args.split, output_relative)
    config_path, config_relative = _repository_relative_path(args.config, "config")
    base_config = load_paper_lost_sales_config(config_path)
    config, settings = _resolved_config(base_config, args.profile, 0)
    generator = PaperLostSalesProductGenerator(config)
    train_products = _safe_generate_products(generator, 128, "train")
    validation_products = _safe_generate_products(generator, 32, "validation")
    feature_stats = compute_paper_fcnn_feature_stats(train_products)
    fingerprints = data_fingerprints(config, settings, train_products, validation_products)
    shared_fingerprint = _fingerprint(fingerprints)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_dir / "environment.json", _environment_record(config_relative, config))
    fingerprints["shared_data_fingerprint"] = shared_fingerprint
    _atomic_write_json(output_dir / "data_fingerprints.json", fingerprints)

    audit, audit_passed = initialization_audit(
        base_config, args.profile, train_products, validation_products, feature_stats
    )
    _atomic_write_csv(output_dir / "initialization_audit.csv", audit)
    _atomic_write_json(
        output_dir / "initialization_gate.json",
        {
            "passed": audit_passed,
            "criterion": "CMNN max absolute z, u-1, target-stock correction, and normalized residual <= 1e-6 for every seed/split",
            "zero_residual_float32_tolerance": ZERO_RESIDUAL_FLOAT32_TOLERANCE,
            "fcnn_comparison_center_tolerance": FCNN_COMPARISON_CENTER_TOLERANCE,
            "training_started": bool(args.run and audit_passed),
        },
    )
    if not audit_passed:
        print("initialization_gate=failed training_started=false")
        return 2
    print("initialization_gate=passed")
    if not args.run:
        return 0

    oracle_levels = _oracle_levels(validation_products, int(settings["oracle_mc_samples"]), 10_000)
    completed = 0
    for policy in RESIDUAL_POLICIES:
        for seed in SEEDS:
            destination = _fit_dir(output_dir, policy, seed)
            existing = _fit_complete(destination)
            if existing is not None and args.resume:
                completed += 1
                print(f"completed={completed}/6 policy={policy} seed={seed} reused=true")
                continue
            result = _run_residual_fit(
                base_config=base_config,
                profile=args.profile,
                policy=policy,
                seed=seed,
                output_dir=output_dir,
                output_relative=output_relative,
                train_products=train_products,
                validation_products=validation_products,
                feature_stats=feature_stats,
                oracle_levels=oracle_levels,
                shared_fingerprint=shared_fingerprint,
            )
            completed += 1
            source = REPOSITORY_ROOT / result["source_run_dir"]
            operational = pd.read_csv(source / "operational_metrics.csv")
            reward = float(operational["mean_reward_per_period"].mean())
            print(f"completed={completed}/6 policy={policy} seed={seed} reused=false reward={reward:.6f}")

    rows: list[dict[str, Any]] = []
    for policy in DIRECT_SOURCE_POLICIES:
        for seed in SEEDS:
            source = _direct_source(direct_root, policy, seed)
            completion = json.loads((source / "completion.json").read_text(encoding="utf-8"))
            if completion.get("shared_data_fingerprint") != shared_fingerprint:
                raise RuntimeError(f"direct source fingerprint mismatch: {source}")
            rows.append(_seed_summary(policy, seed, source))
    for policy in RESIDUAL_POLICIES:
        for seed in SEEDS:
            source = _fit_dir(output_dir, policy, seed)
            completion = json.loads((source / "completion.json").read_text(encoding="utf-8"))
            rows.append(_seed_summary(policy, seed, source, int(completion["parameter_count"])))
    _write_reports(output_dir, rows)
    _atomic_write_json(
        output_dir / "analysis_completion.json",
        {
            "completed_new_fits": completed,
            "expected_new_fits": 6,
            "reused_direct_fits": 6,
            "initialization_gate_passed": True,
            "protected_test_accessed": False,
            "selected_direct_hyperparameters_changed": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
