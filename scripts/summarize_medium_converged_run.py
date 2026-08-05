"""Summarize the converged-imitation medium run against the fixed-epoch run."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inventory_control.idp_benchmark import _model_config, _profile_settings
from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.policies.neural.constrained_mlp import (
    make_constrained_monotone_mlp_model,
    make_positive_weight_mlp_model,
)
from inventory_control.policies.neural.fcnn import (
    compute_paper_fcnn_feature_stats,
    make_paper_fcnn_model,
)
from inventory_control.products import PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import (
    make_paper_lost_sales_training_batch,
    paper_lost_sales_raw_order,
    paper_order_diagnostics,
)


PRIMARY_MODELS = {
    "PaperFCNN": {
        "log_name": "PaperFCNN",
        "seed_offset": 0,
        "factory": make_paper_fcnn_model,
        "checkpoint": "paper_fcnn_best_imitation.pt",
    },
    "PenaltyMLP": {
        "log_name": "PenaltyMLP(lambda=10)",
        "seed_offset": 0,
        "factory": make_paper_fcnn_model,
        "checkpoint": "penalty_mlp_best_imitation.pt",
    },
    "PositiveWeightMLP": {
        "log_name": "PositiveWeightMLP",
        "seed_offset": 101,
        "factory": make_positive_weight_mlp_model,
        "checkpoint": "positive_weight_mlp_best_imitation.pt",
    },
    "ConstrainedMonoMLP": {
        "log_name": "ConstrainedMonoMLP",
        "seed_offset": 102,
        "factory": make_constrained_monotone_mlp_model,
        "checkpoint": "constrained_monotone_mlp_best_imitation.pt",
    },
}


def _zero_inventory_raw_order_mean(model, batch, scale_actions: bool) -> float:
    with torch.no_grad():
        raw_order = paper_lost_sales_raw_order(
            model,
            batch,
            torch.zeros_like(batch["initial_inventory"]),
            scale_action_by_history_mean=scale_actions,
        )
    return float(raw_order.mean().item())


def _product_fingerprint(products) -> str:
    payload = json.dumps(
        [asdict(product) for product in products],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fixed-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fixed-run-dir", type=Path, required=True)
    parser.add_argument("--previous-converged-run-dir", type=Path)
    args = parser.parse_args()

    config = load_paper_lost_sales_config(args.config)
    fixed_config = load_paper_lost_sales_config(args.fixed_config)
    settings = _profile_settings(config, "full")
    fixed_settings = _profile_settings(fixed_config, "full")
    model_config = _model_config(config, settings)

    generator = PaperLostSalesProductGenerator(config)
    train_products = generator.generate(int(settings["training_products"]), "train")
    validation_products = generator.generate(int(settings["validation_products"]), "validation")
    fixed_generator = PaperLostSalesProductGenerator(fixed_config)
    fixed_train_products = fixed_generator.generate(int(fixed_settings["training_products"]), "train")
    fixed_validation_products = fixed_generator.generate(
        int(fixed_settings["validation_products"]), "validation"
    )
    fingerprints = {
        "new_train": _product_fingerprint(train_products),
        "fixed_train": _product_fingerprint(fixed_train_products),
        "new_validation_including_future_demand": _product_fingerprint(validation_products),
        "fixed_validation_including_future_demand": _product_fingerprint(fixed_validation_products),
    }
    fingerprints["train_products_identical"] = (
        fingerprints["new_train"] == fingerprints["fixed_train"]
    )
    fingerprints["validation_products_and_demand_traces_identical"] = (
        fingerprints["new_validation_including_future_demand"]
        == fingerprints["fixed_validation_including_future_demand"]
    )

    feature_stats = compute_paper_fcnn_feature_stats(train_products)
    validation_generator = torch.Generator().manual_seed(int(settings["seed"]) + 2_000_003)
    validation_batch = make_paper_lost_sales_training_batch(
        validation_products,
        int(settings["training_horizon"]),
        validation_generator,
        feature_stats=feature_stats,
    )

    training_log = pd.read_csv(args.run_dir / "training_log.csv")
    summary = pd.read_csv(args.run_dir / "summary.csv")
    previous_training_log = (
        pd.read_csv(args.previous_converged_run_dir / "training_log.csv")
        if args.previous_converged_run_dir is not None
        else None
    )
    previous_summary = (
        pd.read_csv(args.previous_converged_run_dir / "summary.csv")
        if args.previous_converged_run_dir is not None
        else None
    )
    imitation_rows = []
    metric_correction_rows = []
    for policy_name, specification in PRIMARY_MODELS.items():
        policy_log = training_log.loc[
            (training_log["policy_name"] == specification["log_name"])
            & (training_log["phase"] == 0)
        ].copy()
        if policy_log.empty:
            raise RuntimeError(f"missing imitation log for {policy_name}")
        best_row = policy_log.loc[policy_log["imitation_validation_rmse"].idxmin()]

        torch.manual_seed(int(settings["seed"]) + int(specification["seed_offset"]))
        initial_model = specification["factory"](model_config)
        zero_before = _zero_inventory_raw_order_mean(
            initial_model,
            validation_batch,
            bool(settings["scale_action_by_history_mean"]),
        )

        checkpoint_path = args.run_dir / "models" / specification["checkpoint"]
        imitation_model = specification["factory"](model_config)
        imitation_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        zero_after = _zero_inventory_raw_order_mean(
            imitation_model,
            validation_batch,
            bool(settings["scale_action_by_history_mean"]),
        )
        corrected_diagnostics = paper_order_diagnostics(
            imitation_model,
            validation_batch,
            max_order_factor=float(settings["max_order_factor"]),
            scale_action_by_history_mean=bool(settings["scale_action_by_history_mean"]),
            inventory_fractions=(0.0, 0.25, 0.5, 0.75, 1.0),
        )
        imitation_rows.append(
            {
                "policy_name": policy_name,
                "imitation_epochs_run": int(len(policy_log)),
                "selected_best_imitation_epoch": int(best_row["epoch"]),
                "best_validation_imitation_mae": float(best_row["imitation_validation_mae"]),
                "best_validation_imitation_rmse": float(best_row["imitation_validation_rmse"]),
                "checkpoint_selection_metric": str(best_row["imitation_checkpoint_metric"]),
                "raw_validation_imitation_mae": corrected_diagnostics["raw_imitation_mae"],
                "raw_validation_imitation_rmse": corrected_diagnostics["raw_imitation_rmse"],
                "deployed_validation_imitation_mae": corrected_diagnostics[
                    "deployed_imitation_mae"
                ],
                "deployed_validation_imitation_rmse": corrected_diagnostics[
                    "deployed_imitation_rmse"
                ],
                "zero_inventory_raw_order_before_imitation": zero_before,
                "zero_inventory_raw_order_after_imitation": zero_after,
                "checkpoint_zero_order_minus_log": zero_after
                - float(best_row["imitation_validation_zero_inventory_raw_order_mean"]),
                "early_stop_triggered": bool(policy_log.iloc[-1]["imitation_early_stop_triggered"]),
                "best_checkpoint_file": str(checkpoint_path),
                "best_checkpoint_exists": checkpoint_path.is_file(),
            }
        )
        if previous_training_log is not None and args.previous_converged_run_dir is not None:
            previous_policy_log = previous_training_log.loc[
                (previous_training_log["policy_name"] == specification["log_name"])
                & (previous_training_log["phase"] == 0)
            ]
            previous_best_row = previous_policy_log.loc[
                previous_policy_log["imitation_validation_rmse"].idxmin()
            ]
            previous_checkpoint = (
                args.previous_converged_run_dir / "models" / specification["checkpoint"]
            )
            previous_model = specification["factory"](model_config)
            previous_model.load_state_dict(torch.load(previous_checkpoint, map_location="cpu"))
            previous_diagnostics = paper_order_diagnostics(
                previous_model,
                validation_batch,
                max_order_factor=float(settings["max_order_factor"]),
                scale_action_by_history_mean=bool(settings["scale_action_by_history_mean"]),
                inventory_fractions=(0.0, 0.25, 0.5, 0.75, 1.0),
            )
            old_operational = previous_summary.loc[
                previous_summary["policy_name"] == policy_name
            ].iloc[0]
            new_operational = summary.loc[summary["policy_name"] == policy_name].iloc[0]
            metric_correction_rows.append(
                {
                    "policy_name": policy_name,
                    "old_selection_semantics": "raw_prediction_vs_capped_target_rmse",
                    "new_selection_semantics": "deployed_prediction_vs_capped_target_rmse",
                    "old_selected_epoch": int(previous_best_row["epoch"]),
                    "new_selected_epoch": int(best_row["epoch"]),
                    "old_raw_validation_mae_correctly_recomputed": previous_diagnostics[
                        "raw_imitation_mae"
                    ],
                    "new_raw_validation_mae": corrected_diagnostics["raw_imitation_mae"],
                    "old_raw_validation_rmse_correctly_recomputed": previous_diagnostics[
                        "raw_imitation_rmse"
                    ],
                    "new_raw_validation_rmse": corrected_diagnostics["raw_imitation_rmse"],
                    "old_deployed_validation_mae_correctly_recomputed": previous_diagnostics[
                        "deployed_imitation_mae"
                    ],
                    "new_deployed_validation_mae": corrected_diagnostics[
                        "deployed_imitation_mae"
                    ],
                    "old_deployed_validation_rmse_correctly_recomputed": previous_diagnostics[
                        "deployed_imitation_rmse"
                    ],
                    "new_deployed_validation_rmse": corrected_diagnostics[
                        "deployed_imitation_rmse"
                    ],
                    "old_mean_reward_per_period": old_operational["mean_reward_per_period"],
                    "new_mean_reward_per_period": new_operational["mean_reward_per_period"],
                    "old_service_level": old_operational["service_level"],
                    "new_service_level": new_operational["service_level"],
                    "old_mean_inventory": old_operational["mean_inventory"],
                    "new_mean_inventory": new_operational["mean_inventory"],
                    "old_mean_order_quantity": old_operational["mean_order_quantity"],
                    "new_mean_order_quantity": new_operational["mean_order_quantity"],
                    "old_raw_violation_rate": old_operational["raw_violation_rate"],
                    "new_raw_violation_rate": new_operational["raw_violation_rate"],
                    "old_deployed_violation_rate": old_operational["deployed_violation_rate"],
                    "new_deployed_violation_rate": new_operational[
                        "deployed_violation_rate"
                    ],
                }
            )
    imitation = pd.DataFrame(imitation_rows)

    requested_metrics = [
        "mean_reward_per_period",
        "service_level",
        "lost_sales_rate",
        "mean_inventory",
        "mean_order_quantity",
        "raw_mae",
        "raw_rmse",
        "deployed_mae",
        "deployed_rmse",
        "raw_zero_inventory_order_up_to_mae",
        "raw_zero_inventory_order_up_to_rmse",
        "raw_zero_inventory_order_up_to_median_absolute_error",
        "raw_zero_inventory_order_up_to_p90_absolute_error",
        "raw_zero_inventory_order_up_to_max_absolute_error",
        "raw_zero_inventory_order_up_to_largest_product_contribution",
        "raw_zero_inventory_order_up_to_top3_product_contribution",
        "deployed_zero_inventory_order_up_to_mae",
        "deployed_zero_inventory_order_up_to_rmse",
        "deployed_zero_inventory_order_up_to_median_absolute_error",
        "deployed_zero_inventory_order_up_to_p90_absolute_error",
        "deployed_zero_inventory_order_up_to_max_absolute_error",
        "deployed_zero_inventory_order_up_to_largest_product_contribution",
        "deployed_zero_inventory_order_up_to_top3_product_contribution",
        "raw_order_up_to_mae",
        "raw_order_up_to_rmse",
        "deployed_order_up_to_mae",
        "deployed_order_up_to_rmse",
        "raw_violation_rate",
        "deployed_violation_rate",
        "parameter_count",
        "training_seconds",
    ]
    available_report_metrics = [metric for metric in requested_metrics if metric in summary.columns]
    primary_summary = summary.loc[summary["policy_name"].isin(PRIMARY_MODELS)].copy()
    primary_report = imitation.merge(
        primary_summary[["policy_name", *available_report_metrics]], on="policy_name", how="left"
    )
    primary_report.to_csv(args.run_dir / "primary_neural_report.csv", index=False)
    if metric_correction_rows:
        pd.DataFrame(metric_correction_rows).to_csv(
            args.run_dir / "metric_correction_comparison.csv", index=False
        )

    fixed_summary = pd.read_csv(args.fixed_run_dir / "summary.csv")
    comparison_metrics = [
        metric for metric in available_report_metrics if metric in fixed_summary.columns
    ]
    comparison = summary[["policy_name", *comparison_metrics]].merge(
        fixed_summary[["policy_name", *comparison_metrics]],
        on="policy_name",
        suffixes=("_converged", "_fixed5"),
    )
    for metric in comparison_metrics:
        comparison[f"{metric}_delta"] = (
            comparison[f"{metric}_converged"] - comparison[f"{metric}_fixed5"]
        )
    comparison.to_csv(args.run_dir / "comparison_with_fixed5.csv", index=False)

    audit = {
        "split_used": "validation",
        "protected_test_split_accessed": False,
        "training_product_count": len(train_products),
        "validation_product_count": len(validation_products),
        "validation_grid_seed": int(settings["seed"]) + 2_000_003,
        "training_horizon": int(settings["training_horizon"]),
        "evaluation_horizon": int(settings["evaluation_horizon"]),
        "burn_in": int(settings["burn_in"]),
        "max_order_factor": float(settings["max_order_factor"]),
        "imitation_early_stopping": bool(settings["imitation_early_stopping"]),
        "imitation_max_epochs": int(settings["imitation_max_epochs"]),
        "imitation_patience": int(settings["imitation_patience"]),
        "imitation_min_delta": float(settings["imitation_min_delta"]),
        "reset_optimizer_after_imitation": bool(settings["reset_optimizer_after_imitation"]),
        **fingerprints,
    }
    (args.run_dir / "comparison_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(primary_report.to_string(index=False))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
