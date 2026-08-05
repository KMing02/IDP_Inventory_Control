"""Run the medium seed-0 CMNN imitation/reset ablation on validation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inventory_control.idp_benchmark import (
    _model_config,
    _operational_table,
    _oracle_levels,
    _prediction_summary,
    _prediction_table,
    _profile_settings,
    _train_model,
)
from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.policies.neural.constrained_mlp import make_constrained_monotone_mlp_model
from inventory_control.policies.neural.fcnn import compute_paper_fcnn_feature_stats
from inventory_control.products import PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import (
    make_paper_lost_sales_training_batch,
    paper_lost_sales_raw_order,
    paper_order_diagnostics,
)


def _zero_inventory_mean(model, validation_batch, scale_actions: bool) -> float:
    with torch.no_grad():
        raw = paper_lost_sales_raw_order(
            model,
            validation_batch,
            torch.zeros_like(validation_batch["initial_inventory"]),
            scale_action_by_history_mean=scale_actions,
        )
    return float(raw.mean().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/paper_lost_sales_medium_seed0.yaml"))
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=Path("outputs/idp_benchmark/cmnn_medium_seed0_validation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/diagnostics/cmnn_imitation_ablation_seed0"),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing ablation output: {args.output_dir}")

    config = load_paper_lost_sales_config(args.config)
    base_settings = _profile_settings(config, "full")
    model_config = _model_config(config, base_settings)
    generator = PaperLostSalesProductGenerator(config)
    train_products = generator.generate(int(base_settings["training_products"]), "train")
    validation_products = generator.generate(int(base_settings["validation_products"]), "validation")
    feature_stats = compute_paper_fcnn_feature_stats(train_products)
    validation_generator = torch.Generator().manual_seed(int(base_settings["seed"]) + 2_000_003)
    validation_batch = make_paper_lost_sales_training_batch(
        validation_products,
        int(base_settings["training_horizon"]),
        validation_generator,
        feature_stats=feature_stats,
    )
    oracle_levels = _oracle_levels(
        validation_products,
        int(base_settings["oracle_mc_samples"]),
        int(base_settings["seed"]) + 10_000,
    )
    variants = [
        (
            "A_baseline_5_no_reset",
            {
                "imitation_epochs": 5,
                "imitation_early_stopping": False,
                "track_imitation_validation": True,
                "reset_optimizer_after_imitation": False,
            },
        ),
        (
            "B_converged_no_reset",
            {
                "imitation_early_stopping": True,
                "imitation_max_epochs": 100,
                "imitation_patience": 10,
                "imitation_min_delta": 0.5,
                "track_imitation_validation": True,
                "reset_optimizer_after_imitation": False,
            },
        ),
        (
            "C_converged_with_reset",
            {
                "imitation_early_stopping": True,
                "imitation_max_epochs": 100,
                "imitation_patience": 10,
                "imitation_min_delta": 0.5,
                "track_imitation_validation": True,
                "reset_optimizer_after_imitation": True,
            },
        ),
    ]

    args.output_dir.mkdir(parents=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir()
    summary_rows = []
    curve_frames = []
    baseline_checkpoint = torch.load(
        args.benchmark_output / "models" / "constrained_monotone_mlp.pt",
        map_location="cpu",
    )
    for variant_name, overrides in variants:
        settings = {**base_settings, **overrides}
        torch.manual_seed(int(settings["seed"]) + 102)
        model = make_constrained_monotone_mlp_model(model_config)
        zero_before = _zero_inventory_mean(
            model,
            validation_batch,
            bool(settings["scale_action_by_history_mean"]),
        )
        result = _train_model(
            config,
            settings,
            train_products,
            validation_products,
            feature_stats,
            model,
            variant_name,
            "validation",
        )
        if result.imitation_end_state_dict is None:
            raise RuntimeError(f"{variant_name} did not produce an imitation-end checkpoint")
        imitation_model = make_constrained_monotone_mlp_model(model_config)
        imitation_model.load_state_dict(result.imitation_end_state_dict)
        zero_after = _zero_inventory_mean(
            imitation_model,
            validation_batch,
            bool(settings["scale_action_by_history_mean"]),
        )
        imitation_validation = paper_order_diagnostics(
            imitation_model,
            validation_batch,
            max_order_factor=float(settings["max_order_factor"]),
            scale_action_by_history_mean=bool(settings["scale_action_by_history_mean"]),
            inventory_fractions=(0.0, 0.25, 0.5, 0.75, 1.0),
        )
        predictions = _prediction_table(
            validation_products,
            {"ConstrainedMonoMLP": model},
            feature_stats,
            settings,
            oracle_levels,
        )
        prediction_metrics = _prediction_summary(predictions)
        prediction_row = prediction_metrics.loc[
            prediction_metrics["policy_name"] == "ConstrainedMonoMLP"
        ].iloc[0]
        operational = _operational_table(
            validation_products,
            {"ConstrainedMonoMLP": model},
            feature_stats,
            settings,
            oracle_levels,
            "validation",
        )
        operational_row = (
            operational.loc[operational["policy_name"] == "ConstrainedMonoMLP"]
            .mean(numeric_only=True)
        )
        baseline_difference = np.nan
        if variant_name.startswith("A_"):
            baseline_difference = max(
                float((model.state_dict()[key] - baseline_checkpoint[key]).abs().max().item())
                for key in baseline_checkpoint
            )
        summary_rows.append(
            {
                "variant": variant_name,
                "selected_imitation_epoch": (
                    result.best_imitation_epoch
                    if bool(settings.get("imitation_early_stopping", False))
                    else result.imitation_epochs_completed
                ),
                "imitation_epochs_completed": result.imitation_epochs_completed,
                "imitation_early_stopped": result.imitation_early_stopped,
                "optimizer_reset_after_imitation": result.optimizer_reset_after_imitation,
                "imitation_validation_mae": imitation_validation["raw_imitation_mae"],
                "imitation_validation_rmse": imitation_validation["raw_imitation_rmse"],
                "zero_inventory_raw_order_before_imitation": zero_before,
                "zero_inventory_raw_order_after_imitation": zero_after,
                "mean_reward_per_period": operational_row["mean_reward_per_period"],
                "service_level": operational_row["service_level"],
                "lost_sales_rate": operational_row["lost_sales_rate"],
                "mean_inventory": operational_row["mean_inventory"],
                "mean_order_quantity": operational_row["mean_order_quantity"],
                "deployed_mae": prediction_row["deployed_mae"],
                "deployed_rmse": prediction_row["deployed_rmse"],
                "raw_violation_rate": prediction_row["raw_violation_rate"],
                "deployed_violation_rate": prediction_row["deployed_violation_rate"],
                "training_seconds": result.elapsed_seconds,
                "baseline_saved_checkpoint_max_abs_difference": baseline_difference,
            }
        )
        log = pd.DataFrame(result.training_log or [])
        log["variant"] = variant_name
        log.to_csv(args.output_dir / f"{variant_name}_training_log.csv", index=False)
        imitation_curve = log.loc[log["phase"] == 0].copy()
        curve_frames.append(
            imitation_curve[
                [
                    "variant",
                    "epoch",
                    "imitation_validation_zero_inventory_raw_order_mean",
                    "imitation_validation_mae",
                    "imitation_validation_rmse",
                ]
            ]
        )
        torch.save(model.state_dict(), model_dir / f"{variant_name}_final.pt")
        torch.save(result.best_imitation_state_dict, model_dir / f"{variant_name}_best_imitation.pt")

    summary = pd.DataFrame(summary_rows)
    curves = pd.concat(curve_frames, ignore_index=True)
    summary.to_csv(args.output_dir / "ablation_summary.csv", index=False)
    curves.to_csv(args.output_dir / "imitation_curves.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for variant_name, rows in curves.groupby("variant", sort=False):
        axes[0].plot(
            rows["epoch"],
            rows["imitation_validation_zero_inventory_raw_order_mean"],
            label=variant_name,
        )
        axes[1].plot(rows["epoch"], rows["imitation_validation_rmse"], label=variant_name)
    axes[0].set_title("CMNN zero-inventory raw order during imitation")
    axes[0].set_xlabel("Imitation epoch")
    axes[0].set_ylabel("Mean raw order")
    axes[1].set_title("Validation imitation RMSE")
    axes[1].set_xlabel("Imitation epoch")
    axes[1].set_ylabel("RMSE")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output_dir / "imitation_convergence.png", dpi=160)
    plt.close(fig)

    metadata = {
        "config": str(args.config),
        "profile": "full",
        "split": "validation",
        "seed": int(base_settings["seed"]),
        "training_products": int(base_settings["training_products"]),
        "validation_products": int(base_settings["validation_products"]),
        "evaluation_horizon": int(base_settings["evaluation_horizon"]),
        "burn_in": int(base_settings["burn_in"]),
        "imitation_max_epochs": 100,
        "imitation_patience": 10,
        "imitation_min_delta": 0.5,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"saved_ablation={args.output_dir}")


if __name__ == "__main__":
    main()
