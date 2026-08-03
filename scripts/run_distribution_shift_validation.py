"""Evaluate saved residual policies under product-distribution shifts."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from inventory_control.idp_benchmark import (
    _model_config,
    _operational_table,
    _oracle_levels,
    _prediction_summary,
    _prediction_table,
    _profile_settings,
    load_paper_lost_sales_config,
)
from inventory_control.policies.neural.fcnn import compute_paper_fcnn_feature_stats
from inventory_control.policies.neural.registry import (
    RESIDUAL_COMPARISON_POLICY_NAMES,
    checkpoint_filenames,
    policy_factories,
)
from inventory_control.products import PaperLostSalesProductGenerator


SHIFT_OVERRIDES: dict[str, dict[str, Any]] = {
    "in_distribution": {},
    "low_cv": {"coefficient_of_variation_range": [0.0, 0.3]},
    "high_cv": {"coefficient_of_variation_range": [0.8, 1.5]},
    "low_mean_demand": {"mean_demand_mean": 50.0},
    "high_mean_demand": {"mean_demand_mean": 200.0},
    "high_lost_sale_penalty": {"lost_sale_penalty_max": 25.0},
    "high_holding_cost": {"holding_cost_mean": 15.0},
    "mixed_stress": {
        "mean_demand_mean": 200.0,
        "holding_cost_mean": 15.0,
        "lost_sale_penalty_max": 25.0,
        "coefficient_of_variation_range": [0.8, 1.5],
    },
}


def _shifted_config(base_config: dict[str, Any], shift_name: str) -> dict[str, Any]:
    if shift_name not in SHIFT_OVERRIDES:
        raise ValueError(f"unknown shift {shift_name!r}; supported shifts: {', '.join(SHIFT_OVERRIDES)}")
    config = copy.deepcopy(base_config)
    config["product_generation"] = {
        **config["product_generation"],
        **SHIFT_OVERRIDES[shift_name],
    }
    return config


def _load_models(benchmark_dir: Path, model_config: dict[str, Any], policy_names: list[str]) -> dict[str, torch.nn.Module]:
    models: dict[str, torch.nn.Module] = {}
    factories = policy_factories(policy_names)
    filenames = checkpoint_filenames()
    for policy_name in policy_names:
        model_path = benchmark_dir / "models" / filenames[policy_name]
        if not model_path.exists():
            raise FileNotFoundError(f"missing checkpoint for {policy_name}: {model_path}")
        model = factories[policy_name](model_config)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        models[policy_name] = model
    return models


def _summarize_shift(
    operational: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    shift_name: str,
) -> pd.DataFrame:
    summary = (
        operational.groupby("policy_name", as_index=False)
        .agg(
            mean_total_reward=("mean_total_reward", "mean"),
            std_total_reward=("mean_total_reward", "std"),
            mean_reward_per_period=("mean_reward_per_period", "mean"),
            service_level=("service_level", "mean"),
            lost_sales_rate=("lost_sales_rate", "mean"),
            mean_order_quantity=("mean_order_quantity", "mean"),
            mean_inventory=("mean_inventory", "mean"),
        )
    )
    if not prediction_summary.empty:
        summary = summary.merge(prediction_summary, on="policy_name", how="left")
    non_omniscient = summary.loc[summary["policy_name"] == "NonOmniscientOptimal", "mean_reward_per_period"]
    omniscient = summary.loc[summary["policy_name"] == "OmniscientOptimal", "mean_reward_per_period"]
    non_omniscient_value = float(non_omniscient.iloc[0]) if not non_omniscient.empty else float("nan")
    omniscient_value = float(omniscient.iloc[0]) if not omniscient.empty else float("nan")
    summary.insert(0, "shift", shift_name)
    summary["regret_vs_non_omniscient"] = non_omniscient_value - summary["mean_reward_per_period"]
    summary["regret_vs_omniscient"] = omniscient_value - summary["mean_reward_per_period"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paper_lost_sales_fcnn_alignment_medium.yaml"))
    parser.add_argument("--source-benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/monokan_distribution_shift_validation"))
    parser.add_argument("--profile", choices=["compact", "full", "paper_replication"], default="full")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--eval-products", type=int, default=500)
    parser.add_argument("--diagnostic-products", type=int, default=200)
    parser.add_argument("--inventory-states", type=int, default=41)
    parser.add_argument("--oracle-mc-samples", type=int, default=200000)
    parser.add_argument("--shifts", default=",".join(SHIFT_OVERRIDES))
    parser.add_argument(
        "--models",
        default=",".join(RESIDUAL_COMPARISON_POLICY_NAMES),
    )
    args = parser.parse_args()

    base_config = load_paper_lost_sales_config(args.config)
    base_settings = _profile_settings(base_config, args.profile)
    base_settings = {
        **base_settings,
        "validation_products": int(args.eval_products),
        "order_up_to_products": int(args.diagnostic_products),
        "inventory_states": int(args.inventory_states),
        "oracle_mc_samples": int(args.oracle_mc_samples),
    }
    model_config = _model_config(base_config, base_settings)
    policy_names = [name.strip() for name in args.models.split(",") if name.strip()]
    models = _load_models(args.source_benchmark_dir, model_config, policy_names)

    train_products = PaperLostSalesProductGenerator(base_config).generate(int(base_settings["training_products"]), split="train")
    feature_stats = compute_paper_fcnn_feature_stats(train_products)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "config": str(args.config),
        "source_benchmark_dir": str(args.source_benchmark_dir),
        "profile": args.profile,
        "split": args.split,
        "eval_products": int(args.eval_products),
        "diagnostic_products": int(args.diagnostic_products),
        "inventory_states": int(args.inventory_states),
        "oracle_mc_samples": int(args.oracle_mc_samples),
        "models": policy_names,
        "shifts": [name.strip() for name in args.shifts.split(",") if name.strip()],
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8")
    all_summary = []
    for shift_name in run_metadata["shifts"]:
        config = _shifted_config(base_config, shift_name)
        generator = PaperLostSalesProductGenerator(config)
        eval_products = generator.generate(int(args.eval_products), split=args.split)
        diagnostic_products = eval_products[: int(args.diagnostic_products)]
        oracle_levels = _oracle_levels(
            eval_products,
            int(args.oracle_mc_samples),
            int(base_settings["seed"]) + 10_000,
        )
        operational = _operational_table(
            eval_products,
            models,
            feature_stats,
            base_settings,
            oracle_levels,
            args.split,
        )
        diagnostic_oracle_levels = {product.product_id: oracle_levels[product.product_id] for product in diagnostic_products}
        predictions = _prediction_table(
            diagnostic_products,
            models,
            feature_stats,
            base_settings,
            diagnostic_oracle_levels,
        )
        prediction_summary = _prediction_summary(predictions)
        shift_summary = _summarize_shift(operational, prediction_summary, shift_name)
        shift_dir = args.output_dir / shift_name
        shift_dir.mkdir(parents=True, exist_ok=True)
        (shift_dir / "resolved_shift_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        operational.to_csv(shift_dir / "operational_metrics.csv", index=False)
        predictions.to_csv(shift_dir / "predictions.csv", index=False)
        shift_summary.to_csv(shift_dir / "summary.csv", index=False)
        all_summary.append(shift_summary)
        print(f"shift={shift_name} completed products={len(eval_products)} diagnostics={len(diagnostic_products)}", flush=True)

    combined = pd.concat(all_summary, ignore_index=True)
    combined.to_csv(args.output_dir / "summary_all_shifts.csv", index=False)
    learned = combined[combined["policy_name"].isin(policy_names)].copy()
    robustness = (
        learned.groupby("policy_name", as_index=False)
        .agg(
            shifts=("shift", "count"),
            mean_reward_per_period=("mean_reward_per_period", "mean"),
            worst_reward_per_period=("mean_reward_per_period", "min"),
            mean_regret_vs_non_omniscient=("regret_vs_non_omniscient", "mean"),
            worst_regret_vs_non_omniscient=("regret_vs_non_omniscient", "max"),
            mean_regret_vs_omniscient=("regret_vs_omniscient", "mean"),
            mean_service_level=("service_level", "mean"),
            mean_lost_sales_rate=("lost_sales_rate", "mean"),
            mean_order_up_to_mae=("order_up_to_mae", "mean"),
            mean_cap_hit_rate=("cap_hit_rate", "mean"),
            mean_raw_violation_rate=("raw_violation_rate", "mean"),
        )
        .sort_values(["mean_regret_vs_non_omniscient", "mean_reward_per_period"], ascending=[True, False])
    )
    robustness.to_csv(args.output_dir / "robustness_summary.csv", index=False)
    print(robustness.to_string(index=False))
    print(f"saved_summary={args.output_dir / 'summary_all_shifts.csv'}")
    print(f"saved_robustness={args.output_dir / 'robustness_summary.csv'}")


if __name__ == "__main__":
    main()
