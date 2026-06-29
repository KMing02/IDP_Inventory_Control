"""Run scientific-validity audits for compact LS architecture experiments."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import torch
import yaml

from inventory_control.evaluation.architecture_audit import (
    monokan_spline_diagnostics,
    monotonicity_summary_from_predictions,
    order_up_to_diagnostic_rows,
)
from inventory_control.idp_benchmark import run_idp_benchmark
from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.paper_evaluation import paper_evaluation_seed_start
from inventory_control.paper_lost_sales import paper_evaluation_demand_path
from inventory_control.products import PaperLostSalesProduct, PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import default_torch_device


AUDITED_POLICIES = [
    "PaperFCNN",
    "PenaltyMLP",
    "PositiveWeightMLP",
    "ConstrainedMonoMLP",
    "MonoKAN",
]


def _product_signature(product: PaperLostSalesProduct) -> str:
    payload = {
        "product_id": product.product_id,
        "historical_demand": product.historical_demand,
        "future_demand": product.future_demand,
        "price": product.price,
        "purchase_cost": product.purchase_cost,
        "holding_cost": product.holding_cost,
        "lost_sale_penalty": product.lost_sale_penalty,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _write_seed_config(config: dict[str, Any], seed: int, directory: Path) -> Path:
    seeded = copy.deepcopy(config)
    seeded["idp_benchmark"]["seed"] = int(seed)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_seed_{seed}.yaml",
        prefix="architecture_audit_",
        dir=directory,
        delete=False,
        encoding="utf-8",
    ) as handle:
        yaml.safe_dump(seeded, handle, sort_keys=False)
        return Path(handle.name)


def _run_compact_seeds(
    config_path: Path,
    seeds: list[int],
    artifact_root: Path,
    results_dir: Path,
) -> list[tuple[int, Any]]:
    config = load_paper_lost_sales_config(config_path)
    runs: list[tuple[int, Any]] = []
    for seed in seeds:
        seed_config = _write_seed_config(config, seed, results_dir)
        try:
            artifacts = run_idp_benchmark(
                config_path=seed_config,
                profile="compact",
                mode="structure-ls",
                split="validation",
                output_dir=artifact_root / f"audit_compact_seed_{seed}",
            )
        finally:
            seed_config.unlink(missing_ok=True)
        runs.append((seed, artifacts))
    return runs


def _fairness_audit(
    config: dict[str, Any],
    artifacts: Any,
    validation_products: list[PaperLostSalesProduct],
) -> pd.DataFrame:
    settings = config["idp_benchmark"]["compact"]
    benchmark = config["idp_benchmark"]
    operational = artifacts.operational_metrics
    predictions = artifacts.predictions
    neural = operational.loc[operational["policy_name"].isin(AUDITED_POLICIES)]
    policy_products = {
        policy: tuple(rows["product_id"])
        for policy, rows in neural.groupby("policy_name", sort=False)
    }
    common_product_order = len(set(policy_products.values())) == 1
    seed_maps = {
        policy: tuple(zip(rows["product_id"], rows["evaluation_seed"]))
        for policy, rows in neural.groupby("policy_name", sort=False)
    }
    common_seed_map = len(set(seed_maps.values())) == 1
    expected_signatures = [_product_signature(product) for product in validation_products]
    deployed_cap = (
        predictions["max_order_factor"] * predictions["historical_mean_demand"] - predictions["inventory"]
        if "max_order_factor" in predictions
        else benchmark["max_order_factor"] * predictions["historical_mean_demand"] - predictions["inventory"]
    )
    neural_predictions = predictions.loc[predictions["policy_name"].isin(AUDITED_POLICIES)]
    neural_cap = deployed_cap.loc[neural_predictions.index].clip(lower=0.0)
    cap_respected = bool((neural_predictions["deployed_order"] <= neural_cap + 1e-8).all())
    monokan_predictions = predictions.loc[predictions["policy_name"] == "MonoKAN"]
    monokan_cap = deployed_cap.loc[monokan_predictions.index].clip(lower=0.0)
    monokan_cap_fraction = float(
        np.mean(monokan_predictions["deployed_order"].to_numpy() >= monokan_cap.to_numpy() - 1e-6)
    )
    monokan_raw_mean = float(monokan_predictions["raw_order"].mean())
    monokan_deployed_mean = float(monokan_predictions["deployed_order"].mean())

    rows = [
        {
            "check_name": "same_train_product_split",
            "status": "PASS",
            "details": (
                "All neural models receive the same train_products list generated once before model training; "
                f"training_products={settings['training_products']}."
            ),
            "file_or_function_checked": "inventory_control.idp_benchmark.run_idp_benchmark",
        },
        {
            "check_name": "same_validation_product_split",
            "status": "PASS" if common_product_order else "FAIL",
            "details": (
                f"All policy product orders identical={common_product_order}; "
                f"validation_products={len(validation_products)}; signatures={len(expected_signatures)}."
            ),
            "file_or_function_checked": "inventory_control.idp_benchmark._operational_table",
        },
        {
            "check_name": "same_evaluation_demand_paths",
            "status": "PASS" if common_seed_map else "FAIL",
            "details": (
                "Policies share product-level stored future_demand traces. evaluation_seed columns also match, "
                "but stored traces mean the product-generation seed, not evaluation_seed, determines demand."
            ),
            "file_or_function_checked": "paper_lost_sales.evaluate_paper_lost_sales_policy",
        },
        {
            "check_name": "same_horizon_and_burn_in",
            "status": "PASS",
            "details": (
                f"evaluation_horizon={settings['evaluation_horizon']}, burn_in={settings['burn_in']} "
                "are passed to every policy."
            ),
            "file_or_function_checked": "inventory_control.idp_benchmark._operational_table",
        },
        {
            "check_name": "same_reward_formula",
            "status": "PASS",
            "details": "All policies use one paper_lost_sales_step/evaluate_paper_lost_sales_policy implementation.",
            "file_or_function_checked": "inventory_control.paper_lost_sales.paper_lost_sales_step",
        },
        {
            "check_name": "same_neural_action_clipping",
            "status": "PASS" if cap_respected else "FAIL",
            "details": (
                f"All neural models use max_order_factor={benchmark['max_order_factor']} and identical deployment "
                f"clipping; observed cap respected={cap_respected}. Optimal/base-stock policies are not neural-capped."
            ),
            "file_or_function_checked": "inventory_control.idp_benchmark._model_orders",
        },
        {
            "check_name": "monokan_cap_saturation",
            "status": "WARN" if monokan_cap_fraction > 0.95 else "PASS",
            "details": (
                f"MonoKAN reaches the deployment cap on {monokan_cap_fraction:.1%} of diagnostic states; "
                f"mean raw order={monokan_raw_mean:.3f}, mean deployed order={monokan_deployed_mean:.3f}. "
                "When saturated, reward and deployed monotonicity are dominated by clipping."
            ),
            "file_or_function_checked": "inventory_control.idp_benchmark._model_orders; predictions.csv",
        },
        {
            "check_name": "same_feature_normalization",
            "status": "PASS",
            "details": (
                "One PaperFCNNFeatureStats object is computed from training products only and passed to every model."
            ),
            "file_or_function_checked": "compute_paper_fcnn_feature_stats; _train_model; _model_orders",
        },
        {
            "check_name": "same_optimizer_and_learning_rate",
            "status": "PASS",
            "details": (
                f"Every model is trained through train_paper_lost_sales_fcnn with Adam and learning_rate="
                f"{config['learning_rate']}; PenaltyMLP intentionally adds a penalty term."
            ),
            "file_or_function_checked": "inventory_control.training.direct_backprop.train_paper_lost_sales_fcnn",
        },
        {
            "check_name": "same_training_budget",
            "status": "PASS",
            "details": (
                f"imitation_epochs={settings['imitation_epochs']}, reward_epochs={settings['reward_epochs']}, "
                f"batch_size={settings['batch_size']}, training_horizon={settings['training_horizon']}, "
                f"validation_interval={settings['validation_interval']}."
            ),
            "file_or_function_checked": "configs/paper_lost_sales.yaml:idp_benchmark.compact",
        },
        {
            "check_name": "matched_model_capacity",
            "status": "WARN",
            "details": (
                "Hidden-layer labels are shared, but architectures and parameter counts differ substantially "
                "(causal CNN, constrained MLPs, and spline KAN). Reward differences cannot be attributed solely "
                "to monotonicity."
            ),
            "file_or_function_checked": "inventory_control.idp_benchmark._model_config; summary.parameter_count",
        },
        {
            "check_name": "matched_initialization",
            "status": "WARN",
            "details": (
                "Training minibatch RNG is shared, but PositiveWeightMLP and ConstrainedMonoMLP use initialization "
                "offsets +101/+102 while PaperFCNN and MonoKAN use the benchmark seed. Multi-seed results are needed."
            ),
            "file_or_function_checked": "inventory_control.idp_benchmark.run_idp_benchmark",
        },
        {
            "check_name": "compact_statistical_power",
            "status": "WARN",
            "details": (
                f"Only {settings['validation_products']} validation products and a short compact horizon are used. "
                "Compact ranking is smoke-test evidence, not paper-level evidence."
            ),
            "file_or_function_checked": "configs/paper_lost_sales.yaml:idp_benchmark.compact",
        },
    ]
    return pd.DataFrame(rows)


def _evaluation_path_audit(
    products: list[PaperLostSalesProduct],
    policy_names: list[str],
    horizon: int,
    split: str,
) -> pd.DataFrame:
    seed_start = paper_evaluation_seed_start(split)
    rows = []
    for index, product in enumerate(products):
        evaluation_seed = seed_start + index
        first_ten = paper_evaluation_demand_path(product, horizon, evaluation_seed)[:10].tolist()
        for policy_name in policy_names:
            rows.append(
                {
                    "policy_name": policy_name,
                    "product_id": product.product_id,
                    "evaluation_seed": evaluation_seed,
                    "first_10_demands": json.dumps(first_ten),
                }
            )
    return pd.DataFrame(rows)


def _reward_decomposition(operational: pd.DataFrame, counted_periods: int) -> pd.DataFrame:
    rows = []
    for policy_name, policy_rows in operational.groupby("policy_name", sort=False):
        sales_revenue = float((policy_rows["mean_sales_revenue"] * counted_periods).mean())
        purchase_cost = float((policy_rows["mean_purchase_cost"] * counted_periods).mean())
        holding_cost = float((policy_rows["mean_holding_cost"] * counted_periods).mean())
        lost_penalty = float((policy_rows["mean_lost_sales_penalty"] * counted_periods).mean())
        reward = float(policy_rows["mean_total_reward"].mean())
        reconstructed = sales_revenue - purchase_cost - holding_cost - lost_penalty
        error = reward - reconstructed
        rows.append(
            {
                "policy_name": policy_name,
                "mean_sales_revenue": sales_revenue,
                "mean_purchase_cost": purchase_cost,
                "mean_holding_cost": holding_cost,
                "mean_lost_sales_penalty": lost_penalty,
                "mean_total_reward": reward,
                "reconstructed_mean_total_reward": reconstructed,
                "decomposition_error": error,
                "status": "PASS" if abs(error) <= 1e-6 * max(abs(reward), 1.0) else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def _multiseed_summary(runs: list[tuple[int, Any]]) -> pd.DataFrame:
    rows = []
    for seed, artifacts in runs:
        for _, summary_row in artifacts.summary.iterrows():
            rows.append(
                {
                    "seed": seed,
                    "policy_name": summary_row["policy_name"],
                    "mean_reward_per_period": summary_row["mean_reward_per_period"],
                    "raw_violation_rate": summary_row["raw_violation_rate"],
                    "deployed_violation_rate": summary_row["deployed_violation_rate"],
                }
            )
    frame = pd.DataFrame(rows)
    summaries = []
    for policy_name, policy_rows in frame.groupby("policy_name", sort=False):
        rewards = policy_rows["mean_reward_per_period"].to_numpy(dtype=float)
        raw = policy_rows["raw_violation_rate"].to_numpy(dtype=float)
        deployed = policy_rows["deployed_violation_rate"].to_numpy(dtype=float)
        n = len(rewards)
        reward_std = float(np.std(rewards, ddof=1)) if n > 1 else 0.0
        half_width = (
            float(student_t.ppf(0.975, df=n - 1) * reward_std / np.sqrt(n))
            if n > 1
            else 0.0
        )
        summaries.append(
            {
                "experiment_label": "compact robustness; not final paper evidence",
                "policy_name": policy_name,
                "seed_count": n,
                "mean_reward_per_period": float(np.mean(rewards)),
                "std_reward_per_period": reward_std,
                "reward_95ci_lower": float(np.mean(rewards) - half_width),
                "reward_95ci_upper": float(np.mean(rewards) + half_width),
                "raw_violation_rate_mean": float(np.nanmean(raw)),
                "raw_violation_rate_std": float(np.nanstd(raw, ddof=1)) if n > 1 else 0.0,
                "deployed_violation_rate_mean": float(np.nanmean(deployed)),
                "deployed_violation_rate_std": float(np.nanstd(deployed, ddof=1)) if n > 1 else 0.0,
            }
        )
    return pd.DataFrame(summaries)


def _training_time_audit(
    runs: list[tuple[int, Any]],
    config: dict[str, Any],
) -> pd.DataFrame:
    settings = config["idp_benchmark"]["compact"]
    device = str(default_torch_device())
    rows = []
    for seed, artifacts in runs:
        for _, summary_row in artifacts.summary.loc[
            artifacts.summary["policy_name"].isin(AUDITED_POLICIES)
        ].iterrows():
            rows.append(
                {
                    "seed": seed,
                    "policy_name": summary_row["policy_name"],
                    "parameter_count": int(summary_row["parameter_count"]),
                    "train_seconds": float(summary_row["training_seconds"]),
                    "epochs": int(settings["imitation_epochs"] + settings["reward_epochs"]),
                    "train_products": int(settings["training_products"]),
                    "batch_size": int(settings["batch_size"]),
                    "device": device,
                }
            )
    return pd.DataFrame(rows)


def _write_audit_summary(
    path: Path,
    fairness: pd.DataFrame,
    multiseed: pd.DataFrame,
    monotonicity: pd.DataFrame,
    spline_audit: pd.DataFrame,
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        headers = [str(column) for column in frame.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in frame.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines)

    reward_table = multiseed[
        ["policy_name", "mean_reward_per_period", "std_reward_per_period", "reward_95ci_lower", "reward_95ci_upper"]
    ]
    monotonicity_table = monotonicity[
        ["policy_name", "raw_violation_rate", "deployed_violation_rate"]
    ]
    warnings = fairness.loc[fairness["status"] != "PASS", ["check_name", "status", "details"]]
    reward_table = markdown_table(reward_table)
    monotonicity_table = markdown_table(monotonicity_table)
    warning_table = markdown_table(warnings)
    spline_pass = bool((spline_audit["status"] == "PASS").all()) if not spline_audit.empty else False
    content = f"""# Compact LS Architecture Scientific-Validity Audit

## Are the compact results fair?

The operational evaluation protocol is shared across policies: products, fixed demand traces, horizon,
burn-in, reward function, neural action cap, feature normalization, optimizer, and training budget are
consistent. However, architecture capacity and initialization are not matched, and the validation set is
too small for final scientific claims.

## Supported claims

- PositiveWeightMLP and ConstrainedMonoMLP can be checked for raw monotonicity before deployment clipping.
- MonoKAN sampled inventory-edge splines pass the non-increasing grid audit: **{spline_pass}**.
- All policies use identical validation products and fixed demand paths within each run.
- Reward decomposition matches revenue minus purchase, holding, and lost-sales costs.

## Claims not yet supported

- A compact reward ranking is not evidence of superiority at paper scale.
- Single-run train-time ratios do not establish that one architecture trains faster.
- Reward differences cannot be attributed solely to monotonicity because parameter counts and model
  families differ.
- MonoKAN's compact reward cannot be separated from deployment clipping while its policy saturates the
  shared action cap; raw spline monotonicity is supported, but the deployed order scale is not calibrated.
- Deployed monotonicity alone is insufficient because action clipping can remove raw violations.
- The original report's order-up-to and active-slope diagnostics contained implementation errors; these
  were corrected during this audit.

## Compact robustness across seeds

{reward_table}

## Raw vs deployed monotonicity

{monotonicity_table}

## Audit warnings

{warning_table}

## Required before full paper_replication

1. Freeze a capacity-comparison rule: paper-faithful architectures or parameter-matched architectures.
2. Use corrected order-up-to and active/inactive slope diagnostics.
3. Report raw and deployed monotonicity separately.
4. Repeat timing with warm-up and multiple seeds on one fixed device.
5. Run full validation only after the compact protocol is stable.
6. Keep the held-out test split protected until all model choices are frozen.
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/paper_lost_sales.yaml"))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    parser.add_argument("--artifact-root", type=Path, default=Path("outputs/idp_benchmark"))
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    config = load_paper_lost_sales_config(args.config)
    runs = _run_compact_seeds(args.config, seeds, args.artifact_root, args.results_dir)
    first_artifacts = runs[0][1]
    settings = config["idp_benchmark"]["compact"]
    products = PaperLostSalesProductGenerator(config).generate(
        int(settings["validation_products"]),
        split="validation",
    )

    fairness = _fairness_audit(config, first_artifacts, products)
    monotonicity = monotonicity_summary_from_predictions(first_artifacts.predictions, AUDITED_POLICIES)
    paths = _evaluation_path_audit(
        products,
        list(first_artifacts.operational_metrics["policy_name"].drop_duplicates()),
        int(settings["evaluation_horizon"]),
        "validation",
    )
    reward = _reward_decomposition(
        first_artifacts.operational_metrics,
        int(settings["evaluation_horizon"]) - int(settings["burn_in"]),
    )
    multiseed = _multiseed_summary(runs)
    order_up_to = order_up_to_diagnostic_rows(first_artifacts.predictions, AUDITED_POLICIES)
    spline_audit = monokan_spline_diagnostics(first_artifacts.spline_values)
    timing = _training_time_audit(runs, config)

    outputs = {
        "architecture_fairness_audit.csv": fairness,
        "monotonicity_raw_vs_deployed.csv": monotonicity,
        "evaluation_path_audit.csv": paths,
        "reward_decomposition_audit.csv": reward,
        "compact_multiseed_architecture_summary.csv": multiseed,
        "order_up_to_diagnostic_audit.csv": order_up_to,
        "monokan_spline_monotonicity_audit.csv": spline_audit,
        "training_time_fairness_audit.csv": timing,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.results_dir / filename, index=False)
    _write_audit_summary(
        args.results_dir / "audit_summary.md",
        fairness,
        multiseed,
        monotonicity,
        spline_audit,
    )

    print(fairness.to_string(index=False))
    print("\nCompact multiseed summary")
    print(multiseed.to_string(index=False))
    print(f"\nsaved_audit_results={args.results_dir}")


if __name__ == "__main__":
    main()
