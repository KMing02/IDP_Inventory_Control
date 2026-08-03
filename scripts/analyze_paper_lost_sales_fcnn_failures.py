"""Legacy raw-action FCNN failure analysis.

Prefer residual order-up-to benchmark summaries for current architecture
comparisons.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.paper_evaluation import (
    paper_evaluation_product_count,
    paper_evaluation_seed_start,
    require_final_test_confirmation,
)
from inventory_control.products import PaperLostSalesProductGenerator


def _load_policy_results(split: str) -> pd.DataFrame:
    suffix = "" if split == "test" else "_validation"
    baseline_path = Path(f"outputs/results/paper_lost_sales_baselines{suffix}.csv")
    fcnn_path = Path(f"outputs/results/paper_lost_sales_fcnn_eval{suffix}.csv")
    if not baseline_path.exists() or not fcnn_path.exists():
        raise FileNotFoundError(
            "Run scripts/run_paper_lost_sales_baselines.py and "
            "scripts/evaluate_paper_lost_sales_fcnn.py before failure analysis."
        )
    return pd.concat([pd.read_csv(baseline_path), pd.read_csv(fcnn_path)], ignore_index=True)


def _assert_common_products_and_seeds(results: pd.DataFrame, config: dict, split: str) -> None:
    expected_count = paper_evaluation_product_count(config, split)
    expected_products = [f"{split}_{idx:06d}" for idx in range(expected_count)]
    seed_start = paper_evaluation_seed_start(split)
    expected_seeds = {product_id: seed_start + idx for idx, product_id in enumerate(expected_products)}
    for policy_name, policy_results in results.groupby("policy_name"):
        product_ids = policy_results["product_id"].tolist()
        if product_ids != expected_products:
            raise ValueError(f"{policy_name} is not evaluated on the common ordered {split} product set.")
        if "evaluation_seed" in policy_results.columns:
            observed = dict(zip(policy_results["product_id"], policy_results["evaluation_seed"]))
            if observed != expected_seeds:
                raise ValueError(f"{policy_name} is not evaluated with the common demand seeds.")
        if "evaluation_horizon" in policy_results.columns:
            if not (policy_results["evaluation_horizon"] == int(config["evaluation_horizon"])).all():
                raise ValueError(f"{policy_name} has inconsistent evaluation horizon.")
        if "burn_in" in policy_results.columns:
            if not (policy_results["burn_in"] == int(config["burn_in"])).all():
                raise ValueError(f"{policy_name} has inconsistent burn-in.")


def _comparison_summary(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby("policy_name")["mean_total_reward"]
        .agg(
            mean_total_reward="mean",
            median_total_reward="median",
            std_total_reward="std",
        )
        .reset_index()
    )
    pivot = results.pivot(index="product_id", columns="policy_name", values="mean_total_reward")
    fcnn_beats_historical = float((pivot["FCNN"] > pivot["NonOmniscientOptimal"]).mean())
    fcnn_beats_oracle = float((pivot["FCNN"] > pivot["OmniscientOptimal"]).mean())
    summary["pct_products_fcnn_beats_historical"] = np.nan
    summary["pct_products_fcnn_beats_oracle"] = np.nan
    fcnn_row = summary["policy_name"] == "FCNN"
    summary.loc[fcnn_row, "pct_products_fcnn_beats_historical"] = 100.0 * fcnn_beats_historical
    summary.loc[fcnn_row, "pct_products_fcnn_beats_oracle"] = 100.0 * fcnn_beats_oracle
    return summary


def _product_features(config: dict, split: str) -> pd.DataFrame:
    products = PaperLostSalesProductGenerator(config).generate(paper_evaluation_product_count(config, split), split=split)
    rows = []
    for product in products:
        history = np.asarray(product.historical_demand, dtype=float)
        rows.append(
            {
                "product_id": product.product_id,
                "price": product.price,
                "purchase_cost": product.purchase_cost,
                "holding_cost": product.holding_cost,
                "lost_sale_penalty": product.lost_sale_penalty,
                "initial_inventory": product.initial_inventory,
                "historical_demand_mean": float(history.mean()),
                "historical_demand_std": float(history.std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def _worst_cases(results: pd.DataFrame, config: dict, split: str) -> pd.DataFrame:
    pivot = results.pivot(index="product_id", columns="policy_name", values="mean_total_reward").reset_index()
    pivot = pivot.rename(
        columns={
            "FCNN": "fcnn_mean_total_reward",
            "NonOmniscientOptimal": "historical_mean_total_reward",
            "OmniscientOptimal": "oracle_mean_total_reward",
        }
    )
    order_stats = {}
    for policy_name, prefix in [
        ("FCNN", "fcnn"),
        ("NonOmniscientOptimal", "historical"),
        ("OmniscientOptimal", "oracle"),
    ]:
        columns = [
            "product_id",
            "mean_order_quantity",
            "min_order_quantity",
            "max_order_quantity",
            "mean_inventory",
            "service_level",
            "lost_sales_rate",
        ]
        frame = results.loc[results["policy_name"] == policy_name, columns].copy()
        frame = frame.rename(columns={column: f"{prefix}_{column}" for column in columns if column != "product_id"})
        order_stats[policy_name] = frame
    worst = pivot.merge(_product_features(config, split), on="product_id", how="left")
    for frame in order_stats.values():
        worst = worst.merge(frame, on="product_id", how="left")
    worst["fcnn_minus_historical_reward"] = worst["fcnn_mean_total_reward"] - worst["historical_mean_total_reward"]
    worst["fcnn_minus_oracle_reward"] = worst["fcnn_mean_total_reward"] - worst["oracle_mean_total_reward"]
    return worst.sort_values("fcnn_mean_total_reward", ascending=True).head(10)


def main() -> None:
    """Create aggregate comparison and FCNN worst-case analysis CSVs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--final-test", action="store_true", help="Required when analyzing held-out test results.")
    args = parser.parse_args()
    require_final_test_confirmation(args.split, args.final_test)

    config = load_paper_lost_sales_config()
    results = _load_policy_results(args.split)
    required = {"FCNN", "NonOmniscientOptimal", "OmniscientOptimal"}
    observed = set(results["policy_name"])
    if not required.issubset(observed):
        raise ValueError(f"Missing required policies: {sorted(required - observed)}")
    _assert_common_products_and_seeds(results, config, args.split)

    summary = _comparison_summary(results)
    worst = _worst_cases(results, config, args.split)

    suffix = "" if args.split == "test" else "_validation"
    summary_path = Path(f"outputs/results/paper_lost_sales_policy_comparison_summary{suffix}.csv")
    worst_path = Path(f"outputs/results/paper_lost_sales_fcnn_worst_cases{suffix}.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    worst.to_csv(worst_path, index=False)

    print(summary.to_string(index=False))
    print(f"saved_summary={summary_path}")
    print(f"saved_worst_cases={worst_path}")


if __name__ == "__main__":
    main()
