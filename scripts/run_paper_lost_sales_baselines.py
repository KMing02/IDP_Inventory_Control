"""Run Maggiar-style lost-sales base-stock baselines."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.paper_evaluation import (
    paper_evaluation_product_count,
    paper_evaluation_seed_start,
    require_final_test_confirmation,
)
from inventory_control.paper_lost_sales import (
    evaluate_paper_lost_sales_policy,
    historical_demand_base_stock_policy,
    non_omniscient_gamma_base_stock_policy,
    omniscient_gamma_base_stock_policy,
)
from inventory_control.products import PaperLostSalesProductGenerator


def main() -> None:
    """Evaluate oracle and historical base-stock policies."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--final-test", action="store_true", help="Required when evaluating the held-out test set.")
    args = parser.parse_args()
    require_final_test_confirmation(args.split, args.final_test)

    config = load_paper_lost_sales_config()
    generator = PaperLostSalesProductGenerator(config)
    n_products = paper_evaluation_product_count(config, args.split)
    seed_start = paper_evaluation_seed_start(args.split)
    products = generator.generate(n_products, split=args.split)
    rows = []
    for index, product in enumerate(products):
        policies = [
            omniscient_gamma_base_stock_policy(product),
            non_omniscient_gamma_base_stock_policy(product),
            historical_demand_base_stock_policy(product, fit_gamma=False),
        ]
        for policy in policies:
            metrics = evaluate_paper_lost_sales_policy(
                product,
                policy,
                horizon=int(config["evaluation_horizon"]),
                n_episodes=1,
                seed=seed_start + index,
                burn_in=int(config["burn_in"]),
            )
            rows.append(
                {
                    "product_id": product.product_id,
                    "policy_name": policy.name,
                    "evaluation_split": args.split,
                    "evaluation_seed": seed_start + index,
                    "evaluation_horizon": int(config["evaluation_horizon"]),
                    "burn_in": int(config["burn_in"]),
                    "tuned_parameters": json.dumps({"S": policy.S}, sort_keys=True),
                    **metrics,
                }
            )
    results = pd.DataFrame(rows)
    suffix = "" if args.split == "test" else "_validation"
    output_path = Path(f"outputs/results/paper_lost_sales_baselines{suffix}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(
        results.groupby("policy_name")[
            [
                "mean_total_reward",
                "service_level",
                "lost_sales_rate",
                "mean_order_quantity",
                "mean_inventory",
            ]
        ]
        .mean()
        .to_string()
    )


if __name__ == "__main__":
    main()
