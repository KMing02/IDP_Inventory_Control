"""Evaluate trained paper lost-sales MonoKAN."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.paper_evaluation import (
    paper_evaluation_product_count,
    paper_evaluation_seed_start,
    require_final_test_confirmation,
)
from inventory_control.paper_lost_sales import evaluate_paper_lost_sales_policy, historical_demand_base_stock_policy
from inventory_control.policies.neural.fcnn import PaperFCNNFeatureStats, compute_paper_fcnn_feature_stats
from inventory_control.policies.neural.fcnn import infer_model_device
from inventory_control.policies.neural.monokan import PaperMonoKANPolicyAdapter, make_paper_monokan_model
from inventory_control.products import PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import default_torch_device


def main() -> None:
    """Evaluate saved MonoKAN on validation products by default."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--final-test", action="store_true", help="Required when evaluating the held-out test set.")
    args = parser.parse_args()
    require_final_test_confirmation(args.split, args.final_test)

    config = load_paper_lost_sales_config()
    model_path = Path("outputs/models/paper_lost_sales_monokan.pt")
    metadata_path = Path("outputs/models/paper_lost_sales_monokan_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    device = default_torch_device()
    model = make_paper_monokan_model(config, metadata=metadata).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

    generator = PaperLostSalesProductGenerator(config)
    n_training_products = min(int(config["training_products"]), int(config["runtime"]["fcnn_train_products"]))
    training_products = generator.generate(n_training_products, split="train")
    if "feature_demand_scale" in metadata and "feature_price_scale" in metadata:
        feature_stats = PaperFCNNFeatureStats(
            demand_scale=float(metadata["feature_demand_scale"]),
            price_scale=float(metadata["feature_price_scale"]),
        )
    else:
        feature_stats = compute_paper_fcnn_feature_stats(training_products)

    monokan_config = config.get("monokan", {})
    max_order_factor = float(metadata.get("max_order_factor", monokan_config.get("max_order_factor", 2.0)))
    scale_action_by_history_mean = bool(
        metadata.get("scale_action_by_history_mean", monokan_config.get("scale_action_by_history_mean", False))
    )
    n_products = paper_evaluation_product_count(config, args.split)
    seed_start = paper_evaluation_seed_start(args.split)
    products = generator.generate(n_products, split=args.split)

    rows = []
    for index, product in enumerate(products):
        policy = PaperMonoKANPolicyAdapter(
            model,
            product,
            feature_stats=feature_stats,
            max_order_factor=max_order_factor,
            scale_action_by_history_mean=scale_action_by_history_mean,
        )
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
                **metrics,
            }
        )
        if index < 5:
            historical_policy = historical_demand_base_stock_policy(product)
            inventory = float(product.initial_inventory)
            print(
                "action_scale "
                f"product_id={product.product_id} "
                f"inventory={inventory:.3f} "
                f"monokan_q={policy.act(inventory, 0):.3f} "
                f"historical_q={historical_policy.act(inventory, 0):.3f} "
                f"historical_S={historical_policy.S:.3f}"
            )

    results = pd.DataFrame(rows)
    suffix = "" if args.split == "test" else "_validation"
    output_path = Path(f"outputs/results/paper_lost_sales_monokan_eval{suffix}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(
        results[
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
    print(f"device={infer_model_device(model)}")
    print(f"saved_results={output_path}")


if __name__ == "__main__":
    main()
