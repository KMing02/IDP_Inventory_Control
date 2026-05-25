"""Run a tiny paper lost-sales baseline and FCNN smoke test."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.paper_evaluation import PAPER_TEST_EVALUATION_SEED_START
from inventory_control.paper_lost_sales import (
    evaluate_paper_lost_sales_policy,
    historical_demand_base_stock_policy,
    oracle_gamma_base_stock_policy,
)
from inventory_control.policies.neural.fcnn import (
    PaperFCNNPolicyAdapter,
    compute_paper_fcnn_feature_stats,
    make_paper_fcnn_model,
)
from inventory_control.products import PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import train_paper_lost_sales_fcnn


def main() -> None:
    """Run a fast end-to-end check without relying on generated artifacts."""
    config = load_paper_lost_sales_config("configs/paper_lost_sales_debug.yaml")
    generator = PaperLostSalesProductGenerator(config)
    train_products = generator.generate(int(config["runtime"]["fcnn_train_products"]), split="train")
    test_products = generator.generate(int(config["runtime"]["fcnn_test_products"]), split="test")
    feature_stats = compute_paper_fcnn_feature_stats(train_products)
    max_order_factor = float(config["fcnn"]["max_order_factor"])
    scale_action_by_history_mean = bool(config["fcnn"].get("scale_action_by_history_mean", False))

    baseline_product = test_products[0]
    baseline_policies = [
        historical_demand_base_stock_policy(baseline_product),
        oracle_gamma_base_stock_policy(
            baseline_product,
            n_samples=int(config["tuning"]["oracle_mc_samples"]),
            seed=10_000,
        ),
    ]
    for policy in baseline_policies:
        metrics = evaluate_paper_lost_sales_policy(
            baseline_product,
            policy,
            horizon=int(config["evaluation_horizon"]),
            n_episodes=1,
            seed=PAPER_TEST_EVALUATION_SEED_START,
            burn_in=int(config["burn_in"]),
        )
        print(f"{policy.name}: mean_total_reward={metrics['mean_total_reward']:.3f}")

    torch.manual_seed(int(config["product_generation"]["seed"]))
    model = make_paper_fcnn_model(config)
    result = train_paper_lost_sales_fcnn(
        train_products,
        model,
        horizon=int(config["training_horizon_T"]),
        batch_size=int(config["batch_size"]),
        epochs=int(config["epochs"]),
        learning_rate=float(config["learning_rate"]),
        seed=int(config["product_generation"]["seed"]),
        imitation_epochs=int(config["fcnn"]["imitation_epochs"]),
        feature_stats=feature_stats,
        max_order_factor=max_order_factor,
        scale_action_by_history_mean=scale_action_by_history_mean,
    )
    fcnn_policy = PaperFCNNPolicyAdapter(
        model,
        baseline_product,
        feature_stats=feature_stats,
        max_order_factor=max_order_factor,
        scale_action_by_history_mean=scale_action_by_history_mean,
    )
    fcnn_metrics = evaluate_paper_lost_sales_policy(
        baseline_product,
        fcnn_policy,
        horizon=int(config["evaluation_horizon"]),
        n_episodes=1,
        seed=PAPER_TEST_EVALUATION_SEED_START,
        burn_in=int(config["burn_in"]),
    )
    print(f"FCNN: mean_total_reward={fcnn_metrics['mean_total_reward']:.3f}")
    print(f"training_loss_initial={result.training_losses[0]:.3f}")
    print(f"training_loss_final={result.training_losses[-1]:.3f}")
    print("quick_smoke_test=passed")


if __name__ == "__main__":
    main()
