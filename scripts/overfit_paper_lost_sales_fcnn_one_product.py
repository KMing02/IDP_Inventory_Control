"""One-product FCNN overfit diagnostic for paper lost-sales training."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.paper_lost_sales import (
    evaluate_paper_lost_sales_policy,
    historical_demand_base_stock_policy,
)
from inventory_control.policies.neural.fcnn import FCNNPolicy, PaperFCNNPolicyAdapter, compute_paper_fcnn_feature_stats
from inventory_control.products import PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import train_paper_lost_sales_fcnn


def main() -> None:
    """Train on one product and compare against historical base-stock reward."""
    config = load_paper_lost_sales_config()
    product = PaperLostSalesProductGenerator(config).generate(1, split="train")[0]
    feature_stats = compute_paper_fcnn_feature_stats([product])
    max_order_factor = float(config["fcnn"].get("max_order_factor", 2.0))
    model = FCNNPolicy(
        input_dim=1 + int(config["history_length_H"]) + 4,
        hidden_layers=int(config["fcnn"]["hidden_layers"]),
        hidden_units=int(config["fcnn"]["hidden_units"]),
        activation=str(config["fcnn"]["activation"]),
    )
    result = train_paper_lost_sales_fcnn(
        [product],
        model,
        horizon=int(config["training_horizon_T"]),
        batch_size=1,
        epochs=100,
        learning_rate=float(config["learning_rate"]),
        seed=int(config["product_generation"]["seed"]),
        imitation_epochs=50,
        feature_stats=feature_stats,
        max_order_factor=max_order_factor,
    )
    fcnn_metrics = evaluate_paper_lost_sales_policy(
        product,
        PaperFCNNPolicyAdapter(model, product, feature_stats=feature_stats, max_order_factor=max_order_factor),
        horizon=int(config["evaluation_horizon"]),
        n_episodes=1,
        seed=55_000,
        burn_in=int(config["burn_in"]),
    )
    historical_metrics = evaluate_paper_lost_sales_policy(
        product,
        historical_demand_base_stock_policy(product),
        horizon=int(config["evaluation_horizon"]),
        n_episodes=1,
        seed=55_000,
        burn_in=int(config["burn_in"]),
    )
    print(f"initial_training_loss={result.training_losses[0]:.4f}")
    print(f"final_training_loss={result.training_losses[-1]:.4f}")
    print(f"fcnn_mean_total_reward={fcnn_metrics['mean_total_reward']:.4f}")
    print(f"historical_mean_total_reward={historical_metrics['mean_total_reward']:.4f}")


if __name__ == "__main__":
    main()
