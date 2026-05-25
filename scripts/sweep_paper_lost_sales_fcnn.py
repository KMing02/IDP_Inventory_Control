"""Sweep FCNN sizes for the paper lost-sales replication."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.paper_evaluation import (
    PAPER_TEST_EVALUATION_SEED_START,
    PAPER_VALIDATION_EVALUATION_SEED_START,
    paper_test_product_count,
)
from inventory_control.paper_lost_sales import evaluate_paper_lost_sales_policy, historical_demand_base_stock_policy
from inventory_control.policies.neural.fcnn import (
    PaperFCNNPolicyAdapter,
    compute_paper_fcnn_feature_stats,
    make_paper_fcnn_model,
)
from inventory_control.products import PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import train_paper_lost_sales_fcnn


def _mean_policy_reward(config, products, policy_factory, seed_start: int) -> float:
    rewards = [
        evaluate_paper_lost_sales_policy(
            product,
            policy_factory(product),
            horizon=int(config["evaluation_horizon"]),
            n_episodes=1,
            seed=seed_start + idx,
            burn_in=int(config["burn_in"]),
        )["mean_total_reward"]
        for idx, product in enumerate(products)
    ]
    return sum(rewards) / len(rewards)


def main() -> None:
    """Train several FCNN configurations and save validation/test rewards."""
    config = load_paper_lost_sales_config()
    runtime = config.get("runtime", {})
    generator = PaperLostSalesProductGenerator(config)
    n_products = min(int(config["training_products"]), int(config["runtime"]["fcnn_train_products"]))
    train_products = generator.generate(n_products, split="train")
    validation_products = generator.generate(
        min(int(runtime.get("fcnn_validation_products", 20)), int(runtime["fcnn_test_products"])),
        split="validation",
    )
    test_products = generator.generate(paper_test_product_count(config), split="test")
    feature_stats = compute_paper_fcnn_feature_stats(train_products)
    max_order_factor = float(config["fcnn"].get("max_order_factor", 2.0))
    scale_action_by_history_mean = bool(config["fcnn"].get("scale_action_by_history_mean", False))

    sweep_configs = [(2, 32), (3, 64), (3, 128)]
    rows = []
    historical_validation_reward = _mean_policy_reward(
        config,
        validation_products,
        historical_demand_base_stock_policy,
        seed_start=PAPER_VALIDATION_EVALUATION_SEED_START,
    )
    historical_test_reward = _mean_policy_reward(
        config,
        test_products,
        historical_demand_base_stock_policy,
        seed_start=PAPER_TEST_EVALUATION_SEED_START,
    )

    for hidden_layers, hidden_units in sweep_configs:
        torch.manual_seed(int(config["product_generation"]["seed"]))
        model_config = {**config, "fcnn": {**config["fcnn"], "hidden_layers": hidden_layers, "hidden_units": hidden_units}}
        model = make_paper_fcnn_model(model_config)

        def validation_callback(epoch: int, current_model) -> float:
            return _mean_policy_reward(
                config,
                validation_products,
                lambda product: PaperFCNNPolicyAdapter(
                    current_model,
                    product,
                    feature_stats=feature_stats,
                    max_order_factor=max_order_factor,
                    scale_action_by_history_mean=scale_action_by_history_mean,
                ),
                seed_start=PAPER_VALIDATION_EVALUATION_SEED_START,
            )

        result = train_paper_lost_sales_fcnn(
            train_products,
            model,
            horizon=int(config["training_horizon_T"]),
            batch_size=min(int(config["batch_size"]), n_products),
            epochs=min(int(config["epochs"]), int(runtime.get("fcnn_train_epochs", 50))),
            learning_rate=float(config["learning_rate"]),
            seed=int(config["product_generation"]["seed"]),
            imitation_epochs=int(config["fcnn"].get("imitation_epochs", 0)),
            validation_products=validation_products,
            validation_interval=int(config["fcnn"].get("validation_interval", 0)),
            validation_callback=validation_callback,
            feature_stats=feature_stats,
            max_order_factor=max_order_factor,
            scale_action_by_history_mean=scale_action_by_history_mean,
        )
        if result.best_state_dict is not None:
            model.load_state_dict(result.best_state_dict)
        validation_reward = _mean_policy_reward(
            config,
            validation_products,
            lambda product: PaperFCNNPolicyAdapter(
                model,
                product,
                feature_stats=feature_stats,
                max_order_factor=max_order_factor,
                scale_action_by_history_mean=scale_action_by_history_mean,
            ),
            seed_start=PAPER_VALIDATION_EVALUATION_SEED_START,
        )
        test_reward = _mean_policy_reward(
            config,
            test_products,
            lambda product: PaperFCNNPolicyAdapter(
                model,
                product,
                feature_stats=feature_stats,
                max_order_factor=max_order_factor,
                scale_action_by_history_mean=scale_action_by_history_mean,
            ),
            seed_start=PAPER_TEST_EVALUATION_SEED_START,
        )
        rows.append(
            {
                "hidden_layers": hidden_layers,
                "hidden_units": hidden_units,
                "best_validation_reward": result.best_validation_reward,
                "validation_mean_total_reward": validation_reward,
                "test_mean_total_reward": test_reward,
                "validation_historical_mean_total_reward": historical_validation_reward,
                "test_historical_mean_total_reward": historical_test_reward,
            }
        )
        print(
            f"hidden_layers={hidden_layers} hidden_units={hidden_units} "
            f"validation_reward={validation_reward:.4f} test_reward={test_reward:.4f}"
        )

    output_path = Path("outputs/results/paper_lost_sales_fcnn_sweep.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"saved_sweep={output_path}")


if __name__ == "__main__":
    main()
