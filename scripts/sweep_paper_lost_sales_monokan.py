"""Validation-only sweep for adapted MonoKAN paper lost-sales policies."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.paper_evaluation import PAPER_VALIDATION_EVALUATION_SEED_START
from inventory_control.paper_lost_sales import evaluate_paper_lost_sales_policy, historical_demand_base_stock_policy
from inventory_control.policies.neural.fcnn import compute_paper_fcnn_feature_stats
from inventory_control.policies.neural.monokan import (
    PaperMonoKANPolicyAdapter,
    make_paper_monokan_model,
)
from inventory_control.products import PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import train_paper_lost_sales_fcnn


def _mean_policy_metrics(config: dict, products: list, policy_factory) -> dict[str, float]:
    metrics = [
        evaluate_paper_lost_sales_policy(
            product,
            policy_factory(product),
            horizon=int(config["evaluation_horizon"]),
            n_episodes=1,
            seed=PAPER_VALIDATION_EVALUATION_SEED_START + index,
            burn_in=int(config["burn_in"]),
        )
        for index, product in enumerate(products)
    ]
    columns = [
        "mean_total_reward",
        "service_level",
        "lost_sales_rate",
        "mean_order_quantity",
        "mean_inventory",
        "mean_holding_cost",
        "mean_purchase_cost",
        "mean_sales_revenue",
        "mean_lost_sales_penalty",
    ]
    return {column: float(sum(row[column] for row in metrics) / len(metrics)) for column in columns}


def main() -> None:
    """Train several MonoKAN settings and save validation-only metrics."""
    config = load_paper_lost_sales_config()
    runtime = config.get("runtime", {})
    generator = PaperLostSalesProductGenerator(config)
    n_products = min(
        int(config["training_products"]),
        int(runtime.get("monokan_sweep_train_products", min(128, int(runtime.get("fcnn_train_products", 128))))),
    )
    train_products = generator.generate(n_products, split="train")
    validation_products = generator.generate(
        min(int(runtime.get("fcnn_validation_products", 20)), int(runtime.get("fcnn_test_products", 20))),
        split="validation",
    )
    feature_stats = compute_paper_fcnn_feature_stats(train_products)
    historical_metrics = _mean_policy_metrics(config, validation_products, historical_demand_base_stock_policy)

    base = config.get("monokan", {})
    sweep_configs = [
        {
            "label": "small_baseline_cap4",
            "hidden_layers": int(base.get("hidden_layers", 1)),
            "hidden_units": 8,
            "grid": 3,
            "max_order_factor": float(base.get("max_order_factor", 4.0)),
            "learning_rate": 5e-4,
            "imitation_epochs": 8,
        },
        {
            "label": "tighter_cap",
            "hidden_layers": int(base.get("hidden_layers", 1)),
            "hidden_units": 8,
            "grid": 3,
            "max_order_factor": 2.0,
            "learning_rate": 5e-4,
            "imitation_epochs": 8,
        },
        {
            "label": "small_grid_tighter_cap",
            "hidden_layers": 1,
            "hidden_units": 8,
            "grid": 3,
            "max_order_factor": 2.0,
            "learning_rate": 2e-4,
            "imitation_epochs": 16,
        },
        {
            "label": "small_grid_tightest_cap",
            "hidden_layers": 1,
            "hidden_units": 8,
            "grid": 3,
            "max_order_factor": 1.5,
            "learning_rate": 2e-4,
            "imitation_epochs": 16,
        },
    ]
    train_horizon = min(int(config["training_horizon_T"]), int(runtime.get("monokan_sweep_horizon", 40)))
    train_epochs = min(int(config["epochs"]), int(runtime.get("monokan_sweep_epochs", 8)))
    batch_size = min(int(config["batch_size"]), int(runtime.get("monokan_sweep_batch_size", 64)), n_products)

    rows = []
    for sweep in sweep_configs:
        model_config = {
            **config,
            "monokan": {
                **base,
                "hidden_layers": sweep["hidden_layers"],
                "hidden_units": sweep["hidden_units"],
                "grid": sweep["grid"],
            },
        }
        torch.manual_seed(int(config["product_generation"]["seed"]))
        model = make_paper_monokan_model(model_config)

        def validation_callback(epoch: int, current_model) -> float:
            return _mean_policy_metrics(
                config,
                validation_products,
                lambda product: PaperMonoKANPolicyAdapter(
                    current_model,
                    product,
                    feature_stats=feature_stats,
                    max_order_factor=float(sweep["max_order_factor"]),
                    scale_action_by_history_mean=bool(base.get("scale_action_by_history_mean", True)),
                ),
            )["mean_total_reward"]

        result = train_paper_lost_sales_fcnn(
            train_products,
            model,
            horizon=train_horizon,
            batch_size=batch_size,
            epochs=train_epochs,
            learning_rate=float(sweep["learning_rate"]),
            seed=int(config["product_generation"]["seed"]),
            imitation_epochs=int(sweep["imitation_epochs"]),
            validation_products=validation_products,
            validation_interval=int(config["fcnn"].get("validation_interval", 0)),
            validation_callback=validation_callback,
            feature_stats=feature_stats,
            max_order_factor=float(sweep["max_order_factor"]),
            scale_action_by_history_mean=bool(base.get("scale_action_by_history_mean", True)),
        )
        if result.best_state_dict is not None:
            model.load_state_dict(result.best_state_dict)

        validation_metrics = _mean_policy_metrics(
            config,
            validation_products,
            lambda product: PaperMonoKANPolicyAdapter(
                model,
                product,
                feature_stats=feature_stats,
                max_order_factor=float(sweep["max_order_factor"]),
                scale_action_by_history_mean=bool(base.get("scale_action_by_history_mean", True)),
            ),
        )
        row = {
            **sweep,
            "train_products": n_products,
            "train_horizon": train_horizon,
            "train_epochs": train_epochs,
            "batch_size": batch_size,
            "best_validation_reward": result.best_validation_reward,
            "validation_historical_mean_total_reward": historical_metrics["mean_total_reward"],
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        rows.append(row)
        print(
            f"label={sweep['label']} "
            f"reward={validation_metrics['mean_total_reward']:.4f} "
            f"inventory={validation_metrics['mean_inventory']:.4f} "
            f"order={validation_metrics['mean_order_quantity']:.4f}"
        )

    results = pd.DataFrame(rows).sort_values("validation_mean_total_reward", ascending=False)
    output_path = Path("outputs/results/paper_lost_sales_monokan_sweep_validation.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(results.to_string(index=False))
    print(f"saved_sweep={output_path}")


if __name__ == "__main__":
    main()
