"""Train one MLP across generated lead-time products."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from inventory_control.evaluation.runner import evaluate_policy
from inventory_control.neural.policies import LeadTimeTorchPolicyAdapter, TorchMLPPolicy
from inventory_control.neural.training import train_lead_time_product_policy
from inventory_control.products import (
    Product,
    ProductGenerator,
    make_product_env_factory,
    product_generator_config_from_dict,
)
from scripts.run_product_benchmark_baselines import load_config


def _lead_time_products(products: list[Product]) -> list[Product]:
    return [product for product in products if product.environment_type == "lost_sales_lead_time"]


def _evaluate_split(
    split_name: str,
    products: list[Product],
    policy: TorchMLPPolicy,
    horizon: int,
    n_episodes: int,
    max_lead_time: int,
    seed: int,
) -> list[dict[str, float | str]]:
    rows = []
    for index, product in enumerate(products):
        historical_mean = float(sum(product.historical_demand) / len(product.historical_demand))
        historical_std = float(pd.Series(product.historical_demand).std(ddof=0))
        adapter = LeadTimeTorchPolicyAdapter(
            model=policy,
            historical_mean_demand=historical_mean,
            historical_std_demand=historical_std,
            holding_cost=product.holding_cost,
            lost_sales_cost=product.lost_sales_cost,
            unit_order_cost=product.unit_order_cost,
            lead_time=product.lead_time or 1,
            max_lead_time=max_lead_time,
        )
        metrics, _ = evaluate_policy(
            make_product_env_factory(product, horizon=horizon),
            adapter,
            horizon=horizon,
            n_episodes=n_episodes,
            seed=seed + index,
        )
        rows.append(
            {
                "split": split_name,
                "product_id": product.product_id,
                "policy_name": adapter.name,
                "mean_total_cost": metrics["mean_total_cost"],
                "mean_cost_per_period": metrics["mean_cost_per_period"],
                "service_level": metrics["mean_service_level"],
                "lost_sales_rate": metrics["mean_lost_sales_rate"],
            }
        )
    return rows


def main() -> None:
    """Train and evaluate product-level lead-time MLP."""
    config = load_config()
    generator = ProductGenerator(product_generator_config_from_dict(config), seed=int(config["seed"]))
    splits = generator.generate_splits()
    train_products = _lead_time_products(splits["train"])
    validation_products = _lead_time_products(splits["validation"])
    test_products = _lead_time_products(splits["test"])
    all_lead_time_products = train_products + validation_products + test_products
    if not train_products:
        raise ValueError("product benchmark config generated no lead-time training products")

    horizon = int(config["horizon"])
    max_lead_time = max(product.lead_time or 1 for product in all_lead_time_products)
    input_dim = 1 + max_lead_time + 6
    policy = TorchMLPPolicy(input_dim=input_dim, hidden_dims=(64, 64))
    result = train_lead_time_product_policy(
        train_products,
        policy,
        horizon=horizon,
        max_lead_time=max_lead_time,
        n_steps=200,
        batch_size=16,
        learning_rate=1e-3,
        seed=int(config["seed"]),
    )

    rows = []
    rows.extend(
        _evaluate_split(
            "train",
            train_products,
            result.policy,
            horizon,
            n_episodes=int(config["evaluation_episodes"]),
            max_lead_time=max_lead_time,
            seed=int(config["seed"]) + 10_000,
        )
    )
    rows.extend(
        _evaluate_split(
            "validation",
            validation_products,
            result.policy,
            horizon,
            n_episodes=int(config["evaluation_episodes"]),
            max_lead_time=max_lead_time,
            seed=int(config["seed"]) + 20_000,
        )
    )
    rows.extend(
        _evaluate_split(
            "test",
            test_products,
            result.policy,
            horizon,
            n_episodes=int(config["evaluation_episodes"]),
            max_lead_time=max_lead_time,
            seed=int(config["seed"]) + 30_000,
        )
    )

    output_path = Path("outputs/results/product_lead_time_mlp.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(output_path, index=False)
    print(f"initial_training_loss={result.losses[0]:.4f}")
    print(f"final_training_loss={result.losses[-1]:.4f}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
