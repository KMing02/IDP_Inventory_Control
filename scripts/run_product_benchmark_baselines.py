"""Run product-level generalization benchmark for heuristic baselines."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from inventory_control.envs.demand import DeterministicDemand
from inventory_control.evaluation.runner import evaluate_policy
from inventory_control.evaluation.tuning import (
    base_stock_grid_search,
    inventory_position_base_stock_grid_search,
)
from inventory_control.policies.classical import (
    BaseStockPolicy,
    DualSourcingHeuristicPolicy,
    InventoryPositionBaseStockPolicy,
)
from inventory_control.products import (
    Product,
    ProductGenerator,
    make_product_env_factory,
    product_generator_config_from_dict,
)


def load_config(path: str | Path = "configs/product_benchmark.yaml") -> dict[str, Any]:
    """Load product benchmark config."""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _historical_demand_for_horizon(product: Product, horizon: int) -> list[int]:
    if len(product.historical_demand) < horizon:
        repeats = horizon // len(product.historical_demand) + 1
        return (product.historical_demand * repeats)[:horizon]
    return product.historical_demand[:horizon]


def _tune_from_historical(product: Product, horizon: int, min_stock: int, max_stock: int) -> int:
    historical_demand = DeterministicDemand(_historical_demand_for_horizon(product, horizon))
    env_factory = make_product_env_factory(product, horizon=horizon, demand=historical_demand)
    if product.environment_type == "lost_sales_zero_lead_time":
        grid = base_stock_grid_search(
            env_factory,
            horizon=horizon,
            n_episodes=1,
            seed=0,
            min_stock_level=min_stock,
            max_stock_level=max_stock,
        )
    elif product.environment_type == "lost_sales_lead_time":
        grid = inventory_position_base_stock_grid_search(
            env_factory,
            horizon=horizon,
            n_episodes=1,
            seed=0,
            min_stock_level=min_stock,
            max_stock_level=max_stock,
        )
    else:
        raise ValueError(f"unsupported product environment for base-stock tuning: {product.environment_type}")
    return int(grid.loc[grid["is_best"]].iloc[0]["S"])


def _tune_oracle(
    product: Product,
    horizon: int,
    n_episodes: int,
    seed: int,
    min_stock: int,
    max_stock: int,
) -> int:
    env_factory = make_product_env_factory(product, horizon=horizon)
    if product.environment_type == "lost_sales_zero_lead_time":
        grid = base_stock_grid_search(
            env_factory,
            horizon=horizon,
            n_episodes=n_episodes,
            seed=seed,
            min_stock_level=min_stock,
            max_stock_level=max_stock,
        )
    elif product.environment_type == "lost_sales_lead_time":
        grid = inventory_position_base_stock_grid_search(
            env_factory,
            horizon=horizon,
            n_episodes=n_episodes,
            seed=seed,
            min_stock_level=min_stock,
            max_stock_level=max_stock,
        )
    else:
        raise ValueError(f"unsupported product environment for oracle tuning: {product.environment_type}")
    return int(grid.loc[grid["is_best"]].iloc[0]["S"])


def _evaluate_s(product: Product, s_value: int, horizon: int, n_episodes: int, seed: int) -> dict[str, float]:
    env_factory = make_product_env_factory(product, horizon=horizon)
    if product.environment_type == "lost_sales_zero_lead_time":
        policy = BaseStockPolicy(S=s_value)
    elif product.environment_type == "lost_sales_lead_time":
        policy = InventoryPositionBaseStockPolicy(S=s_value)
    else:
        raise ValueError(f"unsupported product environment: {product.environment_type}")
    metrics, _ = evaluate_policy(env_factory, policy, horizon=horizon, n_episodes=n_episodes, seed=seed)
    return metrics


def _true_mean_demand(product: Product) -> float:
    if product.demand_process_type == "poisson":
        return float(product.demand_parameters["lam"])
    if product.demand_process_type == "negative_binomial":
        return float(product.demand_parameters["mean"])
    return float(sum(product.historical_demand) / len(product.historical_demand))


def _dual_sourcing_parameters_from_mean(product: Product, mean_demand: float) -> dict[str, int]:
    regular_lead_time = product.regular_lead_time or 1
    expedited_lead_time = product.expedited_lead_time or 0
    return {
        "S_regular": int(round(mean_demand * (regular_lead_time + 1))),
        "S_expedited": int(round(mean_demand * (expedited_lead_time + 1))),
    }


def _evaluate_dual_sourcing(
    product: Product,
    parameters: dict[str, int],
    horizon: int,
    n_episodes: int,
    seed: int,
) -> dict[str, float]:
    if product.regular_lead_time is None or product.expedited_lead_time is None:
        raise ValueError("dual-sourcing product is missing lead times")
    policy = DualSourcingHeuristicPolicy(
        S_regular=parameters["S_regular"],
        S_expedited=parameters["S_expedited"],
        regular_lead_time=product.regular_lead_time,
        expedited_lead_time=product.expedited_lead_time,
    )
    metrics, _ = evaluate_policy(
        make_product_env_factory(product, horizon=horizon),
        policy,
        horizon=horizon,
        n_episodes=n_episodes,
        seed=seed,
    )
    return metrics


def _row(
    split: str,
    product: Product,
    policy_name: str,
    tuned_parameters: dict[str, int],
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "split": split,
        "product_id": product.product_id,
        "environment_type": product.environment_type,
        "demand_process_type": product.demand_process_type,
        "policy_name": policy_name,
        "tuned_parameters": json.dumps(tuned_parameters, sort_keys=True),
        "mean_total_cost": metrics["mean_total_cost"],
        "mean_cost_per_period": metrics["mean_cost_per_period"],
        "service_level": metrics["mean_service_level"],
        "lost_sales_rate": metrics["mean_lost_sales_rate"],
    }


def main() -> None:
    """Generate products, evaluate heuristic baselines, and save results."""
    config = load_config()
    generator = ProductGenerator(
        product_generator_config_from_dict(config),
        seed=int(config["seed"]),
    )
    splits = generator.generate_splits()

    horizon = int(config["horizon"])
    evaluation_episodes = int(config["evaluation_episodes"])
    oracle_tuning_episodes = int(config["oracle_tuning_episodes"])
    min_stock = int(config["tuning"]["min_stock_level"])
    max_stock = int(config["tuning"]["max_stock_level"])
    seed = int(config["seed"])

    rows: list[dict[str, Any]] = []
    for split, products in splits.items():
        for product_index, product in enumerate(products):
            eval_seed = seed + 10_000 * (product_index + 1)
            if product.environment_type == "dual_sourcing":
                historical_mean = float(sum(product.historical_demand) / len(product.historical_demand))
                historical_parameters = _dual_sourcing_parameters_from_mean(product, historical_mean)
                historical_metrics = _evaluate_dual_sourcing(
                    product,
                    historical_parameters,
                    horizon,
                    evaluation_episodes,
                    seed=eval_seed,
                )
                rows.append(
                    _row(
                        split,
                        product,
                        "HistoricalDualSourcingHeuristicPolicy",
                        historical_parameters,
                        historical_metrics,
                    )
                )

                oracle_parameters = _dual_sourcing_parameters_from_mean(product, _true_mean_demand(product))
                oracle_metrics = _evaluate_dual_sourcing(
                    product,
                    oracle_parameters,
                    horizon,
                    evaluation_episodes,
                    seed=eval_seed,
                )
                rows.append(
                    _row(
                        split,
                        product,
                        "OracleDualSourcingHeuristicPolicy",
                        oracle_parameters,
                        oracle_metrics,
                    )
                )
            else:
                historical_s = _tune_from_historical(product, horizon, min_stock, max_stock)
                historical_metrics = _evaluate_s(
                    product,
                    historical_s,
                    horizon,
                    evaluation_episodes,
                    seed=eval_seed,
                )
                rows.append(
                    _row(
                        split,
                        product,
                        "HistoricalBaseStockPolicy",
                        {"S": historical_s},
                        historical_metrics,
                    )
                )

                oracle_s = _tune_oracle(
                    product,
                    horizon,
                    oracle_tuning_episodes,
                    seed=seed + 50_000 + product_index,
                    min_stock=min_stock,
                    max_stock=max_stock,
                )
                oracle_metrics = _evaluate_s(
                    product,
                    oracle_s,
                    horizon,
                    evaluation_episodes,
                    seed=eval_seed,
                )
                rows.append(
                    _row(
                        split,
                        product,
                        "OracleBaseStockPolicy",
                        {"S": oracle_s},
                        oracle_metrics,
                    )
                )

    results = pd.DataFrame(rows)
    output_path = Path("outputs/results/product_benchmark_baselines.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
