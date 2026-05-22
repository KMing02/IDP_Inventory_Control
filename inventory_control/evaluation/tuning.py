"""Hyperparameter tuning helpers for classical policies."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from inventory_control.envs.lost_sales import LostSalesEnv
from inventory_control.evaluation.runner import evaluate_policy
from inventory_control.policies.base import Policy
from inventory_control.policies.classical import BaseStockPolicy, InventoryPositionBaseStockPolicy


def base_stock_grid_search(
    env_factory: Callable[[int | None], LostSalesEnv],
    horizon: int,
    n_episodes: int,
    seed: int,
    stock_levels: Sequence[int] | None = None,
    min_stock_level: int = 0,
    max_stock_level: int | None = None,
) -> pd.DataFrame:
    """Evaluate base-stock levels over a fixed seed grid.

    Each candidate ``S`` is evaluated with the same episode seeds
    ``seed, seed + 1, ..., seed + n_episodes - 1`` through
    :func:`evaluate_policy`, which makes candidate comparisons use common
    random numbers.
    """
    if stock_levels is None:
        if max_stock_level is None:
            raise ValueError("max_stock_level is required when stock_levels is not provided")
        if min_stock_level < 0:
            raise ValueError("min_stock_level must be non-negative")
        if max_stock_level < min_stock_level:
            raise ValueError("max_stock_level must be at least min_stock_level")
        stock_levels = list(range(min_stock_level, max_stock_level + 1))
    else:
        stock_levels = [int(level) for level in stock_levels]
        if not stock_levels:
            raise ValueError("stock_levels must not be empty")
        if any(level < 0 for level in stock_levels):
            raise ValueError("stock_levels must be non-negative")

    rows = []
    for stock_level in stock_levels:
        metrics, _ = evaluate_policy(
            env_factory=env_factory,
            policy=BaseStockPolicy(S=stock_level),
            horizon=horizon,
            n_episodes=n_episodes,
            seed=seed,
        )
        rows.append({"S": stock_level, **metrics})

    results = pd.DataFrame(rows).sort_values(["mean_total_cost", "S"], ignore_index=True)
    best_s = int(results.loc[0, "S"])
    results["is_best"] = results["S"] == best_s
    return results


def policy_parameter_grid_search(
    env_factory: Callable[[int | None], LostSalesEnv],
    policy_factory: Callable[[int], Policy],
    parameter_name: str,
    horizon: int,
    n_episodes: int,
    seed: int,
    min_value: int,
    max_value: int,
) -> pd.DataFrame:
    """Evaluate a one-dimensional integer policy parameter grid."""
    if min_value < 0:
        raise ValueError("min_value must be non-negative")
    if max_value < min_value:
        raise ValueError("max_value must be at least min_value")

    rows = []
    for value in range(min_value, max_value + 1):
        metrics, _ = evaluate_policy(
            env_factory=env_factory,
            policy=policy_factory(value),
            horizon=horizon,
            n_episodes=n_episodes,
            seed=seed,
        )
        rows.append({parameter_name: value, **metrics})

    results = pd.DataFrame(rows).sort_values(["mean_total_cost", parameter_name], ignore_index=True)
    best_value = int(results.loc[0, parameter_name])
    results["is_best"] = results[parameter_name] == best_value
    return results


def inventory_position_base_stock_grid_search(
    env_factory: Callable[[int | None], LostSalesEnv],
    horizon: int,
    n_episodes: int,
    seed: int,
    min_stock_level: int,
    max_stock_level: int,
) -> pd.DataFrame:
    """Tune inventory-position base-stock level for lead-time systems."""
    return policy_parameter_grid_search(
        env_factory=env_factory,
        policy_factory=lambda value: InventoryPositionBaseStockPolicy(S=value),
        parameter_name="S",
        horizon=horizon,
        n_episodes=n_episodes,
        seed=seed,
        min_value=min_stock_level,
        max_value=max_stock_level,
    )
