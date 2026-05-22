import pytest

from inventory_control.envs.demand import DeterministicDemand
from inventory_control.envs.lost_sales import LostSalesEnv, LostSalesEnvConfig
from inventory_control.evaluation.tuning import base_stock_grid_search


def test_base_stock_grid_search_finds_best_stock_level() -> None:
    config = LostSalesEnvConfig(
        horizon=1,
        initial_inventory=0,
        holding_cost=1.0,
        lost_sales_cost=10.0,
        unit_order_cost=0.0,
        fixed_order_cost=0.0,
    )

    def env_factory(seed: int | None = None) -> LostSalesEnv:
        return LostSalesEnv(config, DeterministicDemand([2]), seed=seed)

    results = base_stock_grid_search(
        env_factory=env_factory,
        horizon=1,
        n_episodes=3,
        seed=11,
        min_stock_level=0,
        max_stock_level=4,
    )

    best = results.loc[results["is_best"]].iloc[0]
    assert int(best["S"]) == 2
    assert best["mean_total_cost"] == 0.0
    assert set(results["S"]) == {0, 1, 2, 3, 4}


def test_base_stock_grid_search_uses_same_seed_grid_for_candidates() -> None:
    config = LostSalesEnvConfig(
        horizon=2,
        initial_inventory=0,
        holding_cost=1.0,
        lost_sales_cost=5.0,
        unit_order_cost=0.0,
    )

    def env_factory(seed: int | None = None) -> LostSalesEnv:
        return LostSalesEnv(config, DeterministicDemand([1, 3]), seed=seed)

    first = base_stock_grid_search(env_factory, 2, 4, seed=5, stock_levels=[0, 1, 2])
    second = base_stock_grid_search(env_factory, 2, 4, seed=5, stock_levels=[0, 1, 2])

    assert first.to_dict("records") == second.to_dict("records")


def test_base_stock_grid_search_rejects_invalid_ranges() -> None:
    config = LostSalesEnvConfig(
        horizon=1,
        initial_inventory=0,
        holding_cost=1.0,
        lost_sales_cost=1.0,
        unit_order_cost=1.0,
    )

    def env_factory(seed: int | None = None) -> LostSalesEnv:
        return LostSalesEnv(config, DeterministicDemand([0]), seed=seed)

    with pytest.raises(ValueError, match="max_stock_level"):
        base_stock_grid_search(env_factory, 1, 1, seed=0)

    with pytest.raises(ValueError, match="at least"):
        base_stock_grid_search(env_factory, 1, 1, seed=0, min_stock_level=3, max_stock_level=2)

    with pytest.raises(ValueError, match="non-negative"):
        base_stock_grid_search(env_factory, 1, 1, seed=0, stock_levels=[-1, 0])
