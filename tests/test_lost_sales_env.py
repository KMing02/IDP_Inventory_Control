import pytest

from inventory_control.envs.demand import DeterministicDemand
from inventory_control.envs.lost_sales import (
    DualSourcingEnv,
    DualSourcingEnvConfig,
    LostSalesEnv,
    LostSalesEnvConfig,
    LostSalesLeadTimeEnv,
    LostSalesLeadTimeEnvConfig,
)


def test_one_step_lost_sales_dynamics_known_values() -> None:
    config = LostSalesEnvConfig(
        horizon=1,
        initial_inventory=5,
        holding_cost=1.0,
        lost_sales_cost=10.0,
        unit_order_cost=2.0,
        fixed_order_cost=3.0,
    )
    env = LostSalesEnv(config, DeterministicDemand([8]))
    state = env.reset(seed=0)
    assert state == 5

    next_state, cost, done, info = env.step(2)

    assert next_state == 0
    assert done is True
    assert info["demand"] == 8
    assert info["sales"] == 7
    assert info["lost_sales"] == 1
    assert info["ending_inventory"] == 0
    assert info["order_quantity"] == 2
    assert info["holding_cost"] == 0.0
    assert info["lost_sales_cost"] == 10.0
    assert info["order_cost"] == 7.0
    assert cost == 17.0
    assert info["total_cost"] == 17.0


def test_one_step_holding_cost_without_lost_sales_or_fixed_order_cost() -> None:
    config = LostSalesEnvConfig(
        horizon=1,
        initial_inventory=5,
        holding_cost=2.0,
        lost_sales_cost=10.0,
        unit_order_cost=3.0,
        fixed_order_cost=7.0,
    )
    env = LostSalesEnv(config, DeterministicDemand([3]))

    next_state, cost, done, info = env.step(0)

    assert next_state == 2
    assert done is True
    assert info["sales"] == 3
    assert info["lost_sales"] == 0
    assert info["ending_inventory"] == 2
    assert info["holding_cost"] == 4.0
    assert info["lost_sales_cost"] == 0.0
    assert info["order_cost"] == 0.0
    assert cost == 4.0


def test_negative_order_quantity_is_invalid() -> None:
    config = LostSalesEnvConfig(
        horizon=1,
        initial_inventory=0,
        holding_cost=1.0,
        lost_sales_cost=1.0,
        unit_order_cost=1.0,
    )
    env = LostSalesEnv(config, DeterministicDemand([1]))

    with pytest.raises(ValueError, match="non-negative"):
        env.step(-1)


def test_lead_time_pipeline_shifting() -> None:
    config = LostSalesLeadTimeEnvConfig(
        horizon=3,
        initial_inventory=0,
        holding_cost=0.0,
        lost_sales_cost=0.0,
        unit_order_cost=0.0,
        lead_time=2,
    )
    env = LostSalesLeadTimeEnv(config, DeterministicDemand([0, 0, 0]))

    state, _, _, info = env.step(5)
    assert state == [0, 0, 5]
    assert info["arrivals"] == 0

    state, _, _, info = env.step(0)
    assert state == [0, 5, 0]
    assert info["arrivals"] == 0

    state, _, _, info = env.step(0)
    assert state == [5, 0, 0]
    assert info["arrivals"] == 5


def test_lead_time_lost_sales_dynamics_with_deterministic_demand() -> None:
    config = LostSalesLeadTimeEnvConfig(
        horizon=1,
        initial_inventory=3,
        holding_cost=1.0,
        lost_sales_cost=4.0,
        unit_order_cost=2.0,
        fixed_order_cost=1.0,
        lead_time=1,
    )
    env = LostSalesLeadTimeEnv(config, DeterministicDemand([5]))
    env.pipeline = [2]

    next_state, cost, done, info = env.step(3)

    assert next_state == [0, 3]
    assert done is True
    assert info["sales"] == 5
    assert info["lost_sales"] == 0
    assert info["ending_inventory"] == 0
    assert info["order_cost"] == 7.0
    assert cost == 7.0


def test_dual_sourcing_pipeline_shifting() -> None:
    config = DualSourcingEnvConfig(
        horizon=2,
        initial_inventory=0,
        holding_cost=0.0,
        lost_sales_cost=0.0,
        regular_unit_order_cost=0.0,
        expedited_unit_order_cost=0.0,
        regular_lead_time=2,
        expedited_lead_time=1,
    )
    env = DualSourcingEnv(config, DeterministicDemand([0, 0]))

    state, _, _, info = env.step([5, 2])
    assert state == [0, 0, 5, 2]
    assert info["regular_arrivals"] == 0
    assert info["expedited_arrivals"] == 0

    state, _, _, info = env.step([0, 0])
    assert state == [2, 5, 0, 0]
    assert info["regular_arrivals"] == 0
    assert info["expedited_arrivals"] == 2


def test_dual_sourcing_cost_decomposition() -> None:
    config = DualSourcingEnvConfig(
        horizon=1,
        initial_inventory=1,
        holding_cost=1.0,
        lost_sales_cost=5.0,
        regular_unit_order_cost=2.0,
        expedited_unit_order_cost=4.0,
        regular_fixed_order_cost=1.0,
        expedited_fixed_order_cost=3.0,
        regular_lead_time=1,
        expedited_lead_time=0,
    )
    env = DualSourcingEnv(config, DeterministicDemand([5]))

    _, cost, _, info = env.step([2, 3])

    assert info["sales"] == 4
    assert info["lost_sales"] == 1
    assert info["regular_order_cost"] == 5.0
    assert info["expedited_order_cost"] == 15.0
    assert info["order_cost"] == 20.0
    assert cost == 25.0
