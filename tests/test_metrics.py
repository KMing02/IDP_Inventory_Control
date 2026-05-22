from inventory_control.envs.demand import DeterministicDemand
from inventory_control.envs.lost_sales import LostSalesEnv, LostSalesEnvConfig
from inventory_control.evaluation.metrics import compute_summary_metrics
from inventory_control.evaluation.runner import run_episode
from inventory_control.policies.classical import ConstantOrderPolicy


def test_metrics_on_small_deterministic_example() -> None:
    config = LostSalesEnvConfig(
        horizon=2,
        initial_inventory=0,
        holding_cost=1.0,
        lost_sales_cost=5.0,
        unit_order_cost=1.0,
        fixed_order_cost=0.0,
    )
    env = LostSalesEnv(config, DeterministicDemand([2, 2]))
    trace = run_episode(env, ConstantOrderPolicy(quantity=1), horizon=2, seed=0)
    metrics = compute_summary_metrics([trace])

    assert metrics["mean_total_cost"] == 12.0
    assert metrics["std_total_cost"] == 0.0
    assert metrics["mean_cost_per_period"] == 6.0
    assert metrics["mean_holding_cost"] == 0.0
    assert metrics["mean_lost_sales_cost"] == 10.0
    assert metrics["mean_order_cost"] == 2.0
    assert metrics["mean_service_level"] == 0.5
    assert metrics["mean_lost_sales_rate"] == 0.5


def test_cost_components_decompose_mean_total_cost() -> None:
    config = LostSalesEnvConfig(
        horizon=2,
        initial_inventory=0,
        holding_cost=1.0,
        lost_sales_cost=5.0,
        unit_order_cost=1.0,
        fixed_order_cost=2.0,
    )
    env = LostSalesEnv(config, DeterministicDemand([0, 4]))
    trace = run_episode(env, ConstantOrderPolicy(quantity=2), horizon=2, seed=0)
    metrics = compute_summary_metrics([trace])

    component_sum = (
        metrics["mean_holding_cost"]
        + metrics["mean_lost_sales_cost"]
        + metrics["mean_order_cost"]
    )
    assert component_sum == metrics["mean_total_cost"]


def test_service_level_uses_total_sales_over_total_demand() -> None:
    config = LostSalesEnvConfig(
        horizon=1,
        initial_inventory=0,
        holding_cost=0.0,
        lost_sales_cost=1.0,
        unit_order_cost=0.0,
    )
    episode_a = run_episode(
        LostSalesEnv(config, DeterministicDemand([1])),
        ConstantOrderPolicy(quantity=1),
        horizon=1,
        seed=0,
    )
    episode_b = run_episode(
        LostSalesEnv(config, DeterministicDemand([9])),
        ConstantOrderPolicy(quantity=0),
        horizon=1,
        seed=0,
    )

    metrics = compute_summary_metrics([episode_a, episode_b])

    assert metrics["mean_service_level"] == 0.1
    assert metrics["mean_lost_sales_rate"] == 0.9
