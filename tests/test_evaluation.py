import pandas as pd

from inventory_control.envs.demand import PoissonDemand
from inventory_control.envs.lost_sales import DualSourcingEnv, DualSourcingEnvConfig, LostSalesEnv, LostSalesEnvConfig
from inventory_control.evaluation.runner import evaluate_policy
from inventory_control.policies.classical import RandomPolicy


def test_evaluate_policy_is_reproducible_with_seed() -> None:
    config = LostSalesEnvConfig(
        horizon=5,
        initial_inventory=3,
        holding_cost=1.0,
        lost_sales_cost=4.0,
        unit_order_cost=1.0,
        fixed_order_cost=0.5,
    )

    def env_factory(seed: int | None = None) -> LostSalesEnv:
        return LostSalesEnv(config, PoissonDemand(lam=2.0), seed=seed)

    metrics_a, episodes_a = evaluate_policy(env_factory, RandomPolicy(max_order=4), 5, 3, seed=42)
    metrics_b, episodes_b = evaluate_policy(env_factory, RandomPolicy(max_order=4), 5, 3, seed=42)

    assert metrics_a == metrics_b
    for trace_a, trace_b in zip(episodes_a, episodes_b, strict=True):
        pd.testing.assert_frame_equal(trace_a, trace_b)


def test_evaluate_policy_changes_with_different_seed() -> None:
    config = LostSalesEnvConfig(
        horizon=5,
        initial_inventory=3,
        holding_cost=1.0,
        lost_sales_cost=4.0,
        unit_order_cost=1.0,
    )

    def env_factory(seed: int | None = None) -> LostSalesEnv:
        return LostSalesEnv(config, PoissonDemand(lam=2.0), seed=seed)

    metrics_a, _ = evaluate_policy(env_factory, RandomPolicy(max_order=4), 5, 3, seed=42)
    metrics_b, _ = evaluate_policy(env_factory, RandomPolicy(max_order=4), 5, 3, seed=43)

    assert metrics_a != metrics_b


class FixedVectorPolicy:
    name = "FixedVectorPolicy"

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, state, t: int, rng=None) -> list[int]:
        return [1, 2]


def test_evaluation_runner_supports_vector_actions() -> None:
    config = DualSourcingEnvConfig(
        horizon=2,
        initial_inventory=0,
        holding_cost=1.0,
        lost_sales_cost=2.0,
        regular_unit_order_cost=1.0,
        expedited_unit_order_cost=3.0,
        regular_lead_time=1,
        expedited_lead_time=0,
    )

    def env_factory(seed: int | None = None) -> DualSourcingEnv:
        return DualSourcingEnv(config, PoissonDemand(lam=1.0), seed=seed)

    metrics, episodes = evaluate_policy(env_factory, FixedVectorPolicy(), 2, 2, seed=1)

    assert len(episodes) == 2
    assert "mean_regular_order_quantity" in metrics
    assert "mean_expedited_order_quantity" in metrics
