from scripts.run_baselines import make_env_factory

from inventory_control.evaluation.runner import evaluate_policy
from inventory_control.policies.classical import BaseStockPolicy, ConstantOrderPolicy, RandomPolicy


def test_baseline_evaluation_from_config_runs_all_policies() -> None:
    config = {
        "horizon": 3,
        "n_episodes": 2,
        "seed": 7,
        "environment": {
            "initial_inventory": 1,
            "holding_cost": 1.0,
            "lost_sales_cost": 2.0,
            "unit_order_cost": 1.0,
            "fixed_order_cost": 0.5,
        },
        "demand": {"type": "deterministic", "sequence": [1, 2, 3]},
        "policies": {
            "base_stock": {"S": 2},
            "constant_order": {"quantity": 1},
            "random": {"max_order": 2},
        },
    }
    env_factory = make_env_factory(config)
    policies = [
        BaseStockPolicy(S=2),
        ConstantOrderPolicy(quantity=1),
        RandomPolicy(max_order=2),
    ]

    rows = [
        evaluate_policy(env_factory, policy, horizon=3, n_episodes=2, seed=7)[0]
        for policy in policies
    ]

    assert len(rows) == 3
    assert all("mean_total_cost" in row for row in rows)
    assert all(row["mean_total_cost"] >= 0.0 for row in rows)
