"""Train an unconstrained MLP on differentiable zero-lead-time lost sales."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inventory_control.envs.demand import PoissonDemand
from inventory_control.envs.lost_sales import LostSalesEnvConfig, LostSalesZeroLeadTimeEnv
from inventory_control.evaluation.runner import evaluate_policy
from inventory_control.neural.policies import TorchMLPPolicy, TorchPolicyAdapter
from inventory_control.neural.torch_simulator import (
    DifferentiableLostSalesZeroLeadTimeConfig,
    DifferentiableLostSalesZeroLeadTimeSimulator,
)
from inventory_control.neural.training import train_zero_lead_time_policy


def main() -> None:
    """Train and evaluate a minimal neural zero-lead-time policy."""
    config = DifferentiableLostSalesZeroLeadTimeConfig(
        horizon=30,
        initial_inventory=10.0,
        holding_cost=1.0,
        lost_sales_cost=5.0,
        unit_order_cost=1.0,
        demand_lambda=8.0,
    )
    simulator = DifferentiableLostSalesZeroLeadTimeSimulator(config)
    policy = TorchMLPPolicy(input_dim=4, hidden_dims=(32, 32))
    result = train_zero_lead_time_policy(
        simulator,
        policy,
        n_steps=200,
        batch_size=128,
        learning_rate=1e-3,
        seed=123,
    )

    true_config = LostSalesEnvConfig(
        horizon=config.horizon,
        initial_inventory=int(config.initial_inventory),
        holding_cost=config.holding_cost,
        lost_sales_cost=config.lost_sales_cost,
        unit_order_cost=config.unit_order_cost,
        fixed_order_cost=0.0,
    )

    def env_factory(seed: int | None = None) -> LostSalesZeroLeadTimeEnv:
        return LostSalesZeroLeadTimeEnv(true_config, PoissonDemand(config.demand_lambda), seed=seed)

    adapter = TorchPolicyAdapter(
        model=result.policy,
        demand_scale=max(config.demand_lambda, 1.0),
        holding_cost=config.holding_cost,
        lost_sales_cost=config.lost_sales_cost,
        unit_order_cost=config.unit_order_cost,
    )
    metrics, _ = evaluate_policy(env_factory, adapter, horizon=config.horizon, n_episodes=100, seed=999)
    print(f"initial_training_loss={result.losses[0]:.4f}")
    print(f"final_training_loss={result.losses[-1]:.4f}")
    print(f"true_sim_mean_total_cost={metrics['mean_total_cost']:.4f}")
    print(f"true_sim_service_level={metrics['mean_service_level']:.4f}")


if __name__ == "__main__":
    main()
