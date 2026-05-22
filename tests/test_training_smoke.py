import pytest

torch = pytest.importorskip("torch")

from inventory_control.neural.policies import TorchMLPPolicy
from inventory_control.neural.torch_simulator import (
    DifferentiableLostSalesZeroLeadTimeConfig,
    DifferentiableLostSalesZeroLeadTimeSimulator,
)
from inventory_control.neural.training import train_zero_lead_time_policy
from inventory_control.neural.training import train_lead_time_product_policy
from inventory_control.products import Product


def test_zero_lead_time_training_smoke() -> None:
    config = DifferentiableLostSalesZeroLeadTimeConfig(
        horizon=3,
        initial_inventory=2.0,
        holding_cost=1.0,
        lost_sales_cost=5.0,
        unit_order_cost=1.0,
        demand_lambda=2.0,
    )
    simulator = DifferentiableLostSalesZeroLeadTimeSimulator(config)
    policy = TorchMLPPolicy(input_dim=4, hidden_dims=(8,))

    result = train_zero_lead_time_policy(
        simulator,
        policy,
        n_steps=2,
        batch_size=4,
        learning_rate=1e-3,
        seed=1,
    )

    assert len(result.losses) == 2
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result.losses)


def test_lead_time_product_training_smoke() -> None:
    products = [
        Product(
            product_id=f"p{i}",
            environment_type="lost_sales_lead_time",
            demand_process_type="poisson",
            demand_parameters={"lam": 3.0 + i},
            historical_demand=[2 + i, 3 + i, 4 + i],
            holding_cost=1.0,
            lost_sales_cost=5.0,
            unit_order_cost=1.0,
            fixed_order_cost=0.0,
            initial_inventory=2,
            lead_time=2,
        )
        for i in range(2)
    ]
    policy = TorchMLPPolicy(input_dim=9, hidden_dims=(8,))

    result = train_lead_time_product_policy(
        products,
        policy,
        horizon=3,
        max_lead_time=2,
        n_steps=2,
        batch_size=2,
        learning_rate=1e-3,
        seed=1,
    )

    assert len(result.losses) == 2
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result.losses)
