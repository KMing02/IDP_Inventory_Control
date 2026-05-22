import pytest

torch = pytest.importorskip("torch")

from inventory_control.neural.policies import TorchMLPPolicy
from inventory_control.neural.torch_simulator import (
    DifferentiableLostSalesLeadTimeProductSimulator,
    DifferentiableLostSalesZeroLeadTimeConfig,
    DifferentiableLostSalesZeroLeadTimeSimulator,
    make_lead_time_product_batch,
)
from inventory_control.products import Product


class ConstantTorchPolicy(torch.nn.Module):
    def __init__(self, quantity: float) -> None:
        super().__init__()
        self.quantity = torch.nn.Parameter(torch.tensor(quantity))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.quantity).expand(features.shape[0])


def test_differentiable_simulator_matches_known_zero_lead_time_cost() -> None:
    config = DifferentiableLostSalesZeroLeadTimeConfig(
        horizon=1,
        initial_inventory=5.0,
        holding_cost=1.0,
        lost_sales_cost=10.0,
        unit_order_cost=2.0,
        demand_lambda=8.0,
    )
    simulator = DifferentiableLostSalesZeroLeadTimeSimulator(config)
    policy = ConstantTorchPolicy(quantity=2.0)
    demand_paths = torch.tensor([[8.0]])

    rollout = simulator.rollout(policy, demand_paths)
    order = torch.nn.functional.softplus(torch.tensor(2.0))
    expected_lost_sales = torch.relu(torch.tensor(8.0) - (torch.tensor(5.0) + order))
    expected_cost = 10.0 * expected_lost_sales + 2.0 * order

    assert torch.allclose(rollout["total_cost"], expected_cost.unsqueeze(0))


def test_differentiable_simulator_backpropagates_to_policy() -> None:
    config = DifferentiableLostSalesZeroLeadTimeConfig(
        horizon=2,
        initial_inventory=1.0,
        holding_cost=1.0,
        lost_sales_cost=5.0,
        unit_order_cost=1.0,
        demand_lambda=3.0,
    )
    simulator = DifferentiableLostSalesZeroLeadTimeSimulator(config)
    policy = TorchMLPPolicy(input_dim=4, hidden_dims=(8,))
    demand_paths = torch.tensor([[3.0, 3.0], [2.0, 4.0]])

    loss = simulator.rollout(policy, demand_paths)["total_cost"].mean()
    loss.backward()

    assert any(parameter.grad is not None for parameter in policy.parameters())


def test_lead_time_product_simulator_backpropagates_to_policy() -> None:
    product = Product(
        product_id="p0",
        environment_type="lost_sales_lead_time",
        demand_process_type="poisson",
        demand_parameters={"lam": 3.0},
        historical_demand=[2, 3, 4, 3],
        holding_cost=1.0,
        lost_sales_cost=5.0,
        unit_order_cost=1.0,
        fixed_order_cost=0.0,
        initial_inventory=2,
        lead_time=2,
    )
    simulator = DifferentiableLostSalesLeadTimeProductSimulator(horizon=3, max_lead_time=2)
    batch = make_lead_time_product_batch([product, product], horizon=3, generator=torch.Generator().manual_seed(0))
    policy = TorchMLPPolicy(input_dim=9, hidden_dims=(8,))

    loss = simulator.rollout(policy, batch)["total_cost"].mean()
    loss.backward()

    assert any(parameter.grad is not None for parameter in policy.parameters())
