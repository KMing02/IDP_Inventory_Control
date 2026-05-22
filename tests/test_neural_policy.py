import pytest

torch = pytest.importorskip("torch")

from inventory_control.neural.policies import TorchMLPPolicy, lead_time_product_feature_vector


def test_torch_mlp_policy_outputs_nonnegative_actions() -> None:
    policy = TorchMLPPolicy(input_dim=4, hidden_dims=(8,))
    features = torch.tensor([[1.0, 1.0, 5.0, 1.0], [0.0, 1.0, 5.0, 1.0]])

    actions = policy(features)

    assert actions.shape == (2,)
    assert torch.all(actions >= 0)


def test_lead_time_product_feature_vector_pads_pipeline() -> None:
    features = lead_time_product_feature_vector(
        state=[5, 2],
        historical_mean_demand=10.0,
        historical_std_demand=3.0,
        holding_cost=1.0,
        lost_sales_cost=5.0,
        unit_order_cost=1.0,
        lead_time=1,
        max_lead_time=3,
    )

    assert len(features) == 10
    assert features[:4] == [0.5, 0.2, 0.0, 0.0]
