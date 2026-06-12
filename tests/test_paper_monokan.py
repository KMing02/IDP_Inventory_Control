import pytest

torch = pytest.importorskip("torch")

from inventory_control.policies.neural.fcnn import compute_paper_fcnn_feature_stats
from inventory_control.policies.neural.monokan import (
    MonoKANPolicy,
    PaperMonoKANPolicyAdapter,
    make_paper_monokan_model,
    monokan_inventory_violation_rate,
    sample_monokan_first_layer_splines,
)
from inventory_control.products import PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import train_paper_lost_sales_fcnn
from tests.test_paper_lost_sales import _paper_config


def test_monokan_output_is_nonnegative() -> None:
    model = MonoKANPolicy(input_dim=37, hidden_layers=1, hidden_units=4, grid=3, monotonic_vars={0: -1})
    features = torch.randn(5, 37)

    orders = model(features)

    assert orders.shape == (5,)
    assert torch.all(orders >= 0)


def test_monokan_is_nonincreasing_in_inventory_feature() -> None:
    model = MonoKANPolicy(input_dim=37, hidden_layers=1, hidden_units=4, grid=3, monotonic_vars={0: -1})
    fixed_features = [0.0] + [0.5] * 36
    inventory_grid = [index / 10 for index in range(40)]

    violation_rate = monokan_inventory_violation_rate(model, fixed_features, inventory_grid)

    assert violation_rate == 0.0


def test_monokan_training_reapplies_monotonic_projection() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(4, "train")
    model = MonoKANPolicy(input_dim=37, hidden_layers=1, hidden_units=4, grid=3, monotonic_vars={0: -1})

    train_paper_lost_sales_fcnn(
        products,
        model,
        horizon=3,
        batch_size=2,
        epochs=1,
        learning_rate=1e-3,
        seed=0,
    )

    fixed_features = [0.0] + [0.5] * 36
    inventory_grid = [index / 10 for index in range(40)]
    assert monokan_inventory_violation_rate(model, fixed_features, inventory_grid) == 0.0


def test_paper_monokan_adapter_matches_action_cap() -> None:
    product = PaperLostSalesProductGenerator(_paper_config()).generate(1, "test")[0]
    feature_stats = compute_paper_fcnn_feature_stats([product])
    model = make_paper_monokan_model(_paper_config())
    adapter = PaperMonoKANPolicyAdapter(model, product, feature_stats=feature_stats, max_order_factor=1.0)

    order = adapter.act(inventory=max(product.historical_demand) + 100, t=0)

    assert order == 0.0


def test_monokan_is_monotone_outside_spline_knots() -> None:
    model = MonoKANPolicy(input_dim=37, hidden_layers=1, hidden_units=4, grid=3, monotonic_vars={0: -1})
    fixed_features = [0.0] + [0.5] * 36

    rate = monokan_inventory_violation_rate(model, fixed_features, [-5, -2, 0, 4, 8])

    assert rate == 0.0


def test_sampled_monokan_edge_matches_first_layer_evaluation() -> None:
    model = MonoKANPolicy(input_dim=3, hidden_layers=1, hidden_units=2, grid=3, monotonic_vars={0: -1})
    x, sampled = sample_monokan_first_layer_splines(model, feature_index=0, samples=100)
    layer = model.layers[0]
    inputs = torch.zeros(len(x), 3)
    inputs[:, 0] = x
    zero_input = torch.zeros(1, 3)
    with torch.no_grad():
        layer_values = layer(inputs)
        zero_values = layer(zero_input).squeeze(0)
    zero_index = int(torch.argmin(torch.abs(x)).item())
    other_edge_contribution = zero_values - sampled[:, zero_index]
    reconstructed = layer_values.T - other_edge_contribution.unsqueeze(1)

    assert torch.allclose(reconstructed, sampled, atol=1e-5)
