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
from inventory_control.training.direct_backprop import (
    historical_base_stock_imitation_loss,
    make_paper_lost_sales_training_batch,
    paper_lost_sales_fcnn_order,
    paper_lost_sales_raw_order,
    paper_order_diagnostics,
)
from tests.test_paper_lost_sales import _paper_config


def test_monokan_output_is_nonnegative() -> None:
    model = MonoKANPolicy(input_dim=37, hidden_layers=1, hidden_units=4, grid=3, monotonic_vars={0: -1})
    features = torch.randn(5, 37)

    orders = model(features)

    assert orders.shape == (5,)
    assert torch.all(orders >= 0)


def test_deep_monokan_initial_orders_are_below_shared_cap() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(8, "train")
    feature_stats = compute_paper_fcnn_feature_stats(products)
    batch = make_paper_lost_sales_training_batch(
        products,
        horizon=3,
        generator=torch.Generator().manual_seed(0),
        feature_stats=feature_stats,
    )
    model = MonoKANPolicy(input_dim=37, hidden_layers=2, hidden_units=32, grid=3, monotonic_vars={0: -1})
    inventory = torch.zeros(len(products))

    raw_order = paper_lost_sales_raw_order(
        model,
        batch,
        inventory,
        scale_action_by_history_mean=True,
    )
    cap = 4.0 * batch["historical_mean_demand"]

    assert torch.mean((raw_order >= cap).float()).item() < 0.5


def test_paper_monokan_initial_order_stays_within_configured_bound() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(4, "train")
    feature_stats = compute_paper_fcnn_feature_stats(products)
    config = {**_paper_config(), "monokan": {"max_order_factor": 2.0}}
    model = make_paper_monokan_model(config)
    batch = make_paper_lost_sales_training_batch(
        products,
        horizon=3,
        generator=torch.Generator().manual_seed(0),
        feature_stats=feature_stats,
    )
    inventory = torch.zeros(len(products))

    raw_order = paper_lost_sales_raw_order(
        model,
        batch,
        inventory,
        scale_action_by_history_mean=True,
    )
    physical_bound = 2.0 * torch.clamp(batch["historical_mean_demand"], min=1.0)

    assert torch.all(raw_order >= 0)
    assert torch.all(raw_order <= physical_bound + 1e-6)


def test_paper_monokan_raw_orders_are_problem_scaled_before_deployment_cap() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(4, "train")
    feature_stats = compute_paper_fcnn_feature_stats(products)
    config = {**_paper_config(), "monokan": {"max_order_factor": 2.0}}
    model = make_paper_monokan_model(config)
    batch = make_paper_lost_sales_training_batch(
        products,
        horizon=3,
        generator=torch.Generator().manual_seed(0),
        feature_stats=feature_stats,
    )
    inventory = torch.zeros(len(products))

    raw_order = paper_lost_sales_raw_order(
        model,
        batch,
        inventory,
        scale_action_by_history_mean=True,
    )
    physical_bound = 2.0 * torch.clamp(batch["historical_mean_demand"], min=1.0)

    assert torch.all(raw_order <= physical_bound + 1e-6)


def test_raw_imitation_retains_gradient_above_deployment_cap() -> None:
    class OversizedPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.raw_bias = torch.nn.Parameter(torch.tensor(10.0))

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.softplus(self.raw_bias).expand(features.shape[0])

    products = PaperLostSalesProductGenerator(_paper_config()).generate(4, "train")
    batch = make_paper_lost_sales_training_batch(
        products,
        horizon=3,
        generator=torch.Generator().manual_seed(0),
    )
    model = OversizedPolicy()
    clipped_loss = historical_base_stock_imitation_loss(
        model,
        batch,
        torch.Generator().manual_seed(1),
        max_order_factor=1.0,
        scale_action_by_history_mean=True,
        clip_predictions=True,
    )
    clipped_loss.backward()
    clipped_gradient = float(model.raw_bias.grad)

    model.zero_grad()
    raw_loss = historical_base_stock_imitation_loss(
        model,
        batch,
        torch.Generator().manual_seed(1),
        max_order_factor=1.0,
        scale_action_by_history_mean=True,
        clip_predictions=False,
    )
    raw_loss.backward()

    assert clipped_gradient == 0.0
    assert model.raw_bias.grad is not None
    assert model.raw_bias.grad.item() > 0.0


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


def test_monokan_raw_imitation_improves_calibration_and_logs_diagnostics() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(1, "train")
    feature_stats = compute_paper_fcnn_feature_stats(products)
    torch.manual_seed(0)
    model = MonoKANPolicy(input_dim=37, hidden_layers=1, hidden_units=4, grid=3, monotonic_vars={0: -1})
    diagnostic_batch = make_paper_lost_sales_training_batch(
        products,
        horizon=3,
        generator=torch.Generator().manual_seed(123),
        feature_stats=feature_stats,
    )
    inventory_fractions = tuple(index / 32 for index in range(33))
    pre_training_rmse = paper_order_diagnostics(
        model,
        diagnostic_batch,
        max_order_factor=4.0,
        scale_action_by_history_mean=True,
        inventory_fractions=inventory_fractions,
    )["raw_imitation_rmse"]

    result = train_paper_lost_sales_fcnn(
        products,
        model,
        horizon=3,
        batch_size=1,
        epochs=0,
        learning_rate=1e-3,
        seed=0,
        imitation_epochs=20,
        feature_stats=feature_stats,
        max_order_factor=4.0,
        scale_action_by_history_mean=True,
        clip_imitation_predictions=False,
    )
    post_training_rmse = paper_order_diagnostics(
        result.model,
        diagnostic_batch,
        max_order_factor=4.0,
        scale_action_by_history_mean=True,
        inventory_fractions=inventory_fractions,
    )["raw_imitation_rmse"]

    assert result.training_log is not None
    assert post_training_rmse < pre_training_rmse
    assert {
        "raw_order_mean",
        "raw_order_min",
        "raw_order_max",
        "deployed_order_mean",
        "cap_hit_rate",
        "raw_imitation_mae",
        "raw_imitation_rmse",
    }.issubset(result.training_log[-1])


def test_paper_monokan_adapter_matches_action_cap() -> None:
    product = PaperLostSalesProductGenerator(_paper_config()).generate(1, "test")[0]
    feature_stats = compute_paper_fcnn_feature_stats([product])
    model = make_paper_monokan_model(_paper_config())
    adapter = PaperMonoKANPolicyAdapter(model, product, feature_stats=feature_stats, max_order_factor=1.0)

    order = adapter.act(inventory=max(product.historical_demand) + 100, t=0)

    assert order == 0.0


def test_paper_monokan_training_and_evaluation_use_same_scaled_action_path() -> None:
    product = PaperLostSalesProductGenerator(_paper_config()).generate(1, "test")[0]
    feature_stats = compute_paper_fcnn_feature_stats([product])
    config = {**_paper_config(), "monokan": {"max_order_factor": 2.0}}
    model = make_paper_monokan_model(config)
    adapter = PaperMonoKANPolicyAdapter(
        model,
        product,
        feature_stats=feature_stats,
        max_order_factor=2.0,
        scale_action_by_history_mean=True,
    )
    batch = make_paper_lost_sales_training_batch(
        [product],
        horizon=1,
        generator=torch.Generator().manual_seed(0),
        feature_stats=feature_stats,
    )
    inventory = torch.tensor([float(product.initial_inventory)])

    training_order = float(
        paper_lost_sales_fcnn_order(
            model,
            batch,
            inventory,
            max_order_factor=2.0,
            scale_action_by_history_mean=True,
        ).item()
    )
    evaluation_order = adapter.act(float(product.initial_inventory), t=0)

    assert training_order == pytest.approx(evaluation_order)


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
