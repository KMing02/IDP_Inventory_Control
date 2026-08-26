import pytest

torch = pytest.importorskip("torch")

from inventory_control.policies.neural.fcnn import compute_paper_fcnn_feature_stats
from inventory_control.policies.neural.monokan import (
    MonoKANPolicy,
    _evaluate_hermite_curves,
    monokan_inventory_violation_rate,
    sample_monokan_first_layer_splines,
)
from inventory_control.products import PaperLostSalesProduct, PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import (
    historical_base_stock_imitation_loss,
    make_paper_lost_sales_training_batch,
    paper_action_cap_excess_penalty,
    paper_active_slope_penalty,
    paper_lost_sales_fcnn_rollout,
    paper_lost_sales_raw_order,
    paper_named_parameter_l2_penalty,
    paper_order_diagnostics,
    paper_target_stock_penalty,
    train_paper_lost_sales_fcnn,
)
from tests.test_paper_lost_sales import _paper_config


def test_training_batch_initial_inventory_uses_historical_mean_not_last_demand() -> None:
    product = PaperLostSalesProduct(
        product_id="spiky_last_demand",
        price=10.0,
        purchase_cost=2.0,
        holding_cost=1.0,
        lost_sale_penalty=1.0,
        true_mean_demand=10.0,
        true_coefficient_of_variation=0.5,
        gamma_shape=4.0,
        gamma_scale=2.5,
        demand_distribution_type="Gamma",
        historical_demand=[1, 1, 1, 100],
        initial_inventory=0,
        future_demand=[1, 1, 1],
    )

    batch = make_paper_lost_sales_training_batch([product], horizon=3, generator=torch.Generator().manual_seed(0))

    historical_mean = sum(product.historical_demand) / len(product.historical_demand)
    assert batch["initial_inventory"].item() <= 2.0 * historical_mean


def test_training_batch_can_use_last_historical_demand_for_initial_inventory() -> None:
    product = PaperLostSalesProduct(
        product_id="spiky_last_demand",
        price=10.0,
        purchase_cost=2.0,
        holding_cost=1.0,
        lost_sale_penalty=1.0,
        true_mean_demand=10.0,
        true_coefficient_of_variation=0.5,
        gamma_shape=4.0,
        gamma_scale=2.5,
        demand_distribution_type="Gamma",
        historical_demand=[1, 1, 1, 100],
        initial_inventory=0,
        future_demand=[1, 1, 1],
    )

    batch = make_paper_lost_sales_training_batch(
        [product],
        horizon=3,
        generator=torch.Generator().manual_seed(0),
        initial_inventory_mode="last_historical_demand",
    )

    expected = float(torch.rand((), generator=torch.Generator().manual_seed(0)).item()) * 200.0
    assert batch["initial_inventory"].item() == pytest.approx(expected)


def test_monokan_output_is_nonnegative() -> None:
    model = MonoKANPolicy(input_dim=37, hidden_layers=1, hidden_units=4, grid=3, monotonic_vars={0: -1})
    features = torch.randn(5, 37)

    orders = model(features)

    assert orders.shape == (5,)
    assert torch.all(orders >= 0)


def test_raw_imitation_retains_gradient_above_deployment_cap() -> None:
    class OversizedPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.raw_bias = torch.nn.Parameter(torch.tensor(10.0))

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.softplus(self.raw_bias).expand(features.shape[0])

    products = PaperLostSalesProductGenerator(_paper_config()).generate(4, "train")
    batch = make_paper_lost_sales_training_batch(products, horizon=3, generator=torch.Generator().manual_seed(0))
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


def test_cap_excess_penalty_pushes_oversized_raw_orders_down() -> None:
    class OversizedPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.raw_bias = torch.nn.Parameter(torch.tensor(10.0))

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.softplus(self.raw_bias).expand(features.shape[0])

    products = PaperLostSalesProductGenerator(_paper_config()).generate(4, "train")
    batch = make_paper_lost_sales_training_batch(products, horizon=3, generator=torch.Generator().manual_seed(0))
    model = OversizedPolicy()

    penalty = paper_action_cap_excess_penalty(
        model,
        batch,
        torch.Generator().manual_seed(1),
        samples_per_product=2,
        max_order_factor=1.0,
        scale_action_by_history_mean=True,
    )
    penalty.backward()

    assert penalty.item() > 0.0
    assert model.raw_bias.grad is not None
    assert model.raw_bias.grad.item() > 0.0


def test_target_stock_penalty_pushes_implied_level_toward_base_stock() -> None:
    class LowOrderPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.raw_bias = torch.nn.Parameter(torch.tensor(-5.0))

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.softplus(self.raw_bias).expand(features.shape[0])

    products = PaperLostSalesProductGenerator(_paper_config()).generate(4, "train")
    batch = make_paper_lost_sales_training_batch(products, horizon=3, generator=torch.Generator().manual_seed(0))
    model = LowOrderPolicy()

    penalty = paper_target_stock_penalty(
        model,
        batch,
        torch.Generator().manual_seed(1),
        samples_per_product=2,
        max_inventory_factor=4.0,
        scale_action_by_history_mean=True,
    )
    penalty.backward()

    assert penalty.item() > 0.0
    assert model.raw_bias.grad is not None
    assert model.raw_bias.grad.item() < 0.0


def test_active_slope_penalty_pushes_flat_policy_toward_negative_inventory_slope() -> None:
    class LinearInventoryPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inventory_weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return 1.0 + self.inventory_weight * features[:, 0]

    products = PaperLostSalesProductGenerator(_paper_config()).generate(4, "train")
    feature_stats = compute_paper_fcnn_feature_stats(products)
    batch = make_paper_lost_sales_training_batch(
        products,
        horizon=3,
        generator=torch.Generator().manual_seed(0),
        feature_stats=feature_stats,
    )
    model = LinearInventoryPolicy()

    penalty = paper_active_slope_penalty(
        model,
        batch,
        torch.Generator().manual_seed(1),
        samples_per_product=2,
        max_inventory_factor=4.0,
        scale_action_by_history_mean=True,
    )
    penalty.backward()

    assert penalty.item() > 0.0
    assert model.inventory_weight.grad is not None
    assert model.inventory_weight.grad.item() > 0.0


def test_order_up_to_decoder_uses_target_minus_inventory() -> None:
    class ConstantTarget(torch.nn.Module):
        paper_action_parameterization = "order_up_to"
        paper_scale_target_by_history_mean = True

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.full((features.shape[0],), 1.5)

    products = PaperLostSalesProductGenerator(_paper_config()).generate(3, "train")
    batch = make_paper_lost_sales_training_batch(products, horizon=3, generator=torch.Generator().manual_seed(0))
    inventory = 0.5 * batch["historical_mean_demand"]

    order = paper_lost_sales_raw_order(ConstantTarget(), batch, inventory)

    assert order.tolist() == pytest.approx(batch["historical_mean_demand"].tolist())


def test_order_up_to_rollout_reuses_static_context_target() -> None:
    class CountingTarget(torch.nn.Module):
        paper_action_parameterization = "order_up_to"
        paper_scale_target_by_history_mean = True

        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(1.0))
            self.forward_calls = 0

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            self.forward_calls += 1
            return self.bias.expand(features.shape[0])

    products = PaperLostSalesProductGenerator(_paper_config()).generate(3, "train")
    batch = make_paper_lost_sales_training_batch(products, horizon=5, generator=torch.Generator().manual_seed(0))
    model = CountingTarget()

    reward = paper_lost_sales_fcnn_rollout(model, batch, max_order_factor=4.0)

    assert reward.shape == (3,)
    assert model.forward_calls == 1


def test_named_parameter_l2_penalty_targets_kan_spline_and_scale_parameters() -> None:
    model = MonoKANPolicy(input_dim=3, hidden_layers=1, hidden_units=2, grid=3, monotonic_vars={0: -1})

    spline_penalty = paper_named_parameter_l2_penalty(model, ("coef",))
    scale_penalty = paper_named_parameter_l2_penalty(model, ("scale_base", "scale_spline"))

    assert spline_penalty.item() >= 0.0
    assert scale_penalty.item() > 0.0


def test_monokan_is_nonincreasing_in_inventory_feature() -> None:
    model = MonoKANPolicy(input_dim=37, hidden_layers=1, hidden_units=4, grid=3, monotonic_vars={0: -1})
    fixed_features = [0.0] + [0.5] * 36
    inventory_grid = [index / 10 for index in range(40)]

    violation_rate = monokan_inventory_violation_rate(model, fixed_features, inventory_grid)

    assert violation_rate == 0.0


def test_monokan_training_reapplies_monotonic_projection() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(4, "train")
    model = MonoKANPolicy(input_dim=37, hidden_layers=1, hidden_units=4, grid=3, monotonic_vars={0: -1})

    train_paper_lost_sales_fcnn(products, model, horizon=3, batch_size=2, epochs=1, learning_rate=1e-3, seed=0)

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
    assert {"raw_order_mean", "deployed_order_mean", "cap_hit_rate", "raw_imitation_rmse"}.issubset(result.training_log[-1])


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


def test_arithmetic_hermite_interval_lookup_matches_mask_reference_and_gradients() -> None:
    def reference(x_eval: torch.Tensor, grid: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
        from inventory_control.policies.neural.monokan import _hermite_basis

        points = grid.shape[1]
        idx = (x_eval.unsqueeze(1) >= grid.unsqueeze(2)).sum(dim=1) - 1
        idx = idx.clamp(min=0, max=points - 2)
        idx_right = idx + 1
        x0 = grid.gather(1, idx)
        x1 = grid.gather(1, idx_right)
        interval = torch.clamp(x1 - x0, min=1e-8)
        t = (x_eval - x0) / interval
        h00, h10, h01, h11 = _hermite_basis(t)
        y0 = coef[:, :, 0].gather(1, idx)
        dy0 = coef[:, :, 1].gather(1, idx)
        y1 = coef[:, :, 0].gather(1, idx_right)
        dy1 = coef[:, :, 1].gather(1, idx_right)
        result = y0 * h00 + dy0 * interval * h10 + y1 * h01 + dy1 * interval * h11
        result = torch.where(x_eval < grid[:, [0]], coef[:, [0], 0] + coef[:, [0], 1] * (x_eval - grid[:, [0]]), result)
        return torch.where(x_eval > grid[:, [-1]], coef[:, [-1], 0] + coef[:, [-1], 1] * (x_eval - grid[:, [-1]]), result)

    grid = torch.linspace(-2.0, 4.0, 8).repeat(3, 1)
    x_reference = torch.tensor(
        [[-3.0, -2.0, -0.5, 4.0, 5.0], [-1.9, 0.0, 1.5, 3.9, 4.1], [-4.0, -1.0, 2.0, 3.0, 6.0]],
        requires_grad=True,
    )
    x_optimized = x_reference.detach().clone().requires_grad_(True)
    coef_reference = torch.randn(3, 8, 2, generator=torch.Generator().manual_seed(8), requires_grad=True)
    coef_optimized = coef_reference.detach().clone().requires_grad_(True)

    expected = reference(x_reference, grid, coef_reference)
    actual = _evaluate_hermite_curves(x_optimized, grid, coef_optimized)
    expected.sum().backward()
    actual.sum().backward()

    assert torch.allclose(expected, actual, atol=1e-6)
    assert torch.allclose(x_reference.grad, x_optimized.grad, atol=1e-6)
    assert torch.allclose(coef_reference.grad, coef_optimized.grad, atol=1e-6)


def test_arithmetic_hermite_interval_lookup_matches_mask_at_knots_and_boundaries() -> None:
    def reference(x_eval: torch.Tensor, grid: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
        from inventory_control.policies.neural.monokan import _hermite_basis

        points = grid.shape[1]
        idx = (x_eval.unsqueeze(1) >= grid.unsqueeze(2)).sum(dim=1) - 1
        idx = idx.clamp(min=0, max=points - 2)
        right = idx + 1
        x0 = grid.gather(1, idx)
        x1 = grid.gather(1, right)
        interval = torch.clamp(x1 - x0, min=1e-8)
        t = (x_eval - x0) / interval
        h00, h10, h01, h11 = _hermite_basis(t)
        y0 = coef[:, :, 0].gather(1, idx)
        dy0 = coef[:, :, 1].gather(1, idx)
        y1 = coef[:, :, 0].gather(1, right)
        dy1 = coef[:, :, 1].gather(1, right)
        result = y0 * h00 + dy0 * interval * h10 + y1 * h01 + dy1 * interval * h11
        result = torch.where(x_eval < grid[:, [0]], coef[:, [0], 0] + coef[:, [0], 1] * (x_eval - grid[:, [0]]), result)
        return torch.where(x_eval > grid[:, [-1]], coef[:, [-1], 0] + coef[:, [-1], 1] * (x_eval - grid[:, [-1]]), result)

    knots = torch.linspace(-2.0, 4.0, 8)
    epsilon = 1e-6
    probes = torch.cat([knots, knots[1:-1] - epsilon, knots[1:-1] + epsilon, torch.tensor([knots[0] + epsilon, knots[-1] - epsilon])]).unsqueeze(0)
    grid = knots.unsqueeze(0)
    coef = torch.randn(1, 8, 2, generator=torch.Generator().manual_seed(19))

    expected = reference(probes, grid, coef)
    actual = _evaluate_hermite_curves(probes, grid, coef)

    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)
