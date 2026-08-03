import pytest

torch = pytest.importorskip("torch")

from inventory_control.policies.neural.fcnn import compute_paper_fcnn_feature_stats
from inventory_control.policies.neural.constrained_mlp import make_paper_residual_order_up_to_constrained_mlp_model
from inventory_control.policies.neural.monokan import (
    MonoKANPolicy,
    PaperMonoKANPolicyAdapter,
    make_paper_monokan_model,
    make_paper_residual_order_up_to_kan_model,
    make_paper_residual_order_up_to_monokan_model,
    monokan_inventory_violation_rate,
    sample_monokan_first_layer_splines,
)
from inventory_control.products import PaperLostSalesProduct, PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import train_paper_lost_sales_fcnn
from inventory_control.training.direct_backprop import (
    historical_base_stock_imitation_loss,
    make_paper_lost_sales_training_batch,
    paper_active_slope_penalty,
    paper_action_cap_excess_penalty,
    paper_lost_sales_fcnn_rollout,
    paper_lost_sales_fcnn_order,
    paper_lost_sales_context_features,
    paper_lost_sales_target_level,
    paper_lost_sales_raw_order,
    paper_named_parameter_l2_penalty,
    paper_order_diagnostics,
    paper_target_stock_penalty,
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

    batch = make_paper_lost_sales_training_batch(
        [product],
        horizon=3,
        generator=torch.Generator().manual_seed(0),
    )

    historical_mean = sum(product.historical_demand) / len(product.historical_demand)
    assert batch["initial_inventory"].item() <= 2.0 * historical_mean


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


def test_monokan_factory_honors_stability_initialization_knobs() -> None:
    config = {
        **_paper_config(),
        "monokan": {
            "max_order_factor": 2.0,
            "noise_scale": 0.0,
            "base_scale_init": 0.2,
            "spline_scale_init": 0.3,
            "bounded_base_scale_init": 0.4,
            "bounded_spline_scale_init": 0.5,
            "bounded_spline_slope_init": 0.6,
        },
    }

    model = make_paper_monokan_model(config)
    first_layer = model.layers[0]

    assert torch.max(first_layer.coef[:, :, 1]).item() == pytest.approx(0.6)
    assert torch.max(first_layer.scale_base.abs()).item() == pytest.approx(0.4)
    assert torch.allclose(first_layer.scale_spline, torch.full_like(first_layer.scale_spline, 0.5))


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


def test_cap_excess_penalty_pushes_oversized_raw_orders_down() -> None:
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
    batch = make_paper_lost_sales_training_batch(
        products,
        horizon=3,
        generator=torch.Generator().manual_seed(0),
    )
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
    batch = make_paper_lost_sales_training_batch(
        products,
        horizon=3,
        generator=torch.Generator().manual_seed(0),
    )
    inventory = 0.5 * batch["historical_mean_demand"]

    order = paper_lost_sales_raw_order(ConstantTarget(), batch, inventory)

    assert order.tolist() == pytest.approx(batch["historical_mean_demand"].tolist())


def test_residual_order_up_to_decoder_starts_from_historical_base_stock() -> None:
    class ZeroResidual(torch.nn.Module):
        paper_action_parameterization = "residual_order_up_to"
        paper_residual_scale = 0.5

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.zeros(features.shape[0])

    products = PaperLostSalesProductGenerator(_paper_config()).generate(3, "train")
    batch = make_paper_lost_sales_training_batch(
        products,
        horizon=3,
        generator=torch.Generator().manual_seed(0),
    )
    inventory = 0.25 * batch["historical_base_stock_level"]

    order = paper_lost_sales_raw_order(ZeroResidual(), batch, inventory)

    assert order.tolist() == pytest.approx((0.75 * batch["historical_base_stock_level"]).tolist())


def test_residual_kan_ablation_is_unconstrained_but_monokan_constrains_context() -> None:
    config = _paper_config()
    history_length = int(config["history_length_H"])

    kan = make_paper_residual_order_up_to_kan_model(config)
    monokan = make_paper_residual_order_up_to_monokan_model(config)

    assert kan.monotonic_vars == {}
    assert all(monokan.monotonic_vars[index] == 1 for index in range(history_length))
    assert monokan.monotonic_vars[history_length] == 1
    assert monokan.monotonic_vars[history_length + 1] == -1
    assert monokan.monotonic_vars[history_length + 2] == -1
    assert monokan.monotonic_vars[history_length + 3] == 1


def test_constrained_mlp_residual_constrains_same_context_directions() -> None:
    config = _paper_config()
    products = PaperLostSalesProductGenerator(config).generate(3, "train")
    feature_stats = compute_paper_fcnn_feature_stats(products)
    batch = make_paper_lost_sales_training_batch(
        products,
        horizon=2,
        generator=torch.Generator().manual_seed(0),
        feature_stats=feature_stats,
    )
    model = make_paper_residual_order_up_to_constrained_mlp_model(config)
    base_target = paper_lost_sales_target_level(model, batch)
    context = paper_lost_sales_context_features(batch)
    history_length = int(config["history_length_H"])

    for feature_index, direction in {
        history_length: 1,
        history_length + 1: -1,
        history_length + 2: -1,
        history_length + 3: 1,
    }.items():
        shifted_context = context.clone()
        shifted_context[:, feature_index] += 0.1
        shifted_target = (
            batch["historical_base_stock_level"]
            + model.paper_residual_scale
            * (model(shifted_context) - model.paper_residual_center)
            * torch.clamp(batch["historical_mean_demand"], min=1.0)
        )
        assert torch.all(direction * (shifted_target - base_target) >= -1e-6)


def test_residual_order_up_to_rollout_reuses_context_target() -> None:
    class CountingResidual(torch.nn.Module):
        paper_action_parameterization = "residual_order_up_to"
        paper_residual_center = 1.0
        paper_residual_scale = 0.05

        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(1.0))
            self.forward_calls = 0

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            self.forward_calls += 1
            return self.bias.expand(features.shape[0])

    products = PaperLostSalesProductGenerator(_paper_config()).generate(3, "train")
    batch = make_paper_lost_sales_training_batch(
        products,
        horizon=5,
        generator=torch.Generator().manual_seed(0),
    )
    model = CountingResidual()

    reward = paper_lost_sales_fcnn_rollout(model, batch, max_order_factor=4.0)

    assert reward.shape == (3,)
    assert model.forward_calls == 1


def test_training_batch_supports_empirical_residual_anchor() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(8, "train")
    gamma_batch = make_paper_lost_sales_training_batch(
        products,
        horizon=3,
        generator=torch.Generator().manual_seed(0),
        residual_anchor="gamma",
    )
    empirical_batch = make_paper_lost_sales_training_batch(
        products,
        horizon=3,
        generator=torch.Generator().manual_seed(0),
        residual_anchor="empirical",
    )

    assert not torch.allclose(
        gamma_batch["historical_base_stock_level"],
        empirical_batch["historical_base_stock_level"],
    )


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
