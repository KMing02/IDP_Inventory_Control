import pytest
import inspect

torch = pytest.importorskip("torch")

from inventory_control.policies.neural.fcnn import (
    FCNNPolicy,
    PaperCausalCNNPolicy,
    PaperFCNNPolicyAdapter,
    compute_paper_fcnn_feature_stats,
)
from inventory_control.training.direct_backprop import (
    make_paper_lost_sales_training_batch,
    paper_inventory_monotonicity_penalty,
    paper_lost_sales_fcnn_order,
    paper_lost_sales_fcnn_rollout,
    paper_reward_normalization_scale,
    train_paper_lost_sales_fcnn,
)
from inventory_control.products import PaperLostSalesProductGenerator
from tests.test_paper_lost_sales import _paper_config


def test_training_package_exports_legacy_and_neutral_helpers() -> None:
    from inventory_control.training import (
        paper_lost_sales_fcnn_order as package_old_order,
        paper_lost_sales_fcnn_rollout as package_old_rollout,
        paper_lost_sales_policy_order as package_neutral_order,
        paper_lost_sales_policy_rollout as package_neutral_rollout,
        train_paper_lost_sales_fcnn as package_old_train,
        train_paper_lost_sales_policy as package_neutral_train,
    )

    assert inspect.signature(package_old_order) == inspect.signature(package_neutral_order)
    assert inspect.signature(package_old_rollout) == inspect.signature(package_neutral_rollout)
    train_parameters = inspect.signature(package_neutral_train).parameters.values()
    assert {parameter.kind for parameter in train_parameters} == {
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    }
    assert package_old_train is not None

    products = PaperLostSalesProductGenerator(_paper_config()).generate(2, "train")
    model = FCNNPolicy(input_dim=37, hidden_layers=1, hidden_units=8, activation="ELU")
    batch = make_paper_lost_sales_training_batch(products, horizon=3, generator=torch.Generator().manual_seed(0))
    inventory = batch["initial_inventory"]

    assert torch.allclose(package_old_order(model, batch, inventory), package_neutral_order(model, batch, inventory))
    assert torch.allclose(package_old_rollout(model, batch), package_neutral_rollout(model, batch))


def test_fcnn_output_is_nonnegative() -> None:
    model = FCNNPolicy(input_dim=37, hidden_layers=2, hidden_units=8, activation="ELU")
    features = torch.randn(4, 37)

    orders = model(features)

    assert orders.shape == (4,)
    assert torch.all(orders >= 0)


def test_paper_causal_cnn_policy_output_is_nonnegative() -> None:
    model = PaperCausalCNNPolicy(history_length=32, hidden_layers=1, hidden_units=8, cnn_channels=4, cnn_dilations=(1, 2))
    features = torch.randn(4, 37)

    orders = model(features)

    assert orders.shape == (4,)
    assert torch.all(orders >= 0)


def test_training_loss_is_negative_mean_total_reward() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(4, "train")
    model = FCNNPolicy(input_dim=37, hidden_layers=1, hidden_units=8, activation="ELU")
    result = train_paper_lost_sales_fcnn(
        products,
        model,
        horizon=3,
        batch_size=2,
        epochs=1,
        learning_rate=1e-3,
        seed=0,
    )

    assert result.training_losses[0] == -result.mean_total_rewards[0]


def test_paper_rollout_returns_reward_not_negative_reward() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(2, "train")
    model = FCNNPolicy(input_dim=37, hidden_layers=1, hidden_units=8, activation="ELU")
    batch = make_paper_lost_sales_training_batch(products, horizon=3, generator=torch.Generator().manual_seed(0))
    rewards = paper_lost_sales_fcnn_rollout(model, batch)

    assert rewards.shape == (2,)


def test_paper_rollout_includes_terminal_salvage_by_default() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(1, "train")
    product = products[0].__class__(
        **{
            **products[0].__dict__,
            "future_demand": [0, 0],
            "purchase_cost": 3.0,
            "initial_inventory": 5,
        }
    )
    model = FCNNPolicy(input_dim=37, hidden_layers=1, hidden_units=8, activation="ELU")
    batch = make_paper_lost_sales_training_batch([product], horizon=2, generator=torch.Generator().manual_seed(0))
    with_salvage = paper_lost_sales_fcnn_rollout(model, batch, include_terminal_salvage=True)
    without_salvage = paper_lost_sales_fcnn_rollout(model, batch, include_terminal_salvage=False)

    assert with_salvage.item() > without_salvage.item()


def test_training_batch_uses_stored_future_demand_trace() -> None:
    product = PaperLostSalesProductGenerator(_paper_config()).generate(1, "train")[0]
    product = product.__class__(
        **{
            **product.__dict__,
            "future_demand": [3, 4, 5, 6],
        }
    )

    batch = make_paper_lost_sales_training_batch([product], horizon=3, generator=torch.Generator().manual_seed(0))

    assert batch["demand_paths"].tolist() == [[3.0, 4.0, 5.0]]


def test_training_batch_can_resample_demand_instead_of_using_stored_trace() -> None:
    product = PaperLostSalesProductGenerator(_paper_config()).generate(1, "train")[0]
    product = product.__class__(
        **{
            **product.__dict__,
            "future_demand": [3, 4, 5, 6],
        }
    )

    batch = make_paper_lost_sales_training_batch(
        [product],
        horizon=3,
        generator=torch.Generator().manual_seed(0),
        demand_mode="resample",
    )

    assert batch["demand_paths"].tolist() != [[3.0, 4.0, 5.0]]


def test_fcnn_adapter_state_dependent_cap_stops_orders_at_high_inventory() -> None:
    product = PaperLostSalesProductGenerator(_paper_config()).generate(1, "test")[0]
    model = FCNNPolicy(input_dim=37, hidden_layers=1, hidden_units=8, activation="ELU")
    adapter = PaperFCNNPolicyAdapter(model, product, max_order_multiplier=1.0)

    order = adapter.act(inventory=max(product.historical_demand) + 100, t=0)

    assert order == 0.0


def test_training_and_evaluation_use_same_fcnn_preprocessing() -> None:
    product = PaperLostSalesProductGenerator(_paper_config()).generate(1, "test")[0]
    feature_stats = compute_paper_fcnn_feature_stats([product])
    model = FCNNPolicy(input_dim=37, hidden_layers=1, hidden_units=8, activation="ELU")
    adapter = PaperFCNNPolicyAdapter(model, product, feature_stats=feature_stats, max_order_factor=2.0)
    batch = make_paper_lost_sales_training_batch(
        [product],
        horizon=1,
        generator=torch.Generator().manual_seed(0),
        feature_stats=feature_stats,
    )
    inventory = torch.tensor([float(product.initial_inventory)])

    training_order = float(paper_lost_sales_fcnn_order(model, batch, inventory, max_order_factor=2.0).item())
    evaluation_order = adapter.act(float(product.initial_inventory), t=0)

    assert training_order == pytest.approx(evaluation_order)


def test_scaled_action_path_matches_training_and_evaluation() -> None:
    product = PaperLostSalesProductGenerator(_paper_config()).generate(1, "test")[0]
    feature_stats = compute_paper_fcnn_feature_stats([product])
    model = FCNNPolicy(input_dim=37, hidden_layers=1, hidden_units=8, activation="ELU")
    adapter = PaperFCNNPolicyAdapter(
        model,
        product,
        feature_stats=feature_stats,
        max_order_factor=4.0,
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
            max_order_factor=4.0,
            scale_action_by_history_mean=True,
        ).item()
    )
    evaluation_order = adapter.act(float(product.initial_inventory), t=0)

    assert training_order == pytest.approx(evaluation_order)


def test_fcnn_feature_stats_are_computed_from_training_products_only() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(2, "train")

    stats = compute_paper_fcnn_feature_stats(products)

    all_history = [value for product in products for value in product.historical_demand]
    assert stats.demand_scale == pytest.approx(sum(all_history) / len(all_history))
    assert stats.price_scale == pytest.approx(sum(product.price for product in products) / len(products))


def test_train_paper_lost_sales_fcnn_tracks_best_validation_checkpoint() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(4, "train")
    model = FCNNPolicy(input_dim=37, hidden_layers=1, hidden_units=8, activation="ELU")

    result = train_paper_lost_sales_fcnn(
        products,
        model,
        horizon=3,
        batch_size=2,
        epochs=2,
        learning_rate=1e-3,
        seed=0,
        validation_products=products[:2],
        validation_interval=1,
        validation_callback=lambda epoch, current_model: float(epoch),
    )

    assert result.best_validation_reward == 2.0
    assert result.best_state_dict is not None
    assert result.training_log is not None
    assert result.training_log[-1]["validation_mean_total_reward"] == 2.0


def test_training_always_validates_final_epoch() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(4, "train")
    model = FCNNPolicy(input_dim=37, hidden_layers=1, hidden_units=8, activation="ELU")
    validated_epochs = []

    result = train_paper_lost_sales_fcnn(
        products,
        model,
        horizon=3,
        batch_size=2,
        epochs=3,
        learning_rate=1e-3,
        seed=0,
        imitation_epochs=2,
        validation_products=products[:2],
        validation_interval=3,
        validation_callback=lambda epoch, current_model: validated_epochs.append(epoch) or float(epoch),
    )

    assert validated_epochs == [3, 5]
    assert result.validation_epochs == [3, 5]
    assert result.best_validation_reward == 5.0


def test_training_tracks_best_imitation_checkpoint() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(4, "train")
    model = FCNNPolicy(input_dim=37, hidden_layers=1, hidden_units=8, activation="ELU")

    result = train_paper_lost_sales_fcnn(
        products,
        model,
        horizon=3,
        batch_size=2,
        epochs=0,
        learning_rate=1e-3,
        seed=0,
        imitation_epochs=3,
    )

    assert result.best_imitation_loss is not None
    assert result.best_imitation_epoch in {1, 2, 3}
    assert result.best_imitation_state_dict is not None


def test_inventory_monotonicity_penalty_has_gradient_for_increasing_policy() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(2, "train")
    batch = make_paper_lost_sales_training_batch(
        products,
        horizon=2,
        generator=torch.Generator().manual_seed(0),
    )
    model = torch.nn.Sequential(torch.nn.Linear(37, 1, bias=False), torch.nn.Flatten(start_dim=0))
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].weight[0, 0] = 1.0

    penalty = paper_inventory_monotonicity_penalty(
        model,
        batch,
        torch.Generator().manual_seed(1),
        samples_per_product=2,
    )
    penalty.backward()

    assert penalty.item() > 0
    assert model[0].weight.grad is not None
    assert model[0].weight.grad[0, 0].item() > 0


def test_inventory_monotonicity_penalty_is_zero_for_decreasing_policy() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(2, "train")
    batch = make_paper_lost_sales_training_batch(
        products,
        horizon=2,
        generator=torch.Generator().manual_seed(0),
    )
    model = torch.nn.Sequential(torch.nn.Linear(37, 1, bias=False), torch.nn.Flatten(start_dim=0))
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].weight[0, 0] = -1.0

    penalty = paper_inventory_monotonicity_penalty(
        model,
        batch,
        torch.Generator().manual_seed(1),
        samples_per_product=2,
    )

    assert penalty.item() == 0.0


def test_reward_normalization_scale_uses_price_demand_and_horizon() -> None:
    batch = {
        "price": torch.tensor([10.0, 20.0]),
        "historical_mean_demand": torch.tensor([2.0, 4.0]),
    }

    scale = paper_reward_normalization_scale(batch, horizon=5)

    assert scale.item() == pytest.approx((10 * 2 * 5 + 20 * 4 * 5) / 2)


def test_normalized_reward_training_loss_uses_reported_scale() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(4, "train")
    model = FCNNPolicy(input_dim=37, hidden_layers=1, hidden_units=8, activation="ELU")
    generator = torch.Generator().manual_seed(3)
    indices = torch.randint(0, len(products), (2,), generator=generator).tolist()
    expected_batch = make_paper_lost_sales_training_batch(
        [products[index] for index in indices],
        horizon=3,
        generator=generator,
    )
    expected_scale = paper_reward_normalization_scale(expected_batch, horizon=3).item()

    result = train_paper_lost_sales_fcnn(
        products,
        model,
        horizon=3,
        batch_size=2,
        epochs=1,
        learning_rate=1e-3,
        seed=3,
        normalize_reward_loss=True,
    )

    assert result.training_losses[0] == pytest.approx(-result.mean_total_rewards[0] / expected_scale)


def test_disabled_penalty_preserves_training_behavior() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(4, "train")
    first = FCNNPolicy(input_dim=37, hidden_layers=1, hidden_units=8, activation="ELU")
    second = FCNNPolicy(input_dim=37, hidden_layers=1, hidden_units=8, activation="ELU")
    second.load_state_dict(first.state_dict())

    first_result = train_paper_lost_sales_fcnn(
        products,
        first,
        horizon=3,
        batch_size=2,
        epochs=1,
        learning_rate=1e-3,
        seed=7,
    )
    second_result = train_paper_lost_sales_fcnn(
        products,
        second,
        horizon=3,
        batch_size=2,
        epochs=1,
        learning_rate=1e-3,
        seed=7,
        monotonicity_penalty_weight=0.0,
        normalize_reward_loss=False,
    )

    assert first_result.training_losses == second_result.training_losses
    for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
        assert torch.equal(first_parameter, second_parameter)
