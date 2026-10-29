import pytest
import inspect

torch = pytest.importorskip("torch")

from inventory_control.policies.neural.fcnn import (
    FCNNPolicy,
    PaperCausalCNNPolicy,
    PaperContextCausalCNNPolicy,
    PaperFCNNFeatureStats,
    PaperFCNNPolicyAdapter,
    build_paper_history_features,
    compute_paper_fcnn_feature_stats,
    make_paper_direct_target_stock_fcnn_model,
    paper_temporal_summary_features,
)
from inventory_control.training.direct_backprop import (
    make_paper_lost_sales_training_batch,
    paper_inventory_monotonicity_penalty,
    paper_lost_sales_fcnn_order,
    paper_lost_sales_fcnn_rollout,
    paper_lost_sales_target_level,
    paper_rolling_target_levels,
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


def test_direct_target_stock_fcnn_uses_raw_history_causal_cnn_when_configured() -> None:
    config = {
        **_paper_config(),
        "paper_history_feature_mode": "raw_lags",
        "fcnn": {
            "history_encoder": "causal_cnn",
            "hidden_layers": 2,
            "hidden_units": 32,
            "activation": "ELU",
            "cnn_channels": 4,
            "cnn_dilations": [1, 2],
        },
    }

    model = make_paper_direct_target_stock_fcnn_model(
        config,
        metadata={**config["fcnn"], "initial_normalized_target": 1.5},
    )
    target_factors = model(torch.randn(3, config["history_length_H"] + 4))

    assert isinstance(model, PaperContextCausalCNNPolicy)
    assert model.paper_action_parameterization == "order_up_to"
    assert model.paper_history_encoder == "causal_cnn"
    assert target_factors.tolist() == pytest.approx([1.5, 1.5, 1.5])


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


def test_raw_paper_fcnn_rollout_uses_only_rolling_observed_history() -> None:
    class RecordingRawPolicy(torch.nn.Module):
        paper_action_parameterization = "raw"
        paper_roll_history = True

        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))
            self.inputs: list[torch.Tensor] = []

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            self.inputs.append(features.detach().clone())
            return self.bias.expand(features.shape[0])

    product = PaperLostSalesProductGenerator(_paper_config()).generate(1, "train")[0]
    product = product.__class__(
        **{
            **product.__dict__,
            "historical_demand": [1, 2, 3],
            "future_demand": [4, 99],
        }
    )
    model = RecordingRawPolicy()
    batch = make_paper_lost_sales_training_batch(
        [product],
        horizon=2,
        generator=torch.Generator().manual_seed(0),
        feature_stats=PaperFCNNFeatureStats(demand_scale=1.0, price_scale=1.0),
    )
    batch["initial_inventory"] = torch.tensor([10.0])

    paper_lost_sales_fcnn_rollout(
        model,
        batch,
        include_terminal_salvage=False,
        enforce_action_cap=False,
    )

    assert len(model.inputs) == 2
    assert model.inputs[0][0, :4].tolist() == pytest.approx([10.0, 1.0, 2.0, 3.0])
    assert model.inputs[1][0, :4].tolist() == pytest.approx([6.0, 2.0, 3.0, 4.0])
    assert 99.0 not in model.inputs[1][0].tolist()


def test_direct_target_rollout_recomputes_target_from_causal_rolling_history() -> None:
    class RecordingTargetPolicy(torch.nn.Module):
        paper_action_parameterization = "order_up_to"
        paper_scale_target_by_history_mean = True
        paper_roll_history = True

        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))
            self.inputs: list[torch.Tensor] = []

        def forward(self, context: torch.Tensor) -> torch.Tensor:
            self.inputs.append(context.detach().clone())
            return context[:, 2] + self.bias

    product = PaperLostSalesProductGenerator(_paper_config()).generate(1, "train")[0]
    product = product.__class__(
        **{
            **product.__dict__,
            "historical_demand": [1, 2, 3],
            "future_demand": [4, 99],
        }
    )
    model = RecordingTargetPolicy()
    batch = make_paper_lost_sales_training_batch(
        [product],
        horizon=2,
        generator=torch.Generator().manual_seed(0),
        feature_stats=PaperFCNNFeatureStats(demand_scale=1.0, price_scale=1.0),
    )
    batch["initial_inventory"] = torch.tensor([10.0])

    paper_lost_sales_fcnn_rollout(
        model,
        batch,
        include_terminal_salvage=False,
        enforce_action_cap=False,
    )

    observed_inputs = torch.cat(model.inputs, dim=0)
    assert observed_inputs.shape == (2, 7)  # Three history values plus four economics; no inventory.
    assert observed_inputs[0, :3].tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert observed_inputs[1, :3].tolist() == pytest.approx([2.0, 3.0, 4.0])
    assert 10.0 not in observed_inputs[0, :3].tolist()
    assert 99.0 not in observed_inputs[1].tolist()


def test_temporal_history_features_use_documented_causal_definitions() -> None:
    history = torch.arange(1, 33, dtype=torch.float32)
    summaries = paper_temporal_summary_features(history)
    features = build_paper_history_features(history.tolist(), demand_scale=2.0, mode="raw_lags_temporal")

    assert summaries[:3].tolist() == pytest.approx([16.5, 28.5, 30.5])
    assert summaries[3].item() == pytest.approx(history.std(unbiased=False).item())
    assert summaries[4].item() == pytest.approx(1.0)
    assert features[:32] == pytest.approx((history / 2.0).tolist())
    assert features[32:] == pytest.approx((summaries / 2.0).tolist())


def test_temporal_rolling_targets_match_sequential_trajectory_and_gradients() -> None:
    class TinyTemporalTargetPolicy(torch.nn.Module):
        paper_action_parameterization = "order_up_to"
        paper_scale_target_by_history_mean = True
        paper_history_feature_mode = "raw_lags_temporal"

        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(41, 1)
            torch.nn.init.constant_(self.linear.weight, 0.002)
            torch.nn.init.constant_(self.linear.bias, 0.5)

        def forward(self, context: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.softplus(self.linear(context)).squeeze(-1)

    product = PaperLostSalesProductGenerator(_paper_config()).generate(1, "train")[0]
    product = product.__class__(
        **{
            **product.__dict__,
            "historical_demand": list(range(1, 33)),
            "future_demand": [33, 999],
            "initial_inventory": 0,
        }
    )
    batch = make_paper_lost_sales_training_batch(
        [product],
        horizon=2,
        generator=torch.Generator().manual_seed(0),
        feature_stats=PaperFCNNFeatureStats(demand_scale=1.0, price_scale=1.0),
        history_feature_mode="raw_lags_temporal",
    )
    batch["initial_inventory"] = torch.zeros(1)
    sequential_model = TinyTemporalTargetPolicy()
    vectorized_model = TinyTemporalTargetPolicy()
    vectorized_model.load_state_dict(sequential_model.state_dict())

    rolling_batch = dict(batch)
    sequential_targets = []
    for period in range(2):
        sequential_targets.append(paper_lost_sales_target_level(sequential_model, rolling_batch))
        observed = batch["demand_paths"][:, period] / batch["feature_demand_scale"]
        rolling_batch["history"] = torch.cat([rolling_batch["history"][:, 1:], observed.unsqueeze(1)], dim=1)
        rolling_batch["temporal_summary_features"] = paper_temporal_summary_features(rolling_batch["history"])
    sequential_targets = torch.stack(sequential_targets, dim=1)
    vectorized_targets = paper_rolling_target_levels(
        vectorized_model,
        batch,
        batch["demand_paths"],
        checkpoint_model=True,
        chunk_size=1,
    )

    def trajectory(targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inventory = batch["initial_inventory"]
        actions, inventories, rewards = [], [], []
        for period in range(2):
            action = torch.relu(targets[:, period] - inventory)
            demand = batch["demand_paths"][:, period]
            available = inventory + action
            sales = torch.minimum(demand, available)
            lost_sales = torch.relu(demand - available)
            inventory = torch.relu(available - demand)
            reward = (
                batch["price"] * sales
                - batch["purchase_cost"] * action
                - batch["lost_sale_penalty"] * lost_sales
                - batch["holding_cost"] * inventory
            )
            actions.append(action)
            inventories.append(inventory)
            rewards.append(reward)
        return torch.stack(actions, 1), torch.stack(inventories, 1), torch.stack(rewards, 1)

    expected = trajectory(sequential_targets)
    actual = trajectory(vectorized_targets)
    expected[2].sum().backward()
    actual[2].sum().backward()

    assert torch.allclose(sequential_targets, vectorized_targets)
    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        assert torch.allclose(expected_tensor, actual_tensor)
    assert 999.0 not in batch["history"].tolist()[0]
    assert torch.isfinite(vectorized_model.linear.weight.grad).all()
    assert torch.allclose(sequential_model.linear.weight.grad, vectorized_model.linear.weight.grad, atol=1e-5)


def test_checkpointed_direct_target_has_equivalent_output_and_gradient() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(2, "train")
    batch = make_paper_lost_sales_training_batch(
        products,
        horizon=2,
        generator=torch.Generator().manual_seed(0),
    )
    plain = make_paper_direct_target_stock_fcnn_model(_paper_config())
    checkpointed = make_paper_direct_target_stock_fcnn_model(_paper_config())
    checkpointed.load_state_dict(plain.state_dict())

    plain_target = paper_lost_sales_target_level(plain, batch)
    checkpointed_target = paper_lost_sales_target_level(checkpointed, batch, checkpoint_model=True)
    plain_target.sum().backward()
    checkpointed_target.sum().backward()

    assert torch.allclose(plain_target, checkpointed_target)
    for plain_parameter, checkpointed_parameter in zip(plain.parameters(), checkpointed.parameters(), strict=True):
        assert torch.allclose(plain_parameter.grad, checkpointed_parameter.grad)


def test_chunked_rolling_targets_match_periodwise_outputs_and_gradients() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(2, "train")
    batch = make_paper_lost_sales_training_batch(
        products,
        horizon=3,
        generator=torch.Generator().manual_seed(0),
    )
    periodwise = make_paper_direct_target_stock_fcnn_model(_paper_config())
    chunked = make_paper_direct_target_stock_fcnn_model(_paper_config())
    chunked.load_state_dict(periodwise.state_dict())
    rolling_batch = dict(batch)
    periodwise_targets = []
    for period in range(3):
        periodwise_targets.append(paper_lost_sales_target_level(periodwise, rolling_batch))
        observed = batch["demand_paths"][:, period] / torch.clamp(batch["feature_demand_scale"], min=1.0)
        rolling_batch["history"] = torch.cat([rolling_batch["history"][:, 1:], observed.unsqueeze(1)], dim=1)
    expected = torch.stack(periodwise_targets, dim=1)
    actual = paper_rolling_target_levels(
        chunked,
        batch,
        batch["demand_paths"],
        checkpoint_model=True,
        chunk_size=2,
    )
    expected.sum().backward()
    actual.sum().backward()

    assert torch.allclose(expected, actual)
    for expected_parameter, actual_parameter in zip(periodwise.parameters(), chunked.parameters(), strict=True):
        assert torch.allclose(expected_parameter.grad, actual_parameter.grad, atol=1e-6)


def test_vectorized_rolling_rollout_matches_sequential_trajectory_reward_and_gradients() -> None:
    class TinyTargetPolicy(torch.nn.Module):
        paper_action_parameterization = "order_up_to"
        paper_scale_target_by_history_mean = True
        paper_history_feature_mode = "raw_lags"

        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(7, 1)
            torch.nn.init.constant_(self.linear.weight, 0.01)
            torch.nn.init.constant_(self.linear.bias, 0.5)

        def forward(self, context: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.softplus(self.linear(context)).squeeze(-1)

    products = PaperLostSalesProductGenerator(_paper_config()).generate(2, "train")
    products = [
        product.__class__(
            **{
                **product.__dict__,
                "historical_demand": [1 + index, 2 + index, 3 + index],
                "future_demand": [4 + index, 2 + index, 5 + index],
                "initial_inventory": 0,
            }
        )
        for index, product in enumerate(products)
    ]
    batch = make_paper_lost_sales_training_batch(
        products,
        horizon=3,
        generator=torch.Generator().manual_seed(0),
        feature_stats=PaperFCNNFeatureStats(demand_scale=1.0, price_scale=1.0),
    )
    batch["initial_inventory"] = torch.zeros(2)
    sequential_model = TinyTargetPolicy()
    vectorized_model = TinyTargetPolicy()
    vectorized_model.load_state_dict(sequential_model.state_dict())

    rolling_batch = dict(batch)
    sequential_targets = []
    for period in range(3):
        sequential_targets.append(paper_lost_sales_target_level(sequential_model, rolling_batch))
        observed = batch["demand_paths"][:, period] / batch["feature_demand_scale"]
        rolling_batch["history"] = torch.cat([rolling_batch["history"][:, 1:], observed.unsqueeze(1)], dim=1)
    sequential_targets_tensor = torch.stack(sequential_targets, dim=1)
    vectorized_targets = paper_rolling_target_levels(
        vectorized_model,
        batch,
        batch["demand_paths"],
        checkpoint_model=True,
        chunk_size=4,
    )

    def trajectory(targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inventory = batch["initial_inventory"]
        inventories = [inventory]
        actions = []
        rewards = []
        for period in range(3):
            action = torch.relu(targets[:, period] - inventory)
            demand = batch["demand_paths"][:, period]
            available = inventory + action
            sales = torch.minimum(demand, available)
            lost_sales = torch.relu(demand - available)
            inventory = torch.relu(available - demand)
            reward = (
                batch["price"] * sales
                - batch["purchase_cost"] * action
                - batch["lost_sale_penalty"] * lost_sales
                - batch["holding_cost"] * inventory
            )
            actions.append(action)
            inventories.append(inventory)
            rewards.append(reward)
        return (
            torch.stack(actions, dim=1),
            torch.stack(inventories, dim=1),
            torch.stack(rewards, dim=1),
            torch.stack(rewards, dim=1).sum(dim=1) + batch["purchase_cost"] * inventory,
        )

    sequential_actions, sequential_inventories, sequential_rewards, sequential_total = trajectory(
        sequential_targets_tensor
    )
    vectorized_actions, vectorized_inventories, vectorized_rewards, vectorized_total = trajectory(
        vectorized_targets
    )
    sequential_total.sum().backward()
    vectorized_total.sum().backward()

    assert torch.allclose(sequential_targets_tensor, vectorized_targets)
    assert torch.allclose(sequential_actions, vectorized_actions)
    assert torch.allclose(sequential_inventories, vectorized_inventories)
    assert torch.allclose(sequential_rewards, vectorized_rewards)
    assert torch.allclose(sequential_total, vectorized_total)
    for sequential_parameter, vectorized_parameter in zip(
        sequential_model.parameters(), vectorized_model.parameters(), strict=True
    ):
        assert torch.allclose(sequential_parameter.grad, vectorized_parameter.grad, atol=1e-5)


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


def test_direct_target_stock_rollout_can_disable_implementation_cap() -> None:
    class ConstantTarget(torch.nn.Module):
        paper_action_parameterization = "order_up_to"
        paper_scale_target_by_history_mean = True

        def __init__(self) -> None:
            super().__init__()
            self.target = torch.nn.Parameter(torch.tensor(10.0))

        def forward(self, context: torch.Tensor) -> torch.Tensor:
            return self.target.expand(context.shape[0])

    product = PaperLostSalesProductGenerator(_paper_config()).generate(1, "train")[0]
    product = product.__class__(**{**product.__dict__, "future_demand": [0], "initial_inventory": 0})
    batch = make_paper_lost_sales_training_batch([product], horizon=1, generator=torch.Generator().manual_seed(0))
    model = ConstantTarget()

    capped = paper_lost_sales_fcnn_rollout(
        model,
        batch,
        max_order_factor=4.0,
        include_terminal_salvage=False,
        enforce_action_cap=True,
    )
    uncapped = paper_lost_sales_fcnn_rollout(
        model,
        batch,
        max_order_factor=4.0,
        include_terminal_salvage=False,
        enforce_action_cap=False,
    )

    assert uncapped.item() < capped.item()


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
