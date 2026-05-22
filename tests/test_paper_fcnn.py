import pytest

torch = pytest.importorskip("torch")

from inventory_control.policies.neural.fcnn import FCNNPolicy, PaperFCNNPolicyAdapter, compute_paper_fcnn_feature_stats
from inventory_control.training.direct_backprop import (
    make_paper_lost_sales_training_batch,
    paper_lost_sales_fcnn_order,
    paper_lost_sales_fcnn_rollout,
    train_paper_lost_sales_fcnn,
)
from inventory_control.products import PaperLostSalesProductGenerator
from tests.test_paper_lost_sales import _paper_config


def test_fcnn_output_is_nonnegative() -> None:
    model = FCNNPolicy(input_dim=37, hidden_layers=2, hidden_units=8, activation="ELU")
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
