from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from inventory_control.idp_benchmark import _model_orders
from inventory_control.paper_lost_sales import non_omniscient_gamma_base_stock_policy
from inventory_control.policies.neural.fcnn import (
    PaperFCNNFeatureStats,
    PaperFCNNTargetStock,
    make_paper_target_stock_fcnn_model,
    target_stock_informed_scalar,
)
from inventory_control.products import PaperLostSalesProduct
from inventory_control.training.direct_backprop import (
    paper_lost_sales_context_features,
    paper_lost_sales_fcnn_order,
    paper_lost_sales_raw_order,
    paper_lost_sales_target_stock_level,
)


class _ConstantNormalizedTarget(nn.Module):
    paper_action_parameterization = "target_stock"
    paper_target_stock_scale = "historical_mean_demand"

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.value.expand(context.shape[0])


def _batch(batch_size: int = 1, history_length: int = 3) -> dict[str, torch.Tensor]:
    return {
        "initial_inventory": torch.zeros(batch_size),
        "history": torch.arange(1, history_length + 1, dtype=torch.float32).repeat(batch_size, 1) / 10.0,
        "feature_demand_scale": torch.full((batch_size,), 10.0),
        "feature_price_scale": torch.full((batch_size,), 100.0),
        "historical_mean_demand": torch.full((batch_size,), 2.0),
        "price": torch.full((batch_size,), 100.0),
        "purchase_cost": torch.full((batch_size,), 40.0),
        "holding_cost": torch.full((batch_size,), 5.0),
        "lost_sale_penalty": torch.full((batch_size,), 10.0),
        "demand_paths": torch.ones(batch_size, 2),
    }


def _product(product_id: str, history: list[int]) -> PaperLostSalesProduct:
    mean = float(np.mean(history))
    return PaperLostSalesProduct(
        product_id=product_id,
        price=100.0,
        purchase_cost=40.0,
        holding_cost=5.0,
        lost_sale_penalty=10.0,
        true_mean_demand=mean,
        true_coefficient_of_variation=0.2,
        gamma_shape=25.0,
        gamma_scale=mean / 25.0,
        demand_distribution_type="Gamma",
        historical_demand=history,
        initial_inventory=0,
    )


def _config() -> dict:
    return {
        "history_length_H": 3,
        "fcnn": {
            "history_encoder": "causal_cnn",
            "hidden_layers": 2,
            "hidden_units": 8,
            "cnn_channels": 2,
            "cnn_dilations": [1],
            "activation": "ELU",
        },
    }


def test_target_stock_network_input_excludes_inventory() -> None:
    model = PaperFCNNTargetStock(
        history_length=3,
        hidden_layers=2,
        hidden_units=8,
        cnn_channels=2,
        cnn_dilations=(1,),
    )
    context = paper_lost_sales_context_features(_batch(batch_size=4))

    assert context.shape == (4, 7)
    assert model.mlp[0].in_features == 2 + 4
    assert model(context).shape == (4,)
    assert torch.all(model(context) > 0.0)


def test_target_level_scales_once_and_is_independent_of_inventory() -> None:
    model = _ConstantNormalizedTarget(2.5)
    batch = _batch(batch_size=5)
    inventory = torch.tensor([1.0, 2.0, 4.0, 5.0, 7.0])

    target = paper_lost_sales_target_stock_level(model, batch)
    raw_unscaled_flag = paper_lost_sales_raw_order(
        model,
        batch,
        inventory,
        scale_action_by_history_mean=False,
    )
    raw_scaled_flag = paper_lost_sales_raw_order(
        model,
        batch,
        inventory,
        scale_action_by_history_mean=True,
    )

    assert torch.equal(target, torch.full((5,), 5.0))
    assert torch.equal(raw_unscaled_flag, torch.tensor([4.0, 3.0, 1.0, 0.0, 0.0]))
    assert torch.equal(raw_scaled_flag, raw_unscaled_flag)
    assert torch.allclose(
        torch.diff(raw_unscaled_flag[:4]) / torch.diff(inventory[:4]),
        torch.full((3,), -1.0),
    )
    assert torch.equal(raw_unscaled_flag[inventory >= target], torch.zeros(2))
    assert torch.equal(
        raw_unscaled_flag[inventory < target] + inventory[inventory < target],
        target[inventory < target],
    )


def test_target_stock_deployment_uses_shared_factor_four_cap() -> None:
    model = _ConstantNormalizedTarget(10.0)
    batch = _batch(batch_size=2)
    inventory = torch.tensor([0.0, 3.0])

    raw = paper_lost_sales_raw_order(model, batch, inventory)
    deployed = paper_lost_sales_fcnn_order(model, batch, inventory, max_order_factor=4.0)

    assert torch.equal(raw, torch.tensor([20.0, 17.0]))
    assert torch.equal(deployed, torch.tensor([8.0, 5.0]))


def test_target_stock_inference_does_not_use_historical_gamma_anchor() -> None:
    model = _ConstantNormalizedTarget(1.5)
    batch = _batch(batch_size=1)
    assert "historical_base_stock_level" not in batch
    assert paper_lost_sales_target_stock_level(model, batch).item() == pytest.approx(3.0)

    product = _product("synthetic", [1, 2, 3])
    stats = PaperFCNNFeatureStats(demand_scale=2.0, price_scale=100.0)
    with patch(
        "inventory_control.idp_benchmark.non_omniscient_gamma_base_stock_policy",
        side_effect=AssertionError("target-stock inference must not construct a Gamma anchor"),
    ):
        raw, deployed = _model_orders(
            model,
            product,
            stats,
            np.asarray([0.0, 2.0, 8.0]),
            max_order_factor=4.0,
            scale_action_by_history_mean=True,
        )

    assert np.allclose(raw, [3.0, 1.0, 0.0])
    assert np.allclose(deployed, raw)


def test_standard_and_informed_initialization_modes() -> None:
    products = [
        _product("train_a", [8, 10, 12]),
        _product("train_b", [16, 20, 24]),
    ]
    torch.manual_seed(0)
    standard = make_paper_target_stock_fcnn_model(_config(), initialization_mode="standard")
    standard_final = standard.mlp[-1]
    assert isinstance(standard_final, nn.Linear)
    assert torch.count_nonzero(standard_final.weight).item() > 0
    assert standard.target_stock_initialization_mode == "standard"
    assert standard.target_stock_initialization_scalar is None

    informed = make_paper_target_stock_fcnn_model(
        _config(),
        initialization_mode="target_stock_informed",
        training_products=products,
    )
    informed_final = informed.mlp[-1]
    scalar = target_stock_informed_scalar(products)
    expected = sum(
        max(float(np.mean(product.historical_demand)), 1.0)
        * float(non_omniscient_gamma_base_stock_policy(product).S)
        for product in products
    ) / sum(max(float(np.mean(product.historical_demand)), 1.0) ** 2 for product in products)
    context = torch.randn(16, 7)

    assert isinstance(informed_final, nn.Linear)
    assert torch.count_nonzero(informed_final.weight).item() == 0
    assert scalar == pytest.approx(expected)
    assert informed.target_stock_initialization_mode == "target_stock_informed"
    assert informed.target_stock_initialization_scalar == pytest.approx(scalar)
    assert torch.allclose(informed(context), torch.full((16,), scalar), atol=1e-6)


def test_informed_initialization_requires_training_products_and_valid_mode() -> None:
    with pytest.raises(ValueError, match="requires training_products"):
        make_paper_target_stock_fcnn_model(_config(), initialization_mode="target_stock_informed")
    with pytest.raises(ValueError, match="initialization_mode"):
        make_paper_target_stock_fcnn_model(_config(), initialization_mode="unknown")
