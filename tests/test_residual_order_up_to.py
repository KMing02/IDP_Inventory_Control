from __future__ import annotations

import torch
from torch import nn

from inventory_control.policies.neural.constrained_mlp import ConstrainedMLPResidualOrderUpTo
from inventory_control.policies.neural.fcnn import PaperFCNNResidualOrderUpTo
from inventory_control.training.direct_backprop import (
    historical_base_stock_imitation_loss,
    paper_lost_sales_context_features,
    paper_lost_sales_fcnn_order,
    paper_lost_sales_raw_order,
    paper_lost_sales_target_level,
    paper_order_diagnostics,
)


class _ConstantResidual(nn.Module):
    paper_action_parameterization = "residual_order_up_to"
    paper_residual_center = 1.0
    paper_residual_scale = 0.5
    paper_residual_anchor = "gamma"

    def __init__(self, output: float) -> None:
        super().__init__()
        self.output = nn.Parameter(torch.tensor(float(output)))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.output.expand(context.shape[0])


def _batch(batch_size: int = 2, history_length: int = 3) -> dict[str, torch.Tensor]:
    return {
        "initial_inventory": torch.zeros(batch_size),
        "history": torch.arange(1, batch_size * history_length + 1, dtype=torch.float32).reshape(batch_size, history_length),
        "feature_demand_scale": torch.full((batch_size,), 10.0),
        "feature_price_scale": torch.full((batch_size,), 100.0),
        "historical_mean_demand": torch.tensor([2.0, 4.0])[:batch_size],
        "historical_base_stock_level": torch.tensor([10.0, 20.0])[:batch_size],
        "price": torch.full((batch_size,), 100.0),
        "purchase_cost": torch.full((batch_size,), 40.0),
        "holding_cost": torch.full((batch_size,), 5.0),
        "lost_sale_penalty": torch.full((batch_size,), 10.0),
        "demand_paths": torch.ones(batch_size, 2),
    }


def test_context_features_exclude_inventory_and_have_history_plus_economics() -> None:
    batch = _batch()
    context = paper_lost_sales_context_features(batch)
    assert context.shape == (2, 7)
    assert torch.equal(context[:, :3], batch["history"])
    assert torch.allclose(context[:, 3:], torch.tensor([[1.0, 0.4, 0.05, 0.1]]).repeat(2, 1))


def test_residual_models_have_common_open_bound_and_fixed_semantics() -> None:
    torch.manual_seed(0)
    models = [
        PaperFCNNResidualOrderUpTo(history_length=3, hidden_layers=2, hidden_units=8, cnn_channels=2, cnn_dilations=(1,)),
        ConstrainedMLPResidualOrderUpTo(input_dim=7, hidden_layers=2, hidden_units=8),
    ]
    context = torch.randn(32, 7)
    for model in models:
        output = model(context)
        assert torch.all(output > 0.0)
        assert torch.all(output < 2.0)
        assert model.paper_residual_center == 1.0
        assert model.paper_residual_scale == 0.5
        assert model.paper_residual_anchor == "gamma"


def test_cmnn_residual_readout_is_exactly_zero_for_every_context_at_initialization() -> None:
    torch.manual_seed(7)
    model = ConstrainedMLPResidualOrderUpTo(input_dim=7, hidden_layers=2, hidden_units=8)
    context = torch.randn(128, 7) * 10.0
    with torch.no_grad():
        preactivation = model.preactivation(context)
        output = model(context)
    assert torch.count_nonzero(model.residual_readout.weight).item() == 0
    assert torch.count_nonzero(model.residual_readout.bias).item() == 0
    assert torch.equal(preactivation, torch.zeros_like(preactivation))
    assert torch.equal(output, torch.ones_like(output))


def test_zero_residual_reproduces_anchor_and_mean_enters_exactly_once() -> None:
    batch = _batch()
    centered = paper_lost_sales_target_level(_ConstantResidual(1.0), batch)
    assert torch.equal(centered, batch["historical_base_stock_level"])

    upper = paper_lost_sales_target_level(_ConstantResidual(2.0), batch)
    lower = paper_lost_sales_target_level(_ConstantResidual(0.0), batch)
    expected_half_mean = 0.5 * batch["historical_mean_demand"]
    assert torch.allclose(upper - batch["historical_base_stock_level"], expected_half_mean)
    assert torch.allclose(lower - batch["historical_base_stock_level"], -expected_half_mean)


def test_raw_decoder_has_exact_order_up_to_shape() -> None:
    batch = _batch(batch_size=1)
    model = _ConstantResidual(1.0)
    inventory = torch.tensor([1.0, 3.0, 9.0, 10.0, 12.0])
    expanded = {key: value.repeat(5, *([1] * (value.ndim - 1))) for key, value in batch.items()}
    raw = paper_lost_sales_raw_order(model, expanded, inventory, scale_action_by_history_mean=True)
    assert torch.allclose(raw, torch.relu(torch.tensor(10.0) - inventory))
    active_slopes = torch.diff(raw[:4]) / torch.diff(inventory[:4])
    assert torch.allclose(active_slopes, torch.full_like(active_slopes, -1.0))
    assert raw[-2:].tolist() == [0.0, 0.0]


def test_deployed_decoder_uses_shared_factor_four_cap() -> None:
    batch = _batch(batch_size=1)
    model = _ConstantResidual(1.0)
    inventory = torch.tensor([0.0])
    deployed = paper_lost_sales_fcnn_order(
        model,
        batch,
        inventory,
        max_order_factor=4.0,
        scale_action_by_history_mean=True,
    )
    assert deployed.item() == 8.0


def test_imitation_and_diagnostics_preserve_matched_raw_and_deployed_semantics() -> None:
    batch = _batch(batch_size=1)
    # mean=2 gives a zero-inventory cap of 8.  u=0 gives S_theta=9 while
    # the uncapped historical target is 10, so deployed errors are zero and
    # raw errors are exactly one at all sampled active states.
    model = _ConstantResidual(0.0)
    deployed_loss = historical_base_stock_imitation_loss(
        model,
        batch,
        torch.Generator().manual_seed(1),
        samples_per_product=4,
        max_order_factor=4.0,
        clip_predictions=True,
    )
    raw_loss = historical_base_stock_imitation_loss(
        model,
        batch,
        torch.Generator().manual_seed(1),
        samples_per_product=4,
        max_order_factor=4.0,
        clip_predictions=False,
    )
    diagnostics = paper_order_diagnostics(
        model,
        batch,
        max_order_factor=4.0,
        scale_action_by_history_mean=True,
        inventory_fractions=(0.0, 0.25, 0.5, 0.75),
    )
    assert deployed_loss.item() == 0.0
    assert raw_loss.item() == 1.0
    assert diagnostics["deployed_imitation_rmse"] == 0.0
    assert diagnostics["raw_imitation_rmse"] == 1.0
