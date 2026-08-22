from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

from inventory_control.policies.neural.fcnn import PaperFCNNFeatureStats
from inventory_control.policies.neural.monokan import (
    make_paper_deep_grid7_raw_action_monokan_model,
    paper_raw_action_monotonic_vars,
)
from inventory_control.raw_monokan_experiment import validate_raw_protocol
from inventory_control.training.direct_backprop import (
    make_paper_lost_sales_training_batch,
    paper_lost_sales_fcnn_rollout,
    paper_lost_sales_raw_order,
)
from inventory_control.products import PaperLostSalesProductGenerator
from tests.test_paper_lost_sales import _paper_config


def _raw_config() -> dict:
    return yaml.safe_load(
        Path("configs/paper_lost_sales_raw_monokan_deepgrid7_5k.yaml").read_text(encoding="utf-8")
    )


def test_raw_feature_direction_map_matches_canonical_feature_order() -> None:
    directions = paper_raw_action_monotonic_vars(32)

    assert directions == {
        0: -1,
        **{index: 1 for index in range(1, 33)},
        33: 1,
        34: -1,
        35: -1,
        36: 1,
    }


def test_raw_deepgrid7_factory_freezes_architecture_and_action_semantics() -> None:
    model = make_paper_deep_grid7_raw_action_monokan_model(_raw_config())

    assert [layer.in_dim for layer in model.layers] == [37, 16, 16]
    assert [layer.out_dim for layer in model.layers] == [16, 16, 1]
    assert all(layer.num_grid_intervals == 7 for layer in model.layers)
    assert model.paper_action_parameterization == "raw"
    assert model.paper_roll_history is True
    assert model.paper_enforce_action_cap is False
    assert model.output_upper_bound is None


def test_raw_monokan_protocol_matches_validated_raw_fcnn() -> None:
    audit = validate_raw_protocol()

    assert audit["protocol"]["training_products"] == 5_000
    assert audit["protocol"]["batch_size"] == 2_500
    assert audit["protocol"]["reward_epochs"] == 1_000
    assert audit["protocol"]["validation_interval"] == 1_000


def test_raw_monokan_rollout_is_causal_and_rolls_only_observed_demand() -> None:
    config = {**_paper_config(), "history_length_H": 3, "monokan": {"grid_range": [-2.0, 4.0]}}
    model = make_paper_deep_grid7_raw_action_monokan_model(config)
    product = PaperLostSalesProductGenerator(_paper_config()).generate(1, "train")[0]
    product = product.__class__(
        **{
            **product.__dict__,
            "historical_demand": [1, 2, 3],
            "future_demand": [4, 99],
        }
    )
    batch = make_paper_lost_sales_training_batch(
        [product],
        horizon=2,
        generator=torch.Generator().manual_seed(4),
        feature_stats=PaperFCNNFeatureStats(demand_scale=1.0, price_scale=1.0),
    )
    batch["initial_inventory"] = torch.tensor([10.0])
    changed = {name: value.clone() for name, value in batch.items()}
    changed["demand_paths"][0, 1] = 123.0
    first_inputs: list[torch.Tensor] = []
    second_inputs: list[torch.Tensor] = []
    hook = model.register_forward_pre_hook(lambda _module, args: first_inputs.append(args[0].detach().clone()))
    first_reward = paper_lost_sales_fcnn_rollout(
        model, batch, include_terminal_salvage=False, enforce_action_cap=False
    )
    hook.remove()
    hook = model.register_forward_pre_hook(lambda _module, args: second_inputs.append(args[0].detach().clone()))
    second_reward = paper_lost_sales_fcnn_rollout(
        model, changed, include_terminal_salvage=False, enforce_action_cap=False
    )
    hook.remove()

    assert len(first_inputs) == len(second_inputs) == 2
    assert torch.allclose(first_inputs[0], second_inputs[0])
    assert torch.allclose(first_inputs[1], second_inputs[1])
    assert first_inputs[0][0, :4].tolist() == pytest.approx([10.0, 1.0, 2.0, 3.0])
    assert first_inputs[1][0, 1:4].tolist() == pytest.approx([2.0, 3.0, 4.0])
    assert first_reward.item() != second_reward.item()  # Only the later realized reward changes.


def test_raw_monokan_inventory_feedback_and_all_projected_directions() -> None:
    model = make_paper_deep_grid7_raw_action_monokan_model(_raw_config())
    inputs = torch.randn(8, 37, generator=torch.Generator().manual_seed(7))
    base = model(inputs)

    for feature_index, direction in model.monotonic_vars.items():
        changed = inputs.clone()
        changed[:, feature_index] += 0.1
        delta = model(changed) - base
        assert torch.all(float(direction) * delta >= -1e-5), feature_index

    higher_inventory = inputs.clone()
    higher_inventory[:, 0] += 0.5
    inventory_delta = model(higher_inventory) - base
    assert torch.all(inventory_delta <= 1e-5)
    assert torch.any(torch.abs(inventory_delta) > 1e-8)


def test_raw_rollout_matches_transparent_sequential_reference_with_salvage() -> None:
    class FixedRawPolicy(torch.nn.Module):
        paper_action_parameterization = "raw"
        paper_roll_history = True

        def __init__(self) -> None:
            super().__init__()
            self.order = torch.nn.Parameter(torch.tensor(2.0))
            self.inputs: list[torch.Tensor] = []

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            self.inputs.append(features.detach().clone())
            return self.order.expand(features.shape[0])

    model = FixedRawPolicy()
    batch = {
        "initial_inventory": torch.tensor([5.0]),
        "demand_paths": torch.tensor([[3.0, 3.0]]),
        "history": torch.tensor([[1.0, 2.0, 3.0]]),
        "feature_demand_scale": torch.tensor([1.0]),
        "feature_price_scale": torch.tensor([1.0]),
        "historical_mean_demand": torch.tensor([2.0]),
        "historical_base_stock_level": torch.tensor([0.0]),
        "price": torch.tensor([10.0]),
        "purchase_cost": torch.tensor([2.0]),
        "holding_cost": torch.tensor([1.0]),
        "lost_sale_penalty": torch.tensor([5.0]),
    }

    reward = paper_lost_sales_fcnn_rollout(
        model,
        batch,
        include_terminal_salvage=True,
        enforce_action_cap=False,
    )

    assert [inputs[0, 0].item() for inputs in model.inputs] == pytest.approx([5.0, 4.0])
    assert model.inputs[1][0, 1:4].tolist() == pytest.approx([2.0, 3.0, 3.0])
    period_rewards = [10.0 * 3.0 - 2.0 * 2.0 - 1.0 * 4.0, 10.0 * 3.0 - 2.0 * 2.0 - 1.0 * 3.0]
    terminal_salvage = 2.0 * 3.0
    assert period_rewards == pytest.approx([22.0, 23.0])
    assert reward.item() == pytest.approx(sum(period_rewards) + terminal_salvage)


def test_raw_monokan_reward_has_finite_nonzero_parameter_gradients() -> None:
    config = {**_paper_config(), "history_length_H": 3, "monokan": {"grid_range": [-2.0, 4.0]}}
    model = make_paper_deep_grid7_raw_action_monokan_model(config)
    products = PaperLostSalesProductGenerator(_paper_config()).generate(2, "train")
    products = [product.__class__(**{**product.__dict__, "historical_demand": [1, 2, 3]}) for product in products]
    batch = make_paper_lost_sales_training_batch(
        products,
        horizon=3,
        generator=torch.Generator().manual_seed(8),
        feature_stats=PaperFCNNFeatureStats(demand_scale=1.0, price_scale=1.0),
    )

    paper_lost_sales_fcnn_rollout(model, batch, enforce_action_cap=False).sum().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]

    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.any(torch.abs(gradient) > 0) for gradient in gradients)


def test_raw_action_checkpointing_preserves_reward_and_gradients() -> None:
    config = {**_paper_config(), "history_length_H": 3, "monokan": {"grid_range": [-2.0, 4.0]}}
    plain = make_paper_deep_grid7_raw_action_monokan_model(config)
    checkpointed = copy.deepcopy(plain)
    products = PaperLostSalesProductGenerator(_paper_config()).generate(2, "train")
    products = [product.__class__(**{**product.__dict__, "historical_demand": [1, 2, 3]}) for product in products]
    batch = make_paper_lost_sales_training_batch(
        products,
        horizon=3,
        generator=torch.Generator().manual_seed(9),
        feature_stats=PaperFCNNFeatureStats(demand_scale=1.0, price_scale=1.0),
    )

    plain_reward = paper_lost_sales_fcnn_rollout(
        plain, batch, enforce_action_cap=False, checkpoint_raw_action=False
    ).sum()
    checkpointed_reward = paper_lost_sales_fcnn_rollout(
        checkpointed, batch, enforce_action_cap=False, checkpoint_raw_action=True
    ).sum()
    plain_reward.backward()
    checkpointed_reward.backward()

    assert torch.allclose(plain_reward, checkpointed_reward, atol=1e-5, rtol=1e-5)
    for plain_parameter, checkpointed_parameter in zip(
        plain.parameters(), checkpointed.parameters(), strict=True
    ):
        assert torch.allclose(
            plain_parameter.grad,
            checkpointed_parameter.grad,
            atol=1e-5,
            rtol=1e-5,
        )
