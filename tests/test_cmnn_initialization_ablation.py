import numpy as np
import pytest
import torch

from inventory_control.idp_benchmark import _model_config, _profile_settings
from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.paper_lost_sales import historical_demand_base_stock_policy
from inventory_control.policies.neural.constrained_mlp import NonnegativeLinear
from inventory_control.products import PaperLostSalesProductGenerator
from scripts.run_cmnn_initialization_ablation import (
    EFFECTIVE_ZERO_WEIGHT,
    empirical_base_stock_normalized_order,
    initialize_base_stock_informed_cmnn,
)
from scripts.run_no_imitation_ablation import resolve_no_imitation_config
from scripts.tune_primary_neural_policies import _make_model


def test_base_stock_bias_is_training_only_least_squares_solution() -> None:
    base = load_paper_lost_sales_config(
        "configs/paper_lost_sales_medium_seed0_converged.yaml"
    )
    products = PaperLostSalesProductGenerator(base).generate(128, "train")
    means = np.asarray([max(np.mean(product.historical_demand), 1.0) for product in products])
    targets = np.asarray([historical_demand_base_stock_policy(product).S for product in products])
    expected = float(np.dot(means, targets) / np.dot(means, means))
    assert empirical_base_stock_normalized_order(products) == expected


def test_informed_initialization_preserves_architecture_and_numerically_zeros_readout() -> None:
    base = load_paper_lost_sales_config(
        "configs/paper_lost_sales_medium_seed0_converged.yaml"
    )
    products = PaperLostSalesProductGenerator(base).generate(128, "train")
    config, _ = resolve_no_imitation_config(base, "full", "ConstrainedMonoMLP", 3)
    settings = _profile_settings(config, "full")
    model = _make_model("ConstrainedMonoMLP", _model_config(config, settings), 3)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    metadata = initialize_base_stock_informed_cmnn(model, products)
    final_layer = model.network[-1]

    assert isinstance(final_layer, NonnegativeLinear)
    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count
    assert torch.allclose(
        final_layer.effective_weight(),
        torch.full_like(final_layer.effective_weight(), EFFECTIVE_ZERO_WEIGHT),
        rtol=1e-5,
        atol=1e-12,
    )
    assert final_layer.bias.item() == pytest.approx(metadata["output_bias"], abs=1e-6)
    random_features = torch.randn(9, model.input_dim)
    expected_output = torch.full(
        (9,), metadata["empirical_base_stock_normalized_order"], dtype=torch.float32
    )
    assert torch.allclose(model(random_features), expected_output, atol=2e-5, rtol=1e-5)
