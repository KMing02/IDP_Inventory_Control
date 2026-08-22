"""Shared neural policy metadata for paper lost-sales benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import nn

from inventory_control.policies.neural.constrained_mlp import (
    make_constrained_monotone_mlp_model,
    make_paper_residual_order_up_to_constrained_mlp_model,
    make_positive_weight_mlp_model,
)
from inventory_control.policies.neural.fcnn import (
    make_paper_base_stock_output_fcnn_model,
    make_paper_direct_target_stock_fcnn_model,
    make_paper_fcnn_model,
    make_paper_order_up_to_fcnn_model,
    make_paper_residual_order_up_to_fcnn_model,
)
from inventory_control.policies.neural.monokan import (
    make_paper_base_stock_output_kan_model,
    make_paper_base_stock_output_monokan_model,
    make_paper_direct_target_stock_kan_model,
    make_paper_direct_target_stock_monokan_model,
    make_paper_deep_grid7_direct_target_stock_kan_model,
    make_paper_deep_grid7_direct_target_stock_monokan_model,
    make_paper_selective_direct_target_stock_kan_model,
    make_paper_selective_direct_target_stock_monokan_model,
    make_paper_monokan_model,
    make_paper_order_up_to_monokan_model,
    make_paper_residual_order_up_to_kan_model,
    make_paper_residual_order_up_to_monokan_model,
)


PolicyFactory = Callable[[dict], nn.Module]


@dataclass(frozen=True)
class NeuralPolicySpec:
    """Canonical metadata for one trainable neural benchmark policy."""

    name: str
    factory: PolicyFactory
    checkpoint_filename: str | None
    category: str
    order: int
    seed_offset: int = 0


NEURAL_POLICY_SPECS: tuple[NeuralPolicySpec, ...] = (
    NeuralPolicySpec("PaperFCNN", make_paper_fcnn_model, "paper_fcnn.pt", "fcnn", 0, 0),
    NeuralPolicySpec("PaperFCNNOrderUpTo", make_paper_order_up_to_fcnn_model, "paper_fcnn_order_up_to.pt", "fcnn", 1, 201),
    NeuralPolicySpec(
        "PaperFCNNBaseStockOrderUpTo",
        make_paper_base_stock_output_fcnn_model,
        "paper_fcnn_base_stock_order_up_to.pt",
        "fcnn",
        2,
        211,
    ),
    NeuralPolicySpec(
        "PaperFCNNResidualOrderUpTo",
        make_paper_residual_order_up_to_fcnn_model,
        "paper_fcnn_residual_order_up_to.pt",
        "fcnn",
        3,
        202,
    ),
    NeuralPolicySpec(
        "ConstrainedMLPResidualOrderUpTo",
        make_paper_residual_order_up_to_constrained_mlp_model,
        "constrained_mlp_residual_order_up_to.pt",
        "monotone_mlp",
        4,
        203,
    ),
    NeuralPolicySpec("PositiveWeightMLP", make_positive_weight_mlp_model, "positive_weight_mlp.pt", "monotone_mlp", 5, 101),
    NeuralPolicySpec(
        "ConstrainedMonoMLP",
        make_constrained_monotone_mlp_model,
        "constrained_monotone_mlp.pt",
        "monotone_mlp",
        6,
        102,
    ),
    NeuralPolicySpec("MonoKAN", make_paper_monokan_model, "monokan.pt", "monokan", 7, 0),
    NeuralPolicySpec("MonoKANOrderUpTo", make_paper_order_up_to_monokan_model, "monokan_order_up_to.pt", "monokan", 8, 301),
    NeuralPolicySpec("KANBaseStockOrderUpTo", make_paper_base_stock_output_kan_model, "kan_base_stock_order_up_to.pt", "kan", 9, 311),
    NeuralPolicySpec(
        "MonoKANBaseStockOrderUpTo",
        make_paper_base_stock_output_monokan_model,
        "monokan_base_stock_order_up_to.pt",
        "monokan",
        10,
        312,
    ),
    NeuralPolicySpec(
        "KANResidualOrderUpTo",
        make_paper_residual_order_up_to_kan_model,
        "kan_residual_order_up_to.pt",
        "kan",
        11,
        303,
    ),
    NeuralPolicySpec(
        "MonoKANResidualOrderUpTo",
        make_paper_residual_order_up_to_monokan_model,
        "monokan_residual_order_up_to.pt",
        "monokan",
        12,
        302,
    ),
    NeuralPolicySpec(
        "PaperFCNNDirectTargetStock",
        make_paper_direct_target_stock_fcnn_model,
        "paper_fcnn_direct_target_stock.pt",
        "fcnn",
        13,
        401,
    ),
    NeuralPolicySpec(
        "KANDirectTargetStock",
        make_paper_direct_target_stock_kan_model,
        "kan_direct_target_stock.pt",
        "kan",
        14,
        402,
    ),
    NeuralPolicySpec(
        "MonoKANDirectTargetStock",
        make_paper_direct_target_stock_monokan_model,
        "monokan_direct_target_stock.pt",
        "monokan",
        15,
        402,
    ),
    NeuralPolicySpec(
        "KANSelectiveDirectTargetStock",
        make_paper_selective_direct_target_stock_kan_model,
        "kan_selective_direct_target_stock.pt",
        "kan",
        16,
        402,
    ),
    NeuralPolicySpec(
        "MonoKANSelectiveDirectTargetStock",
        make_paper_selective_direct_target_stock_monokan_model,
        "monokan_selective_direct_target_stock.pt",
        "monokan",
        17,
        402,
    ),
    NeuralPolicySpec(
        "KANDirectTargetStockDeepGrid7",
        make_paper_deep_grid7_direct_target_stock_kan_model,
        "kan_direct_target_stock_deep_grid7.pt",
        "kan",
        18,
        402,
    ),
    NeuralPolicySpec(
        "MonoKANDirectTargetStockDeepGrid7",
        make_paper_deep_grid7_direct_target_stock_monokan_model,
        "monokan_direct_target_stock_deep_grid7.pt",
        "monokan",
        19,
        402,
    ),
    NeuralPolicySpec("PenaltyMLP", make_paper_fcnn_model, "penalty_mlp.pt", "penalty", 20, 0),
)

POLICY_SPECS_BY_NAME = {spec.name: spec for spec in NEURAL_POLICY_SPECS}
TRAINABLE_POLICY_NAMES = frozenset(POLICY_SPECS_BY_NAME)
RESIDUAL_COMPARISON_POLICY_NAMES = (
    "PaperFCNNResidualOrderUpTo",
    "KANResidualOrderUpTo",
    "ConstrainedMLPResidualOrderUpTo",
    "MonoKANResidualOrderUpTo",
)
BASE_STOCK_OUTPUT_POLICY_NAMES = (
    "PaperFCNNBaseStockOrderUpTo",
    "KANBaseStockOrderUpTo",
    "MonoKANBaseStockOrderUpTo",
)
DIRECT_TARGET_STOCK_POLICY_NAMES = (
    "PaperFCNNDirectTargetStock",
    "KANDirectTargetStock",
    "MonoKANDirectTargetStock",
)
SELECTIVE_DIRECT_TARGET_STOCK_POLICY_NAMES = (
    "KANSelectiveDirectTargetStock",
    "MonoKANSelectiveDirectTargetStock",
)
CAPACITY_DIRECT_TARGET_STOCK_POLICY_NAMES = (
    "KANDirectTargetStockDeepGrid7",
    "MonoKANDirectTargetStockDeepGrid7",
)
DEFAULT_STRUCTURE_POLICY_NAMES = frozenset(
    {
        "PaperFCNN",
        "PenaltyMLP",
        "PositiveWeightMLP",
        "ConstrainedMonoMLP",
        "MonoKAN",
    }
)


def checkpoint_filenames() -> dict[str, str]:
    """Return checkpoint filenames for policies with persisted torch state."""
    return {
        spec.name: spec.checkpoint_filename
        for spec in NEURAL_POLICY_SPECS
        if spec.checkpoint_filename is not None
    }


def policy_factories(names: list[str] | tuple[str, ...]) -> dict[str, PolicyFactory]:
    """Return factories for requested policy names."""
    return {name: POLICY_SPECS_BY_NAME[name].factory for name in names}
