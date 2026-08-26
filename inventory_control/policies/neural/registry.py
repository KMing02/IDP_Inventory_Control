"""Shared neural policy metadata for paper lost-sales benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import nn

from inventory_control.policies.neural.fcnn import (
    make_paper_direct_target_stock_fcnn_model,
    make_paper_fcnn_model,
)
from inventory_control.policies.neural.monokan import (
    make_paper_direct_target_stock_kan_model,
    make_paper_direct_target_stock_monokan_model,
    make_paper_deep_grid7_direct_target_stock_kan_model,
    make_paper_deep_grid7_direct_target_stock_monokan_model,
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
    NeuralPolicySpec(
        "PaperFCNNDirectTargetStock",
        make_paper_direct_target_stock_fcnn_model,
        "paper_fcnn_direct_target_stock.pt",
        "fcnn",
        1,
        401,
    ),
    NeuralPolicySpec(
        "KANDirectTargetStock",
        make_paper_direct_target_stock_kan_model,
        "kan_direct_target_stock.pt",
        "kan",
        2,
        402,
    ),
    NeuralPolicySpec(
        "MonoKANDirectTargetStock",
        make_paper_direct_target_stock_monokan_model,
        "monokan_direct_target_stock.pt",
        "monokan",
        3,
        402,
    ),
    NeuralPolicySpec(
        "KANDirectTargetStockDeepGrid7",
        make_paper_deep_grid7_direct_target_stock_kan_model,
        "kan_direct_target_stock_deep_grid7.pt",
        "kan",
        4,
        402,
    ),
    NeuralPolicySpec(
        "MonoKANDirectTargetStockDeepGrid7",
        make_paper_deep_grid7_direct_target_stock_monokan_model,
        "monokan_direct_target_stock_deep_grid7.pt",
        "monokan",
        5,
        402,
    ),
)

POLICY_SPECS_BY_NAME = {spec.name: spec for spec in NEURAL_POLICY_SPECS}
TRAINABLE_POLICY_NAMES = frozenset(POLICY_SPECS_BY_NAME)
DIRECT_TARGET_STOCK_POLICY_NAMES = (
    "PaperFCNNDirectTargetStock",
    "KANDirectTargetStock",
    "MonoKANDirectTargetStock",
)
CAPACITY_DIRECT_TARGET_STOCK_POLICY_NAMES = (
    "KANDirectTargetStockDeepGrid7",
    "MonoKANDirectTargetStockDeepGrid7",
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
