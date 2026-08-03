"""Minimal residual order-up-to policy example.

This example shows the architecture-specific code a teammate needs to write.
It reuses the shared product generation, historical base-stock anchor,
DirectBackprop training loop, and order-up-to decoder.

To make this policy available in benchmark CLIs, add a `NeuralPolicySpec` entry
in `inventory_control/policies/neural/registry.py` and a matching branch in
`inventory_control/idp_benchmark.py` until the benchmark registry is fully
declarative.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.policies.neural.fcnn import compute_paper_fcnn_feature_stats, infer_model_device
from inventory_control.products import PaperLostSalesProductGenerator
from inventory_control.training import train_paper_lost_sales_policy
from inventory_control.training.direct_backprop import (
    make_paper_lost_sales_training_batch,
    move_batch_to_device,
    paper_lost_sales_policy_order,
)


class CustomResidualPolicy(nn.Module):
    """Tiny context-only residual network for target-stock corrections."""

    paper_action_parameterization = "residual_order_up_to"
    paper_residual_center = 0.0
    paper_residual_scale = 0.05
    paper_residual_anchor = "gamma"

    def __init__(self, context_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(context_dim, 8),
            nn.ELU(),
            nn.Linear(8, 1),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return a bounded residual factor from context features only."""
        return self.network(features).squeeze(-1)


def main() -> None:
    config = load_paper_lost_sales_config("configs/paper_lost_sales_debug.yaml")
    generator = PaperLostSalesProductGenerator(config)
    products = generator.generate(8, split="train")
    feature_stats = compute_paper_fcnn_feature_stats(products)
    context_dim = int(config["history_length_H"]) + 4
    model = CustomResidualPolicy(context_dim)

    result = train_paper_lost_sales_policy(
        products,
        model,
        horizon=3,
        batch_size=4,
        epochs=1,
        learning_rate=1e-3,
        seed=0,
        imitation_epochs=1,
        feature_stats=feature_stats,
        max_order_factor=float(config["fcnn"]["max_order_factor"]),
        scale_action_by_history_mean=True,
        residual_anchor=model.paper_residual_anchor,
        include_terminal_salvage=False,
    )

    batch = make_paper_lost_sales_training_batch(
        products[:2],
        horizon=1,
        generator=torch.Generator().manual_seed(0),
        feature_stats=feature_stats,
        residual_anchor=model.paper_residual_anchor,
    )
    batch = move_batch_to_device(batch, infer_model_device(result.model))
    inventory = batch["initial_inventory"]
    order = paper_lost_sales_policy_order(
        result.model,
        batch,
        inventory,
        max_order_factor=float(config["fcnn"]["max_order_factor"]),
        scale_action_by_history_mean=True,
    )
    print(f"trained_epochs={len(result.training_losses)}")
    print(f"example_order_mean={float(order.mean().item()):.6f}")
    print("custom_residual_policy_example=passed")


if __name__ == "__main__":
    main()
