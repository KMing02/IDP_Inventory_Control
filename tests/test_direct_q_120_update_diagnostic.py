from __future__ import annotations

from pathlib import Path

import torch

from inventory_control.paper_configs import load_paper_lost_sales_config
from scripts.run_direct_q_120_update_diagnostic import (
    COMPARISON_UPDATES,
    EVALUATION_UPDATES,
    optimizer_checkpoint_hook,
    resolve_direct_q_120_config,
)


def test_direct_q_120_configuration_is_reward_only_and_fixed() -> None:
    config = load_paper_lost_sales_config(
        Path("configs/paper_lost_sales_medium_seed0_converged.yaml")
    )
    _, resolved = resolve_direct_q_120_config(config, "full", 2)

    assert resolved["training_seed"] == 2
    assert resolved["hidden_layers"] == 2
    assert resolved["hidden_units"] == 32
    assert resolved["learning_rate"] == 0.001
    assert resolved["reward_optimizer_updates"] == 120
    assert resolved["imitation_optimizer_updates"] == 0
    assert resolved["checkpoint_selection"] is None
    assert resolved["early_stopping"] is False
    assert tuple(resolved["checkpoint_updates"]) == EVALUATION_UPDATES
    assert COMPARISON_UPDATES == (15, 30, 60, 90, 120)


def test_checkpoint_hook_observes_updates_without_changing_adam_step() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    observed: list[int] = []
    with optimizer_checkpoint_hook(observed.append) as state:
        for _ in range(5):
            optimizer.zero_grad()
            parameter.square().sum().backward()
            optimizer.step()

    assert state["updates"] == 5
    assert observed == [1, 5]
    assert float(parameter.item()) < 1.0
