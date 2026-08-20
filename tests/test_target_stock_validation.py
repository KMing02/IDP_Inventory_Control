from __future__ import annotations

import pandas as pd
import pytest

from inventory_control.idp_benchmark import _profile_settings
from inventory_control.paper_configs import load_paper_lost_sales_config
from scripts.run_target_stock_validation import (
    paired_statistics,
    resolve_target_stock_config,
)


def test_target_stock_followup_preserves_reward_only_direct_q_budget() -> None:
    base = load_paper_lost_sales_config(
        "configs/paper_lost_sales_medium_seed0_converged.yaml"
    )
    config, resolved = resolve_target_stock_config(base, "full", "standard", 7)
    settings = _profile_settings(config, "full")

    assert settings["seed"] == 7
    assert settings["training_products"] == 128
    assert settings["validation_products"] == 32
    assert settings["training_horizon"] == 50
    assert settings["evaluation_horizon"] == 120
    assert settings["burn_in"] == 20
    assert settings["imitation_epochs"] == 0
    assert settings["reward_epochs"] == 15
    assert settings["max_order_factor"] == 4.0
    assert config["learning_rate"] == 0.001
    assert resolved["action_parameterization"] == "target_stock"
    assert resolved["inventory_network_input"] is False
    assert resolved["initialization_mode"] == "standard"


def test_target_stock_followup_changes_only_initialization_mode() -> None:
    base = load_paper_lost_sales_config(
        "configs/paper_lost_sales_medium_seed0_converged.yaml"
    )
    _, standard = resolve_target_stock_config(base, "full", "standard", 0)
    _, informed = resolve_target_stock_config(
        base, "full", "target_stock_informed", 0
    )
    ignored = {"policy_name", "initialization_mode"}

    assert {
        key: value for key, value in standard.items() if key not in ignored
    } == {key: value for key, value in informed.items() if key not in ignored}


def test_paired_statistics_uses_left_minus_right_and_all_ten_seeds() -> None:
    rows = []
    for seed in range(10):
        rows.extend(
            [
                {
                    "policy_name": "left",
                    "training_seed": seed,
                    "mean_reward_per_period": float(seed + 2),
                },
                {
                    "policy_name": "right",
                    "training_seed": seed,
                    "mean_reward_per_period": float(seed + 1),
                },
            ]
        )

    summary, detail = paired_statistics(
        pd.DataFrame(rows), "left", "right", "left_minus_right"
    )

    assert summary["paired_mean_difference"] == pytest.approx(1.0)
    assert summary["left_wins"] == 10
    assert summary["right_wins"] == 0
    assert summary["ties"] == 0
    assert (detail["paired_difference_left_minus_right"] == 1.0).all()
