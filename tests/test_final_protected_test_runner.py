from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_final_protected_test_evaluation import (
    FROZEN_TEST_OUTPUT,
    _assert_frozen_resolved_config,
    _paired_statistics,
    _reward_reconstruction,
    _safe_generate_products,
    _validate_evaluation_request,
)


def test_protected_test_request_fails_closed_without_authorization() -> None:
    with pytest.raises(ValueError, match="without --final-test"):
        _validate_evaluation_request("test", False, None, FROZEN_TEST_OUTPUT)


def test_protected_product_generator_is_not_called_without_authorization() -> None:
    class Generator:
        called = False

        def generate(self, count: int, split: str):
            self.called = True
            return [count, split]

    generator = Generator()
    with pytest.raises(RuntimeError, match="not explicitly authorized"):
        _safe_generate_products(generator, 32, "test", False)  # type: ignore[arg-type]
    assert generator.called is False


def test_frozen_configuration_guard_accepts_only_matched_specification() -> None:
    record = {
        "policy_name": "ConstrainedMonoMLP",
        "training_seed": 7,
        "hidden_layers": 2,
        "hidden_units": 32,
        "learning_rate": 0.001,
        "reward_epochs": 15,
        "imitation_early_stopping": True,
        "imitation_max_epochs": 100,
        "imitation_patience": 10,
        "imitation_min_delta": 0.5,
        "reset_optimizer_after_imitation": False,
        "deployment_action_cap_factor": 4.0,
        "evaluation_horizon": 120,
        "burn_in": 20,
        "inventory_states": 50,
        "imitation_semantics": "deployed_prediction_vs_capped_target",
        "imitation_checkpoint_metric": "deployed_imitation_rmse",
        "activation_weights": [1, 1, 1],
    }
    _assert_frozen_resolved_config(record, "ConstrainedMonoMLP", 7)
    record["reward_epochs"] = 30
    with pytest.raises(RuntimeError, match="reward_epochs"):
        _assert_frozen_resolved_config(record, "ConstrainedMonoMLP", 7)


def test_reward_reconstruction_includes_terminal_value_once() -> None:
    operational = pd.DataFrame(
        [
            {
                "policy_name": "PaperFCNN",
                "training_seed": 0,
                "product_id": "validation_000000",
                "mean_reward_per_period": 8.5,
                "mean_sales_revenue": 12.0,
                "mean_purchase_cost": 2.0,
                "mean_lost_sales_penalty": 1.0,
                "mean_holding_cost": 1.0,
                "mean_terminal_inventory_value": 50.0,
            }
        ]
    )
    result = _reward_reconstruction(operational, scored_periods=100)
    assert result.iloc[0]["reconstructed_reward_per_period"] == pytest.approx(8.5)
    assert result.iloc[0]["reconstruction_error"] == pytest.approx(0.0)


def test_paired_statistics_use_cmnn_minus_fcnn_differences() -> None:
    rows = []
    for seed, fcnn, cmnn in [(0, 1.0, 2.0), (1, 3.0, 2.0), (2, 4.0, 4.5)]:
        rows.extend(
            [
                {"policy_name": "PaperFCNN", "training_seed": seed, "mean_reward_per_period": fcnn},
                {
                    "policy_name": "ConstrainedMonoMLP",
                    "training_seed": seed,
                    "mean_reward_per_period": cmnn,
                },
            ]
        )
    result = _paired_statistics(pd.DataFrame(rows))
    assert np.asarray(list(result["paired_differences_cmnn_minus_paper_fcnn"].values())) == pytest.approx(
        [1.0, -1.0, 0.5]
    )
    assert result["cmnn_seed_wins"] == 2
    assert result["paper_fcnn_seed_wins"] == 1
    assert result["ties"] == 0
