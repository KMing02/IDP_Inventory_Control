import numpy as np
import pytest

from scripts.run_policy_behavior_extrapolation_analysis import (
    policy_active_slope_error,
    reconstructed_observed_upper_factor,
    region_diagnostics,
)


def test_observed_upper_factor_uses_fixed_995_quantile() -> None:
    states = np.arange(1.0, 1001.0)
    assert reconstructed_observed_upper_factor(states) == pytest.approx(
        np.quantile(states, 0.995)
    )


def test_ideal_order_up_to_has_no_violations_and_exact_active_slope() -> None:
    inventory = np.linspace(0.0, 80.0, 801)
    level = 35.0
    raw = np.maximum(level - inventory, 0.0)
    deployed = np.minimum(raw, np.maximum(40.0 - inventory, 0.0))
    metrics = region_diagnostics(inventory, raw, deployed, level)

    assert metrics["raw_violation_rate"] == 0.0
    assert metrics["deployed_violation_rate"] == 0.0
    assert metrics["raw_maximum_positive_violation"] == 0.0
    assert metrics["raw_active_slope_deviation_sum"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["order_up_to_absolute_error_sum"] == pytest.approx(0.0, abs=1e-10)


def test_increasing_raw_action_is_detected_separately_from_deployed_action() -> None:
    inventory = np.arange(5.0)
    raw = np.array([2.0, 1.0, 1.5, 1.0, 0.0])
    deployed = np.minimum(raw, np.maximum(2.0 - inventory, 0.0))
    metrics = region_diagnostics(inventory, raw, deployed, reference_level=2.0)

    assert metrics["raw_violating_interval_count"] == 1
    assert metrics["raw_maximum_positive_violation"] == pytest.approx(0.5)
    assert metrics["deployed_violation_rate"] == 0.0


def test_policy_active_slope_error_distinguishes_order_up_to_from_flat_action() -> None:
    inventory = np.arange(6.0)
    exact_order_up_to = np.maximum(4.0 - inventory, 0.0)
    flat_active_action = np.full_like(inventory, 4.0)

    assert policy_active_slope_error(inventory, exact_order_up_to) == pytest.approx(0.0)
    assert policy_active_slope_error(inventory, flat_active_action) == pytest.approx(1.0)
