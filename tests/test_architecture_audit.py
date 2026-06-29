import pandas as pd
import pytest

from inventory_control.evaluation.architecture_audit import (
    monokan_spline_diagnostics,
    paired_policy_monotonicity_rates,
)


def test_raw_and_deployed_diagnostic_detects_non_monotone_policy() -> None:
    raw_rate, deployed_rate = paired_policy_monotonicity_rates(
        raw_policy=lambda inventory: inventory,
        deployed_policy=lambda inventory: min(inventory, 10.0),
        inventory_low=[1.0, 2.0, 3.0],
        inventory_high=[2.0, 3.0, 4.0],
    )

    assert raw_rate == 1.0
    assert deployed_rate == 1.0


def test_monokan_spline_audit_flags_positive_segments() -> None:
    splines = pd.DataFrame(
        {
            "hidden_unit": [0, 0, 0, 1, 1, 1],
            "inventory_feature": [0.0, 1.0, 2.0] * 2,
            "edge_value": [2.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        }
    )

    audit = monokan_spline_diagnostics(splines).set_index("hidden_unit")

    assert audit.loc[0, "status"] == "PASS"
    assert audit.loc[0, "max_positive_finite_difference"] == 0.0
    assert audit.loc[1, "status"] == "FAIL"
    assert audit.loc[1, "fraction_violating_segments"] == pytest.approx(0.5)
