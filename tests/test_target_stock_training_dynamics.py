from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_target_stock_training_dynamics import (
    EVALUATION_UPDATES,
    GRADIENT_UPDATES,
    _target_summary,
)


def test_target_stock_diagnostic_uses_only_predefined_updates() -> None:
    assert EVALUATION_UPDATES == (0, 1, 5, 15, 30, 60, 90, 120)
    assert GRADIENT_UPDATES == (1, 2, 5, 15, 30, 60, 120)


def test_target_summary_reports_context_variation_and_initial_distance() -> None:
    targets = pd.DataFrame({"normalized_target_stock": [1.0, 2.0, 3.0]})
    historical = np.asarray([1.5, 2.0, 2.5])
    initial = np.asarray([1.0, 1.0, 1.0])

    result = _target_summary(targets, historical, initial)

    assert result["normalized_target_stock_mean"] == pytest.approx(2.0)
    assert result["normalized_target_stock_across_product_sd"] == pytest.approx(1.0)
    assert result["historical_gamma_multiple_rmse"] == pytest.approx(
        np.sqrt((0.25 + 0.0 + 0.25) / 3.0)
    )
    assert result["historical_gamma_multiple_pearson"] == pytest.approx(1.0)
    assert result["distance_from_initial_target_rmse"] == pytest.approx(
        np.sqrt(5.0 / 3.0)
    )


def test_target_summary_marks_constant_prediction_correlation_undefined() -> None:
    targets = pd.DataFrame({"normalized_target_stock": [2.0, 2.0, 2.0]})
    result = _target_summary(
        targets,
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([2.0, 2.0, 2.0]),
    )

    assert np.isnan(result["historical_gamma_multiple_pearson"])
    assert result["distance_from_initial_target_rmse"] == 0.0
