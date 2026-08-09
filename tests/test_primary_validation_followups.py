from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_primary_structural_ood_diagnostic import (
    aggregate_seed_region_summary,
    positive_violation_metrics,
    sample_ordered_pairs,
    stratified_inventory_grid,
)
from scripts.run_primary_validation_followups import (
    _build_plan,
    optimizer_update_counts,
    output_diagnostics,
)


def test_followup_fit_budgets_match_preregistered_design() -> None:
    equal = _build_plan("equal-capacity", [128], [0, 1, 2, 3, 4])
    efficiency = _build_plan("data-efficiency", [16, 32, 64, 128], [0, 1, 2])
    extension = _build_plan("matched-width-extension", [128], range(10))
    assert len(equal) == 20
    assert len(efficiency) == 48
    assert len(extension) == 20
    assert {row[4] for row in equal} == {0, 1, 2, 3, 4}
    assert {row[1] for row in efficiency} == {16, 32, 64, 128}
    assert {row[0] for row in extension} == {"PaperFCNN", "ConstrainedMonoMLP"}


def test_optimizer_updates_count_actual_training_log_rows() -> None:
    log = pd.DataFrame({"phase": [0, 0, 0, 1, 1]})
    assert optimizer_update_counts(log) == {
        "imitation_optimizer_updates": 3,
        "reward_optimizer_updates": 2,
        "total_optimizer_updates": 5,
    }


def test_extreme_action_definition_is_preregistered() -> None:
    predictions = pd.DataFrame(
        {
            "raw_order": [-2e-6, 80.0, 80.01, np.nan],
            "deployed_order": [0.0, 10.0, 10.0, 0.0],
            "historical_mean_demand": [10.0, 10.0, 10.0, 10.0],
        }
    )
    result = output_diagnostics(predictions)
    assert result["extreme_raw_action_count"] == 3
    assert result["extreme_raw_action_rate"] == pytest.approx(0.75)
    assert result["non_finite_output_count"] == 1


def test_stratified_grid_has_exact_size_and_required_boundaries() -> None:
    first = stratified_inventory_grid(10.0, 8.0, 4096, np.random.default_rng(3_000_003))
    second = stratified_inventory_grid(10.0, 8.0, 4096, np.random.default_rng(3_000_003))
    assert len(first) == 4096
    assert np.array_equal(first, second)
    assert first[0] == 0.0
    assert 40.0 in first
    assert first[-1] == 80.0


def test_ordered_pairs_are_deterministic_ordered_and_separated() -> None:
    first = sample_ordered_pairs(np.random.default_rng(7), 40.0, 80.0, 2048, 0.1)
    second = sample_ordered_pairs(np.random.default_rng(7), 40.0, 80.0, 2048, 0.1)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert np.all(first[0] < first[1])
    assert np.all(first[1] - first[0] >= 0.1)


def test_positive_violation_metrics_use_strict_positive_tail() -> None:
    result = positive_violation_metrics(
        np.array([3.0, 3.0, 3.0, 3.0]),
        np.array([2.0, 3.0 + 5e-7, 4.0, 5.0]),
    )
    assert result["violating_pair_count"] == 2
    assert result["raw_violation_rate"] == pytest.approx(0.5)
    assert result["maximum_positive_violation"] == 2.0
    assert result["p95_strictly_positive_violation"] == pytest.approx(1.95)


def test_structural_script_does_not_call_operational_reward_evaluation() -> None:
    source = Path("scripts/run_primary_structural_ood_diagnostic.py").read_text(encoding="utf-8")
    assert "_operational_table" not in source
    assert "evaluate_paper_lost_sales_policy" not in source


def test_structural_seed_aggregation_preserves_totals_and_true_maximum() -> None:
    seed_rows = pd.DataFrame(
        {
            "policy_name": ["P", "P"],
            "candidate_id": ["C", "C"],
            "training_seed": [0, 1],
            "region": ["full_range", "full_range"],
            "raw_violation_rate": [0.1, 0.2],
            "violating_pair_count": [1, 3],
            "ordered_pair_count": [10, 10],
            "maximum_positive_violation": [2.0, 5.0],
            "non_finite_pair_difference_count": [0, 0],
        }
    )
    result = aggregate_seed_region_summary(seed_rows).iloc[0]
    assert result["violating_pair_count"] == 4
    assert result["ordered_pair_count"] == 20
    assert result["raw_violation_rate"] == pytest.approx(0.2)
    assert result["maximum_positive_violation"] == 5.0
