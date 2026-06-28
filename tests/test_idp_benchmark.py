from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from inventory_control.idp_benchmark import (
    _oracle_levels,
    _prediction_summary,
    _select_penalty_candidate,
    run_idp_benchmark,
)
from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.products import PaperLostSalesProductGenerator


def test_prediction_summary_reports_accuracy_and_monotonicity() -> None:
    import pandas as pd

    predictions = pd.DataFrame(
        {
            "product_id": ["p"] * 3,
            "policy_name": ["model"] * 3,
            "state_index": [0, 1, 2],
            "inventory": [0.0, 1.0, 2.0],
            "raw_order": [2.0, 3.0, 1.0],
            "deployed_order": [2.0, 2.5, 1.0],
            "error": [0.0, 1.0, -1.0],
        }
    )

    summary = _prediction_summary(predictions).iloc[0]

    assert summary["mae"] == pytest.approx(2 / 3)
    assert summary["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert summary["raw_violation_rate"] == 0.5
    assert summary["raw_max_violation"] == 1.0
    assert summary["deployed_max_violation"] == 0.5


def test_penalty_selection_prefers_compliant_high_reward_candidate() -> None:
    candidates = [
        {"raw_violation_rate": 0.02, "validation_reward": 20.0},
        {"raw_violation_rate": 0.0, "validation_reward": 10.0},
        {"raw_violation_rate": 0.005, "validation_reward": 15.0},
    ]

    selected = _select_penalty_candidate(candidates, threshold=0.01)

    assert selected["validation_reward"] == 15.0


def test_oracle_levels_are_deterministic() -> None:
    config = load_paper_lost_sales_config("configs/paper_lost_sales_debug.yaml")
    products = PaperLostSalesProductGenerator(config).generate(2, split="validation")

    first = _oracle_levels(products, oracle_mc_samples=100, seed=99)
    second = _oracle_levels(products, oracle_mc_samples=100, seed=99)

    assert first == second


def test_idp_test_split_requires_explicit_confirmation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Refusing test-set evaluation"):
        run_idp_benchmark(
            config_path="configs/paper_lost_sales_debug.yaml",
            profile="compact",
            mode="structure-ls",
            split="test",
            output_dir=tmp_path,
        )


def test_compact_idp_benchmark_smoke(tmp_path: Path) -> None:
    artifacts = run_idp_benchmark(
        config_path="configs/paper_lost_sales_debug.yaml",
        profile="compact",
        mode="structure-ls",
        output_dir=tmp_path,
    )

    expected_policies = {
        "HistoricalDemandBaseStock",
        "OmniscientOptimal",
        "NonOmniscientOptimal",
        "PaperFCNN",
        "PenaltyMLP",
        "PositiveWeightMLP",
        "ConstrainedMonoMLP",
        "MonoKAN",
    }
    assert set(artifacts.summary["policy_name"]) == expected_policies
    assert {"order_up_to_mae", "raw_violation_rate", "parameter_count", "training_seconds"}.issubset(
        artifacts.summary.columns
    )
    assert artifacts.predictions.groupby(["product_id", "policy_name"]).size().nunique() == 1
    assert (
        artifacts.predictions["deployed_order"]
        <= 4.0 * artifacts.predictions["historical_mean_demand"] - artifacts.predictions["inventory"] + 1e-8
    ).all()
    assert artifacts.summary.loc[
        artifacts.summary["policy_name"] == "MonoKAN", "raw_violation_rate"
    ].item() == 0.0
    for policy_name in ["PositiveWeightMLP", "ConstrainedMonoMLP"]:
        assert artifacts.summary.loc[
            artifacts.summary["policy_name"] == policy_name, "raw_violation_rate"
        ].item() == 0.0
        assert artifacts.summary.loc[
            artifacts.summary["policy_name"] == policy_name, "deployed_violation_rate"
        ].item() == 0.0
    for filename in [
        "summary.csv",
        "paper_table3_replication.csv",
        "operational_metrics.csv",
        "predictions.csv",
        "training_log.csv",
        "monokan_splines.csv",
        "metadata.json",
    ]:
        assert (tmp_path / filename).exists()
    assert (tmp_path / "plots" / "figure2_policy_curves.png").exists()


def test_replicate_ls_mode_smoke(tmp_path: Path) -> None:
    artifacts = run_idp_benchmark(
        config_path="configs/paper_lost_sales_debug.yaml",
        profile="compact",
        mode="replicate-ls",
        output_dir=tmp_path,
    )

    assert {"PaperFCNN", "OmniscientOptimal", "NonOmniscientOptimal"}.issubset(
        set(artifacts.summary["policy_name"])
    )
    assert (tmp_path / "paper_table3_replication.csv").exists()


def test_sample_efficiency_mode_smoke(tmp_path: Path) -> None:
    artifacts = run_idp_benchmark(
        config_path="configs/paper_lost_sales_debug.yaml",
        profile="compact",
        mode="sample-efficiency",
        output_dir=tmp_path,
    )

    assert artifacts.sample_efficiency is not None
    assert {"history_window", "training_products", "policy_name", "mean_total_reward"}.issubset(
        artifacts.sample_efficiency.columns
    )
    assert (tmp_path / "sample_efficiency.csv").exists()
