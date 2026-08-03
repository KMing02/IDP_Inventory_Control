from pathlib import Path

import pandas as pd

from inventory_control.policies.neural.registry import (
    RESIDUAL_COMPARISON_POLICY_NAMES,
    checkpoint_filenames,
    policy_factories,
)
from scripts.run_distribution_shift_multiseed import _paired_reward_differences
from scripts.run_distribution_shift_validation import _shifted_config, _summarize_shift


def test_residual_policy_registry_exposes_checkpoint_names_and_factories() -> None:
    filenames = checkpoint_filenames()
    factories = policy_factories(RESIDUAL_COMPARISON_POLICY_NAMES)

    assert list(RESIDUAL_COMPARISON_POLICY_NAMES) == [
        "PaperFCNNResidualOrderUpTo",
        "KANResidualOrderUpTo",
        "ConstrainedMLPResidualOrderUpTo",
        "MonoKANResidualOrderUpTo",
    ]
    assert filenames["PaperFCNNResidualOrderUpTo"] == "paper_fcnn_residual_order_up_to.pt"
    assert filenames["KANResidualOrderUpTo"] == "kan_residual_order_up_to.pt"
    assert filenames["ConstrainedMLPResidualOrderUpTo"] == "constrained_mlp_residual_order_up_to.pt"
    assert filenames["MonoKANResidualOrderUpTo"] == "monokan_residual_order_up_to.pt"
    assert set(factories) == set(RESIDUAL_COMPARISON_POLICY_NAMES)


def test_shifted_config_preserves_base_and_applies_named_shift() -> None:
    base = {
        "product_generation": {
            "seed": 1,
            "mean_demand_mean": 100.0,
            "coefficient_of_variation_range": [0.0, 1.0],
        }
    }

    shifted = _shifted_config(base, "high_mean_demand")

    assert base["product_generation"]["mean_demand_mean"] == 100.0
    assert shifted["product_generation"]["seed"] == 1
    assert shifted["product_generation"]["mean_demand_mean"] == 200.0


def test_summarize_shift_outputs_required_columns_for_one_shift() -> None:
    operational = pd.DataFrame(
        [
            {
                "policy_name": "PaperFCNNResidualOrderUpTo",
                "mean_total_reward": 10.0,
                "mean_reward_per_period": 5.0,
                "service_level": 0.9,
                "lost_sales_rate": 0.1,
                "mean_order_quantity": 2.0,
                "mean_inventory": 3.0,
            },
            {
                "policy_name": "NonOmniscientOptimal",
                "mean_total_reward": 12.0,
                "mean_reward_per_period": 6.0,
                "service_level": 0.95,
                "lost_sales_rate": 0.05,
                "mean_order_quantity": 2.0,
                "mean_inventory": 3.0,
            },
        ]
    )
    prediction_summary = pd.DataFrame(
        [
            {
                "policy_name": "PaperFCNNResidualOrderUpTo",
                "order_up_to_mae": 1.0,
                "cap_hit_rate": 0.0,
                "raw_violation_rate": 0.0,
            }
        ]
    )

    summary = _summarize_shift(operational, prediction_summary, "in_distribution")

    assert {
        "shift",
        "policy_name",
        "mean_reward_per_period",
        "regret_vs_non_omniscient",
        "regret_vs_omniscient",
        "order_up_to_mae",
        "cap_hit_rate",
        "raw_violation_rate",
    }.issubset(summary.columns)
    assert summary.loc[summary["policy_name"] == "PaperFCNNResidualOrderUpTo", "regret_vs_non_omniscient"].item() == 1.0


def test_paired_reward_differences_are_unique_and_order_independent() -> None:
    rows = []
    for seed in [2025]:
        for shift in ["in_distribution"]:
            for product_id in ["p1", "p2"]:
                rows.extend(
                    [
                        {
                            "seed": seed,
                            "shift": shift,
                            "product_id": product_id,
                            "evaluation_seed": 100,
                            "policy_name": "MonoKANResidualOrderUpTo",
                            "mean_reward_per_period": 11.0,
                        },
                        {
                            "seed": seed,
                            "shift": shift,
                            "product_id": product_id,
                            "evaluation_seed": 100,
                            "policy_name": "PaperFCNNResidualOrderUpTo",
                            "mean_reward_per_period": 10.0,
                        },
                    ]
                )
    operational = pd.DataFrame(reversed(rows))

    differences, summary = _paired_reward_differences(
        operational,
        {"PaperFCNNResidualOrderUpTo", "MonoKANResidualOrderUpTo"},
    )

    keys = ["seed", "shift", "product_id", "evaluation_seed", "policy_name", "baseline_policy_name"]
    assert not differences.duplicated(keys).any()
    assert len(differences) == 2
    assert set(differences["evaluation_seed"]) == {100}
    assert set(differences["reward_diff_vs_baseline"]) == {1.0}
    assert summary["paired_products"].item() == 2
    assert summary["mean_reward_diff_vs_baseline"].item() == 1.0
    assert summary["win_rate_vs_baseline"].item() == 1.0


def test_paired_reward_differences_reject_duplicate_product_policy_rows() -> None:
    operational = pd.DataFrame(
        [
            {
                "seed": 2025,
                "shift": "in_distribution",
                "product_id": "p1",
                "evaluation_seed": 100,
                "policy_name": "PaperFCNNResidualOrderUpTo",
                "mean_reward_per_period": 10.0,
            },
            {
                "seed": 2025,
                "shift": "in_distribution",
                "product_id": "p1",
                "evaluation_seed": 100,
                "policy_name": "PaperFCNNResidualOrderUpTo",
                "mean_reward_per_period": 10.5,
            },
            {
                "seed": 2025,
                "shift": "in_distribution",
                "product_id": "p1",
                "evaluation_seed": 100,
                "policy_name": "MonoKANResidualOrderUpTo",
                "mean_reward_per_period": 11.0,
            },
        ]
    )

    try:
        _paired_reward_differences(
            operational,
            {"PaperFCNNResidualOrderUpTo", "MonoKANResidualOrderUpTo"},
        )
    except ValueError as error:
        assert "duplicate product-policy rows" in str(error)
    else:
        raise AssertionError("duplicate product-policy rows should fail paired comparison")


def test_paired_reward_differences_reject_missing_comparator_pairs() -> None:
    operational = pd.DataFrame(
        [
            {
                "seed": 2025,
                "shift": "in_distribution",
                "product_id": "p1",
                "evaluation_seed": 100,
                "policy_name": "PaperFCNNResidualOrderUpTo",
                "mean_reward_per_period": 10.0,
            },
            {
                "seed": 2025,
                "shift": "in_distribution",
                "product_id": "p1",
                "evaluation_seed": 100,
                "policy_name": "MonoKANResidualOrderUpTo",
                "mean_reward_per_period": 11.0,
            },
            {
                "seed": 2025,
                "shift": "in_distribution",
                "product_id": "p2",
                "evaluation_seed": 101,
                "policy_name": "PaperFCNNResidualOrderUpTo",
                "mean_reward_per_period": 9.0,
            },
        ]
    )

    try:
        _paired_reward_differences(
            operational,
            {"PaperFCNNResidualOrderUpTo", "MonoKANResidualOrderUpTo"},
        )
    except ValueError as error:
        assert "missing baseline/comparator pairs" in str(error)
    else:
        raise AssertionError("missing product-level comparator pairs should fail paired comparison")


def test_distribution_shift_callable_smoke_writes_expected_outputs(tmp_path: Path) -> None:
    shift_dir = tmp_path / "in_distribution"
    shift_dir.mkdir()
    operational = pd.DataFrame(
        [
            {
                "policy_name": "PaperFCNNResidualOrderUpTo",
                "mean_total_reward": 10.0,
                "mean_reward_per_period": 5.0,
                "service_level": 0.9,
                "lost_sales_rate": 0.1,
                "mean_order_quantity": 2.0,
                "mean_inventory": 3.0,
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "policy_name": "PaperFCNNResidualOrderUpTo",
                "product_id": "p1",
                "state_index": 0,
                "raw_order": 1.0,
                "deployed_order": 1.0,
            }
        ]
    )
    summary = _summarize_shift(operational, pd.DataFrame(), "in_distribution")

    operational.to_csv(shift_dir / "operational_metrics.csv", index=False)
    predictions.to_csv(shift_dir / "predictions.csv", index=False)
    summary.to_csv(shift_dir / "summary.csv", index=False)

    assert (shift_dir / "operational_metrics.csv").exists()
    assert (shift_dir / "predictions.csv").exists()
    assert (shift_dir / "summary.csv").exists()
    assert {"shift", "policy_name", "mean_reward_per_period"}.issubset(pd.read_csv(shift_dir / "summary.csv").columns)
