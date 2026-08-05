from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from inventory_control.idp_benchmark import (
    _SharedActionCapPolicyAdapter,
    _operational_table,
    _oracle_levels,
    _prediction_table,
    _prediction_summary,
    _select_penalty_candidate,
    run_idp_benchmark,
)
from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.paper_lost_sales import BaseStockOrderPolicy, evaluate_paper_lost_sales_policy
from inventory_control.policies.neural.fcnn import compute_paper_fcnn_feature_stats
from inventory_control.products import PaperLostSalesProduct, PaperLostSalesProductGenerator


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
            "raw_oracle_order": [3.0, 2.0, 1.0],
            "deployed_oracle_order": [2.0, 0.0, 0.0],
            "oracle_order": [2.0, 0.0, 0.0],
            "raw_error": [-1.0, 1.0, 0.0],
            "deployed_error": [0.0, 2.5, 1.0],
            "error": [0.0, 2.5, 1.0],
            "raw_order_up_to_level": [2.0, 4.0, 3.0],
            "deployed_order_up_to_level": [2.0, 3.5, 3.0],
            "order_up_to_level": [2.0, 3.5, 3.0],
            "raw_omniscient_order_up_to_level": [3.0, 3.0, 3.0],
            "deployed_omniscient_order_up_to_level": [2.0, 1.0, 2.0],
            "omniscient_order_up_to_level": [2.0, 1.0, 2.0],
        }
    )

    summary = _prediction_summary(predictions).iloc[0]

    assert summary["raw_mae"] == pytest.approx(2 / 3)
    assert summary["raw_rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert summary["deployed_mae"] == pytest.approx(3.5 / 3)
    assert summary["deployed_rmse"] == pytest.approx(np.sqrt(7.25 / 3))
    assert summary["mae"] == summary["deployed_mae"]
    assert summary["rmse"] == summary["deployed_rmse"]
    assert summary["raw_violation_rate"] == 0.5
    assert summary["raw_max_violation"] == 1.0
    assert summary["deployed_max_violation"] == 0.5
    assert summary["raw_zero_inventory_order_up_to_mae"] == 1.0
    assert summary["raw_zero_inventory_order_up_to_rmse"] == 1.0
    assert summary["deployed_zero_inventory_order_up_to_mae"] == 0.0
    assert summary["deployed_zero_inventory_order_up_to_rmse"] == 0.0
    assert summary["raw_order_up_to_mae"] == 1.0
    assert summary["raw_order_up_to_mae"] == summary["raw_zero_inventory_order_up_to_mae"]
    assert summary["raw_order_up_to_rmse"] == summary["raw_zero_inventory_order_up_to_rmse"]
    assert summary["deployed_order_up_to_mae"] == 0.0
    assert (
        summary["deployed_order_up_to_mae"]
        == summary["deployed_zero_inventory_order_up_to_mae"]
    )
    assert (
        summary["deployed_order_up_to_rmse"]
        == summary["deployed_zero_inventory_order_up_to_rmse"]
    )
    assert summary["order_up_to_mae"] == summary["deployed_order_up_to_mae"]
    assert summary["order_up_to_rmse"] == summary["deployed_order_up_to_rmse"]
    assert summary["raw_zero_inventory_order_up_to_median_absolute_error"] == 1.0
    assert summary["raw_zero_inventory_order_up_to_p90_absolute_error"] == 1.0
    assert summary["raw_zero_inventory_order_up_to_max_absolute_error"] == 1.0
    assert summary["raw_zero_inventory_order_up_to_largest_product_contribution"] == 1.0
    assert summary["raw_zero_inventory_order_up_to_top3_product_contribution"] == 1.0
    assert summary["deployed_zero_inventory_order_up_to_median_absolute_error"] == 0.0
    assert summary["deployed_zero_inventory_order_up_to_p90_absolute_error"] == 0.0
    assert summary["deployed_zero_inventory_order_up_to_max_absolute_error"] == 0.0
    assert summary["deployed_zero_inventory_order_up_to_largest_product_contribution"] == 0.0
    assert summary["deployed_zero_inventory_order_up_to_top3_product_contribution"] == 0.0
    assert summary["active_slope_deviation_from_minus_one"] == pytest.approx(1.5)
    assert summary["inactive_slope_deviation_from_zero"] == pytest.approx(1.5)


def test_zero_inventory_order_up_to_diagnostics_are_robust_and_ignore_other_states() -> None:
    import pandas as pd

    raw_zero_errors = np.asarray([1.0, -2.0, 3.0, -10.0])
    deployed_zero_errors = np.asarray([1.0, -2.0, 3.0, -4.0])
    rows = []
    for product_index, (raw_error, deployed_error) in enumerate(
        zip(raw_zero_errors, deployed_zero_errors)
    ):
        for state_index in (0, 1):
            raw_level_error = raw_error if state_index == 0 else 10_000.0
            deployed_level_error = deployed_error if state_index == 0 else -10_000.0
            rows.append(
                {
                    "product_id": f"p{product_index}",
                    "policy_name": "model",
                    "state_index": state_index,
                    "inventory": float(state_index),
                    "raw_order": 5.0 - state_index,
                    "deployed_order": 5.0 - state_index,
                    "raw_oracle_order": 5.0 - state_index,
                    "deployed_oracle_order": 5.0 - state_index,
                    "raw_error": 0.0,
                    "deployed_error": 0.0,
                    "raw_order_up_to_level": 20.0 + raw_level_error,
                    "deployed_order_up_to_level": 20.0 + deployed_level_error,
                    "raw_omniscient_order_up_to_level": 20.0,
                    "deployed_omniscient_order_up_to_level": 20.0,
                }
            )

    summary = _prediction_summary(pd.DataFrame(rows)).iloc[0]
    raw_absolute_errors = np.abs(raw_zero_errors)
    deployed_absolute_errors = np.abs(deployed_zero_errors)

    assert summary["raw_zero_inventory_order_up_to_mae"] == pytest.approx(
        raw_absolute_errors.mean()
    )
    assert summary["raw_zero_inventory_order_up_to_rmse"] == pytest.approx(
        np.sqrt(np.mean(raw_absolute_errors**2))
    )
    assert summary["raw_zero_inventory_order_up_to_median_absolute_error"] == pytest.approx(
        np.median(raw_absolute_errors)
    )
    assert summary["raw_zero_inventory_order_up_to_p90_absolute_error"] == pytest.approx(
        np.quantile(raw_absolute_errors, 0.9)
    )
    assert summary["raw_zero_inventory_order_up_to_max_absolute_error"] == 10.0
    assert summary["raw_zero_inventory_order_up_to_largest_product_contribution"] == pytest.approx(
        10.0 / 16.0
    )
    assert summary["raw_zero_inventory_order_up_to_top3_product_contribution"] == pytest.approx(
        15.0 / 16.0
    )
    assert summary["deployed_zero_inventory_order_up_to_mae"] == pytest.approx(
        deployed_absolute_errors.mean()
    )
    assert summary["deployed_zero_inventory_order_up_to_rmse"] == pytest.approx(
        np.sqrt(np.mean(deployed_absolute_errors**2))
    )
    assert summary["deployed_zero_inventory_order_up_to_median_absolute_error"] == pytest.approx(
        np.median(deployed_absolute_errors)
    )
    assert summary["deployed_zero_inventory_order_up_to_p90_absolute_error"] == pytest.approx(
        np.quantile(deployed_absolute_errors, 0.9)
    )
    assert summary["deployed_zero_inventory_order_up_to_max_absolute_error"] == 4.0
    assert summary[
        "deployed_zero_inventory_order_up_to_largest_product_contribution"
    ] == pytest.approx(4.0 / 10.0)
    assert summary["deployed_zero_inventory_order_up_to_top3_product_contribution"] == pytest.approx(
        9.0 / 10.0
    )
    assert summary["raw_order_up_to_mae"] == summary["raw_zero_inventory_order_up_to_mae"]
    assert (
        summary["deployed_order_up_to_mae"]
        == summary["deployed_zero_inventory_order_up_to_mae"]
    )


def _fixed_product() -> PaperLostSalesProduct:
    return PaperLostSalesProduct(
        product_id="fixed",
        price=2.0,
        purchase_cost=1.0,
        holding_cost=1.0,
        lost_sale_penalty=0.0,
        true_mean_demand=10.0,
        true_coefficient_of_variation=0.0,
        gamma_shape=1.0,
        gamma_scale=10.0,
        demand_distribution_type="gamma",
        historical_demand=[10] * 8,
        initial_inventory=0,
        future_demand=[0],
    )


def test_shared_cap_preserves_raw_base_stock_action_and_drives_reward() -> None:
    product = _fixed_product()
    raw_policy = BaseStockOrderPolicy(S=100.0, name="OmniscientOptimal")
    deployed_policy = _SharedActionCapPolicyAdapter(raw_policy, product, max_order_factor=2.0)

    assert raw_policy.act(inventory=0.0, t=0) == 100.0
    assert deployed_policy.act(inventory=0.0, t=0) == 20.0

    raw_metrics = evaluate_paper_lost_sales_policy(product, raw_policy, horizon=1, n_episodes=1, seed=0)
    deployed_metrics = evaluate_paper_lost_sales_policy(
        product,
        deployed_policy,
        horizon=1,
        n_episodes=1,
        seed=0,
    )
    assert raw_metrics["mean_order_quantity"] == 100.0
    assert deployed_metrics["mean_order_quantity"] == 20.0
    assert raw_metrics["mean_total_reward"] == -100.0
    assert deployed_metrics["mean_total_reward"] == -20.0

    feature_stats = compute_paper_fcnn_feature_stats([product])
    operational = _operational_table(
        [product],
        models={},
        feature_stats=feature_stats,
        settings={
            "max_order_factor": 2.0,
            "scale_action_by_history_mean": True,
            "evaluation_horizon": 1,
            "burn_in": 0,
        },
        oracle_levels={product.product_id: 100.0},
        split="validation",
    )
    omniscient = operational.loc[operational["policy_name"] == "OmniscientOptimal"].iloc[0]
    assert omniscient["mean_order_quantity"] == 20.0
    assert omniscient["mean_total_reward"] == -20.0


def test_omniscient_prediction_keeps_uncapped_raw_and_capped_deployed_actions() -> None:
    product = _fixed_product()
    feature_stats = compute_paper_fcnn_feature_stats([product])
    predictions = _prediction_table(
        [product],
        models={},
        feature_stats=feature_stats,
        settings={
            "max_order_factor": 2.0,
            "inventory_states": 3,
            "scale_action_by_history_mean": True,
        },
        oracle_levels={product.product_id: 30.0},
    )
    omniscient = predictions.loc[predictions["policy_name"] == "OmniscientOptimal"]

    assert omniscient["raw_order"].tolist() == [30.0, 20.0, 10.0]
    assert omniscient["deployed_order"].tolist() == [20.0, 10.0, 0.0]
    assert omniscient["raw_oracle_order"].tolist() == omniscient["raw_order"].tolist()
    assert omniscient["deployed_oracle_order"].tolist() == omniscient["deployed_order"].tolist()
    assert (omniscient[["raw_error", "deployed_error"]] == 0.0).all().all()


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
    assert {
        "raw_mae",
        "deployed_mae",
        "raw_zero_inventory_order_up_to_mae",
        "deployed_zero_inventory_order_up_to_mae",
        "deployed_zero_inventory_order_up_to_p90_absolute_error",
        "deployed_zero_inventory_order_up_to_top3_product_contribution",
        "order_up_to_mae",
        "raw_violation_rate",
        "active_slope_deviation_from_minus_one",
        "inactive_slope_deviation_from_zero",
        "parameter_count",
        "training_seconds",
    }.issubset(artifacts.summary.columns)
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
