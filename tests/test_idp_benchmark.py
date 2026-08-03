from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from inventory_control.idp_benchmark import (
    _TorchPaperPolicyAdapter,
    _batched_torch_policy_metrics,
    _model_config,
    _oracle_levels,
    _prediction_summary,
    _select_penalty_candidate,
    _train_model,
    run_idp_benchmark,
)
from inventory_control.paper_evaluation import paper_evaluation_seed_start
from inventory_control.paper_lost_sales import evaluate_paper_lost_sales_policy
from inventory_control.policies.neural.fcnn import compute_paper_fcnn_feature_stats, make_paper_fcnn_model
from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.products import PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import DirectBackpropResult


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
            "oracle_order": [2.0, 0.0, 0.0],
            "order_up_to_level": [2.0, 3.5, 3.0],
            "omniscient_order_up_to_level": [3.0, 3.0, 3.0],
            "error": [0.0, 1.0, -1.0],
        }
    )

    summary = _prediction_summary(predictions).iloc[0]

    assert summary["mae"] == pytest.approx(2 / 3)
    assert summary["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert summary["raw_violation_rate"] == 0.5
    assert summary["raw_max_violation"] == 1.0
    assert summary["deployed_max_violation"] == 0.5
    assert summary["order_up_to_mae"] == 1.0
    assert summary["order_up_to_bias"] == -1.0
    assert summary["cap_hit_rate"] == pytest.approx(1 / 3)
    assert summary["active_slope_deviation_from_minus_one"] == pytest.approx(1.5)
    assert summary["inactive_slope_deviation_from_zero"] == pytest.approx(1.5)


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


def test_model_config_preserves_monokan_specific_settings() -> None:
    config = load_paper_lost_sales_config("configs/paper_lost_sales_debug.yaml")
    config["idp_benchmark"]["hidden_layers"] = 2
    config["idp_benchmark"]["hidden_units"] = 32
    config["idp_benchmark"]["max_order_factor"] = 4.0
    config["monokan"]["hidden_layers"] = 1
    config["monokan"]["hidden_units"] = 8
    config["monokan"]["max_order_factor"] = 2.0
    settings = {**config["idp_benchmark"], **config["idp_benchmark"]["compact"]}

    model_config = _model_config(config, settings)

    assert model_config["monokan"]["hidden_layers"] == 1
    assert model_config["monokan"]["hidden_units"] == 8
    assert model_config["monokan"]["max_order_factor"] == 2.0
    assert model_config["positive_weight_mlp"]["hidden_units"] == 32


def test_batched_torch_policy_metrics_match_scalar_adapter() -> None:
    config = load_paper_lost_sales_config("configs/paper_lost_sales_debug.yaml")
    settings = {**config["idp_benchmark"], **config["idp_benchmark"]["compact"]}
    settings["evaluation_horizon"] = 5
    settings["burn_in"] = 1
    products = PaperLostSalesProductGenerator(config).generate(2, split="validation")
    feature_stats = compute_paper_fcnn_feature_stats(products)
    model = make_paper_fcnn_model(config)

    batched = _batched_torch_policy_metrics(
        products,
        model,
        feature_stats,
        settings,
        split="validation",
    )
    seed_start = paper_evaluation_seed_start("validation")
    scalar_rewards = []
    for index, product in enumerate(products):
        policy = _TorchPaperPolicyAdapter(
            "PaperFCNN",
            model,
            product,
            feature_stats,
            float(settings["max_order_factor"]),
            bool(settings["scale_action_by_history_mean"]),
        )
        scalar_rewards.append(
            evaluate_paper_lost_sales_policy(
                product,
                policy,
                horizon=int(settings["evaluation_horizon"]),
                n_episodes=1,
                seed=seed_start + index,
                burn_in=int(settings["burn_in"]),
            )["mean_total_reward"]
        )

    assert batched["mean_total_reward"].tolist() == pytest.approx(scalar_rewards)


def test_benchmark_training_disables_terminal_salvage_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_paper_lost_sales_config("configs/paper_lost_sales_debug.yaml")
    settings = {**config["idp_benchmark"], **config["idp_benchmark"]["compact"]}
    settings["learning_rate"] = 5e-4
    products = PaperLostSalesProductGenerator(config).generate(2, split="train")
    feature_stats = compute_paper_fcnn_feature_stats(products)
    model = make_paper_fcnn_model(config)
    captured: dict[str, object] = {}

    def fake_train(*args: object, **kwargs: object) -> DirectBackpropResult:
        del args
        captured.update(kwargs)
        return DirectBackpropResult(model=model, training_losses=[], mean_total_rewards=[])

    monkeypatch.setattr("inventory_control.idp_benchmark.train_paper_lost_sales_policy", fake_train)

    _train_model(
        config,
        settings,
        products,
        products,
        feature_stats,
        model,
        "PaperFCNN",
        "validation",
    )

    assert captured["include_terminal_salvage"] is False
    assert captured["learning_rate"] == 5e-4


def test_idp_test_split_requires_explicit_confirmation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Refusing test-set evaluation"):
        run_idp_benchmark(
            config_path="configs/paper_lost_sales_debug.yaml",
            profile="compact",
            mode="structure-ls",
            split="test",
            output_dir=tmp_path,
        )


def test_idp_benchmark_rejects_unknown_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown trainable model"):
        run_idp_benchmark(
            config_path="configs/paper_lost_sales_debug.yaml",
            profile="compact",
            mode="structure-ls",
            output_dir=tmp_path,
            models=["NoSuchModel"],
        )


def test_idp_benchmark_model_filter_runs_fcnn_only(tmp_path: Path) -> None:
    artifacts = run_idp_benchmark(
        config_path="configs/paper_lost_sales_debug.yaml",
        profile="compact",
        mode="structure-ls",
        output_dir=tmp_path,
        models=["PaperFCNN"],
    )

    expected_policies = {
        "HistoricalDemandBaseStock",
        "OmniscientOptimal",
        "NonOmniscientOptimal",
        "PaperFCNN",
    }
    assert set(artifacts.summary["policy_name"]) == expected_policies
    assert set(artifacts.training_log["policy_name"]) == {"PaperFCNN"}
    assert (tmp_path / "models" / "paper_fcnn.pt").exists()
    assert not (tmp_path / "models" / "monokan.pt").exists()


def test_idp_benchmark_model_filter_runs_fcnn_order_up_to_only(tmp_path: Path) -> None:
    artifacts = run_idp_benchmark(
        config_path="configs/paper_lost_sales_debug.yaml",
        profile="compact",
        mode="structure-ls",
        output_dir=tmp_path,
        models=["PaperFCNNOrderUpTo"],
    )

    expected_policies = {
        "HistoricalDemandBaseStock",
        "OmniscientOptimal",
        "NonOmniscientOptimal",
        "PaperFCNNOrderUpTo",
    }
    assert set(artifacts.summary["policy_name"]) == expected_policies
    assert set(artifacts.training_log["policy_name"]) == {"PaperFCNNOrderUpTo"}
    assert (tmp_path / "models" / "paper_fcnn_order_up_to.pt").exists()
    assert not (tmp_path / "models" / "paper_fcnn.pt").exists()


def test_idp_benchmark_penalty_mlp_uses_target_stock_penalty(tmp_path: Path) -> None:
    artifacts = run_idp_benchmark(
        config_path="configs/paper_lost_sales_debug.yaml",
        profile="compact",
        mode="structure-ls",
        output_dir=tmp_path,
        models=["PaperFCNN", "PenaltyMLP"],
    )

    penalty_rows = artifacts.training_log[artifacts.training_log["policy_name"].str.startswith("PenaltyMLP")]
    assert not penalty_rows.empty
    assert penalty_rows["target_stock_penalty_weight"].max() > 0.0
    assert penalty_rows["monotonicity_penalty_weight"].max() == 0.0


def test_idp_benchmark_penalty_mlp_can_use_active_slope_penalty(tmp_path: Path) -> None:
    config = load_paper_lost_sales_config("configs/paper_lost_sales_debug.yaml")
    config["idp_benchmark"]["structural_penalty_kind"] = "active_slope"
    config_path = tmp_path / "active_slope.yaml"
    config_path.write_text(yaml.safe_dump(config))

    artifacts = run_idp_benchmark(
        config_path=config_path,
        profile="compact",
        mode="structure-ls",
        output_dir=tmp_path / "out",
        models=["PaperFCNN", "PenaltyMLP"],
    )

    penalty_rows = artifacts.training_log[artifacts.training_log["policy_name"].str.startswith("PenaltyMLP")]
    assert not penalty_rows.empty
    assert penalty_rows["active_slope_penalty_weight"].max() > 0.0
    assert penalty_rows["target_stock_penalty_weight"].max() == 0.0


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
        "target_structure_diagnostics.csv",
        "monokan_spline_certification.csv",
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


def test_residual_comparison_exports_structure_diagnostics(tmp_path: Path) -> None:
    artifacts = run_idp_benchmark(
        config_path="configs/paper_lost_sales_debug.yaml",
        profile="compact",
        mode="structure-ls",
        output_dir=tmp_path,
        models=[
            "PaperFCNNResidualOrderUpTo",
            "KANResidualOrderUpTo",
            "ConstrainedMLPResidualOrderUpTo",
            "MonoKANResidualOrderUpTo",
        ],
    )

    expected = {
        "PaperFCNNResidualOrderUpTo",
        "KANResidualOrderUpTo",
        "ConstrainedMLPResidualOrderUpTo",
        "MonoKANResidualOrderUpTo",
    }
    assert expected.issubset(set(artifacts.summary["policy_name"]))
    assert expected.issubset(set(artifacts.target_structure["policy_name"]))
    assert {"finite_diff_violation_rate", "autograd_violation_rate", "residual_correction_abs_mean"}.issubset(
        artifacts.target_structure.columns
    )
    assert set(artifacts.spline_certification["policy_name"]) == {"MonoKANResidualOrderUpTo"}
    assert (tmp_path / "models" / "constrained_mlp_residual_order_up_to.pt").exists()
    assert (tmp_path / "target_structure_diagnostics.csv").exists()
    assert (tmp_path / "monokan_spline_certification.csv").exists()


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
