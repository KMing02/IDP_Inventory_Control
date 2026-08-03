from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from inventory_control.policies.neural.registry import RESIDUAL_COMPARISON_POLICY_NAMES, checkpoint_filenames, policy_factories
from scripts import plot_controlled_policy_surfaces as controlled_surfaces
from scripts import plot_policy_contours_from_predictions as contour_plots
from scripts import run_monokan_multiseed_validation as multiseed_validation
from scripts import run_monokan_optimization_sweep as optimization_sweep


def _tiny_factorial_config() -> dict:
    return {
        "history_length_H": 8,
        "training_horizon_T": 3,
        "evaluation_horizon": 5,
        "burn_in": 0,
        "training_products": 4,
        "test_products": 2,
        "batch_size": 2,
        "epochs": 1,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "fcnn": {"max_order_factor": 4.0},
        "monokan": {"hidden_units": 4, "residual_scale": 0.05},
        "idp_benchmark": {
            "seed": 2025,
            "hidden_layers": 1,
            "hidden_units": 4,
            "activation": "ELU",
            "max_order_factor": 4.0,
            "scale_action_by_history_mean": True,
            "normalize_reward_loss": True,
            "include_terminal_salvage": False,
            "compact": {
                "training_products": 4,
                "validation_products": 2,
                "batch_size": 2,
                "training_horizon": 3,
                "imitation_epochs": 0,
                "reward_epochs": 1,
                "validation_interval": 1,
                "evaluation_horizon": 5,
                "burn_in": 0,
                "inventory_states": 3,
                "structure_products": 1,
                "order_up_to_products": 1,
                "oracle_mc_samples": 10,
            },
        },
        "product_generation": {
            "seed": 2025,
            "mode": "paper_table_1",
            "price_mean": 20.0,
            "holding_cost_mean": 1.0,
            "mean_demand_mean": 10.0,
            "lost_sale_penalty_max": 3.0,
            "coefficient_of_variation_range": [0.0, 1.0],
        },
    }


def test_factorial_small_config_is_exploratory_and_loadable() -> None:
    config = yaml.safe_load(Path("configs/paper_lost_sales_factorial_small.yaml").read_text())

    assert config["idp_benchmark"]["compact"]["training_products"] < config["idp_benchmark"]["paper_replication"]["training_products"]
    assert config["idp_benchmark"]["compact"]["validation_products"] < config["idp_benchmark"]["paper_replication"]["validation_products"]
    assert config["idp_benchmark"]["include_terminal_salvage"] is False
    assert config["idp_benchmark"]["residual_anchor"] == "gamma"


def test_registry_metadata_supports_four_residual_policies() -> None:
    factories = policy_factories(RESIDUAL_COMPARISON_POLICY_NAMES)
    filenames = checkpoint_filenames()

    assert list(factories) == list(RESIDUAL_COMPARISON_POLICY_NAMES)
    assert set(RESIDUAL_COMPARISON_POLICY_NAMES).issubset(filenames)
    assert "constrained_mlp_residual_order_up_to.pt" == filenames["ConstrainedMLPResidualOrderUpTo"]


def test_controlled_surface_plot_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "surface.png"
    data = {
        "PaperFCNNResidualOrderUpTo": np.array([[1.0, 0.5], [1.5, 0.7]]),
        "ConstrainedMLPResidualOrderUpTo": np.array([[0.9, 0.4], [1.3, 0.6]]),
    }

    controlled_surfaces._plot_surface(
        data,
        np.array([0.0, 1.0]),
        np.array([0.5, 1.0]),
        "tiny controlled target stock S_theta",
        "target stock S_theta",
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_controlled_surface_loader_fails_on_missing_checkpoint(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmark"
    (benchmark_dir / "models").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="missing checkpoint"):
        controlled_surfaces._load_models(
            benchmark_dir,
            {"history_length_H": 8},
            ["PaperFCNNResidualOrderUpTo"],
        )


def test_contour_plots_from_synthetic_predictions(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        [
            {
                "product_id": "p1",
                "policy_name": "PaperFCNNResidualOrderUpTo",
                "state_index": state,
                "inventory": float(state),
                "inventory_normalized": float(state),
                "historical_mean_demand": float(mean),
                "raw_order": float(mean - state + 1),
                "deployed_order": float(max(mean - state, 0)),
                "oracle_order": float(max(mean - state + 0.5, 0)),
            }
            for mean in [1.0, 2.0, 3.0]
            for state in [0, 1, 2]
        ]
    )

    contour_path = tmp_path / "contours.png"
    curves_path = tmp_path / "curves.png"
    contour_plots._plot_inventory_demand_contours(predictions, ["PaperFCNNResidualOrderUpTo"], contour_path)
    contour_plots._plot_raw_vs_deployed_curves(predictions, ["PaperFCNNResidualOrderUpTo"], curves_path)

    assert contour_path.exists()
    assert curves_path.exists()
    assert contour_path.stat().st_size > 0
    assert curves_path.stat().st_size > 0


def test_contour_plot_fails_on_missing_required_column() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        contour_plots._plot_inventory_demand_contours(pd.DataFrame({"policy_name": ["x"]}), ["x"], Path("unused.png"))


def test_contour_plot_cli_fails_clearly_on_empty_predictions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    (benchmark_dir / "predictions.csv").write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["plot_policy_contours_from_predictions.py", str(benchmark_dir)])

    with pytest.raises(ValueError, match="predictions file is empty"):
        contour_plots.main()


def test_multiseed_runner_one_seed_records_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_tiny_factorial_config()), encoding="utf-8")
    output_root = tmp_path / "multiseed"

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess:
        assert check is True
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "policy_name": "PaperFCNNResidualOrderUpTo",
                    "mean_reward_per_period": 1.0,
                    "service_level": 0.9,
                    "lost_sales_rate": 0.1,
                    "mean_inventory": 2.0,
                    "training_seconds": 0.01,
                }
            ]
        ).to_csv(output_dir / "summary.csv", index=False)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_monokan_multiseed_validation.py",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--seeds",
            "2025",
            "--models",
            "PaperFCNNResidualOrderUpTo",
        ],
    )

    multiseed_validation.main()

    assert (output_root / "run_metadata.json").exists()
    assert (output_root / "per_seed_summary.csv").exists()
    aggregate = pd.read_csv(output_root / "aggregate_summary.csv")
    assert aggregate["policy"].tolist() == ["PaperFCNNResidualOrderUpTo"]


def test_optimization_sweep_one_variant_records_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_tiny_factorial_config()), encoding="utf-8")
    output_root = tmp_path / "sweep"

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess:
        assert check is True
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "policy_name": "MonoKANResidualOrderUpTo",
                    "mean_reward_per_period": 1.0,
                    "order_up_to_mae": 0.1,
                    "cap_hit_rate": 0.0,
                    "mean_inventory": 2.0,
                    "service_level": 0.9,
                    "training_seconds": 0.01,
                }
            ]
        ).to_csv(output_dir / "summary.csv", index=False)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_monokan_optimization_sweep.py",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--models",
            "MonoKANResidualOrderUpTo",
            "--max-variants",
            "1",
        ],
    )

    optimization_sweep.main()

    assert (output_root / "run_metadata.json").exists()
    assert (output_root / "aggregate_summary.csv").exists()
    aggregate = pd.read_csv(output_root / "aggregate_summary.csv")
    assert aggregate["policy"].tolist() == ["MonoKANResidualOrderUpTo"]
