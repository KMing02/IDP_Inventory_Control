import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")

from inventory_control.base_stock_output_experiment import run_base_stock_output_experiment
from inventory_control.policies.neural.fcnn import (
    build_paper_lost_sales_features,
    compute_paper_fcnn_feature_stats,
    empirical_newsvendor_target_ratio,
    make_paper_base_stock_output_fcnn_model_for_products,
    paper_history_feature_count,
)
from inventory_control.policies.neural.monokan import make_paper_base_stock_output_monokan_model_for_products
from inventory_control.policies.neural.monokan import make_paper_base_stock_output_kan_model
from inventory_control.policies.neural.registry import BASE_STOCK_OUTPUT_POLICY_NAMES, checkpoint_filenames
from inventory_control.products import PaperLostSalesProductGenerator
from tests.test_paper_lost_sales import _paper_config


def test_base_stock_registry_exposes_current_monokan_comparison() -> None:
    filenames = checkpoint_filenames()

    assert BASE_STOCK_OUTPUT_POLICY_NAMES == (
        "PaperFCNNBaseStockOrderUpTo",
        "KANBaseStockOrderUpTo",
        "MonoKANBaseStockOrderUpTo",
    )
    assert filenames["PaperFCNNBaseStockOrderUpTo"] == "paper_fcnn_base_stock_order_up_to.pt"
    assert filenames["KANBaseStockOrderUpTo"] == "kan_base_stock_order_up_to.pt"
    assert filenames["MonoKANBaseStockOrderUpTo"] == "monokan_base_stock_order_up_to.pt"


def test_base_stock_fcnn_initializes_to_empirical_newsvendor_target() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(8, "train")
    feature_stats = compute_paper_fcnn_feature_stats(products)
    target_ratio = empirical_newsvendor_target_ratio(products, feature_stats)
    model = make_paper_base_stock_output_fcnn_model_for_products(
        _paper_config(),
        products,
        feature_stats=feature_stats,
    )

    predicted = model(torch.zeros(3, 36))

    assert torch.allclose(predicted, torch.full((3,), target_ratio), atol=1e-5)


def test_base_stock_monokan_initializes_to_empirical_newsvendor_target() -> None:
    products = PaperLostSalesProductGenerator(_paper_config()).generate(8, "train")
    feature_stats = compute_paper_fcnn_feature_stats(products)
    target_ratio = empirical_newsvendor_target_ratio(products, feature_stats)
    config = {
        **_paper_config(),
        "monokan": {
            "hidden_layers": 1,
            "hidden_units": 4,
            "grid": 3,
            "max_order_factor": 4.0,
        },
    }

    model = make_paper_base_stock_output_monokan_model_for_products(config, products, feature_stats=feature_stats)
    predicted = model(torch.zeros(1, 36))

    assert predicted.item() == pytest.approx(target_ratio, abs=1e-5)


def test_base_stock_kan_uses_logit_residual_anchor() -> None:
    config = {
        **_paper_config(),
        "monokan": {
            "hidden_layers": 1,
            "hidden_units": 4,
            "grid": 3,
            "max_order_factor": 4.0,
            "base_stock_logit_residual_scale": 0.25,
            "initial_normalized_target": 1.5,
        },
    }

    model = make_paper_base_stock_output_kan_model(config)

    assert model.output_logit_anchor is not None
    assert model.output_logit_residual_scale == pytest.approx(0.25)
    assert model(torch.zeros(1, 36)).item() == pytest.approx(1.5, abs=1e-5)


def test_summary_history_features_have_stable_dimension() -> None:
    features = build_paper_lost_sales_features(
        inventory=5.0,
        historical_demand=list(range(1, 33)),
        price=10.0,
        purchase_cost=4.0,
        holding_cost=1.0,
        lost_sale_penalty=2.0,
        history_feature_mode="summary",
    )

    assert paper_history_feature_count(32, "summary") == 9
    assert len(features) == 1 + 9 + 4
    assert features[1] == pytest.approx(16.5 / 16.5)


def test_base_stock_summary_history_mode_reduces_kan_input_dimension() -> None:
    config = {
        **_paper_config(),
        "paper_history_feature_mode": "summary",
        "monokan": {
            "hidden_layers": 1,
            "hidden_units": 4,
            "grid": 3,
            "max_order_factor": 4.0,
            "initial_normalized_target": 1.5,
        },
    }

    model = make_paper_base_stock_output_kan_model(config)

    assert model.layers[0].in_dim == 13
    assert model.paper_history_feature_mode == "summary"
    assert model(torch.zeros(1, 13)).item() == pytest.approx(1.5, abs=1e-5)


def test_base_stock_output_experiment_writes_expected_artifacts(tmp_path) -> None:
    artifacts = run_base_stock_output_experiment(
        config_path="configs/paper_lost_sales_debug.yaml",
        output_dir=tmp_path,
        models=("PaperFCNNBaseStockOrderUpTo", "MonoKANBaseStockOrderUpTo"),
        device="cpu",
    )

    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "training_log.csv").exists()
    assert (tmp_path / "target_stock_diagnostics.csv").exists()
    assert (tmp_path / "metadata.json").exists()
    assert set(artifacts.summary["policy_name"]) == {
        "PaperFCNNBaseStockOrderUpTo",
        "MonoKANBaseStockOrderUpTo",
        "HistoricalDemandBaseStock",
    }
    assert artifacts.metadata["imitation_pretraining"] is False
    assert artifacts.metadata["paper_history_feature_mode"] == "raw_lags"
    assert "NN(z)->S_theta" in artifacts.metadata["action_parameterization"]


def test_base_stock_output_experiment_records_schema(tmp_path) -> None:
    run_base_stock_output_experiment(
        config_path="configs/paper_lost_sales_debug.yaml",
        output_dir=tmp_path,
        models=("KANBaseStockOrderUpTo",),
        device="cpu",
    )

    summary = pd.read_csv(tmp_path / "summary.csv")
    target_diagnostics = pd.read_csv(tmp_path / "target_stock_diagnostics.csv")
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

    assert {
        "policy_name",
        "mean_reward_per_period",
        "service_level",
        "lost_sales_rate",
        "mean_order_quantity",
        "mean_inventory",
        "checkpoint_filename",
        "target_factor_mean",
        "target_minus_historical_base_stock_mean",
        "near_upper_bound_rate",
    }.issubset(summary.columns)
    assert {
        "policy_name",
        "product_id",
        "target_factor",
        "target_stock",
        "historical_base_stock",
        "target_minus_historical_base_stock",
        "near_upper_bound",
    }.issubset(target_diagnostics.columns)
    assert metadata["models"] == ["KANBaseStockOrderUpTo"]
    assert summary.loc[summary["policy_name"] == "KANBaseStockOrderUpTo", "imitation_epochs"].iloc[0] == 0


def test_base_stock_output_experiment_supports_summary_history_mode(tmp_path) -> None:
    config = yaml.safe_load(Path("configs/paper_lost_sales_debug.yaml").read_text(encoding="utf-8"))
    config["paper_history_feature_mode"] = "summary"
    config["idp_benchmark"]["compact"] = {
        "training_products": 8,
        "validation_products": 4,
        "batch_size": 4,
        "training_horizon": 4,
        "reward_epochs": 1,
        "validation_interval": 1,
        "evaluation_horizon": 4,
    }
    config["monokan"] = {
        "hidden_layers": 1,
        "hidden_units": 4,
        "grid": 3,
        "max_order_factor": 4.0,
        "base_stock_logit_residual_scale": 0.03,
    }
    config_path = tmp_path / "summary_history.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    artifacts = run_base_stock_output_experiment(
        config_path=config_path,
        output_dir=tmp_path / "run",
        models=("KANBaseStockOrderUpTo", "MonoKANBaseStockOrderUpTo"),
        device="cpu",
    )

    assert artifacts.metadata["paper_history_feature_mode"] == "summary"
    assert set(artifacts.summary["policy_name"]) == {
        "KANBaseStockOrderUpTo",
        "MonoKANBaseStockOrderUpTo",
        "HistoricalDemandBaseStock",
    }
    assert (tmp_path / "run" / "target_stock_diagnostics.csv").exists()


def test_base_stock_output_cli_smoke(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_base_stock_output_experiment.py",
            "--config",
            "configs/paper_lost_sales_debug.yaml",
            "--output-dir",
            str(tmp_path),
            "--models",
            "PaperFCNNBaseStockOrderUpTo",
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "PaperFCNNBaseStockOrderUpTo" in result.stdout
    assert "saved_artifacts=" in result.stdout
    assert (tmp_path / "summary.csv").exists()
