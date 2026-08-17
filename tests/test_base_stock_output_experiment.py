import json
import subprocess
import sys

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from inventory_control.base_stock_output_experiment import run_base_stock_output_experiment
from inventory_control.policies.neural.fcnn import (
    compute_paper_fcnn_feature_stats,
    empirical_newsvendor_target_ratio,
    make_paper_base_stock_output_fcnn_model_for_products,
)
from inventory_control.policies.neural.monokan import make_paper_base_stock_output_monokan_model_for_products
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


def test_base_stock_output_experiment_writes_expected_artifacts(tmp_path) -> None:
    artifacts = run_base_stock_output_experiment(
        config_path="configs/paper_lost_sales_debug.yaml",
        output_dir=tmp_path,
        models=("PaperFCNNBaseStockOrderUpTo", "MonoKANBaseStockOrderUpTo"),
    )

    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "training_log.csv").exists()
    assert (tmp_path / "metadata.json").exists()
    assert set(artifacts.summary["policy_name"]) == {
        "PaperFCNNBaseStockOrderUpTo",
        "MonoKANBaseStockOrderUpTo",
        "HistoricalDemandBaseStock",
    }
    assert artifacts.metadata["imitation_pretraining"] is False
    assert "NN(z)->S_theta" in artifacts.metadata["action_parameterization"]


def test_base_stock_output_experiment_records_schema(tmp_path) -> None:
    run_base_stock_output_experiment(
        config_path="configs/paper_lost_sales_debug.yaml",
        output_dir=tmp_path,
        models=("KANBaseStockOrderUpTo",),
    )

    summary = pd.read_csv(tmp_path / "summary.csv")
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

    assert {
        "policy_name",
        "mean_reward_per_period",
        "service_level",
        "lost_sales_rate",
        "mean_order_quantity",
        "mean_inventory",
        "checkpoint_filename",
    }.issubset(summary.columns)
    assert metadata["models"] == ["KANBaseStockOrderUpTo"]
    assert summary.loc[summary["policy_name"] == "KANBaseStockOrderUpTo", "imitation_epochs"].iloc[0] == 0


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
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "PaperFCNNBaseStockOrderUpTo" in result.stdout
    assert "saved_artifacts=" in result.stdout
    assert (tmp_path / "summary.csv").exists()
