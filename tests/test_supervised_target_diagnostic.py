from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import torch

from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.policies.neural.fcnn import compute_paper_fcnn_feature_stats
from inventory_control.products import PaperLostSalesProductGenerator
from inventory_control.supervised_target_diagnostic import (
    build_supervised_target_dataset,
    run_supervised_target_diagnostic,
)


def test_supervised_labels_do_not_use_future_demand() -> None:
    config = load_paper_lost_sales_config("configs/paper_lost_sales_debug.yaml")
    products = PaperLostSalesProductGenerator(config).generate(4, split="train")
    changed_future = [replace(product, future_demand=[999] * len(product.future_demand or [])) for product in products]
    stats = compute_paper_fcnn_feature_stats(products)

    original = build_supervised_target_dataset(products, stats, "summary")
    changed = build_supervised_target_dataset(changed_future, stats, "summary")

    assert torch.equal(original.contexts, changed.contexts)
    assert torch.equal(original.target_ratios, changed.target_ratios)


def test_supervised_target_diagnostic_writes_stable_outputs(tmp_path: Path) -> None:
    artifacts = run_supervised_target_diagnostic(
        config_path="configs/paper_lost_sales_debug.yaml",
        output_dir=tmp_path,
        models=("PaperFCNNDirectTargetStock", "KANDirectTargetStock", "MonoKANDirectTargetStock"),
        epochs=2,
        validation_interval=1,
        training_products=6,
        validation_products=3,
        batch_size=3,
        seed=17,
    )

    assert set(path.name for path in tmp_path.iterdir()) == {
        "summary.csv",
        "predictions.csv",
        "training_log.csv",
        "feature_sensitivities.csv",
        "metadata.json",
    }
    assert set(artifacts.summary["model"]) == {
        "PaperFCNNDirectTargetStock",
        "KANDirectTargetStock",
        "MonoKANDirectTargetStock",
    }
    assert set(artifacts.summary["split"]) == {"train", "validation"}
    assert {
        "target_ratio_mae",
        "target_ratio_rmse",
        "target_ratio_r2",
        "predicted_ratio_std",
        "hidden_effective_rank",
        "final_spline_scale_share",
    }.issubset(artifacts.summary.columns)
    assert artifacts.predictions.groupby(["model", "split", "product_id"]).size().max() == 1
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["diagnostic_only"] is True
    assert metadata["used_for_policy_pretraining"] is False
    assert metadata["future_demand_used"] is False
    assert metadata["test_split_used"] is False


def test_supervised_target_diagnostic_is_deterministic(tmp_path: Path) -> None:
    kwargs = {
        "config_path": "configs/paper_lost_sales_debug.yaml",
        "models": ("KANDirectTargetStock",),
        "epochs": 2,
        "validation_interval": 1,
        "training_products": 6,
        "validation_products": 3,
        "batch_size": 3,
        "seed": 23,
    }
    first = run_supervised_target_diagnostic(output_dir=tmp_path / "first", **kwargs)
    second = run_supervised_target_diagnostic(output_dir=tmp_path / "second", **kwargs)

    pd.testing.assert_frame_equal(first.summary, second.summary)
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    pd.testing.assert_frame_equal(first.training_log, second.training_log)


def test_supervised_target_diagnostic_is_model_order_independent(tmp_path: Path) -> None:
    kwargs = {
        "config_path": "configs/paper_lost_sales_debug.yaml",
        "epochs": 2,
        "validation_interval": 1,
        "training_products": 6,
        "validation_products": 3,
        "batch_size": 3,
        "seed": 29,
    }
    alone = run_supervised_target_diagnostic(
        output_dir=tmp_path / "alone",
        models=("KANDirectTargetStock",),
        **kwargs,
    )
    with_comparator = run_supervised_target_diagnostic(
        output_dir=tmp_path / "with-comparator",
        models=("PaperFCNNDirectTargetStock", "KANDirectTargetStock"),
        **kwargs,
    )

    expected = alone.predictions[alone.predictions["model"] == "KANDirectTargetStock"].reset_index(drop=True)
    actual = with_comparator.predictions[
        with_comparator.predictions["model"] == "KANDirectTargetStock"
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, actual)
    assert with_comparator.metadata["model_seeds"]["KANDirectTargetStock"] == 29


def test_supervised_target_cli_smoke(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_supervised_target_diagnostic.py",
            "--config",
            "configs/paper_lost_sales_debug.yaml",
            "--output-dir",
            str(tmp_path),
            "--models",
            "PaperFCNNDirectTargetStock",
            "--epochs",
            "1",
            "--validation-interval",
            "1",
            "--training-products",
            "4",
            "--validation-products",
            "2",
            "--batch-size",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "PaperFCNNDirectTargetStock" in result.stdout
    assert f"saved_artifacts={tmp_path}" in result.stdout
