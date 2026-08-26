from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")

from inventory_control.rolling_direct_target_matched import (
    PAIRING_KEYS,
    POLICY_LABELS,
    _assert_baselines_match,
    _paired_fcnn_monokan,
    deep_grid7_checkpoint_filename,
    protocol_signature,
    validate_development_protocol,
    validate_frozen_protocol,
    validate_temporal_protocol,
    write_benchmark_config,
)


def test_matched_configs_freeze_the_paper_aligned_protocol() -> None:
    signatures = validate_frozen_protocol()

    assert signatures["rolling"]["training_products"] == signatures["raw"]["training_products"] == 5_000
    assert signatures["rolling"]["batch_size"] == 2_500
    assert signatures["rolling"]["reward_epochs"] == 1_000
    assert signatures["rolling"]["training_horizon"] == 100
    assert signatures["rolling"]["history_feature_mode"] == "raw_lags"
    assert deep_grid7_checkpoint_filename() == "monokan_direct_target_stock_deep_grid7.pt"


def test_development_config_matches_final_three_seed_protocol() -> None:
    protocol = validate_development_protocol()

    assert protocol["training_products"] == 1_000
    assert protocol["batch_size"] == 512
    assert protocol["reward_epochs"] == 200
    assert protocol["initial_inventory_mode"] == "last_historical_demand"
    assert protocol["normalize_observations"] is True
    assert protocol["history_feature_mode"] == "raw_lags"


def test_temporal_config_matches_ablation_protocol() -> None:
    protocol = validate_temporal_protocol()

    assert protocol["training_products"] == 1_000
    assert protocol["reward_epochs"] == 200
    assert protocol["history_feature_mode"] == "raw_lags_temporal"
    assert protocol["normalize_observations"] is True


def test_benchmark_config_changes_only_epoch_fields(tmp_path: Path) -> None:
    source = Path("configs/paper_lost_sales_rolling_direct_target_5k.yaml")
    before = yaml.safe_load(source.read_text(encoding="utf-8"))
    generated = write_benchmark_config(tmp_path / "benchmark.yaml", epochs=3, source=source)
    after = yaml.safe_load(generated.read_text(encoding="utf-8"))

    assert after["epochs"] == 3
    assert after["idp_benchmark"]["compact"]["reward_epochs"] == 3
    after["epochs"] = before["epochs"]
    after["idp_benchmark"]["compact"]["reward_epochs"] = before["idp_benchmark"]["compact"]["reward_epochs"]
    assert after == before
    signature = protocol_signature(generated)
    assert signature["training_products"] == 5_000
    assert signature["batch_size"] == 2_500


def test_target_stock_pairing_is_one_to_one_and_uses_declared_sign() -> None:
    fcnn_name = POLICY_LABELS["PaperFCNNDirectTargetStock"]
    monokan_name = POLICY_LABELS["MonoKANDirectTargetStockDeepGrid7"]
    rows = []
    for product_id, fcnn_reward, monokan_reward in (("p0", 12.0, 10.0), ("p1", 7.0, 9.0)):
        keys = {
            "run_seed": 2025,
            "product_generation_seed": 2025,
            "evaluation_seed": 40_000 + len(rows),
            "demand_path_id": f"path-{product_id}",
            "product_id": product_id,
        }
        rows.extend(
            [
                {**keys, "policy_name": fcnn_name, "mean_reward_per_period": fcnn_reward},
                {**keys, "policy_name": monokan_name, "mean_reward_per_period": monokan_reward},
            ]
        )

    paired = _paired_fcnn_monokan(pd.DataFrame(rows))

    assert list(paired.columns[: len(PAIRING_KEYS)]) == list(PAIRING_KEYS)
    assert paired["fcnn_minus_monokan_reward_per_period"].tolist() == pytest.approx([2.0, -2.0])
    assert not paired.duplicated(list(PAIRING_KEYS)).any()


def test_target_stock_pairing_rejects_missing_pairs() -> None:
    row = {
        "run_seed": 2025,
        "product_generation_seed": 2025,
        "evaluation_seed": 40_000,
        "demand_path_id": "path-p0",
        "product_id": "p0",
        "policy_name": POLICY_LABELS["PaperFCNNDirectTargetStock"],
        "mean_reward_per_period": 1.0,
    }

    with pytest.raises(ValueError, match="missing product pairs"):
        _paired_fcnn_monokan(pd.DataFrame([row]))


def test_baseline_protocol_check_rejects_changed_reward() -> None:
    reference = pd.DataFrame(
        [
            {
                "policy_name": "HistoricalDemandBaseStock",
                "product_id": "p0",
                "evaluation_seed": 40_000,
                "mean_reward_per_period": 10.0,
            }
        ]
    )
    changed = reference.copy()
    changed["mean_reward_per_period"] = 9.0

    with pytest.raises(ValueError, match="baseline rewards differ"):
        _assert_baselines_match(changed, reference, "candidate")
