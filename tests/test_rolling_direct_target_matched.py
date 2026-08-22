from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

pytest.importorskip("torch")

from inventory_control.rolling_direct_target_matched import (
    PAIRING_KEYS,
    POLICY_LABELS,
    _paired_monokan,
    _assert_baselines_match,
    deep_grid7_checkpoint_filename,
    frozen_protocol_signature,
    validate_small_development_protocol,
    validate_frozen_protocol,
    write_benchmark_config,
)
from inventory_control.base_stock_output_experiment import _make_model
from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.policies.neural.fcnn import PaperFCNNFeatureStats
from inventory_control.policies.neural.registry import POLICY_SPECS_BY_NAME
from inventory_control.products import PaperLostSalesProductGenerator


def test_matched_configs_freeze_the_paper_aligned_protocol() -> None:
    signatures = validate_frozen_protocol()

    assert signatures["rolling"] == signatures["static"] == signatures["raw"]
    assert signatures["rolling"]["training_products"] == 5_000
    assert signatures["rolling"]["batch_size"] == 2_500
    assert signatures["rolling"]["reward_epochs"] == 1_000
    assert signatures["rolling"]["training_horizon"] == 100
    assert deep_grid7_checkpoint_filename() == "monokan_direct_target_stock_deep_grid7.pt"


def test_development_configs_differ_only_by_rolling_history() -> None:
    protocol = validate_small_development_protocol()
    rolling = load_paper_lost_sales_config("configs/paper_lost_sales_rolling_direct_target_dev_1k.yaml")
    static = load_paper_lost_sales_config("configs/paper_lost_sales_static_direct_target_dev_1k.yaml")

    assert protocol["training_products"] == 1_000
    assert protocol["batch_size"] == 512
    assert protocol["reward_epochs"] == 200
    assert protocol["training_initial_inventory"] == "Uniform(0, 2 * last_historical_demand)"
    assert rolling["fcnn"]["roll_history"] is True
    assert rolling["monokan"]["roll_history"] is True
    assert static["fcnn"]["roll_history"] is False
    assert static["monokan"]["roll_history"] is False


def test_development_rolling_and_static_monokan_start_identically() -> None:
    rolling = load_paper_lost_sales_config("configs/paper_lost_sales_rolling_direct_target_dev_1k.yaml")
    static = load_paper_lost_sales_config("configs/paper_lost_sales_static_direct_target_dev_1k.yaml")
    products = PaperLostSalesProductGenerator(rolling).generate(8, split="train")
    stats = PaperFCNNFeatureStats(demand_scale=1.0, price_scale=1.0)
    model_seed = 2025 + POLICY_SPECS_BY_NAME["MonoKANDirectTargetStockDeepGrid7"].seed_offset

    torch.manual_seed(model_seed)
    rolling_model = _make_model("MonoKANDirectTargetStockDeepGrid7", rolling, products, stats, 1.0)
    torch.manual_seed(model_seed)
    static_model = _make_model("MonoKANDirectTargetStockDeepGrid7", static, products, stats, 1.0)

    assert rolling_model.paper_roll_history is True
    assert static_model.paper_roll_history is False
    assert rolling_model.state_dict().keys() == static_model.state_dict().keys()
    for name, value in rolling_model.state_dict().items():
        assert torch.equal(value, static_model.state_dict()[name]), name


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
    signature = frozen_protocol_signature(generated)
    assert signature["training_products"] == 5_000
    assert signature["batch_size"] == 2_500


def test_rolling_static_pairing_is_one_to_one_and_uses_declared_sign() -> None:
    rolling_name = POLICY_LABELS["rolling:MonoKANDirectTargetStockDeepGrid7"]
    static_name = POLICY_LABELS["static:MonoKANDirectTargetStockDeepGrid7"]
    rows = []
    for product_id, rolling_reward, static_reward in (("p0", 12.0, 10.0), ("p1", 7.0, 9.0)):
        keys = {
            "run_seed": 2025,
            "product_generation_seed": 2025,
            "evaluation_seed": 40_000 + len(rows),
            "demand_path_id": f"path-{product_id}",
            "product_id": product_id,
        }
        rows.extend(
            [
                {**keys, "policy_name": rolling_name, "mean_reward_per_period": rolling_reward},
                {**keys, "policy_name": static_name, "mean_reward_per_period": static_reward},
            ]
        )

    paired = _paired_monokan(pd.DataFrame(rows))

    assert list(paired.columns[: len(PAIRING_KEYS)]) == list(PAIRING_KEYS)
    assert paired["rolling_minus_static_reward_per_period"].tolist() == pytest.approx([2.0, -2.0])
    assert not paired.duplicated(list(PAIRING_KEYS)).any()


def test_rolling_static_pairing_rejects_missing_pairs() -> None:
    row = {
        "run_seed": 2025,
        "product_generation_seed": 2025,
        "evaluation_seed": 40_000,
        "demand_path_id": "path-p0",
        "product_id": "p0",
        "policy_name": POLICY_LABELS["rolling:MonoKANDirectTargetStockDeepGrid7"],
        "mean_reward_per_period": 1.0,
    }

    with pytest.raises(ValueError, match="missing product pairs"):
        _paired_monokan(pd.DataFrame([row]))


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
