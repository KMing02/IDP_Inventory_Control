from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

from inventory_control.base_stock_output_experiment import run_base_stock_output_experiment
from inventory_control.direct_target_structural_audit import run_direct_target_structural_audit


def test_direct_target_structural_audit_loads_checkpoints_and_writes_complete_target_results(tmp_path: Path) -> None:
    config = yaml.safe_load(Path("configs/paper_lost_sales_debug.yaml").read_text(encoding="utf-8"))
    config["paper_history_feature_mode"] = "raw_lags"
    config["paper_include_newsvendor_feature"] = False
    config["idp_benchmark"]["compact"].update(
        {
            "training_products": 6,
            "validation_products": 3,
            "batch_size": 3,
            "training_horizon": 3,
            "reward_epochs": 1,
            "validation_interval": 1,
            "evaluation_horizon": 4,
            "burn_in": 0,
        }
    )
    config["monokan"] = {"hidden_layers": 1, "hidden_units": 3, "grid": 3, "grid_range": [-2.0, 4.0]}
    config_path = tmp_path / "audit_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    checkpoint_root = tmp_path / "checkpoints"
    seed_dir = checkpoint_root / "seed_17"
    run_base_stock_output_experiment(
        config_path=config_path,
        output_dir=seed_dir,
        models=("KANDirectTargetStock", "MonoKANDirectTargetStock"),
        device="cpu",
        seed=17,
    )

    artifacts = run_direct_target_structural_audit(
        config_path,
        checkpoint_root,
        tmp_path / "audit",
        models=("KANDirectTargetStock", "MonoKANDirectTargetStock"),
        products_per_domain=3,
    )

    assert len(artifacts.target_monotonicity) == 2 * 5 * 2
    assert artifacts.target_monotonicity["complete_target"].all()
    assert set(artifacts.target_monotonicity["domain"]) == {"in_domain", "stress_extrapolation"}
    monokan = artifacts.target_monotonicity[
        artifacts.target_monotonicity["policy_name"] == "MonoKANDirectTargetStock"
    ]
    assert monokan["finite_diff_violation_rate"].max() == 0.0
    assert not artifacts.spline_certification.empty
    assert artifacts.spline_certification["max_wrong_direction_increment"].max() <= 1e-6
    assert (tmp_path / "audit" / "complete_target_monotonicity.csv").exists()
    assert (tmp_path / "audit" / "monokan_spline_certification.csv").exists()
    assert (tmp_path / "audit" / "STRUCTURAL_AUDIT.md").exists()
