import pytest

from inventory_control.paper_configs import (
    apply_config_overrides,
    load_paper_lost_sales_config,
    validate_paper_lost_sales_config,
)
from inventory_control.paper_evaluation import require_final_test_confirmation


def test_paper_config_validation_requires_fields() -> None:
    with pytest.raises(ValueError, match="missing required"):
        validate_paper_lost_sales_config({"history_length_H": 32})


def test_test_split_requires_final_test_confirmation() -> None:
    require_final_test_confirmation("validation", final_test=False)
    require_final_test_confirmation("test", final_test=True)

    with pytest.raises(ValueError, match="Refusing test-set evaluation"):
        require_final_test_confirmation("test", final_test=False)


def test_paper_config_can_be_selected_by_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPER_LOST_SALES_CONFIG", "configs/paper_lost_sales_debug.yaml")

    config = load_paper_lost_sales_config()

    assert config["history_length_H"] == 8


def test_config_overrides_parse_values_and_leave_the_original_untouched() -> None:
    config = load_paper_lost_sales_config("configs/paper_lost_sales_soft_quantile_monokan.yaml")

    updated = apply_config_overrides(
        config, ["monokan.level_network=kan", "monokan.roll_history=false", "epochs=1200"]
    )

    assert updated["monokan"]["level_network"] == "kan"
    assert updated["monokan"]["roll_history"] is False
    assert updated["epochs"] == 1200
    assert config["monokan"]["level_network"] == "monokan"


def test_config_overrides_reject_unknown_keys() -> None:
    config = load_paper_lost_sales_config("configs/paper_lost_sales_soft_quantile_monokan.yaml")

    with pytest.raises(KeyError, match="unknown config key"):
        apply_config_overrides(config, ["monokan.level_netwrk=kan"])
    with pytest.raises(ValueError, match="section.key=value"):
        apply_config_overrides(config, ["monokan.level_network"])


def test_final_configs_resolve_to_the_reported_protocol() -> None:
    from inventory_control.base_stock_output_experiment import _compact_settings

    final = load_paper_lost_sales_config("configs/paper_lost_sales_soft_quantile_monokan.yaml")
    matched = load_paper_lost_sales_config("configs/paper_lost_sales_rolling_direct_target_dev_1k_normalized.yaml")

    assert _compact_settings(final)["training_products"] == 5_000
    assert _compact_settings(final)["reward_epochs"] == 2_400
    assert _compact_settings(final)["per_product_reward_weighting"] is True
    assert final["monokan"]["window_mode"] == "expanding"
    assert _compact_settings(matched)["training_products"] == 1_000
    for config in (final, matched):
        settings = _compact_settings(config)
        assert settings["validation_products"] == 300
        assert (settings["evaluation_horizon"], settings["burn_in"], settings["training_horizon"]) == (520, 20, 100)

