import pandas as pd

from inventory_control.idp_benchmark import _profile_settings
from inventory_control.paper_configs import load_paper_lost_sales_config
from scripts.run_no_imitation_ablation import (
    FIXED_CANDIDATE,
    _aggregate,
    resolve_no_imitation_config,
)


def test_no_imitation_config_is_reward_only_and_frozen() -> None:
    base = load_paper_lost_sales_config(
        "configs/paper_lost_sales_medium_seed0_converged.yaml"
    )
    config, resolved = resolve_no_imitation_config(base, "full", "ConstrainedMonoMLP", 7)
    settings = _profile_settings(config, "full")

    assert settings["seed"] == 7
    assert settings["training_products"] == 128
    assert settings["validation_products"] == 32
    assert settings["imitation_epochs"] == 0
    assert settings["imitation_early_stopping"] is False
    assert settings["imitation_max_epochs"] is None
    assert settings["track_imitation_validation"] is False
    assert settings["reset_optimizer_after_imitation"] is False
    assert settings["reward_epochs"] == 15
    assert settings["max_order_factor"] == 4.0
    assert resolved["training_semantics"] == (
        "reward_only_direct_backprop_from_random_initialization"
    )
    assert config["constrained_monotone_mlp"]["activation_weights"] == [1, 1, 1]


def test_no_imitation_penalty_uses_matched_lambda_without_changing_other_policies() -> None:
    base = load_paper_lost_sales_config(
        "configs/paper_lost_sales_medium_seed0_converged.yaml"
    )
    _, penalty = resolve_no_imitation_config(base, "full", "PenaltyMLP", 0)
    _, paper = resolve_no_imitation_config(base, "full", "PaperFCNN", 0)

    assert penalty["penalty_lambda"] == FIXED_CANDIDATE.penalty_lambda == 0.1
    assert paper["penalty_lambda"] is None


def test_aggregate_reports_worst_maximum_violation_across_seeds() -> None:
    rows = []
    for seed, raw_max, deployed_max in ((0, 2.0, 1.0), (1, 5.0, 3.0)):
        rows.append(
            {
                "policy_name": "PaperFCNN",
                "training_seed": seed,
                "parameter_count": 10,
                "mean_reward_per_period": 1.0,
                "mean_inventory": 1.0,
                "service_level": 1.0,
                "lost_sales_rate": 0.0,
                "mean_order_quantity": 1.0,
                "mean_holding_cost": 1.0,
                "deployed_rmse": 1.0,
                "raw_violation_rate": 0.1,
                "deployed_violation_rate": 0.1,
                "raw_max_violation": raw_max,
                "deployed_max_violation": deployed_max,
                "cap_hit_rate": 0.0,
                "non_finite_output_rate": 0.0,
            }
        )

    aggregate = _aggregate(pd.DataFrame(rows)).iloc[0]
    assert aggregate["raw_max_violation"] == 5.0
    assert aggregate["deployed_max_violation"] == 3.0
