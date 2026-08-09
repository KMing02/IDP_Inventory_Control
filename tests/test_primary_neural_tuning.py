import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.products import PaperLostSalesProduct
from inventory_control.training import direct_backprop
from scripts.tune_primary_neural_policies import (
    CANDIDATE_IDS,
    PRIMARY_POLICIES,
    REQUIRED_COMPLETION_FILES,
    _completion_record,
    _fixed_validation_grid_seed,
    _penalty_pair_recorder,
    _pooled_prediction_diagnostics,
    _rank_with_reward_ties,
    _repository_relative_path,
    _safe_generate_products,
    _stage_plan,
    candidate_manifest,
    expected_fit_counts,
    validate_split_and_output,
)


def test_balanced_manifest_has_exact_two_stage_budget_and_assignments() -> None:
    manifest = candidate_manifest()

    assert len(manifest) == 32
    assert expected_fit_counts() == {"stage1": 32, "stage2": 16, "maximum_total": 48}
    assert set(manifest["policy_name"]) == set(PRIMARY_POLICIES)
    assert (manifest.groupby("policy_name")["candidate_id"].nunique() == 8).all()
    assert set(manifest["candidate_id"]) == set(CANDIDATE_IDS)

    penalty = manifest.loc[manifest["policy_name"] == "PenaltyMLP"].set_index("candidate_id")
    assert penalty["penalty_lambda"].value_counts().to_dict() == {
        0.1: 2,
        1.0: 2,
        10.0: 2,
        100.0: 2,
    }
    assert penalty.loc["C2", "penalty_lambda"] == 10.0
    assert penalty.loc["C1", "penalty_lambda"] == penalty.loc["C8", "penalty_lambda"]
    assert penalty.loc["C3", "penalty_lambda"] == penalty.loc["C6", "penalty_lambda"]
    assert penalty.loc["C4", "penalty_lambda"] == penalty.loc["C5", "penalty_lambda"]

    cmnn = manifest.loc[manifest["policy_name"] == "ConstrainedMonoMLP"].set_index(
        "candidate_id"
    )
    assert json.loads(cmnn.loc["C1", "activation_weights"]) == [1, 1, 1]
    assert json.loads(cmnn.loc["C2", "activation_weights"]) == [1, 1, 1]
    assert json.loads(cmnn.loc["C3", "activation_weights"]) == [2, 2, 1]
    assert json.loads(cmnn.loc["C4", "activation_weights"]) == [1, 1, 2]
    assert json.loads(cmnn.loc["C5", "activation_weights"]) == [1, 1, 2]
    assert json.loads(cmnn.loc["C6", "activation_weights"]) == [2, 2, 1]
    assert json.loads(cmnn.loc["C7", "activation_weights"]) == [1, 1, 1]
    assert json.loads(cmnn.loc["C8", "activation_weights"]) == [1, 1, 1]


def _ranking_frame(policy: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "policy_name": [policy, policy, policy],
            "candidate_id": ["C1", "C2", "C3"],
            "mean_reward_per_period": [100.0, 99.7, 98.0],
            "raw_violation_rate": [0.0, 0.0, 0.0],
            "deployed_violation_rate": [0.0, 0.0, 0.0],
            "deployed_rmse": [5.0, 4.0, 1.0],
            "deployed_zero_inventory_order_up_to_p90_absolute_error": [5.0, 4.0, 1.0],
            "deployed_zero_inventory_order_up_to_max_absolute_error": [5.0, 4.0, 1.0],
            "parameter_count": [10, 20, 5],
        }
    )


def test_stage1_ranking_uses_diagnostics_only_inside_reward_tie_band() -> None:
    ranked = _rank_with_reward_ties(
        _ranking_frame("PaperFCNN"),
        "PaperFCNN",
        reward_column="mean_reward_per_period",
        pooled=False,
    )

    assert ranked.sort_values("selection_rank")["candidate_id"].tolist() == ["C2", "C1", "C3"]


def test_penalty_ranking_prefers_lower_raw_violation_inside_reward_tie() -> None:
    frame = _ranking_frame("PenaltyMLP")
    frame.loc[frame["candidate_id"] == "C1", "raw_violation_rate"] = 0.02
    frame.loc[frame["candidate_id"] == "C2", "raw_violation_rate"] = 0.01

    ranked = _rank_with_reward_ties(
        frame,
        "PenaltyMLP",
        reward_column="mean_reward_per_period",
        pooled=False,
    )

    assert ranked.sort_values("selection_rank")["candidate_id"].tolist()[:2] == ["C2", "C1"]


@pytest.mark.parametrize("policy", ["PositiveWeightMLP", "ConstrainedMonoMLP"])
def test_hard_monotone_ranking_rejects_any_nonzero_violation(policy: str) -> None:
    frame = _ranking_frame(policy)
    frame.loc[frame["candidate_id"] == "C1", "raw_violation_rate"] = 1e-3

    ranked = _rank_with_reward_ties(
        frame,
        policy,
        reward_column="mean_reward_per_period",
        pooled=False,
    )

    rejected = ranked.loc[ranked["candidate_id"] == "C1"].iloc[0]
    assert not bool(rejected["eligible"])
    assert np.isnan(rejected["selection_rank"])
    assert ranked.sort_values("selection_rank")["candidate_id"].iloc[0] == "C2"


def test_protected_test_guards_fail_closed() -> None:
    with pytest.raises(ValueError, match="validation-only"):
        validate_split_and_output("test", Path("outputs/idp_tuning/validation"))
    with pytest.raises(ValueError, match="test-labelled"):
        validate_split_and_output("validation", Path("outputs/idp_tuning/protected_test"))
    with pytest.raises(ValueError, match="repository-relative"):
        _repository_relative_path(Path("C:/absolute/config.yaml"), "--config")

    class GeneratorThatMustNotBeCalled:
        def generate(self, count, split):  # pragma: no cover - the guard must run first
            raise AssertionError((count, split))

    with pytest.raises(RuntimeError, match="Protected product split"):
        _safe_generate_products(GeneratorThatMustNotBeCalled(), 1, "test")


def test_screening_plan_contains_exactly_32_seed_zero_fits(tmp_path: Path) -> None:
    plan = _stage_plan(
        "screening",
        candidate_manifest(),
        tmp_path,
        PRIMARY_POLICIES,
        CANDIDATE_IDS,
        (0,),
    )

    assert len(plan) == 32
    assert {seed for _, _, seed in plan} == {0}
    with pytest.raises(ValueError, match="seed 0 only"):
        _stage_plan(
            "screening",
            candidate_manifest(),
            tmp_path,
            PRIMARY_POLICIES,
            CANDIDATE_IDS,
            (1,),
        )


def test_confirmation_plan_uses_only_two_promoted_candidates_per_policy(tmp_path: Path) -> None:
    promotions = pd.DataFrame(
        [
            {"policy_name": policy, "candidate_id": candidate_id}
            for policy in PRIMARY_POLICIES
            for candidate_id in ("C2", "C7")
        ]
    )
    promotions.to_csv(tmp_path / "stage1_promoted_candidates.csv", index=False)
    with pytest.raises(RuntimeError, match="all 32 Stage-1 fits"):
        _stage_plan(
            "confirmation",
            candidate_manifest(),
            tmp_path,
            PRIMARY_POLICIES,
            CANDIDATE_IDS,
            (1, 2),
        )
    for policy in PRIMARY_POLICIES:
        for candidate_id in CANDIDATE_IDS:
            fit_dir = tmp_path / "runs" / policy / candidate_id / "seed_0"
            fit_dir.mkdir(parents=True)
            for filename in REQUIRED_COMPLETION_FILES:
                (fit_dir / filename).write_text("placeholder", encoding="utf-8")
            if policy == "PenaltyMLP":
                (fit_dir / "penalty_pair_diagnostics.csv").write_text(
                    "sampled_pair_count\n128\n",
                    encoding="utf-8",
                )
            (fit_dir / "completion.json").write_text(
                json.dumps(
                    {
                        "stage": "screening",
                        "policy_name": policy,
                        "candidate_id": candidate_id,
                        "training_seed": 0,
                    }
                ),
                encoding="utf-8",
            )

    plan = _stage_plan(
        "confirmation",
        candidate_manifest(),
        tmp_path,
        PRIMARY_POLICIES,
        CANDIDATE_IDS,
        (1, 2),
    )

    assert len(plan) == 16
    assert {candidate.candidate_id for _, candidate, _ in plan} == {"C2", "C7"}
    assert {seed for _, _, seed in plan} == {1, 2}


def test_resume_completion_requires_marker_and_all_fit_files(tmp_path: Path) -> None:
    fit_dir = tmp_path / "runs" / "PaperFCNN" / "C1" / "seed_0"
    fit_dir.mkdir(parents=True)
    (fit_dir / "completion.json").write_text('{"candidate_id":"C1"}', encoding="utf-8")
    assert _completion_record(fit_dir) is None

    for filename in REQUIRED_COMPLETION_FILES:
        if filename != "completion.json":
            (fit_dir / filename).write_text("placeholder", encoding="utf-8")
    assert _completion_record(fit_dir) == {"candidate_id": "C1"}


def test_penalty_pair_instrumentation_preserves_loss_and_generator_state() -> None:
    model = torch.nn.Linear(37, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
        model.weight[0, 0] = 1.0
    batch_size = 3
    batch = {
        "initial_inventory": torch.zeros(batch_size),
        "history": torch.zeros(batch_size, 32),
        "feature_demand_scale": torch.ones(batch_size),
        "feature_price_scale": torch.ones(batch_size),
        "historical_mean_demand": torch.full((batch_size,), 10.0),
        "price": torch.ones(batch_size),
        "purchase_cost": torch.ones(batch_size),
        "holding_cost": torch.ones(batch_size),
        "lost_sale_penalty": torch.ones(batch_size),
    }
    expected_generator = torch.Generator().manual_seed(123)
    expected = direct_backprop.paper_inventory_monotonicity_penalty(
        model,
        batch,
        expected_generator,
        samples_per_product=2,
        max_inventory_factor=4.0,
    )
    expected_state = expected_generator.get_state()

    actual_generator = torch.Generator().manual_seed(123)
    with _penalty_pair_recorder(True) as records:
        actual = direct_backprop.paper_inventory_monotonicity_penalty(
            model,
            batch,
            actual_generator,
            samples_per_product=2,
            max_inventory_factor=4.0,
        )

    assert torch.equal(actual, expected)
    assert torch.equal(actual_generator.get_state(), expected_state)
    assert records[0]["sampled_pair_count"] == 6
    assert records[0]["violating_pair_count"] == 6
    assert records[0]["violating_pair_proportion"] == 1.0


def test_validation_grid_seed_is_decoupled_from_training_generator() -> None:
    product = PaperLostSalesProduct(
        product_id="validation_fixed",
        price=100.0,
        purchase_cost=40.0,
        holding_cost=5.0,
        lost_sale_penalty=10.0,
        true_mean_demand=10.0,
        true_coefficient_of_variation=0.5,
        gamma_shape=4.0,
        gamma_scale=2.5,
        demand_distribution_type="Gamma",
        historical_demand=[10] * 32,
        initial_inventory=0,
        future_demand=[10] * 50,
    )
    validation_products = [product]

    with _fixed_validation_grid_seed(validation_products):
        first = direct_backprop.make_paper_lost_sales_training_batch(
            validation_products,
            50,
            torch.Generator().manual_seed(0),
        )
        second = direct_backprop.make_paper_lost_sales_training_batch(
            validation_products,
            50,
            torch.Generator().manual_seed(2),
        )
        training_seed_zero = direct_backprop.make_paper_lost_sales_training_batch(
            list(validation_products),
            50,
            torch.Generator().manual_seed(0),
        )
        training_seed_two = direct_backprop.make_paper_lost_sales_training_batch(
            list(validation_products),
            50,
            torch.Generator().manual_seed(2),
        )

    assert torch.equal(first["initial_inventory"], second["initial_inventory"])
    assert not torch.equal(
        training_seed_zero["initial_inventory"], training_seed_two["initial_inventory"]
    )


def test_pooled_diagnostics_use_all_seed_product_grid_rows() -> None:
    predictions = pd.DataFrame(
        {
            "training_seed": [0, 0, 1, 1, 2, 2],
            "product_id": ["p"] * 6,
            "state_index": [0, 1, 0, 1, 0, 1],
            "deployed_error": [1.0, -1.0, 2.0, -2.0, 3.0, -3.0],
            "raw_order": [2.0, 1.0, 2.0, 3.0, 2.0, 1.0],
            "deployed_order": [2.0, 1.0, 2.0, 2.5, 2.0, 1.0],
            "deployed_order_up_to_level": [11.0, 0.0, 12.0, 0.0, 13.0, 0.0],
            "deployed_omniscient_order_up_to_level": [10.0, 0.0, 10.0, 0.0, 10.0, 0.0],
        }
    )

    diagnostics = _pooled_prediction_diagnostics(predictions)

    assert diagnostics["pooled_deployed_rmse"] == pytest.approx(np.sqrt(28.0 / 6.0))
    assert diagnostics["pooled_raw_violation_rate"] == pytest.approx(1.0 / 3.0)
    assert diagnostics["pooled_deployed_violation_rate"] == pytest.approx(1.0 / 3.0)
    assert diagnostics[
        "pooled_deployed_zero_inventory_order_up_to_p90_absolute_error"
    ] == pytest.approx(np.quantile([1.0, 2.0, 3.0], 0.9))
    assert diagnostics[
        "pooled_deployed_zero_inventory_order_up_to_max_absolute_error"
    ] == 3.0


def test_training_seed_does_not_change_product_generation_seed() -> None:
    base = load_paper_lost_sales_config("configs/paper_lost_sales_medium_seed0_converged.yaml")
    changed = copy.deepcopy(base)
    changed["idp_benchmark"]["seed"] = 2

    assert base["product_generation"]["seed"] == changed["product_generation"]["seed"] == 0
