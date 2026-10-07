from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inventory_control.analysis import compare_runs, paired_difference, regret_pct, summarize_run
from inventory_control.classical_windows import evaluate_classical_windows

POLICY = "MonoKANSoftQuantileTargetStock"


def test_regret_skips_products_where_the_omniscient_policy_loses_money() -> None:
    omniscient = np.array([100.0, 200.0, -50.0, 0.0])
    policy = np.array([90.0, 200.0, -500.0, -10.0])

    assert regret_pct(policy, omniscient) == pytest.approx(5.0)


def test_paired_difference_reports_the_average_and_the_typical_product() -> None:
    rng = np.random.default_rng(0)
    second = rng.uniform(100.0, 10_000.0, size=200)
    first = second + rng.normal(1.0, 0.5, size=200)

    result = paired_difference(first, second)

    assert result["mean"] == pytest.approx(np.mean(first - second))
    assert result["ci95"][0] < result["mean"] < result["ci95"][1]
    assert result["p_paired_t"] < 1e-6
    assert result["p_wilcoxon"] < 1e-6
    assert result["first_ahead_pct"] > 90.0
    assert result["products"] == 200


def test_paired_difference_of_identical_rewards_is_neutral() -> None:
    rewards = np.linspace(1.0, 2.0, 10)

    result = paired_difference(rewards, rewards)

    assert result["mean"] == 0.0
    assert result["ci95"] == [0.0, 0.0]
    assert result["p_paired_t"] == 1.0
    assert result["first_ahead_pct"] == 0.0


def _write_run(root: Path, offsets: dict[str, float]) -> None:
    omniscient = np.array([100.0, 200.0, 300.0, -20.0])
    for seed, offset in offsets.items():
        frames = []
        for name, rewards in ((POLICY, omniscient - offset), ("OmniscientOptimal", omniscient)):
            frames.append(pd.DataFrame({
                "product_id": [f"p{index}" for index in range(4)],
                "evaluation_seed": [40_000 + index for index in range(4)],
                "policy_name": name,
                "mean_reward_per_period": rewards,
            }))
        (root / seed).mkdir(parents=True)
        pd.concat(frames).to_csv(root / seed / "product_metrics.csv", index=False)


def test_runs_are_summarized_per_seed_and_compared_product_by_product(tmp_path: Path) -> None:
    _write_run(tmp_path / "better", {"seed_2025": 1.0, "seed_2026": 3.0})
    _write_run(tmp_path / "worse", {"seed_2025": 6.0, "seed_2026": 6.0})

    summary = summarize_run(tmp_path / "better", POLICY)
    comparison = compare_runs(tmp_path / "better", tmp_path / "worse", POLICY)

    regret_2025 = np.mean([1 / 100, 1 / 200, 1 / 300]) * 100
    assert summary["per_seed"]["seed_2025"]["regret_pct"] == pytest.approx(regret_2025)
    assert summary["regret_pct"] == pytest.approx(np.mean([regret_2025, 3 * regret_2025]))
    assert comparison["mean"] == pytest.approx(4.0)
    assert comparison["first_ahead_pct"] == 100.0


def test_summary_rejects_a_run_without_the_policy(tmp_path: Path) -> None:
    _write_run(tmp_path / "run", {"seed_2025": 1.0})

    with pytest.raises(ValueError, match="no rows for policy"):
        summarize_run(tmp_path / "run", "SomeOtherPolicy")


def test_classical_windows_cover_every_rule_on_the_shared_products() -> None:
    results = evaluate_classical_windows("configs/paper_lost_sales_debug.yaml")

    assert set(results) == {"omniscient"} | {
        f"{estimator}_{window}" for estimator in ("percentile", "gamma") for window in ("fixed", "rolling", "expanding")
    }
    assert results["omniscient"]["regret_pct"] == 0.0
    assert all(np.isfinite(row["reward"]) for row in results.values())
