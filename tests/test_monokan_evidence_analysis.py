from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inventory_control.evaluation.monokan_evidence import (
    KAN_MODEL,
    MONOKAN_MODEL,
    classify_equivalence,
    fixed_shift_equal_weight_mean,
    hierarchical_paired_bootstrap,
    kan_monokan_product_pairs,
    seed_shift_deltas,
)


def _operational_rows() -> pd.DataFrame:
    rows = []
    for seed in [1, 2]:
        for shift in ["a", "b"]:
            for product in ["p1", "p2"]:
                base = 100.0 + seed
                delta = 1.0 if shift == "a" else -0.5
                rows.extend(
                    [
                        {
                            "seed": seed,
                            "shift": shift,
                            "product_id": product,
                            "evaluation_seed": 10 + seed,
                            "policy_name": KAN_MODEL,
                            "mean_reward_per_period": base,
                        },
                        {
                            "seed": seed,
                            "shift": shift,
                            "product_id": product,
                            "evaluation_seed": 10 + seed,
                            "policy_name": MONOKAN_MODEL,
                            "mean_reward_per_period": base + delta,
                        },
                    ]
                )
    return pd.DataFrame(rows)


def test_kan_monokan_pairs_and_delta_sign() -> None:
    pairs = kan_monokan_product_pairs(_operational_rows())

    assert len(pairs) == 8
    assert set(pairs["delta_reward"]) == {1.0, -0.5}
    assert {"kan_reward", "monokan_reward", "delta_pct_of_kan"}.issubset(pairs.columns)


def test_kan_monokan_pairs_reject_duplicates() -> None:
    rows = pd.concat([_operational_rows(), _operational_rows().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        kan_monokan_product_pairs(rows)


def test_kan_monokan_pairs_reject_missing_pair() -> None:
    rows = _operational_rows()
    rows = rows.drop(rows[(rows["policy_name"] == MONOKAN_MODEL) & (rows["product_id"] == "p1")].index[0])

    with pytest.raises(ValueError, match="missing"):
        kan_monokan_product_pairs(rows)


def test_fixed_shift_equal_weighting() -> None:
    pairs = kan_monokan_product_pairs(_operational_rows())
    cells = seed_shift_deltas(pairs)

    assert fixed_shift_equal_weight_mean(cells) == pytest.approx(0.25)


def test_hierarchical_bootstrap_is_deterministic() -> None:
    pairs = kan_monokan_product_pairs(_operational_rows())

    first = hierarchical_paired_bootstrap(pairs, samples=100, seed=123)
    second = hierarchical_paired_bootstrap(pairs, samples=100, seed=123)

    assert first == second
    assert np.isfinite(first).all()


def test_equivalence_classification() -> None:
    equivalent = classify_equivalence(ci_low=-1.0, ci_high=1.0, margin_abs=2.0, mean_kan_reward=100.0)
    equivalent_but_different = classify_equivalence(ci_low=-1.0, ci_high=-0.2, margin_abs=2.0, mean_kan_reward=100.0)
    different = classify_equivalence(ci_low=2.5, ci_high=3.0, margin_abs=2.0, mean_kan_reward=100.0)
    inconclusive = classify_equivalence(ci_low=-3.0, ci_high=1.0, margin_abs=2.0, mean_kan_reward=100.0)

    assert equivalent.classification == "practically_equivalent"
    assert equivalent.statistically_excludes_zero is False
    assert equivalent_but_different.classification == "practically_equivalent"
    assert equivalent_but_different.statistically_excludes_zero is True
    assert different.classification == "statistically_different"
    assert inconclusive.classification == "statistically_inconclusive"
