"""Evidence utilities for KAN versus MonoKAN residual policy analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


KAN_MODEL = "KANResidualOrderUpTo"
MONOKAN_MODEL = "MonoKANResidualOrderUpTo"


@dataclass(frozen=True)
class EquivalenceResult:
    """Practical/statistical classification for a paired reward interval."""

    classification: Literal["statistically_different", "practically_equivalent", "statistically_inconclusive"]
    margin_abs: float
    margin_pct_of_kan: float
    ci_low: float
    ci_high: float
    statistically_excludes_zero: bool


def kan_monokan_product_pairs(operational: pd.DataFrame) -> pd.DataFrame:
    """Return product-level paired rows with delta = MonoKAN - KAN."""
    required = {"seed", "shift", "product_id", "evaluation_seed", "policy_name", "mean_reward_per_period"}
    missing = required.difference(operational.columns)
    if missing:
        raise ValueError(f"operational data is missing required columns: {sorted(missing)}")

    selected = operational.loc[operational["policy_name"].isin({KAN_MODEL, MONOKAN_MODEL})].copy()
    key = ["seed", "shift", "product_id", "evaluation_seed", "policy_name"]
    duplicates = selected.loc[selected.duplicated(key, keep=False), key]
    if not duplicates.empty:
        raise ValueError(f"duplicate KAN/MonoKAN product-policy rows: {duplicates.head().to_dict('records')}")

    index = ["seed", "shift", "product_id", "evaluation_seed"]
    pivot = selected.pivot(index=index, columns="policy_name", values="mean_reward_per_period")
    missing_pairs = pivot[[KAN_MODEL, MONOKAN_MODEL]].isna().any(axis=1)
    if missing_pairs.any():
        raise ValueError(f"missing KAN/MonoKAN product pairs: {int(missing_pairs.sum())}")

    pairs = pivot.reset_index()
    pairs["kan_reward"] = pairs[KAN_MODEL].astype(float)
    pairs["monokan_reward"] = pairs[MONOKAN_MODEL].astype(float)
    pairs["delta_reward"] = pairs["monokan_reward"] - pairs["kan_reward"]
    pairs["delta_pct_of_kan"] = 100.0 * pairs["delta_reward"] / pairs["kan_reward"].replace(0.0, np.nan)
    return pairs[["seed", "shift", "product_id", "evaluation_seed", "kan_reward", "monokan_reward", "delta_reward", "delta_pct_of_kan"]]


def seed_shift_deltas(pairs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate paired product deltas to the seed-shift level."""
    return (
        pairs.groupby(["seed", "shift"], as_index=False)
        .agg(
            mean_delta=("delta_reward", "mean"),
            median_delta=("delta_reward", "median"),
            bottom_decile_delta=("delta_reward", lambda values: float(np.quantile(values, 0.1))),
            worst_product_delta=("delta_reward", "min"),
            mean_kan_reward=("kan_reward", "mean"),
            products=("delta_reward", "size"),
        )
    )


def fixed_shift_equal_weight_mean(cell_deltas: pd.DataFrame, value_column: str = "mean_delta") -> float:
    """Average seed-shift cells with equal shift weight."""
    per_shift = cell_deltas.groupby("shift", as_index=False)[value_column].mean()
    return float(per_shift[value_column].mean())


def hierarchical_paired_bootstrap(
    pairs: pd.DataFrame,
    *,
    samples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap fixed shifts by resampling seeds and products within seed-shift cells."""
    rng = np.random.default_rng(seed)
    seeds = np.asarray(sorted(pairs["seed"].unique()))
    shifts = list(sorted(pairs["shift"].unique()))
    grouped = {
        (int(seed_value), str(shift)): group["delta_reward"].to_numpy(dtype=float)
        for (seed_value, shift), group in pairs.groupby(["seed", "shift"], sort=False)
    }
    estimates = np.empty(samples, dtype=float)
    for sample_index in range(samples):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        shift_means = []
        for shift in shifts:
            seed_means = []
            for seed_value in sampled_seeds:
                values = grouped[(int(seed_value), shift)]
                seed_means.append(float(rng.choice(values, size=len(values), replace=True).mean()))
            shift_means.append(float(np.mean(seed_means)))
        estimates[sample_index] = float(np.mean(shift_means))
    return (
        float(np.mean(estimates)),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def per_shift_hierarchical_bootstrap(
    pairs: pd.DataFrame,
    *,
    samples: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Bootstrap each fixed shift separately by resampling seeds and products."""
    rows = []
    for offset, (shift, shift_rows) in enumerate(pairs.groupby("shift", sort=True)):
        mean, low, high = hierarchical_paired_bootstrap(shift_rows, samples=samples, seed=seed + offset)
        rows.append({"shift": shift, "bootstrap_mean_delta": mean, "ci95_low": low, "ci95_high": high})
    return pd.DataFrame(rows)


def classify_equivalence(
    *,
    ci_low: float,
    ci_high: float,
    margin_abs: float,
    mean_kan_reward: float,
) -> EquivalenceResult:
    """Classify KAN/MonoKAN difference against a predeclared practical margin."""
    margin_pct = 100.0 * float(margin_abs) / float(mean_kan_reward)
    statistically_excludes_zero = bool(ci_low > 0.0 or ci_high < 0.0)
    if ci_low >= -margin_abs and ci_high <= margin_abs:
        classification: Literal["statistically_different", "practically_equivalent", "statistically_inconclusive"] = "practically_equivalent"
    elif statistically_excludes_zero:
        classification = "statistically_different"
    else:
        classification = "statistically_inconclusive"
    return EquivalenceResult(
        classification=classification,
        margin_abs=float(margin_abs),
        margin_pct_of_kan=margin_pct,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        statistically_excludes_zero=statistically_excludes_zero,
    )
