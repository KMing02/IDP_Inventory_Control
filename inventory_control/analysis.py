"""Regret and paired comparisons over evaluation results.

Every policy in this project is evaluated on the same products and the same demand paths, so policies are compared
product by product: the large product-to-product spread in profit cancels and only the policy difference remains.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

OMNISCIENT_POLICY = "OmniscientOptimal"
PAIRING_KEY = ["product_id", "evaluation_seed"]


def regret_pct(policy_rewards, omniscient_rewards) -> float:
    """Mean per-product shortfall against the omniscient base-stock policy, in percent.

    Products on which the omniscient policy itself earns nothing or loses money have no meaningful relative
    shortfall and are excluded (21 of the 300 validation products); this reproduces the classical baselines'
    1.35% and 1.76%.
    """
    policy = np.asarray(policy_rewards, dtype=float)
    omniscient = np.asarray(omniscient_rewards, dtype=float)
    counted = omniscient > 0
    return float(np.mean((omniscient[counted] - policy[counted]) / omniscient[counted]) * 100)


def paired_difference(first, second) -> dict[str, Any]:
    """Paired comparison of per-product rewards, ``first - second``.

    The paired t-test and its 95% interval speak about the average difference, which large products dominate; the
    Wilcoxon signed-rank test and ``first_ahead_pct`` speak about the typical product. Both are reported because
    they can disagree.
    """
    difference = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    if difference.size < 2:
        raise ValueError("a paired comparison needs at least two products")
    if np.allclose(difference, difference[0]):
        # No spread: the interval collapses to the mean, and the tests have nothing to test.
        low = high = float(difference.mean())
        p_t = p_wilcoxon = 1.0 if np.allclose(difference, 0.0) else 0.0
    else:
        low, high = stats.t.interval(0.95, difference.size - 1, loc=difference.mean(), scale=stats.sem(difference))
        p_t = float(stats.ttest_1samp(difference, 0.0).pvalue)
        p_wilcoxon = float(stats.wilcoxon(difference).pvalue)
    return {
        "mean": float(difference.mean()),
        "ci95": [float(low), float(high)],
        "p_paired_t": p_t,
        "p_wilcoxon": p_wilcoxon,
        "first_ahead_pct": float(np.mean(difference > 0) * 100),
        "products": int(difference.size),
    }


def load_run(run_root: str | Path, policy: str) -> dict[str, pd.DataFrame]:
    """Per-seed rewards of ``policy`` and of the omniscient policy, read from ``seed_*/product_metrics.csv``.

    Each frame is indexed by product and evaluation path and has the columns ``policy`` and ``omniscient``.
    """
    runs: dict[str, pd.DataFrame] = {}
    for path in sorted(Path(run_root).glob("seed_*/product_metrics.csv")):
        metrics = pd.read_csv(path)
        rewards = {}
        for name in (policy, OMNISCIENT_POLICY):
            rows = metrics[metrics["policy_name"] == name]
            if rows.empty:
                raise ValueError(f"{path} has no rows for policy {name!r}")
            rewards[name] = rows.set_index(PAIRING_KEY)["mean_reward_per_period"]
        policy_rewards, omniscient = rewards[policy].align(rewards[OMNISCIENT_POLICY], join="inner")
        runs[path.parent.name] = pd.DataFrame({"policy": policy_rewards, "omniscient": omniscient})
    if not runs:
        raise FileNotFoundError(f"no seed_*/product_metrics.csv under {run_root}")
    return runs


def summarize_run(run_root: str | Path, policy: str) -> dict[str, Any]:
    """Reward and regret per seed and averaged over seeds."""
    per_seed = {
        seed: {
            "reward": float(frame["policy"].mean()),
            "regret_pct": regret_pct(frame["policy"], frame["omniscient"]),
            "products": int(len(frame)),
        }
        for seed, frame in load_run(run_root, policy).items()
    }
    rewards = [row["reward"] for row in per_seed.values()]
    return {
        "per_seed": per_seed,
        "reward": float(np.mean(rewards)),
        "reward_seed_std": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
        "regret_pct": float(np.mean([row["regret_pct"] for row in per_seed.values()])),
    }


def seed_averaged_rewards(run_root: str | Path, policy: str) -> pd.Series:
    """Per-product reward of ``policy``, averaged over the seeds of a run."""
    frames = [frame["policy"].groupby(level="product_id").mean() for frame in load_run(run_root, policy).values()]
    return pd.concat(frames, axis=1).mean(axis=1)


def compare_runs(
    first_root: str | Path,
    second_root: str | Path,
    policy: str,
    second_policy: str | None = None,
) -> dict[str, Any]:
    """Paired ``first - second`` comparison over shared products, each run's rewards averaged over seeds first."""
    first = seed_averaged_rewards(first_root, policy)
    second = seed_averaged_rewards(second_root, second_policy or policy)
    first, second = first.align(second, join="inner")
    return paired_difference(first.to_numpy(), second.to_numpy())
