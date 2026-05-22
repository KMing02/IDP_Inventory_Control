"""Episode runners and policy evaluation loops."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from inventory_control.envs.lost_sales import LostSalesEnv
from inventory_control.evaluation.metrics import compute_summary_metrics
from inventory_control.policies.base import Policy


def run_episode(env: Any, policy: Policy, horizon: int, seed: int | None = None) -> pd.DataFrame:
    """Run one episode and return a per-period trace."""
    state = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset(seed=seed)
    policy_rng = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []
    for t in range(horizon):
        action = policy.act(state, t, rng=policy_rng)
        next_state, cost, done, info = env.step(action)
        rows.append({"state": state, "next_state": next_state, "cost": cost, **info})
        state = next_state
        if done:
            break
    return pd.DataFrame(rows)


def evaluate_policy(
    env_factory: Callable[[int | None], Any],
    policy: Policy,
    horizon: int,
    n_episodes: int,
    seed: int | None = None,
) -> tuple[dict[str, float], list[pd.DataFrame]]:
    """Evaluate a policy over many independent episodes."""
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")

    episodes: list[pd.DataFrame] = []
    base_seed = seed if seed is not None else int(np.random.SeedSequence().entropy)
    for episode_idx in range(n_episodes):
        episode_seed = base_seed + episode_idx
        env = env_factory(episode_seed)
        episodes.append(run_episode(env, policy, horizon=horizon, seed=episode_seed))

    metrics = compute_summary_metrics(episodes)
    return metrics, episodes
