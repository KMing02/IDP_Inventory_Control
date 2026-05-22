"""Generate plots for classical baseline policies."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs/.cache").resolve()))

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inventory_control.evaluation.runner import run_episode
from inventory_control.plotting.plots import (
    plot_cost_per_period,
    plot_inventory_trajectory,
    plot_order_quantity_trajectory,
    plot_policy_curve,
)
from inventory_control.policies.classical import BaseStockPolicy, ConstantOrderPolicy, RandomPolicy
from run_baselines import load_config, make_env_factory


def main() -> None:
    """Create and save baseline diagnostic plots."""
    config = load_config()
    horizon = int(config["horizon"])
    seed = int(config["seed"])
    env_factory = make_env_factory(config)
    output_dir = Path("outputs/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    policies = [
        BaseStockPolicy(S=int(config["policies"]["base_stock"]["S"])),
        ConstantOrderPolicy(quantity=int(config["policies"]["constant_order"]["quantity"])),
        RandomPolicy(max_order=int(config["policies"]["random"]["max_order"])),
    ]

    max_inventory = max(
        int(config["environment"]["initial_inventory"]),
        int(config["policies"]["base_stock"]["S"]),
        int(config["policies"]["random"]["max_order"]),
    ) * 2

    for idx, policy in enumerate(policies):
        slug = policy.name.lower()
        trace = run_episode(env_factory(seed + idx), policy, horizon=horizon, seed=seed + idx)
        plot_cost_per_period(trace, output_dir / f"{slug}_cost_per_period.png")
        plot_inventory_trajectory(trace, output_dir / f"{slug}_inventory.png")
        plot_order_quantity_trajectory(trace, output_dir / f"{slug}_orders.png")
        plot_policy_curve(policy, max_inventory=max_inventory, save_path=output_dir / f"{slug}_policy_curve.png")
        plt.close("all")


if __name__ == "__main__":
    main()
