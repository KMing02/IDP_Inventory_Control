"""Classical base-stock rules under the three information windows, on the report's evaluation path.

A learned policy may only be compared with a classical rule that sees the same data, so each rule is evaluated with
one fixed estimate from the first 32 days (the paper's benchmarks), the last 32 days re-estimated every day, and all
demand observed so far. Products and demand paths are those of the learned policies' validation runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from inventory_control.analysis import regret_pct
from inventory_control.base_stock_output_experiment import _compact_settings, _evaluate_baseline
from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.paper_lost_sales import WindowedBaseStockPolicy, omniscient_gamma_base_stock_policy
from inventory_control.products import PaperLostSalesProductGenerator

CONFIG = Path("configs/paper_lost_sales_rolling_direct_target_dev_1k_normalized.yaml")


def evaluate_classical_windows(
    config_path: str | Path = CONFIG,
    estimators=WindowedBaseStockPolicy.ESTIMATORS,
    windows=WindowedBaseStockPolicy.WINDOWS,
) -> dict[str, dict[str, Any]]:
    """Reward per period and regret of every (estimator, window) rule, plus the omniscient reference."""
    config = load_paper_lost_sales_config(config_path)
    settings = _compact_settings(config)
    products = PaperLostSalesProductGenerator(config).generate(settings["validation_products"], split="validation")

    def rewards(name, factory) -> np.ndarray:
        return np.array([
            _evaluate_baseline(product, name, settings, index, factory)["mean_reward_per_period"]
            for index, product in enumerate(products)
        ])

    omniscient = rewards("OmniscientOptimal", omniscient_gamma_base_stock_policy)
    results = {"omniscient": {"reward": float(omniscient.mean()), "regret_pct": 0.0, "products": len(products)}}
    for estimator in estimators:
        for window in windows:
            policy_rewards = rewards(
                f"{estimator}_{window}",
                lambda product, e=estimator, w=window: WindowedBaseStockPolicy(product, estimator=e, window=w),
            )
            results[f"{estimator}_{window}"] = {
                "estimator": estimator,
                "window": window,
                "reward": float(policy_rewards.mean()),
                "regret_pct": regret_pct(policy_rewards, omniscient),
            }
    return results
