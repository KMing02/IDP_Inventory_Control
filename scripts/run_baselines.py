"""Evaluate classical baseline policies from the default config."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from inventory_control.envs.demand import (
    DeterministicDemand,
    NegativeBinomialDemand,
    NormalDemand,
    PoissonDemand,
    RegimeSwitchingPoissonDemand,
    SeasonalPoissonDemand,
)
from inventory_control.envs.lost_sales import LostSalesEnv, LostSalesEnvConfig
from inventory_control.evaluation.runner import evaluate_policy
from inventory_control.policies.classical import BaseStockPolicy, ConstantOrderPolicy, RandomPolicy


def load_config(path: str | Path = "configs/lost_sales_default.yaml") -> dict[str, Any]:
    """Load a YAML experiment config."""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def make_demand(config: dict[str, Any]):
    """Create a demand distribution from config."""
    demand_type = config["type"]
    if demand_type == "poisson":
        return PoissonDemand(lam=float(config["lam"]))
    if demand_type == "normal":
        return NormalDemand(mean=float(config["mean"]), std=float(config["std"]))
    if demand_type == "negative_binomial":
        if "n" in config and "p" in config:
            return NegativeBinomialDemand(n=float(config["n"]), p=float(config["p"]))
        return NegativeBinomialDemand(
            mean=float(config["mean"]),
            dispersion=float(config["dispersion"]),
        )
    if demand_type == "seasonal_poisson":
        return SeasonalPoissonDemand(
            base_lambda=float(config["base_lambda"]),
            amplitude=float(config["amplitude"]),
            period=int(config["period"]),
        )
    if demand_type == "regime_switching_poisson":
        return RegimeSwitchingPoissonDemand(
            low_lambda=float(config["low_lambda"]),
            high_lambda=float(config["high_lambda"]),
            switch_probability=float(config["switch_probability"]),
            initial_regime=str(config.get("initial_regime", "low")),
        )
    if demand_type == "deterministic":
        return DeterministicDemand(sequence=list(config["sequence"]))
    raise ValueError(f"unknown demand type: {demand_type}")


def make_env_factory(config: dict[str, Any]):
    """Create an environment factory compatible with evaluation."""
    env_config = LostSalesEnvConfig(
        horizon=int(config["horizon"]),
        initial_inventory=int(config["environment"]["initial_inventory"]),
        holding_cost=float(config["environment"]["holding_cost"]),
        lost_sales_cost=float(config["environment"]["lost_sales_cost"]),
        unit_order_cost=float(config["environment"]["unit_order_cost"]),
        fixed_order_cost=float(config["environment"].get("fixed_order_cost", 0.0)),
    )

    def factory(seed: int | None = None) -> LostSalesEnv:
        return LostSalesEnv(env_config, make_demand(config["demand"]), seed=seed)

    return factory


def main() -> None:
    """Run baseline evaluations and write a CSV summary."""
    config = load_config()
    horizon = int(config["horizon"])
    n_episodes = int(config["n_episodes"])
    seed = int(config["seed"])
    env_factory = make_env_factory(config)

    policies = [
        BaseStockPolicy(S=int(config["policies"]["base_stock"]["S"])),
        ConstantOrderPolicy(quantity=int(config["policies"]["constant_order"]["quantity"])),
        RandomPolicy(max_order=int(config["policies"]["random"]["max_order"])),
    ]

    rows = []
    for policy in policies:
        metrics, _ = evaluate_policy(
            env_factory=env_factory,
            policy=policy,
            horizon=horizon,
            n_episodes=n_episodes,
            seed=seed,
        )
        rows.append({"policy": policy.name, **metrics})

    results = pd.DataFrame(rows)
    output_path = Path("outputs/results/baseline_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
