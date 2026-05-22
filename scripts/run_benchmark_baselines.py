"""Run heuristic baselines across benchmark scenarios."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from inventory_control.evaluation.runner import evaluate_policy
from inventory_control.evaluation.tuning import (
    base_stock_grid_search,
    inventory_position_base_stock_grid_search,
)
from inventory_control.policies.classical import DualSourcingHeuristicPolicy
from inventory_control.scenarios import build_env_factory


def load_benchmark_config(path: str | Path = "configs/benchmark_scenarios.yaml") -> dict[str, Any]:
    """Load benchmark scenario configuration."""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _result_row(
    scenario: dict[str, Any],
    metadata: dict[str, Any],
    policy_name: str,
    tuned_parameters: dict[str, Any],
    metrics: dict[str, float],
) -> dict[str, Any]:
    columns = {
        "scenario_name": scenario["name"],
        "environment_type": metadata["environment_type"],
        "demand_type": metadata["demand_type"],
        "cost_regime": metadata["cost_regime"],
        "policy_name": policy_name,
        "tuned_parameters": json.dumps(tuned_parameters, sort_keys=True),
    }
    columns.update(metrics)
    return columns


def main() -> None:
    """Evaluate benchmark baselines and save combined CSV results."""
    config = load_benchmark_config()
    horizon = int(config["horizon"])
    n_episodes = int(config["n_episodes"])
    seed = int(config["seed"])
    tuning = config.get("tuning", {})
    min_stock = int(tuning.get("min_stock_level", 0))
    max_stock = int(tuning.get("max_stock_level", 30))

    rows: list[dict[str, Any]] = []
    for scenario in config["scenarios"]:
        env_factory, metadata = build_env_factory(config, scenario)
        environment_type = metadata["environment_type"]

        if environment_type == "lost_sales_zero_lead_time":
            grid = base_stock_grid_search(
                env_factory,
                horizon=horizon,
                n_episodes=n_episodes,
                seed=seed,
                min_stock_level=min_stock,
                max_stock_level=max_stock,
            )
            best = grid.loc[grid["is_best"]].iloc[0]
            metrics = best.drop(labels=["S", "is_best"]).to_dict()
            rows.append(
                _result_row(
                    scenario,
                    metadata,
                    "TunedBaseStockPolicy",
                    {"S": int(best["S"])},
                    metrics,
                )
            )

        elif environment_type == "lost_sales_lead_time":
            grid = inventory_position_base_stock_grid_search(
                env_factory,
                horizon=horizon,
                n_episodes=n_episodes,
                seed=seed,
                min_stock_level=min_stock,
                max_stock_level=max_stock,
            )
            best = grid.loc[grid["is_best"]].iloc[0]
            metrics = best.drop(labels=["S", "is_best"]).to_dict()
            rows.append(
                _result_row(
                    scenario,
                    metadata,
                    "TunedInventoryPositionBaseStockPolicy",
                    {"S": int(best["S"])},
                    metrics,
                )
            )

        elif environment_type == "dual_sourcing":
            heuristic = scenario.get("heuristic", {})
            policy = DualSourcingHeuristicPolicy(
                S_regular=int(heuristic.get("S_regular", 16)),
                S_expedited=int(heuristic.get("S_expedited", 5)),
                regular_lead_time=int(scenario["regular_lead_time"]),
                expedited_lead_time=int(scenario["expedited_lead_time"]),
            )
            metrics, _ = evaluate_policy(env_factory, policy, horizon, n_episodes, seed=seed)
            rows.append(
                _result_row(
                    scenario,
                    metadata,
                    "DualSourcingHeuristicPolicy",
                    {
                        "S_regular": policy.S_regular,
                        "S_expedited": policy.S_expedited,
                    },
                    metrics,
                )
            )

        else:
            raise ValueError(f"unknown environment_type: {environment_type}")

    results = pd.DataFrame(rows)
    required_columns = [
        "scenario_name",
        "environment_type",
        "demand_type",
        "cost_regime",
        "policy_name",
        "tuned_parameters",
        "mean_total_cost",
        "std_total_cost",
        "mean_cost_per_period",
        "mean_service_level",
        "mean_lost_sales_rate",
    ]
    remaining_columns = [column for column in results.columns if column not in required_columns]
    results = results[required_columns + remaining_columns]

    output_path = Path("outputs/results/benchmark_baselines.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(results[required_columns].to_string(index=False))


if __name__ == "__main__":
    main()
