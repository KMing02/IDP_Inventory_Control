"""Tune the base-stock level by grid search."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inventory_control.evaluation.tuning import base_stock_grid_search
from run_baselines import load_config, make_env_factory


def main() -> None:
    """Run base-stock grid search and save full results."""
    config = load_config()
    tuning_config = config.get("tuning", {}).get("base_stock", {})

    horizon = int(config["horizon"])
    n_episodes = int(config["n_episodes"])
    seed = int(config["seed"])
    min_stock_level = int(tuning_config.get("min_stock_level", 0))
    max_stock_level = int(tuning_config.get("max_stock_level", 30))

    results = base_stock_grid_search(
        env_factory=make_env_factory(config),
        horizon=horizon,
        n_episodes=n_episodes,
        seed=seed,
        min_stock_level=min_stock_level,
        max_stock_level=max_stock_level,
    )

    output_path = Path("outputs/results/base_stock_grid.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    best_row = results.loc[results["is_best"]].iloc[0]
    print(results[["S", "mean_total_cost", "std_total_cost", "is_best"]].to_string(index=False))
    print(f"\nBest S: {int(best_row['S'])}")
    print(f"Best mean total cost: {best_row['mean_total_cost']:.6f}")


if __name__ == "__main__":
    main()
