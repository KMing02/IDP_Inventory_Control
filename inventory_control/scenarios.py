"""Benchmark scenario construction utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from inventory_control.envs.demand import (
    DemandDistribution,
    DeterministicDemand,
    NegativeBinomialDemand,
    NormalDemand,
    PoissonDemand,
    RegimeSwitchingPoissonDemand,
    SeasonalPoissonDemand,
)
from inventory_control.envs.lost_sales import (
    DualSourcingEnv,
    DualSourcingEnvConfig,
    LostSalesLeadTimeEnv,
    LostSalesLeadTimeEnvConfig,
    LostSalesZeroLeadTimeEnv,
    LostSalesEnvConfig,
)


def make_demand(config: dict[str, Any]) -> DemandDistribution:
    """Create a demand process from a config dictionary."""
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


def build_env_factory(
    benchmark_config: dict[str, Any],
    scenario: dict[str, Any],
) -> tuple[Callable[[int | None], Any], dict[str, Any]]:
    """Build an environment factory and metadata for one benchmark scenario."""
    demand_name = scenario["demand"]
    cost_name = scenario["cost"]
    demand_config = benchmark_config["demand_processes"][demand_name]
    cost_config = benchmark_config["cost_regimes"][cost_name]
    environment_type = scenario["environment_type"]
    horizon = int(benchmark_config["horizon"])
    initial_inventory = int(scenario.get("initial_inventory", benchmark_config["initial_inventory"]))

    if environment_type == "lost_sales_zero_lead_time":
        env_config = LostSalesEnvConfig(
            horizon=horizon,
            initial_inventory=initial_inventory,
            holding_cost=float(cost_config["holding_cost"]),
            lost_sales_cost=float(cost_config["lost_sales_cost"]),
            unit_order_cost=float(cost_config["unit_order_cost"]),
            fixed_order_cost=float(cost_config.get("fixed_order_cost", 0.0)),
        )

        def factory(seed: int | None = None) -> LostSalesZeroLeadTimeEnv:
            return LostSalesZeroLeadTimeEnv(env_config, make_demand(demand_config), seed=seed)

    elif environment_type == "lost_sales_lead_time":
        env_config = LostSalesLeadTimeEnvConfig(
            horizon=horizon,
            initial_inventory=initial_inventory,
            holding_cost=float(cost_config["holding_cost"]),
            lost_sales_cost=float(cost_config["lost_sales_cost"]),
            unit_order_cost=float(cost_config["unit_order_cost"]),
            fixed_order_cost=float(cost_config.get("fixed_order_cost", 0.0)),
            lead_time=int(scenario["lead_time"]),
        )

        def factory(seed: int | None = None) -> LostSalesLeadTimeEnv:
            return LostSalesLeadTimeEnv(env_config, make_demand(demand_config), seed=seed)

    elif environment_type == "dual_sourcing":
        env_config = DualSourcingEnvConfig(
            horizon=horizon,
            initial_inventory=initial_inventory,
            holding_cost=float(cost_config["holding_cost"]),
            lost_sales_cost=float(cost_config["lost_sales_cost"]),
            regular_unit_order_cost=float(cost_config["regular_unit_order_cost"]),
            expedited_unit_order_cost=float(cost_config["expedited_unit_order_cost"]),
            regular_fixed_order_cost=float(cost_config.get("regular_fixed_order_cost", 0.0)),
            expedited_fixed_order_cost=float(cost_config.get("expedited_fixed_order_cost", 0.0)),
            regular_lead_time=int(scenario["regular_lead_time"]),
            expedited_lead_time=int(scenario["expedited_lead_time"]),
        )

        def factory(seed: int | None = None) -> DualSourcingEnv:
            return DualSourcingEnv(env_config, make_demand(demand_config), seed=seed)

    else:
        raise ValueError(f"unknown environment_type: {environment_type}")

    metadata = {
        "environment_type": environment_type,
        "demand_name": demand_name,
        "demand_type": demand_config["type"],
        "cost_regime": cost_name,
    }
    return factory, metadata
