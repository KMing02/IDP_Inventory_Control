"""Single-item lost-sales inventory environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from inventory_control.envs.demand import DemandDistribution


@dataclass(frozen=True)
class LostSalesEnvConfig:
    """Configuration for the lost-sales environment."""

    horizon: int
    initial_inventory: int
    holding_cost: float
    lost_sales_cost: float
    unit_order_cost: float
    fixed_order_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.initial_inventory < 0:
            raise ValueError("initial_inventory must be non-negative")
        costs = [
            self.holding_cost,
            self.lost_sales_cost,
            self.unit_order_cost,
            self.fixed_order_cost,
        ]
        if any(cost < 0 for cost in costs):
            raise ValueError("cost parameters must be non-negative")


def _validate_order_quantity(action: float | int, name: str = "order quantity") -> int:
    action_value = float(action)
    if not np.isfinite(action_value):
        raise ValueError(f"{name} must be finite")
    if action_value < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(round(action_value))


def _reset_demand_if_supported(demand: DemandDistribution) -> None:
    reset = getattr(demand, "reset", None)
    if callable(reset):
        reset()


class LostSalesZeroLeadTimeEnv:
    """Gymnasium-like environment for zero-lead-time lost-sales inventory."""

    def __init__(
        self,
        config: LostSalesEnvConfig,
        demand: DemandDistribution,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.demand = demand
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.inventory = config.initial_inventory

    def reset(self, seed: int | None = None) -> int:
        """Reset the environment and return the initial inventory state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        _reset_demand_if_supported(self.demand)
        self.t = 0
        self.inventory = self.config.initial_inventory
        return self.inventory

    def step(self, action: float | int) -> tuple[int, float, bool, dict[str, Any]]:
        """Advance one period using ``action`` as the order quantity."""
        if self.t >= self.config.horizon:
            raise RuntimeError("cannot call step after episode is done")

        order_quantity = _validate_order_quantity(action)
        demand = self.demand.sample(self.rng, self.t)
        available_inventory = self.inventory + order_quantity
        sales = min(available_inventory, demand)
        lost_sales = max(demand - available_inventory, 0)
        ending_inventory = max(available_inventory - demand, 0)

        holding_cost = self.config.holding_cost * ending_inventory
        lost_sales_cost = self.config.lost_sales_cost * lost_sales
        order_cost = (
            self.config.unit_order_cost * order_quantity
            + self.config.fixed_order_cost * int(order_quantity > 0)
        )
        total_cost = holding_cost + lost_sales_cost + order_cost

        self.inventory = ending_inventory
        self.t += 1
        done = self.t >= self.config.horizon

        info = {
            "period": self.t - 1,
            "demand": demand,
            "sales": sales,
            "lost_sales": lost_sales,
            "ending_inventory": ending_inventory,
            "order_quantity": order_quantity,
            "holding_cost": float(holding_cost),
            "lost_sales_cost": float(lost_sales_cost),
            "order_cost": float(order_cost),
            "total_cost": float(total_cost),
        }
        return self.inventory, float(total_cost), done, info


LostSalesEnv = LostSalesZeroLeadTimeEnv


@dataclass(frozen=True)
class LostSalesLeadTimeEnvConfig(LostSalesEnvConfig):
    """Configuration for deterministic lead-time lost-sales inventory."""

    lead_time: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.lead_time < 1:
            raise ValueError("lead_time must be at least 1")


class LostSalesLeadTimeEnv:
    """Single-item lost-sales inventory with a deterministic order pipeline."""

    def __init__(
        self,
        config: LostSalesLeadTimeEnvConfig,
        demand: DemandDistribution,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.demand = demand
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.on_hand_inventory = config.initial_inventory
        self.pipeline = [0 for _ in range(config.lead_time)]

    def reset(self, seed: int | None = None) -> list[int]:
        """Reset the environment and return ``[on_hand, pipeline...]``."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        _reset_demand_if_supported(self.demand)
        self.t = 0
        self.on_hand_inventory = self.config.initial_inventory
        self.pipeline = [0 for _ in range(self.config.lead_time)]
        return self._state()

    def _state(self) -> list[int]:
        return [self.on_hand_inventory, *self.pipeline]

    def step(self, action: float | int) -> tuple[list[int], float, bool, dict[str, Any]]:
        """Advance one period with a scalar order quantity."""
        if self.t >= self.config.horizon:
            raise RuntimeError("cannot call step after episode is done")

        order_quantity = _validate_order_quantity(action)
        arrivals = self.pipeline[0]
        available_inventory = self.on_hand_inventory + arrivals
        demand = self.demand.sample(self.rng, self.t)
        sales = min(available_inventory, demand)
        lost_sales = max(demand - available_inventory, 0)
        ending_inventory = max(available_inventory - demand, 0)

        holding_cost = self.config.holding_cost * ending_inventory
        lost_sales_cost = self.config.lost_sales_cost * lost_sales
        order_cost = (
            self.config.unit_order_cost * order_quantity
            + self.config.fixed_order_cost * int(order_quantity > 0)
        )
        total_cost = holding_cost + lost_sales_cost + order_cost

        self.on_hand_inventory = ending_inventory
        self.pipeline = [*self.pipeline[1:], order_quantity]
        self.t += 1
        done = self.t >= self.config.horizon

        info = {
            "period": self.t - 1,
            "demand": demand,
            "sales": sales,
            "lost_sales": lost_sales,
            "ending_inventory": ending_inventory,
            "on_hand_inventory": ending_inventory,
            "arrivals": arrivals,
            "pipeline": list(self.pipeline),
            "order_quantity": order_quantity,
            "holding_cost": float(holding_cost),
            "lost_sales_cost": float(lost_sales_cost),
            "order_cost": float(order_cost),
            "total_cost": float(total_cost),
        }
        return self._state(), float(total_cost), done, info


@dataclass(frozen=True)
class DualSourcingEnvConfig:
    """Configuration for a dual-sourcing lost-sales environment."""

    horizon: int
    initial_inventory: int
    holding_cost: float
    lost_sales_cost: float
    regular_unit_order_cost: float
    expedited_unit_order_cost: float
    regular_lead_time: int
    expedited_lead_time: int
    regular_fixed_order_cost: float = 0.0
    expedited_fixed_order_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.initial_inventory < 0:
            raise ValueError("initial_inventory must be non-negative")
        if self.regular_lead_time < 1:
            raise ValueError("regular_lead_time must be at least 1")
        if self.expedited_lead_time < 0:
            raise ValueError("expedited_lead_time must be non-negative")
        costs = [
            self.holding_cost,
            self.lost_sales_cost,
            self.regular_unit_order_cost,
            self.expedited_unit_order_cost,
            self.regular_fixed_order_cost,
            self.expedited_fixed_order_cost,
        ]
        if any(cost < 0 for cost in costs):
            raise ValueError("cost parameters must be non-negative")


class DualSourcingEnv:
    """Single-item lost-sales environment with regular and expedited suppliers."""

    def __init__(
        self,
        config: DualSourcingEnvConfig,
        demand: DemandDistribution,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.demand = demand
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.on_hand_inventory = config.initial_inventory
        self.regular_pipeline = [0 for _ in range(config.regular_lead_time)]
        self.expedited_pipeline = [0 for _ in range(config.expedited_lead_time)]

    def reset(self, seed: int | None = None) -> list[int]:
        """Reset and return ``[on_hand, regular_pipeline..., expedited_pipeline...]``."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        _reset_demand_if_supported(self.demand)
        self.t = 0
        self.on_hand_inventory = self.config.initial_inventory
        self.regular_pipeline = [0 for _ in range(self.config.regular_lead_time)]
        self.expedited_pipeline = [0 for _ in range(self.config.expedited_lead_time)]
        return self._state()

    def _state(self) -> list[int]:
        return [self.on_hand_inventory, *self.regular_pipeline, *self.expedited_pipeline]

    def step(self, action: list[float] | tuple[float, float] | np.ndarray) -> tuple[list[int], float, bool, dict[str, Any]]:
        """Advance one period with ``[regular_order, expedited_order]``."""
        if self.t >= self.config.horizon:
            raise RuntimeError("cannot call step after episode is done")
        if len(action) != 2:
            raise ValueError("dual-sourcing action must have length 2")

        regular_order = _validate_order_quantity(action[0], "regular order quantity")
        expedited_order = _validate_order_quantity(action[1], "expedited order quantity")
        regular_arrivals = self.regular_pipeline[0]
        expedited_arrivals = self.expedited_pipeline[0] if self.expedited_pipeline else expedited_order
        available_inventory = self.on_hand_inventory + regular_arrivals + expedited_arrivals

        demand = self.demand.sample(self.rng, self.t)
        sales = min(available_inventory, demand)
        lost_sales = max(demand - available_inventory, 0)
        ending_inventory = max(available_inventory - demand, 0)

        holding_cost = self.config.holding_cost * ending_inventory
        lost_sales_cost = self.config.lost_sales_cost * lost_sales
        regular_order_cost = (
            self.config.regular_unit_order_cost * regular_order
            + self.config.regular_fixed_order_cost * int(regular_order > 0)
        )
        expedited_order_cost = (
            self.config.expedited_unit_order_cost * expedited_order
            + self.config.expedited_fixed_order_cost * int(expedited_order > 0)
        )
        order_cost = regular_order_cost + expedited_order_cost
        total_cost = holding_cost + lost_sales_cost + order_cost

        self.on_hand_inventory = ending_inventory
        self.regular_pipeline = [*self.regular_pipeline[1:], regular_order]
        if self.expedited_pipeline:
            self.expedited_pipeline = [*self.expedited_pipeline[1:], expedited_order]
        self.t += 1
        done = self.t >= self.config.horizon

        info = {
            "period": self.t - 1,
            "demand": demand,
            "sales": sales,
            "lost_sales": lost_sales,
            "ending_inventory": ending_inventory,
            "on_hand_inventory": ending_inventory,
            "regular_arrivals": regular_arrivals,
            "expedited_arrivals": expedited_arrivals,
            "regular_pipeline": list(self.regular_pipeline),
            "expedited_pipeline": list(self.expedited_pipeline),
            "regular_order_quantity": regular_order,
            "expedited_order_quantity": expedited_order,
            "order_quantity": regular_order + expedited_order,
            "holding_cost": float(holding_cost),
            "lost_sales_cost": float(lost_sales_cost),
            "regular_order_cost": float(regular_order_cost),
            "expedited_order_cost": float(expedited_order_cost),
            "order_cost": float(order_cost),
            "total_cost": float(total_cost),
        }
        return self._state(), float(total_cost), done, info
