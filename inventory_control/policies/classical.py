"""Classical baseline policies for lost-sales inventory control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inventory_control.policies.base import State


def _on_hand_inventory(state: State) -> int:
    if isinstance(state, np.ndarray):
        return int(state[0]) if state.ndim > 0 else int(state.item())
    if isinstance(state, list | tuple):
        return int(state[0])
    return int(state)


def _state_values(state: State) -> list[int]:
    if isinstance(state, np.ndarray):
        return [int(value) for value in state.tolist()]
    if isinstance(state, list | tuple):
        return [int(value) for value in state]
    return [int(state)]


@dataclass
class BaseStockPolicy:
    """Order up to a target base-stock level ``S``."""

    S: int
    name: str = "BaseStockPolicy"

    def reset(self, seed: int | None = None) -> None:
        """Reset policy state. This policy is stateless."""

    def act(self, inventory: State, t: int, rng: np.random.Generator | None = None) -> int:
        """Return ``max(S - inventory, 0)``."""
        return max(int(self.S) - _on_hand_inventory(inventory), 0)


@dataclass
class ConstantOrderPolicy:
    """Always order a fixed quantity."""

    quantity: int
    name: str = "ConstantOrderPolicy"

    def reset(self, seed: int | None = None) -> None:
        """Reset policy state. This policy is stateless."""

    def act(self, inventory: State, t: int, rng: np.random.Generator | None = None) -> int:
        """Return the fixed order quantity."""
        return max(int(self.quantity), 0)


@dataclass
class RandomPolicy:
    """Sample an integer order quantity uniformly from ``[0, max_order]``."""

    max_order: int
    name: str = "RandomPolicy"

    def reset(self, seed: int | None = None) -> None:
        """Reset policy state. Randomness is supplied by the runner."""

    def act(self, inventory: State, t: int, rng: np.random.Generator | None = None) -> int:
        """Return a random order quantity."""
        if self.max_order < 0:
            raise ValueError("max_order must be non-negative")
        generator = rng if rng is not None else np.random.default_rng()
        return int(generator.integers(0, self.max_order + 1))


@dataclass
class InventoryPositionBaseStockPolicy:
    """Order up to ``S`` using on-hand inventory plus pipeline inventory."""

    S: int
    name: str = "InventoryPositionBaseStockPolicy"

    def reset(self, seed: int | None = None) -> None:
        """Reset policy state. This policy is stateless."""

    def act(self, state: State, t: int, rng: np.random.Generator | None = None) -> int:
        """Return ``max(S - inventory_position, 0)``."""
        inventory_position = sum(_state_values(state))
        return max(int(self.S) - inventory_position, 0)


@dataclass
class DualSourcingHeuristicPolicy:
    """Simple dual-sourcing heuristic baseline.

    The regular order uses total inventory position. The expedited order uses
    on-hand inventory only. This is a configurable heuristic, not an optimality
    claim.
    """

    S_regular: int
    S_expedited: int
    regular_lead_time: int
    expedited_lead_time: int
    name: str = "DualSourcingHeuristicPolicy"

    def reset(self, seed: int | None = None) -> None:
        """Reset policy state. This policy is stateless."""

    def act(self, state: State, t: int, rng: np.random.Generator | None = None) -> list[int]:
        """Return ``[regular_order, expedited_order]``."""
        values = _state_values(state)
        on_hand = values[0]
        inventory_position = sum(values)
        regular_order = max(int(self.S_regular) - inventory_position, 0)
        expedited_order = max(int(self.S_expedited) - on_hand, 0)
        return [regular_order, expedited_order]
