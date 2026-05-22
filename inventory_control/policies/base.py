"""Policy interface for inventory control algorithms."""

from __future__ import annotations

from typing import Protocol, TypeAlias

import numpy as np

State: TypeAlias = int | float | list[int] | tuple[int, ...] | np.ndarray
Action: TypeAlias = int | list[int]


class Policy(Protocol):
    """Interface implemented by classical and future neural policies."""

    name: str

    def reset(self, seed: int | None = None) -> None:
        """Reset any policy state before a new episode."""

    def act(self, state: State, t: int, rng: np.random.Generator | None = None) -> Action:
        """Return the order quantity for the current state and period."""
