"""Demand distributions used by inventory environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class DemandDistribution(Protocol):
    """Protocol for stochastic or deterministic demand sources."""

    def sample(self, rng: np.random.Generator, t: int) -> int:
        """Return demand for period ``t`` using the provided random generator."""


@dataclass(frozen=True)
class PoissonDemand:
    """Poisson demand with rate parameter ``lam``."""

    lam: float

    def __post_init__(self) -> None:
        if self.lam < 0:
            raise ValueError("lam must be non-negative")

    def sample(self, rng: np.random.Generator, t: int) -> int:
        """Sample non-negative integer demand."""
        return int(rng.poisson(self.lam))


@dataclass(frozen=True)
class NormalDemand:
    """Normal demand rounded to an integer and clipped at zero."""

    mean: float
    std: float

    def __post_init__(self) -> None:
        if self.std < 0:
            raise ValueError("std must be non-negative")

    def sample(self, rng: np.random.Generator, t: int) -> int:
        """Sample demand, round to nearest integer, and clip negative values."""
        return int(max(round(float(rng.normal(self.mean, self.std))), 0))


@dataclass(frozen=True)
class NegativeBinomialDemand:
    """Negative binomial demand parameterized by mean and dispersion.

    The variance is ``mean + mean^2 / dispersion``. Larger dispersion values
    approach Poisson-like demand.
    """

    mean: float | None = None
    dispersion: float | None = None
    n: float | None = None
    p: float | None = None

    def __post_init__(self) -> None:
        has_mean_dispersion = self.mean is not None and self.dispersion is not None
        has_np = self.n is not None and self.p is not None
        if has_mean_dispersion == has_np:
            raise ValueError("provide either mean/dispersion or n/p")
        if has_mean_dispersion:
            if self.mean is None or self.dispersion is None:
                raise ValueError("mean and dispersion must both be provided")
            if self.mean < 0:
                raise ValueError("mean must be non-negative")
            if self.dispersion <= 0:
                raise ValueError("dispersion must be positive")
        if has_np:
            if self.n is None or self.p is None:
                raise ValueError("n and p must both be provided")
            if self.n <= 0:
                raise ValueError("n must be positive")
            if not 0 < self.p <= 1:
                raise ValueError("p must be in (0, 1]")

    def sample(self, rng: np.random.Generator, t: int) -> int:
        """Sample non-negative integer demand."""
        if self.n is not None and self.p is not None:
            return int(rng.negative_binomial(self.n, self.p))
        if self.mean == 0:
            return 0
        if self.mean is None or self.dispersion is None:
            raise RuntimeError("invalid negative binomial parameters")
        p = self.dispersion / (self.dispersion + self.mean)
        return int(rng.negative_binomial(self.dispersion, p))


@dataclass(frozen=True)
class SeasonalPoissonDemand:
    """Poisson demand with sinusoidal seasonality."""

    base_lambda: float
    amplitude: float
    period: int

    def __post_init__(self) -> None:
        if self.base_lambda < 0:
            raise ValueError("base_lambda must be non-negative")
        if self.amplitude < 0:
            raise ValueError("amplitude must be non-negative")
        if self.period <= 0:
            raise ValueError("period must be positive")

    def rate(self, t: int) -> float:
        """Return the Poisson rate for period ``t``."""
        seasonal = self.amplitude * np.sin(2.0 * np.pi * t / self.period)
        return float(max(self.base_lambda + seasonal, 0.0))

    def sample(self, rng: np.random.Generator, t: int) -> int:
        """Sample seasonal Poisson demand."""
        return int(rng.poisson(self.rate(t)))


@dataclass
class RegimeSwitchingPoissonDemand:
    """Two-regime Poisson demand with Markov switching."""

    low_lambda: float
    high_lambda: float
    switch_probability: float
    initial_regime: str = "low"

    def __post_init__(self) -> None:
        if self.low_lambda < 0 or self.high_lambda < 0:
            raise ValueError("lambda values must be non-negative")
        if not 0 <= self.switch_probability <= 1:
            raise ValueError("switch_probability must be in [0, 1]")
        if self.initial_regime not in {"low", "high"}:
            raise ValueError("initial_regime must be 'low' or 'high'")
        self._regime = self.initial_regime

    def reset(self) -> None:
        """Reset to the configured initial regime."""
        self._regime = self.initial_regime

    def sample(self, rng: np.random.Generator, t: int) -> int:
        """Sample demand from the current regime, switching between periods."""
        if t == 0:
            self.reset()
        elif rng.random() < self.switch_probability:
            self._regime = "high" if self._regime == "low" else "low"
        rate = self.low_lambda if self._regime == "low" else self.high_lambda
        return int(rng.poisson(rate))


@dataclass(frozen=True)
class DeterministicDemand:
    """Demand source backed by a fixed sequence."""

    sequence: tuple[int, ...]

    def __init__(self, sequence: list[int] | tuple[int, ...]) -> None:
        if len(sequence) == 0:
            raise ValueError("sequence must not be empty")
        if any(value < 0 for value in sequence):
            raise ValueError("sequence values must be non-negative")
        object.__setattr__(self, "sequence", tuple(int(value) for value in sequence))

    def sample(self, rng: np.random.Generator, t: int) -> int:
        """Return the sequence value for period ``t``."""
        if t >= len(self.sequence):
            raise IndexError("deterministic demand sequence exhausted")
        return self.sequence[t]
