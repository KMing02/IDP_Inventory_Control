import numpy as np

from inventory_control.envs.demand import (
    DeterministicDemand,
    NegativeBinomialDemand,
    NormalDemand,
    PoissonDemand,
    RegimeSwitchingPoissonDemand,
    SeasonalPoissonDemand,
)


def test_poisson_demand_is_non_negative_integer() -> None:
    demand = PoissonDemand(lam=4.0)
    rng = np.random.default_rng(0)
    samples = [demand.sample(rng, t) for t in range(100)]
    assert all(isinstance(sample, int) for sample in samples)
    assert all(sample >= 0 for sample in samples)


def test_normal_demand_is_clipped_at_zero() -> None:
    demand = NormalDemand(mean=-10.0, std=0.0)
    rng = np.random.default_rng(0)
    assert demand.sample(rng, 0) == 0


def test_deterministic_demand_sequence() -> None:
    demand = DeterministicDemand([2, 5, 1])
    rng = np.random.default_rng(0)
    assert [demand.sample(rng, t) for t in range(3)] == [2, 5, 1]


def test_negative_binomial_demand_is_non_negative_integer() -> None:
    demand = NegativeBinomialDemand(mean=5.0, dispersion=2.0)
    rng = np.random.default_rng(0)
    samples = [demand.sample(rng, t) for t in range(100)]
    assert all(isinstance(sample, int) for sample in samples)
    assert all(sample >= 0 for sample in samples)


def test_seasonal_poisson_rate_changes_over_time() -> None:
    demand = SeasonalPoissonDemand(base_lambda=10.0, amplitude=3.0, period=4)
    assert demand.rate(0) == 10.0
    assert demand.rate(1) == 13.0
    assert demand.rate(3) == 7.0


def test_regime_switching_demand_reproducible_with_seed() -> None:
    demand_a = RegimeSwitchingPoissonDemand(2.0, 10.0, switch_probability=0.5)
    demand_b = RegimeSwitchingPoissonDemand(2.0, 10.0, switch_probability=0.5)
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    samples_a = [demand_a.sample(rng_a, t) for t in range(10)]
    samples_b = [demand_b.sample(rng_b, t) for t in range(10)]

    assert samples_a == samples_b
