import pytest

torch = pytest.importorskip("torch")

from inventory_control.monokan_interpretability import (
    ECONOMICS_DRIVERS,
    level_curve_table,
    partial_dependence_table,
)
from inventory_control.policies.neural.monokan import MonoKANSoftQuantileTargetPolicy
from inventory_control.products import PaperLostSalesProduct


HISTORY_LENGTH = 12


def _policy() -> MonoKANSoftQuantileTargetPolicy:
    torch.manual_seed(0)
    return MonoKANSoftQuantileTargetPolicy(
        history_length=HISTORY_LENGTH,
        economics_dim=4,
        demand_scale=50.0,
        hidden_layers=1,
        hidden_units=6,
        grid=5,
        monotonic_vars={0: 1, 1: -1, 2: -1, 3: 1},
    )


def _contexts(rows: int = 64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(4)
    lags = torch.distributions.Exponential(1.0).sample((rows, HISTORY_LENGTH))
    economics = torch.rand((rows, 4), generator=generator) * torch.tensor([1.0, 0.5, 0.05, 0.02])
    return torch.cat([lags, economics], dim=1)


def _product(index: int) -> PaperLostSalesProduct:
    return PaperLostSalesProduct(
        product_id=f"p{index}",
        price=100.0 + index,
        purchase_cost=40.0,
        holding_cost=5.0,
        lost_sale_penalty=3.0,
        true_mean_demand=50.0,
        true_coefficient_of_variation=0.4,
        gamma_shape=6.25,
        gamma_scale=8.0,
        demand_distribution_type="Gamma",
        historical_demand=[50.0] * HISTORY_LENGTH,
        initial_inventory=50.0,
    )


def test_partial_dependence_reports_every_driver_and_its_certified_direction() -> None:
    frame = partial_dependence_table(_policy(), _contexts(), HISTORY_LENGTH, samples=20)

    drivers = list(frame["driver"].unique())
    assert len(drivers) == len(ECONOMICS_DRIVERS) + 1
    assert "demand level (all lags scaled)" in drivers
    # A certified policy must hold every direction it declares.
    assert bool(frame["direction_holds"].all())


def test_partial_dependence_curves_move_in_the_certified_direction() -> None:
    frame = partial_dependence_table(_policy(), _contexts(), HISTORY_LENGTH, samples=20)

    for driver, group in frame.groupby("driver"):
        direction = int(group["certified_direction"].iloc[0])
        increments = group["mean_target"].diff().dropna() * direction
        assert bool((increments >= -1e-6).all()), driver


def test_level_curve_table_compares_against_the_analytic_fractile() -> None:
    products = [_product(index) for index in range(6)]
    contexts = _contexts(len(products))

    frame = level_curve_table(_policy(), contexts, products)

    assert list(frame["product_id"]) == [p.product_id for p in products]
    assert ((frame["learned_level"] > 0) & (frame["learned_level"] < 1)).all()
    expected = frame["learned_level"] - frame["analytic_critical_fractile"]
    assert (frame["difference"] - expected).abs().max() < 1e-9
