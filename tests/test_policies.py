import numpy as np

from inventory_control.policies.classical import (
    BaseStockPolicy,
    ConstantOrderPolicy,
    DualSourcingHeuristicPolicy,
    InventoryPositionBaseStockPolicy,
    RandomPolicy,
)


def test_base_stock_policy_outputs_correct_order_quantity() -> None:
    policy = BaseStockPolicy(S=10)
    assert policy.act(inventory=3, t=0) == 7
    assert policy.act(inventory=10, t=0) == 0
    assert policy.act(inventory=12, t=0) == 0


def test_constant_order_policy_outputs_fixed_quantity() -> None:
    policy = ConstantOrderPolicy(quantity=4)
    assert policy.act(inventory=100, t=0) == 4


def test_random_policy_is_bounded() -> None:
    policy = RandomPolicy(max_order=3)
    rng = np.random.default_rng(0)
    samples = [policy.act(inventory=0, t=t, rng=rng) for t in range(50)]
    assert all(0 <= sample <= 3 for sample in samples)


def test_inventory_position_base_stock_policy() -> None:
    policy = InventoryPositionBaseStockPolicy(S=10)
    assert policy.act([3, 2, 1], t=0) == 4
    assert policy.act([5, 3, 2], t=0) == 0


def test_dual_sourcing_heuristic_policy() -> None:
    policy = DualSourcingHeuristicPolicy(
        S_regular=12,
        S_expedited=4,
        regular_lead_time=2,
        expedited_lead_time=1,
    )
    assert policy.act([3, 2, 1, 0], t=0) == [6, 1]
