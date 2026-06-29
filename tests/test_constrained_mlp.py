import pytest

torch = pytest.importorskip("torch")

from inventory_control.policies.neural.constrained_mlp import (
    ConstrainedMonotoneMLP,
    make_constrained_monotone_mlp_model,
    make_positive_weight_mlp_model,
)


def _features_on_inventory_grid(model: ConstrainedMonotoneMLP, grid: list[float]) -> torch.Tensor:
    features = torch.full((len(grid), model.input_dim), 0.25)
    features[:, 0] = torch.tensor(grid)
    return features


@pytest.mark.parametrize("mixed_activation", [False, True])
def test_constrained_mlp_is_nonincreasing_in_inventory_inside_and_outside_range(mixed_activation: bool) -> None:
    model = ConstrainedMonotoneMLP(
        input_dim=12,
        hidden_layers=2,
        hidden_units=5,
        mixed_activation=mixed_activation,
        monotonic_feature=0,
        monotonic_direction=-1,
    )
    features = _features_on_inventory_grid(model, [-5.0, -2.0, 0.0, 1.0, 4.0, 8.0])

    with torch.no_grad():
        orders = model(features)

    assert torch.all(torch.diff(orders) <= 1e-6)


def test_constrained_mlp_deployed_cap_preserves_nonincreasing_order() -> None:
    model = ConstrainedMonotoneMLP(
        input_dim=12,
        hidden_layers=2,
        hidden_units=5,
        mixed_activation=True,
        monotonic_feature=0,
        monotonic_direction=-1,
    )
    inventory = torch.linspace(0.0, 40.0, steps=20)
    features = _features_on_inventory_grid(model, inventory.tolist())

    with torch.no_grad():
        raw_orders = model(features)
    cap = torch.clamp(40.0 - inventory, min=0.0)
    deployed = torch.minimum(raw_orders, cap)

    assert torch.all(torch.diff(deployed) <= 1e-6)


@pytest.mark.parametrize("factory", [make_positive_weight_mlp_model, make_constrained_monotone_mlp_model])
def test_constrained_mlp_effective_constraints_survive_optimizer_step(factory) -> None:
    config = {
        "history_length_H": 8,
        "idp_benchmark": {"hidden_layers": 2, "hidden_units": 4, "activation": "ELU"},
    }
    model = factory(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    features = torch.randn(6, 13)

    loss = model(features).mean()
    loss.backward()
    optimizer.step()
    model.apply_monotonic_constraints()

    first_weight = model.effective_first_weight()
    assert torch.all(first_weight[:, 0] <= 0.0)
    for downstream_weight in model.effective_downstream_weights():
        assert torch.all(downstream_weight >= 0.0)


def test_constrained_monotone_mlp_mixed_activation_expands_hidden_features() -> None:
    positive_model = ConstrainedMonotoneMLP(input_dim=12, hidden_layers=1, hidden_units=5, mixed_activation=False)
    constrained_model = ConstrainedMonotoneMLP(input_dim=12, hidden_layers=1, hidden_units=5, mixed_activation=True)

    assert positive_model.network[-1].in_features == 5
    assert constrained_model.network[-1].in_features == 15
