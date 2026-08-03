import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F

from inventory_control.policies.neural.constrained_mlp import (
    ConstrainedMonotoneMLP,
    MonotoneActivation,
    _allocate_activation_counts,
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


def test_only_inventory_weight_is_constrained_in_first_layer() -> None:
    model = ConstrainedMonotoneMLP(
        input_dim=6,
        hidden_layers=1,
        hidden_units=4,
        mixed_activation=True,
        monotonic_feature=0,
        monotonic_direction=-1,
    )
    first_layer = model.network[0]
    effective_weight = model.effective_first_weight()

    assert torch.all(effective_weight[:, 0] < 0.0)
    assert torch.equal(effective_weight[:, 1:], first_layer.raw_weight[:, 1:])


@pytest.mark.parametrize(
    ("activation", "base"),
    [("ELU", F.elu), ("softplus", F.softplus)],
)
def test_cmnn_convex_activation_equals_base_activation(activation, base) -> None:
    layer = MonotoneActivation(activation, mixed=True, input_dim=3)
    values = torch.tensor([-2.0, 0.0, 2.0])

    assert torch.allclose(layer.convex(values), base(values))


@pytest.mark.parametrize(
    ("activation", "base"),
    [("ELU", F.elu), ("softplus", F.softplus)],
)
def test_cmnn_concave_activation_equals_negative_reflected_base(activation, base) -> None:
    layer = MonotoneActivation(activation, mixed=True, input_dim=3)
    values = torch.tensor([-2.0, 0.0, 2.0])

    assert torch.allclose(layer.concave(values), -base(-values))


@pytest.mark.parametrize(
    ("activation", "base"),
    [("ELU", F.elu), ("softplus", F.softplus)],
)
def test_cmnn_bounded_activation_follows_piecewise_formula(activation, base) -> None:
    layer = MonotoneActivation(activation, mixed=True, input_dim=5)
    values = torch.tensor([-3.0, -0.5, 0.0, 0.5, 3.0])
    rho_one = base(torch.tensor(1.0))
    expected = torch.where(
        values < 0.0,
        base(values + 1.0) - rho_one,
        -base(1.0 - values) + rho_one,
    )

    assert torch.allclose(layer.bounded(values), expected)


@pytest.mark.parametrize("activation", ["ELU", "softplus"])
def test_cmnn_bounded_activation_is_continuous_at_zero(activation: str) -> None:
    layer = MonotoneActivation(activation, mixed=True, input_dim=3)
    epsilon = 1e-6
    values = layer.bounded(torch.tensor([-epsilon, 0.0, epsilon]))

    assert values[1].item() == pytest.approx(0.0, abs=1e-7)
    assert values[0].item() == pytest.approx(values[1].item(), abs=2e-6)
    assert values[2].item() == pytest.approx(values[1].item(), abs=2e-6)


@pytest.mark.parametrize("activation", ["ELU", "softplus"])
def test_cmnn_bounded_activation_is_monotone_increasing(activation: str) -> None:
    layer = MonotoneActivation(activation, mixed=True, input_dim=401)
    values = torch.linspace(-20.0, 20.0, steps=401)

    activated = layer.bounded(values)

    assert torch.all(torch.diff(activated) >= -1e-7)


def test_cmnn_bounded_elu_saturates_on_both_sides() -> None:
    layer = MonotoneActivation("ELU", mixed=True, input_dim=4)
    activated = layer.bounded(torch.tensor([-40.0, -20.0, 20.0, 40.0]))

    assert activated[:2].tolist() == pytest.approx([-2.0, -2.0], abs=1e-6)
    assert activated[2:].tolist() == pytest.approx([2.0, 2.0], abs=1e-6)


def test_cmnn_mixed_activation_preserves_dimension_and_allocates_groups() -> None:
    layer = MonotoneActivation("ELU", mixed=True, input_dim=5, activation_weights=(1.0, 1.0, 1.0))
    values = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])

    activated = layer(values)
    expected = torch.cat(
        [
            layer.convex(values[..., :2]),
            layer.concave(values[..., 2:4]),
            layer.bounded(values[..., 4:]),
        ],
        dim=-1,
    )

    assert layer.activation_counts == (2, 2, 1)
    assert activated.shape == values.shape
    assert torch.allclose(activated, expected)


@pytest.mark.parametrize(
    ("hidden_units", "expected"),
    [(1, (1, 0, 0)), (2, (1, 1, 0)), (4, (2, 1, 1)), (5, (2, 2, 1)), (32, (11, 11, 10))],
)
def test_cmnn_equal_weight_neuron_counts_sum_to_hidden_units(hidden_units: int, expected) -> None:
    counts = _allocate_activation_counts(hidden_units, (1.0, 1.0, 1.0))

    assert counts == expected
    assert sum(counts) == hidden_units


def test_cmnn_custom_activation_weights_are_allocated_deterministically() -> None:
    assert _allocate_activation_counts(7, (1.0, 2.0, 1.0)) == (2, 3, 2)


def test_cmnn_factory_reads_configured_activation_weights() -> None:
    config = {
        "history_length_H": 8,
        "idp_benchmark": {"hidden_layers": 1, "hidden_units": 5, "activation": "ELU"},
        "constrained_monotone_mlp": {"activation_weights": [0.0, 1.0, 2.0]},
    }

    model = make_constrained_monotone_mlp_model(config)

    assert model.activation_weights == (0.0, 1.0, 2.0)
    assert model.activation_counts == (0, 2, 3)


def test_constrained_monotone_mlp_mixed_activation_preserves_hidden_features() -> None:
    positive_model = ConstrainedMonotoneMLP(input_dim=12, hidden_layers=1, hidden_units=5, mixed_activation=False)
    constrained_model = ConstrainedMonotoneMLP(input_dim=12, hidden_layers=1, hidden_units=5, mixed_activation=True)

    assert positive_model.network[-1].in_features == 5
    assert constrained_model.network[-1].in_features == 5
    assert constrained_model.activation_counts == (2, 2, 1)


def test_positive_weight_mlp_retains_simple_unsplit_activation() -> None:
    config = {
        "history_length_H": 8,
        "idp_benchmark": {"hidden_layers": 1, "hidden_units": 5, "activation": "ELU"},
    }
    model = make_positive_weight_mlp_model(config)
    activation = model.network[1]
    values = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])

    assert isinstance(activation, MonotoneActivation)
    assert activation.mixed is False
    assert activation.activation_counts is None
    assert model.network[-1].in_features == 5
    assert torch.allclose(activation(values), F.elu(values))


def test_exact_cmnn_rejects_nonconvex_tanh_base_activation() -> None:
    with pytest.raises(ValueError, match="requires a convex monotone CMNN base activation: ELU or softplus"):
        ConstrainedMonotoneMLP(
            input_dim=12,
            hidden_layers=1,
            hidden_units=5,
            activation="tanh",
            mixed_activation=True,
        )


def test_simple_mode_still_supports_tanh_for_backward_compatibility() -> None:
    model = ConstrainedMonotoneMLP(
        input_dim=12,
        hidden_layers=1,
        hidden_units=5,
        activation="tanh",
        mixed_activation=False,
    )
    activation = model.network[1]
    values = torch.linspace(-2.0, 2.0, steps=5).unsqueeze(0)

    assert isinstance(activation, MonotoneActivation)
    assert torch.allclose(activation(values), torch.tanh(values))
