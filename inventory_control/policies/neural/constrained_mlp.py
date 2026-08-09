"""Constrained monotone MLP policies for paper-style lost-sales benchmarks."""

from __future__ import annotations

import math
from collections.abc import Sequence

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install neural dependencies with `python -m pip install -r requirements-neural.txt`.") from exc


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    """Return a numerically stable inverse of softplus for positive tensors."""
    return value + torch.log(-torch.expm1(-value))


def _positive_initial_weight(out_features: int, in_features: int) -> torch.Tensor:
    """Initialize positive effective weights with Xavier-like scale."""
    scale = (2.0 / max(in_features + out_features, 1)) ** 0.5
    desired = torch.rand(out_features, in_features) * scale + 1e-3
    return _inverse_softplus(desired)


def _allocate_activation_counts(
    hidden_units: int,
    activation_weights: Sequence[float],
) -> tuple[int, int, int]:
    """Allocate convex, concave, and bounded neurons by largest remainder.

    Proportional quotas are rounded down first. Remaining neurons are assigned
    by descending fractional remainder, with deterministic ties resolved in
    convex, concave, bounded order.
    """
    if hidden_units < 1:
        raise ValueError("hidden_units must be positive")
    weights = tuple(float(weight) for weight in activation_weights)
    if len(weights) != 3:
        raise ValueError("activation_weights must contain convex, concave, and bounded weights")
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("activation_weights must be finite and nonnegative")
    total_weight = sum(weights)
    if total_weight <= 0.0:
        raise ValueError("activation_weights must contain at least one positive weight")

    quotas = [hidden_units * weight / total_weight for weight in weights]
    counts = [math.floor(quota) for quota in quotas]
    remaining = hidden_units - sum(counts)
    priority = sorted(range(3), key=lambda index: (-(quotas[index] - counts[index]), index))
    for index in priority[:remaining]:
        counts[index] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


class FirstMonotoneLinear(nn.Module):
    """Linear layer with one feature constrained to be monotone by sign."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        monotonic_feature: int = 0,
        monotonic_direction: int = -1,
    ) -> None:
        super().__init__()
        if monotonic_feature < 0 or monotonic_feature >= in_features:
            raise ValueError(f"monotonic_feature={monotonic_feature} is outside in_features={in_features}")
        if monotonic_direction not in {-1, 1}:
            raise ValueError("monotonic_direction must be -1 or 1")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.monotonic_feature = int(monotonic_feature)
        self.monotonic_direction = int(monotonic_direction)
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.raw_weight)

    def effective_weight(self) -> torch.Tensor:
        """Return the sign-constrained weight matrix used in forward passes."""
        constrained = self.monotonic_direction * torch.nn.functional.softplus(
            self.raw_weight[:, self.monotonic_feature : self.monotonic_feature + 1]
        )
        pieces = []
        if self.monotonic_feature > 0:
            pieces.append(self.raw_weight[:, : self.monotonic_feature])
        pieces.append(constrained)
        if self.monotonic_feature + 1 < self.in_features:
            pieces.append(self.raw_weight[:, self.monotonic_feature + 1 :])
        return torch.cat(pieces, dim=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the sign-constrained affine map."""
        return torch.nn.functional.linear(features, self.effective_weight(), self.bias)


class NonnegativeLinear(nn.Module):
    """Linear layer whose weights are nonnegative by parameterization."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.raw_weight = nn.Parameter(_positive_initial_weight(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def effective_weight(self) -> torch.Tensor:
        """Return the nonnegative weight matrix used in forward passes."""
        return torch.nn.functional.softplus(self.raw_weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the nonnegative affine map."""
        return torch.nn.functional.linear(features, self.effective_weight(), self.bias)


class MonotoneActivation(nn.Module):
    """Simple monotone activation or faithful CMNN mixed activation."""

    def __init__(
        self,
        activation: str = "ELU",
        mixed: bool = False,
        input_dim: int | None = None,
        activation_weights: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        self.activation = str(activation)
        self.mixed = bool(mixed)
        normalized_activation = self.activation.lower()
        supported = {"elu", "softplus", "tanh"}
        if normalized_activation not in supported:
            raise ValueError(f"unsupported monotone activation: {self.activation}")
        if self.mixed and normalized_activation not in {"elu", "softplus"}:
            raise ValueError(
                "mixed_activation=True requires a convex monotone CMNN base activation: ELU or softplus"
            )
        if self.mixed and input_dim is None:
            raise ValueError("input_dim is required when mixed_activation=True")
        self.input_dim = int(input_dim) if input_dim is not None else None
        self.activation_weights = tuple(float(weight) for weight in activation_weights)
        self.activation_counts = (
            _allocate_activation_counts(self.input_dim, self.activation_weights)
            if self.mixed and self.input_dim is not None
            else None
        )

    def _base(self, values: torch.Tensor) -> torch.Tensor:
        if self.activation.lower() == "elu":
            return torch.nn.functional.elu(values)
        if self.activation.lower() == "softplus":
            return torch.nn.functional.softplus(values)
        if self.activation.lower() == "tanh":
            return torch.tanh(values)
        raise ValueError(f"unsupported monotone activation: {self.activation}")  # pragma: no cover

    def convex(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the convex CMNN activation rho."""
        return self._base(values)

    def concave(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the concave CMNN activation -rho(-x)."""
        return -self._base(-values)

    def bounded(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the paper's bounded/saturated CMNN activation."""
        one = torch.ones((), dtype=values.dtype, device=values.device)
        rho_one = self._base(one)
        negative_branch = self._base(values + one) - rho_one
        nonnegative_branch = self.concave(values - one) + rho_one
        return torch.where(values < 0.0, negative_branch, nonnegative_branch)

    def output_dim(self, input_dim: int) -> int:
        """Return the feature dimension after applying this activation."""
        if self.mixed and int(input_dim) != self.input_dim:
            raise ValueError(f"mixed activation was allocated for {self.input_dim} features, got {input_dim}")
        return int(input_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Apply a simple activation or the allocated CMNN activation mix."""
        if not self.mixed:
            return self._base(values)
        if values.shape[-1] != self.input_dim:
            raise ValueError(
                f"mixed activation was allocated for {self.input_dim} features, got {values.shape[-1]}"
            )
        if self.activation_counts is None:  # pragma: no cover
            raise RuntimeError("mixed activation counts were not initialized")
        functions = (self.convex, self.concave, self.bounded)
        activated_groups = []
        start = 0
        for count, function in zip(self.activation_counts, functions):
            stop = start + count
            activated_groups.append(function(values[..., start:stop]))
            start = stop
        return torch.cat(activated_groups, dim=-1)


class ConstrainedMonotoneMLP(nn.Module):
    """Faithful partially monotone CMNN with nonincreasing inventory response.

    The first layer constrains only the inventory feature. Later layers use
    nonnegative weights and monotone activations, so the full network remains
    nonincreasing in inventory while allowing unconstrained effects from demand
    history and product economics through their first-layer weights. In exact
    CMNN mode, each hidden layer allocates its fixed width across convex,
    concave, and bounded activation groups.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: int = 2,
        hidden_units: int = 32,
        activation: str = "ELU",
        mixed_activation: bool = True,
        activation_weights: Sequence[float] = (1.0, 1.0, 1.0),
        monotonic_feature: int = 0,
        monotonic_direction: int = -1,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("ConstrainedMonotoneMLP requires at least one hidden layer")
        self.input_dim = int(input_dim)
        self.hidden_layers = int(hidden_layers)
        self.hidden_units = int(hidden_units)
        self.monotonic_feature = int(monotonic_feature)
        self.monotonic_direction = int(monotonic_direction)
        self.mixed_activation = bool(mixed_activation)
        self.activation_weights = tuple(float(weight) for weight in activation_weights)
        self.activation_counts = (
            _allocate_activation_counts(self.hidden_units, self.activation_weights)
            if self.mixed_activation
            else None
        )

        layers: list[nn.Module] = [
            FirstMonotoneLinear(
                input_dim,
                hidden_units,
                monotonic_feature=monotonic_feature,
                monotonic_direction=monotonic_direction,
            )
        ]
        activation_layer = MonotoneActivation(
            activation,
            mixed=mixed_activation,
            input_dim=hidden_units,
            activation_weights=self.activation_weights,
        )
        layers.append(activation_layer)
        previous_dim = activation_layer.output_dim(hidden_units)
        for _ in range(hidden_layers - 1):
            layers.append(NonnegativeLinear(previous_dim, hidden_units))
            activation_layer = MonotoneActivation(
                activation,
                mixed=mixed_activation,
                input_dim=hidden_units,
                activation_weights=self.activation_weights,
            )
            layers.append(activation_layer)
            previous_dim = activation_layer.output_dim(hidden_units)
        layers.append(NonnegativeLinear(previous_dim, 1))
        self.network = nn.ModuleList(layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return nonnegative normalized order quantity."""
        values = features
        for layer in self.network:
            values = layer(values)
        return torch.nn.functional.softplus(values).squeeze(-1)

    def apply_monotonic_constraints(self) -> None:
        """Compatibility hook; constraints are enforced by parameterization."""
        return None

    def effective_first_weight(self) -> torch.Tensor:
        """Return the first-layer effective weights for tests and diagnostics."""
        first_layer = self.network[0]
        if not isinstance(first_layer, FirstMonotoneLinear):  # pragma: no cover
            raise TypeError("first layer is not FirstMonotoneLinear")
        return first_layer.effective_weight()

    def effective_downstream_weights(self) -> list[torch.Tensor]:
        """Return effective weights from all nonnegative downstream layers."""
        return [layer.effective_weight() for layer in self.network if isinstance(layer, NonnegativeLinear)]


class ConstrainedMLPResidualOrderUpTo(nn.Module):
    """CMNN-backbone Residual Order-Up-To model.

    Inventory monotonicity is supplied by the external decoder
    ``q = relu(S_theta - inventory)``.  The context-only first layer is
    unconstrained in this initial ablation, so this model makes no
    context-monotonicity guarantee.  It reuses the faithful width-preserving
    CMNN mixed activations and nonnegative downstream layers.
    """

    paper_action_parameterization = "residual_order_up_to"
    paper_residual_center = 1.0
    paper_residual_scale = 0.5
    paper_residual_anchor = "gamma"
    paper_track_residual_gradient_diagnostics = True

    def __init__(
        self,
        input_dim: int,
        hidden_layers: int = 2,
        hidden_units: int = 32,
        activation: str = "ELU",
        activation_weights: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("ConstrainedMLPResidualOrderUpTo requires at least one hidden layer")
        self.input_dim = int(input_dim)
        self.hidden_layers = int(hidden_layers)
        self.hidden_units = int(hidden_units)
        self.activation_weights = tuple(float(weight) for weight in activation_weights)
        self.activation_counts = _allocate_activation_counts(self.hidden_units, self.activation_weights)

        first = nn.Linear(self.input_dim, self.hidden_units)
        nn.init.xavier_uniform_(first.weight)
        nn.init.zeros_(first.bias)
        layers: list[nn.Module] = [first]
        activation_layer = MonotoneActivation(
            activation,
            mixed=True,
            input_dim=self.hidden_units,
            activation_weights=self.activation_weights,
        )
        layers.append(activation_layer)
        for _ in range(self.hidden_layers - 1):
            layers.append(NonnegativeLinear(self.hidden_units, self.hidden_units))
            layers.append(
                MonotoneActivation(
                    activation,
                    mixed=True,
                    input_dim=self.hidden_units,
                    activation_weights=self.activation_weights,
                )
            )
        self.backbone = nn.ModuleList(layers)
        self.residual_readout = nn.Linear(self.hidden_units, 1)
        nn.init.zeros_(self.residual_readout.weight)
        nn.init.zeros_(self.residual_readout.bias)

    def preactivation(self, context: torch.Tensor) -> torch.Tensor:
        values = context
        for layer in self.backbone:
            values = layer(values)
        return self.residual_readout(values).squeeze(-1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Return ``u_theta = 2 sigmoid(z_theta)``."""
        return 2.0 * torch.sigmoid(self.preactivation(context))

    def apply_monotonic_constraints(self) -> None:
        """Compatibility hook; downstream constraints use parameterization."""
        return None

    def effective_downstream_weights(self) -> list[torch.Tensor]:
        return [layer.effective_weight() for layer in self.backbone if isinstance(layer, NonnegativeLinear)]


def _constrained_mlp_source(config: dict, metadata: dict | None, key: str) -> dict:
    if metadata is not None:
        return metadata
    benchmark = config.get("idp_benchmark", {})
    return {
        **benchmark,
        **config.get(key, {}),
    }


def make_constrained_monotone_mlp_model(config: dict, metadata: dict | None = None) -> ConstrainedMonotoneMLP:
    """Build the faithful CMNN for paper lost-sales features."""
    source = _constrained_mlp_source(config, metadata, "constrained_monotone_mlp")
    history_length = int(source.get("history_length_H", config["history_length_H"]))
    return ConstrainedMonotoneMLP(
        input_dim=1 + history_length + 4,
        hidden_layers=int(source.get("hidden_layers", 2)),
        hidden_units=int(source.get("hidden_units", 32)),
        activation=str(source.get("activation", "ELU")),
        mixed_activation=bool(source.get("mixed_activation", True)),
        activation_weights=source.get("activation_weights", (1.0, 1.0, 1.0)),
        monotonic_feature=int(source.get("monotonic_feature", 0)),
        monotonic_direction=int(source.get("monotonic_direction", -1)),
    )


def make_paper_residual_order_up_to_constrained_mlp_model(
    config: dict,
    metadata: dict | None = None,
) -> ConstrainedMLPResidualOrderUpTo:
    """Build the CMNN-backbone Residual Order-Up-To model."""
    source = _constrained_mlp_source(config, metadata, "constrained_monotone_mlp")
    history_length = int(source.get("history_length_H", config["history_length_H"]))
    return ConstrainedMLPResidualOrderUpTo(
        input_dim=history_length + 4,
        hidden_layers=int(source.get("hidden_layers", 2)),
        hidden_units=int(source.get("hidden_units", 32)),
        activation=str(source.get("activation", "ELU")),
        activation_weights=source.get("activation_weights", (1.0, 1.0, 1.0)),
    )


def make_positive_weight_mlp_model(config: dict, metadata: dict | None = None) -> ConstrainedMonotoneMLP:
    """Build the simpler positive-weight monotone MLP diagnostic baseline."""
    source = _constrained_mlp_source(config, metadata, "positive_weight_mlp")
    history_length = int(source.get("history_length_H", config["history_length_H"]))
    return ConstrainedMonotoneMLP(
        input_dim=1 + history_length + 4,
        hidden_layers=int(source.get("hidden_layers", 2)),
        hidden_units=int(source.get("hidden_units", 32)),
        activation=str(source.get("activation", "ELU")),
        mixed_activation=False,
        monotonic_feature=int(source.get("monotonic_feature", 0)),
        monotonic_direction=int(source.get("monotonic_direction", -1)),
    )
