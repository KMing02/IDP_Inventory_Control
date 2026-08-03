"""Constrained monotone MLP policies for paper-style lost-sales benchmarks."""

from __future__ import annotations

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


class FirstPartialMonotoneLinear(nn.Module):
    """Linear layer with selected input columns constrained by sign."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        monotonic_vars: dict[int, int],
    ) -> None:
        super().__init__()
        for feature, direction in monotonic_vars.items():
            if feature < 0 or feature >= in_features:
                raise ValueError(f"monotonic feature={feature} is outside in_features={in_features}")
            if direction not in {-1, 1}:
                raise ValueError("monotonic directions must be -1 or 1")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.monotonic_vars = {int(feature): int(direction) for feature, direction in monotonic_vars.items()}
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.raw_weight)

    def effective_weight(self) -> torch.Tensor:
        """Return the sign-constrained weight matrix used in forward passes."""
        columns = []
        for feature in range(self.in_features):
            raw_column = self.raw_weight[:, feature : feature + 1]
            if feature in self.monotonic_vars:
                columns.append(self.monotonic_vars[feature] * torch.nn.functional.softplus(raw_column))
            else:
                columns.append(raw_column)
        return torch.cat(columns, dim=1)

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
    """Monotone activation with optional C-MNN-style activation mixing."""

    def __init__(self, activation: str = "ELU", mixed: bool = False) -> None:
        super().__init__()
        self.activation = str(activation)
        self.mixed = bool(mixed)

    def _base(self, values: torch.Tensor) -> torch.Tensor:
        if self.activation.upper() == "ELU":
            return torch.nn.functional.elu(values)
        if self.activation.lower() == "softplus":
            return torch.nn.functional.softplus(values)
        if self.activation.lower() == "tanh":
            return torch.tanh(values)
        raise ValueError(f"unsupported monotone activation: {self.activation}")

    def output_dim(self, input_dim: int) -> int:
        """Return the feature dimension after applying this activation."""
        return int(input_dim) * 3 if self.mixed else int(input_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Apply a monotone activation or a monotone convex/concave/linear mix."""
        if not self.mixed:
            return self._base(values)
        return torch.cat([self._base(values), -self._base(-values), values], dim=-1)


class ConstrainedMonotoneMLP(nn.Module):
    """Partially monotone MLP with nonincreasing inventory response.

    The first layer constrains only the inventory feature. Later layers use
    nonnegative weights and monotone activations, so the full network remains
    nonincreasing in inventory while allowing unconstrained effects from demand
    history and product economics through their first-layer weights.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: int = 2,
        hidden_units: int = 32,
        activation: str = "ELU",
        mixed_activation: bool = True,
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

        layers: list[nn.Module] = [
            FirstMonotoneLinear(
                input_dim,
                hidden_units,
                monotonic_feature=monotonic_feature,
                monotonic_direction=monotonic_direction,
            )
        ]
        activation_layer = MonotoneActivation(activation, mixed=mixed_activation)
        layers.append(activation_layer)
        previous_dim = activation_layer.output_dim(hidden_units)
        for _ in range(hidden_layers - 1):
            layers.append(NonnegativeLinear(previous_dim, hidden_units))
            activation_layer = MonotoneActivation(activation, mixed=mixed_activation)
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


class PartialContextMonotoneMLP(nn.Module):
    """Partially monotone MLP over context features for target-stock residuals."""

    def __init__(
        self,
        input_dim: int,
        monotonic_vars: dict[int, int],
        hidden_layers: int = 2,
        hidden_units: int = 32,
        activation: str = "ELU",
        mixed_activation: bool = True,
        output: str = "signed",
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("PartialContextMonotoneMLP requires at least one hidden layer")
        self.input_dim = int(input_dim)
        self.monotonic_vars = {int(feature): int(direction) for feature, direction in monotonic_vars.items()}
        self.output = str(output)
        layers: list[nn.Module] = [
            FirstPartialMonotoneLinear(
                self.input_dim,
                int(hidden_units),
                self.monotonic_vars,
            )
        ]
        activation_layer = MonotoneActivation(activation, mixed=mixed_activation)
        layers.append(activation_layer)
        previous_dim = activation_layer.output_dim(int(hidden_units))
        for _ in range(int(hidden_layers) - 1):
            layers.append(NonnegativeLinear(previous_dim, int(hidden_units)))
            activation_layer = MonotoneActivation(activation, mixed=mixed_activation)
            layers.append(activation_layer)
            previous_dim = activation_layer.output_dim(int(hidden_units))
        layers.append(NonnegativeLinear(previous_dim, 1))
        self.network = nn.ModuleList(layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return a context-dependent scalar."""
        context = features[:, -self.input_dim :]
        values = context
        for layer in self.network:
            values = layer(values)
        raw = values.squeeze(-1)
        if self.output == "signed":
            return torch.tanh(raw)
        return torch.nn.functional.softplus(raw)

    def apply_monotonic_constraints(self) -> None:
        """Compatibility hook; constraints are enforced by parameterization."""
        return None


def _constrained_mlp_source(config: dict, metadata: dict | None, key: str) -> dict:
    if metadata is not None:
        return metadata
    benchmark = config.get("idp_benchmark", {})
    return {
        **benchmark,
        **config.get(key, {}),
    }


def make_constrained_monotone_mlp_model(config: dict, metadata: dict | None = None) -> ConstrainedMonotoneMLP:
    """Build the C-MNN-style constrained monotone MLP for paper features."""
    source = _constrained_mlp_source(config, metadata, "constrained_monotone_mlp")
    history_length = int(source.get("history_length_H", config["history_length_H"]))
    return ConstrainedMonotoneMLP(
        input_dim=1 + history_length + 4,
        hidden_layers=int(source.get("hidden_layers", 2)),
        hidden_units=int(source.get("hidden_units", 32)),
        activation=str(source.get("activation", "ELU")),
        mixed_activation=bool(source.get("mixed_activation", True)),
        monotonic_feature=int(source.get("monotonic_feature", 0)),
        monotonic_direction=int(source.get("monotonic_direction", -1)),
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


def make_paper_residual_order_up_to_constrained_mlp_model(
    config: dict,
    metadata: dict | None = None,
) -> PartialContextMonotoneMLP:
    """Build a context-monotone MLP residual target-stock comparator."""
    source = _constrained_mlp_source(config, metadata, "constrained_monotone_mlp")
    history_length = int(source.get("history_length_H", config["history_length_H"]))
    monotonic_vars = {index: 1 for index in range(history_length)}
    monotonic_vars.update(
        {
            history_length: 1,
            history_length + 1: -1,
            history_length + 2: -1,
            history_length + 3: 1,
        }
    )
    model = PartialContextMonotoneMLP(
        input_dim=history_length + 4,
        monotonic_vars=monotonic_vars,
        hidden_layers=int(source.get("hidden_layers", 2)),
        hidden_units=int(source.get("hidden_units", 32)),
        activation=str(source.get("activation", "ELU")),
        mixed_activation=bool(source.get("mixed_activation", True)),
        output="signed",
    )
    model.paper_action_parameterization = "residual_order_up_to"
    model.paper_residual_center = 0.0
    model.paper_residual_scale = float(source.get("residual_scale", config.get("fcnn", {}).get("residual_scale", 0.5)))
    model.paper_residual_anchor = str(source.get("residual_anchor", config.get("fcnn", {}).get("residual_anchor", "gamma")))
    return model
