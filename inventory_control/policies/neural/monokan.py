"""MonoKAN policy components for paper-style lost-sales replication.

This module adapts the core Hermite-spline monotonicity idea from MonoKAN:
https://github.com/alejandropolo/MonoKAN

The original MonoKAN repository is MIT licensed. This local implementation is
kept intentionally small so it can be trained through the inventory
DirectBackprop loop without pulling in the original notebooks, symbolic KAN
utilities, or supervised-tabular training code.
"""

from __future__ import annotations

from collections.abc import Sequence

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install neural dependencies with `python -m pip install -e '.[dev]'`.") from exc

from inventory_control.policies.neural.fcnn import (
    PaperFCNNFeatureStats,
    empirical_newsvendor_target_ratio,
    infer_model_device,
    paper_history_feature_count,
)
from inventory_control.products import PaperLostSalesProduct


def _positive_elu(x: torch.Tensor) -> torch.Tensor:
    """Positive increasing base function used by MonoKAN-style layers."""
    alpha = 0.9
    return alpha * torch.nn.functional.elu(x) + alpha


def _hermite_basis(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cubic Hermite basis functions for normalized positions."""
    t2 = t * t
    t3 = t2 * t
    return (
        2 * t3 - 3 * t2 + 1,
        t3 - 2 * t2 + t,
        -2 * t3 + 3 * t2,
        t3 - t2,
    )


def _evaluate_hermite_curves(x_eval: torch.Tensor, grid: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    """Evaluate batched cubic Hermite curves.

    Args:
        x_eval: Tensor with shape ``(num_splines, batch)``.
        grid: Tensor with shape ``(num_splines, num_grid_points)``.
        coef: Tensor with shape ``(num_splines, num_grid_points, 2)``.

    Returns:
        Tensor with shape ``(num_splines, batch)``.
    """
    num_grid_points = grid.shape[1]
    mask = x_eval.unsqueeze(1) >= grid.unsqueeze(2)
    idx = mask.sum(dim=1) - 1
    idx = idx.clamp(min=0, max=num_grid_points - 2)
    idx_right = idx + 1

    x0 = grid.gather(1, idx)
    x1 = grid.gather(1, idx_right)
    interval = torch.clamp(x1 - x0, min=1e-8)
    t = (x_eval - x0) / interval
    h00, h10, h01, h11 = _hermite_basis(t)

    y0 = coef[:, :, 0].gather(1, idx)
    dy0 = coef[:, :, 1].gather(1, idx)
    y1 = coef[:, :, 0].gather(1, idx_right)
    dy1 = coef[:, :, 1].gather(1, idx_right)
    y_eval = y0 * h00 + dy0 * interval * h10 + y1 * h01 + dy1 * interval * h11

    left_mask = x_eval < grid[:, [0]]
    right_mask = x_eval > grid[:, [-1]]
    y_eval = torch.where(left_mask, coef[:, [0], 0] + coef[:, [0], 1] * (x_eval - grid[:, [0]]), y_eval)
    y_eval = torch.where(
        right_mask,
        coef[:, [-1], 0] + coef[:, [-1], 1] * (x_eval - grid[:, [-1]]),
        y_eval,
    )
    return y_eval


class HermiteKANLayer(nn.Module):
    """KAN layer with one learnable cubic Hermite spline per input-output edge."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        grid: int = 5,
        grid_range: tuple[float, float] = (-2.0, 4.0),
        noise_scale: float = 0.01,
        base_scale_init: float | None = None,
        spline_scale_init: float | None = None,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_edges = self.in_dim * self.out_dim
        self.num_grid_intervals = int(grid)

        grid_values = torch.linspace(float(grid_range[0]), float(grid_range[1]), steps=self.num_grid_intervals + 1)
        self.register_buffer("grid", grid_values.repeat(self.num_edges, 1))
        coef = torch.zeros(self.num_edges, self.num_grid_intervals + 1, 2)
        coef[:, :, 0] = noise_scale * torch.randn(self.num_edges, self.num_grid_intervals + 1)
        self.coef = nn.Parameter(coef)
        edge_scale = 1.0 / max(self.in_dim, 1)
        self.scale_base = nn.Parameter(torch.full((self.num_edges,), edge_scale if base_scale_init is None else float(base_scale_init)))
        self.scale_spline = nn.Parameter(
            torch.full((self.num_edges,), edge_scale if spline_scale_init is None else float(spline_scale_init))
        )
        self.bias = nn.Parameter(torch.zeros(self.out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return transformed layer activations."""
        batch_size = x.shape[0]
        expanded = x.unsqueeze(1).expand(batch_size, self.out_dim, self.in_dim)
        edge_inputs = expanded.reshape(batch_size, self.num_edges).transpose(0, 1)
        spline_values = _evaluate_hermite_curves(edge_inputs, self.grid, self.coef).transpose(0, 1)
        base_values = _positive_elu(edge_inputs).transpose(0, 1)
        edge_values = self.scale_base * base_values + self.scale_spline * spline_values
        return edge_values.reshape(batch_size, self.out_dim, self.in_dim).sum(dim=2) + self.bias


def _project_edge_coefficients(coef: torch.Tensor, grid: torch.Tensor, direction: int) -> None:
    """Project Hermite coefficients in-place to monotone increasing/decreasing curves."""
    if direction == 1:
        coef[:, :, 0] = torch.cummax(coef[:, :, 0], dim=1)[0]
    elif direction == -1:
        coef[:, :, 0] = torch.cummin(coef[:, :, 0], dim=1)[0]
    else:
        raise ValueError("direction must be 1 or -1")

    delta = (coef[:, 1:, 0] - coef[:, :-1, 0]) / torch.clamp(grid[:, 1:] - grid[:, :-1], min=1e-8)
    zero_delta = delta == 0
    coef[:, :-1, 1] = torch.where(zero_delta, torch.zeros_like(coef[:, :-1, 1]), coef[:, :-1, 1])
    coef[:, 1:, 1] = torch.where(zero_delta, torch.zeros_like(coef[:, 1:, 1]), coef[:, 1:, 1])

    if direction == 1:
        coef[:, :-1, 1] = torch.clamp(coef[:, :-1, 1], min=0.0)
        coef[:, 1:, 1] = torch.clamp(coef[:, 1:, 1], min=0.0)
    else:
        coef[:, :-1, 1] = torch.clamp(coef[:, :-1, 1], max=0.0)
        coef[:, 1:, 1] = torch.clamp(coef[:, 1:, 1], max=0.0)

    safe_delta = torch.where(zero_delta, torch.full_like(delta, float("inf")), delta)
    alpha = coef[:, :-1, 1] / safe_delta
    beta = coef[:, 1:, 1] / safe_delta
    norm = torch.sqrt(alpha**2 + beta**2)
    exceed = norm > 3.0
    tau = torch.where(exceed, 3.0 / torch.clamp(norm, min=1e-8), torch.ones_like(norm))
    coef[:, :-1, 1] = torch.where(exceed, tau * alpha * safe_delta, coef[:, :-1, 1])
    coef[:, 1:, 1] = torch.where(exceed, tau * beta * safe_delta, coef[:, 1:, 1])


class MonoKANPolicy(nn.Module):
    """KAN policy with certified monotonicity in selected observable inputs.

    ``monotonic_vars`` maps feature indices to ``1`` for nondecreasing or ``-1``
    for nonincreasing behavior. For lost-sales ordering, inventory should be
    configured as ``{0: -1}``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_units: int = 16,
        hidden_layers: int = 1,
        grid: int = 5,
        grid_range: tuple[float, float] = (-2.0, 4.0),
        noise_scale: float = 0.01,
        base_scale_init: float | None = None,
        spline_scale_init: float | None = None,
        bounded_base_scale_init: float | None = None,
        bounded_spline_scale_init: float = 1.0,
        bounded_spline_slope_init: float = 1.0,
        monotonic_vars: dict[int, int] | None = None,
        initial_normalized_order: float = 0.5,
        output_upper_bound: float | None = None,
        output_logit_anchor: float | None = None,
        output_logit_residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.monotonic_vars = dict(monotonic_vars or {})
        self.output_upper_bound = float(output_upper_bound) if output_upper_bound is not None else None
        self.output_logit_anchor = float(output_logit_anchor) if output_logit_anchor is not None else None
        self.output_logit_residual_scale = float(output_logit_residual_scale)
        self.bounded_base_scale_init = bounded_base_scale_init
        self.bounded_spline_scale_init = float(bounded_spline_scale_init)
        self.bounded_spline_slope_init = float(bounded_spline_slope_init)
        if self.output_upper_bound is not None and self.output_upper_bound <= 0:
            raise ValueError("output_upper_bound must be positive")
        widths = [int(input_dim)] + [int(hidden_units)] * int(hidden_layers) + [1]
        self.layers = nn.ModuleList(
            [
                HermiteKANLayer(
                    widths[index],
                    widths[index + 1],
                    grid=grid,
                    grid_range=grid_range,
                    noise_scale=noise_scale,
                    base_scale_init=base_scale_init,
                    spline_scale_init=spline_scale_init,
                )
                for index in range(len(widths) - 1)
            ]
        )
        if self.output_upper_bound is not None:
            self._initialize_bounded_layers()
        self.apply_monotonic_constraints()
        self._calibrate_initial_output(float(initial_normalized_order))

    @torch.no_grad()
    def _initialize_bounded_layers(self) -> None:
        """Use expressive spline scales when the paper action bound controls output scale."""
        for layer in self.layers:
            layer.coef[:, :, 1].fill_(self.bounded_spline_slope_init)
            base_scale = 1.0 / max(layer.in_dim, 1) ** 0.5 if self.bounded_base_scale_init is None else self.bounded_base_scale_init
            layer.scale_base.copy_(torch.full_like(layer.scale_base, float(base_scale)))
            layer.scale_spline.fill_(self.bounded_spline_scale_init)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return non-negative normalized order quantity."""
        x = features
        for layer in self.layers:
            x = layer(x)
        if self.output_upper_bound is not None:
            if self.output_logit_anchor is not None:
                x = self.output_logit_anchor + self.output_logit_residual_scale * x
            return (self.output_upper_bound * torch.sigmoid(x)).squeeze(-1)
        return torch.nn.functional.softplus(x).squeeze(-1)

    @torch.no_grad()
    def _calibrate_initial_output(self, target_order: float) -> None:
        """Set the final bias to a stable normalized order at a reference input."""
        if target_order <= 0:
            raise ValueError("initial_normalized_order must be positive")
        if self.output_upper_bound is not None:
            target_order = min(target_order, self.output_upper_bound - 1e-6)
            if target_order <= 0:
                raise ValueError("initial_normalized_order must be below output_upper_bound")
        reference = torch.zeros(1, self.layers[0].in_dim, device=self.layers[0].bias.device)
        x = reference
        for layer in self.layers:
            x = layer(x)
        if self.output_upper_bound is None:
            target_pre_activation = torch.log(torch.expm1(torch.tensor(target_order, device=x.device)))
        elif self.output_logit_anchor is not None:
            target_pre_activation = torch.zeros((), device=x.device)
        else:
            target_fraction = torch.tensor(target_order / self.output_upper_bound, device=x.device)
            target_fraction = torch.clamp(target_fraction, min=1e-6, max=1 - 1e-6)
            target_pre_activation = torch.logit(target_fraction)
        self.layers[-1].bias.add_(target_pre_activation - x.squeeze())

    @torch.no_grad()
    def apply_monotonic_constraints(self) -> None:
        """Project layer parameters to the configured monotone feasible set."""
        if not self.layers:
            return

        first_layer = self.layers[0]
        first_scale_base = first_layer.scale_base.view(first_layer.out_dim, first_layer.in_dim)
        first_scale_spline = first_layer.scale_spline.view(first_layer.out_dim, first_layer.in_dim)
        first_coef = first_layer.coef.view(
            first_layer.out_dim,
            first_layer.in_dim,
            first_layer.num_grid_intervals + 1,
            2,
        )
        first_grid = first_layer.grid.view(first_layer.out_dim, first_layer.in_dim, first_layer.num_grid_intervals + 1)

        for input_index, direction in self.monotonic_vars.items():
            if input_index < 0 or input_index >= first_layer.in_dim:
                raise ValueError(f"monotonic feature index {input_index} is outside input_dim={first_layer.in_dim}")
            if direction not in {-1, 1}:
                raise ValueError("monotonic directions must be 1 or -1")
            if direction == 1:
                first_scale_base[:, input_index].clamp_(min=0.0)
            else:
                first_scale_base[:, input_index].clamp_(max=0.0)
            first_scale_spline[:, input_index].clamp_(min=0.0)
            _project_edge_coefficients(first_coef[:, input_index], first_grid[:, input_index], direction)

        for layer in self.layers[1:]:
            layer.scale_base.clamp_(min=0.0)
            layer.scale_spline.clamp_(min=0.0)
            _project_edge_coefficients(layer.coef, layer.grid, 1)


class PaperMonoKANPolicyAdapter:
    """Adapter for evaluating MonoKAN policies in paper-style NumPy rollouts."""

    name = "MonoKAN"

    def __init__(
        self,
        model: MonoKANPolicy,
        product,
        feature_stats=None,
        max_order_factor: float = 2.0,
        scale_action_by_history_mean: bool = False,
    ) -> None:
        from inventory_control.policies.neural.fcnn import build_paper_lost_sales_features

        self.model = model
        self.product = product
        self.feature_stats = feature_stats
        self.max_order_factor = max_order_factor
        self.scale_action_by_history_mean = scale_action_by_history_mean
        self._feature_builder = build_paper_lost_sales_features

    def act(self, inventory: float, t: int, rng=None) -> float:
        """Return non-negative continuous order quantity."""
        features = torch.tensor(
            [
                self._feature_builder(
                    inventory=inventory,
                    historical_demand=self.product.historical_demand,
                    price=self.product.price,
                    purchase_cost=self.product.purchase_cost,
                    holding_cost=self.product.holding_cost,
                    lost_sale_penalty=self.product.lost_sale_penalty,
                    feature_stats=self.feature_stats,
                )
            ],
            dtype=torch.float32,
            device=infer_model_device(self.model),
        )
        self.model.eval()
        with torch.no_grad():
            order = float(self.model(features).item())
        historical_mean = max(sum(self.product.historical_demand) / len(self.product.historical_demand), 1.0)
        if self.scale_action_by_history_mean:
            order *= historical_mean
        max_inventory_level = self.max_order_factor * historical_mean
        return min(order, max(max_inventory_level - float(inventory), 0.0))


def make_paper_monokan_model(config: dict, metadata: dict | None = None) -> MonoKANPolicy:
    """Build the configured paper MonoKAN policy model."""
    source = metadata or config.get("monokan", {})
    history_length = int(source.get("history_length_H", config["history_length_H"]))
    max_order_factor = float(
        source.get(
            "max_order_factor",
            config.get("monokan", {}).get("max_order_factor", config.get("fcnn", {}).get("max_order_factor", 2.0)),
        )
    )
    return MonoKANPolicy(
        input_dim=1 + history_length + 4,
        hidden_layers=int(source.get("hidden_layers", config.get("monokan", {}).get("hidden_layers", 1))),
        hidden_units=int(source.get("hidden_units", config.get("monokan", {}).get("hidden_units", 16))),
        grid=int(source.get("grid", config.get("monokan", {}).get("grid", 5))),
        grid_range=tuple(source.get("grid_range", config.get("monokan", {}).get("grid_range", [-2.0, 4.0]))),
        noise_scale=float(source.get("noise_scale", config.get("monokan", {}).get("noise_scale", 0.01))),
        base_scale_init=_optional_float(source.get("base_scale_init", config.get("monokan", {}).get("base_scale_init"))),
        spline_scale_init=_optional_float(source.get("spline_scale_init", config.get("monokan", {}).get("spline_scale_init"))),
        bounded_base_scale_init=_optional_float(
            source.get("bounded_base_scale_init", config.get("monokan", {}).get("bounded_base_scale_init"))
        ),
        bounded_spline_scale_init=float(
            source.get("bounded_spline_scale_init", config.get("monokan", {}).get("bounded_spline_scale_init", 1.0))
        ),
        bounded_spline_slope_init=float(
            source.get("bounded_spline_slope_init", config.get("monokan", {}).get("bounded_spline_slope_init", 1.0))
        ),
        monotonic_vars={0: -1},
        initial_normalized_order=float(source.get("initial_normalized_order", 0.5 * max_order_factor)),
        output_upper_bound=max_order_factor,
    )


def make_paper_order_up_to_monokan_model(config: dict, metadata: dict | None = None) -> MonoKANPolicy:
    """Build a MonoKAN model that predicts a normalized order-up-to target."""
    return _make_paper_base_stock_output_kan_model(config, metadata=metadata, monotonic_vars={}, use_logit_residual=False)


def make_paper_base_stock_output_kan_model(config: dict, metadata: dict | None = None) -> MonoKANPolicy:
    """Build an unconstrained KAN base-stock-output policy."""
    return _make_paper_base_stock_output_kan_model(config, metadata=metadata, monotonic_vars={}, use_logit_residual=True)


def make_paper_base_stock_output_monokan_model(config: dict, metadata: dict | None = None) -> MonoKANPolicy:
    """Build a MonoKAN base-stock-output policy with context comparative statics."""
    return _make_paper_base_stock_output_kan_model(config, metadata=metadata, monotonic_vars="context", use_logit_residual=True)


def make_paper_base_stock_output_monokan_model_for_products(
    config: dict,
    products: Sequence[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats | None = None,
    metadata: dict | None = None,
) -> MonoKANPolicy:
    """Build a MonoKAN base-stock policy initialized at the empirical newsvendor target."""
    source = dict(metadata or config.get("monokan", {}))
    source["initial_normalized_target"] = empirical_newsvendor_target_ratio(products, feature_stats=feature_stats)
    return make_paper_base_stock_output_monokan_model(config, metadata=source)


def _make_paper_base_stock_output_kan_model(
    config: dict,
    monotonic_vars: dict[int, int] | str,
    metadata: dict | None = None,
    use_logit_residual: bool = False,
) -> MonoKANPolicy:
    """Build a direct base-stock-output KAN model with optional monotonic constraints."""
    source = metadata or config.get("monokan", {})
    history_length = int(source.get("history_length_H", config["history_length_H"]))
    history_feature_mode = str(source.get("paper_history_feature_mode", config.get("paper_history_feature_mode", "raw_lags")))
    history_features = paper_history_feature_count(history_length, history_feature_mode)
    resolved_monotonic_vars = (
        _paper_context_monotonic_vars(history_features, history_feature_mode)
        if monotonic_vars == "context"
        else dict(monotonic_vars)
    )
    max_order_factor = float(
        source.get(
            "max_order_factor",
            config.get("monokan", {}).get("max_order_factor", config.get("fcnn", {}).get("max_order_factor", 2.0)),
        )
    )
    initial_target = float(source.get("initial_normalized_target", 0.5 * max_order_factor))
    target_fraction = min(max(initial_target / max_order_factor, 1e-6), 1.0 - 1e-6)
    output_logit_anchor = torch.logit(torch.tensor(target_fraction)).item() if use_logit_residual else None
    model = MonoKANPolicy(
        input_dim=history_features + 4,
        hidden_layers=int(source.get("hidden_layers", config.get("monokan", {}).get("hidden_layers", 1))),
        hidden_units=int(source.get("hidden_units", config.get("monokan", {}).get("hidden_units", 16))),
        grid=int(source.get("grid", config.get("monokan", {}).get("grid", 5))),
        grid_range=tuple(source.get("grid_range", config.get("monokan", {}).get("grid_range", [-2.0, 4.0]))),
        noise_scale=float(source.get("noise_scale", config.get("monokan", {}).get("noise_scale", 0.01))),
        base_scale_init=_optional_float(source.get("base_scale_init", config.get("monokan", {}).get("base_scale_init"))),
        spline_scale_init=_optional_float(source.get("spline_scale_init", config.get("monokan", {}).get("spline_scale_init"))),
        bounded_base_scale_init=_optional_float(
            source.get("bounded_base_scale_init", config.get("monokan", {}).get("bounded_base_scale_init"))
        ),
        bounded_spline_scale_init=float(
            source.get("bounded_spline_scale_init", config.get("monokan", {}).get("bounded_spline_scale_init", 1.0))
        ),
        bounded_spline_slope_init=float(
            source.get("bounded_spline_slope_init", config.get("monokan", {}).get("bounded_spline_slope_init", 1.0))
        ),
        monotonic_vars=resolved_monotonic_vars,
        initial_normalized_order=initial_target,
        output_upper_bound=max_order_factor,
        output_logit_anchor=output_logit_anchor,
        output_logit_residual_scale=float(source.get("base_stock_logit_residual_scale", 0.25)) if use_logit_residual else 1.0,
    )
    model.paper_action_parameterization = "order_up_to"
    model.paper_scale_target_by_history_mean = bool(source.get("scale_action_by_history_mean", True))
    model.paper_history_feature_mode = history_feature_mode
    return model


def make_paper_residual_order_up_to_monokan_model(config: dict, metadata: dict | None = None) -> MonoKANPolicy:
    """Build a MonoKAN model that predicts a residual over historical base stock."""
    return _make_paper_residual_order_up_to_kan_model(config, metadata=metadata, monotonic_vars="context")


def make_paper_residual_order_up_to_kan_model(config: dict, metadata: dict | None = None) -> MonoKANPolicy:
    """Build an unconstrained KAN residual model for MONOkan ablations."""
    return _make_paper_residual_order_up_to_kan_model(config, metadata=metadata, monotonic_vars={})


def _make_paper_residual_order_up_to_kan_model(
    config: dict,
    monotonic_vars: dict[int, int] | str,
    metadata: dict | None = None,
) -> MonoKANPolicy:
    """Build a residual order-up-to KAN model with optional monotonic constraints."""
    source = metadata or config.get("monokan", {})
    history_length = int(source.get("history_length_H", config["history_length_H"]))
    history_feature_mode = str(source.get("paper_history_feature_mode", config.get("paper_history_feature_mode", "raw_lags")))
    history_features = paper_history_feature_count(history_length, history_feature_mode)
    resolved_monotonic_vars = (
        _paper_context_monotonic_vars(history_features, history_feature_mode)
        if monotonic_vars == "context"
        else dict(monotonic_vars)
    )
    model = MonoKANPolicy(
        input_dim=history_features + 4,
        hidden_layers=int(source.get("hidden_layers", config.get("monokan", {}).get("hidden_layers", 1))),
        hidden_units=int(source.get("hidden_units", config.get("monokan", {}).get("hidden_units", 16))),
        grid=int(source.get("grid", config.get("monokan", {}).get("grid", 5))),
        grid_range=tuple(source.get("grid_range", config.get("monokan", {}).get("grid_range", [-2.0, 4.0]))),
        noise_scale=float(source.get("noise_scale", config.get("monokan", {}).get("noise_scale", 0.01))),
        base_scale_init=_optional_float(source.get("base_scale_init", config.get("monokan", {}).get("base_scale_init"))),
        spline_scale_init=_optional_float(source.get("spline_scale_init", config.get("monokan", {}).get("spline_scale_init"))),
        bounded_base_scale_init=_optional_float(
            source.get("bounded_base_scale_init", config.get("monokan", {}).get("bounded_base_scale_init"))
        ),
        bounded_spline_scale_init=float(
            source.get("bounded_spline_scale_init", config.get("monokan", {}).get("bounded_spline_scale_init", 1.0))
        ),
        bounded_spline_slope_init=float(
            source.get("bounded_spline_slope_init", config.get("monokan", {}).get("bounded_spline_slope_init", 1.0))
        ),
        monotonic_vars=resolved_monotonic_vars,
        initial_normalized_order=float(source.get("initial_residual_center", 1.0)),
        output_upper_bound=2.0,
    )
    model.paper_action_parameterization = "residual_order_up_to"
    model.paper_residual_center = 1.0
    model.paper_residual_scale = float(source.get("residual_scale", 0.5))
    model.paper_residual_anchor = str(source.get("residual_anchor", "gamma"))
    model.paper_history_feature_mode = history_feature_mode
    return model


def _paper_context_monotonic_vars(history_length: int, history_feature_mode: str = "raw_lags") -> dict[int, int]:
    """Return structural directions for context-only target-stock prediction.

    Context features are historical demand followed by price, purchase cost,
    holding cost, and lost-sale penalty. Higher observed demand, selling price,
    and lost-sale penalty should not reduce the target stock; higher purchase
    and holding cost should not increase it.
    """
    if history_feature_mode == "summary":
        directions = {index: 1 for index in (0, 3, 4, 6, 7, 8)}
    else:
        directions = {index: 1 for index in range(int(history_length))}
    directions.update(
        {
            int(history_length): 1,
            int(history_length) + 1: -1,
            int(history_length) + 2: -1,
            int(history_length) + 3: 1,
        }
    )
    return directions


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def monokan_inventory_violation_rate(
    model: nn.Module,
    fixed_features: Sequence[float],
    inventory_values: Sequence[float],
) -> float:
    """Return fraction of adjacent inventory pairs violating nonincreasing orders."""
    device = infer_model_device(model)
    features = torch.tensor([list(fixed_features) for _ in inventory_values], dtype=torch.float32, device=device)
    features[:, 0] = torch.tensor(list(inventory_values), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        orders = model(features)
    if orders.numel() < 2:
        return 0.0
    violations = orders[1:] > orders[:-1] + 1e-6
    return float(violations.float().mean().item())


def sample_monokan_first_layer_splines(
    model: MonoKANPolicy,
    feature_index: int = 0,
    samples: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample first-layer edge functions for one input feature.

    Returns a common x-grid and one row per hidden/output unit. Values include
    both the base and spline terms for the selected edge, but not layer bias.
    """
    if not model.layers:
        raise ValueError("MonoKAN model has no layers")
    layer = model.layers[0]
    if feature_index < 0 or feature_index >= layer.in_dim:
        raise ValueError(f"feature_index must be in [0, {layer.in_dim})")
    if samples < 2:
        raise ValueError("samples must be at least 2")
    edge_indices = torch.arange(layer.out_dim, device=layer.grid.device) * layer.in_dim + feature_index
    selected_grid = layer.grid[edge_indices]
    x = torch.linspace(
        float(selected_grid[:, 0].min().item()),
        float(selected_grid[:, -1].max().item()),
        steps=samples,
        device=layer.grid.device,
    )
    x_eval = x.repeat(layer.out_dim, 1)
    spline_values = _evaluate_hermite_curves(x_eval, selected_grid, layer.coef[edge_indices])
    base_values = _positive_elu(x_eval)
    values = (
        layer.scale_base[edge_indices].unsqueeze(1) * base_values
        + layer.scale_spline[edge_indices].unsqueeze(1) * spline_values
    )
    return x.detach().cpu(), values.detach().cpu()
