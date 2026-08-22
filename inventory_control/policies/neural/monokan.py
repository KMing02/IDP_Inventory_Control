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
    build_paper_lost_sales_features,
    empirical_newsvendor_target_ratio,
    infer_model_device,
    paper_context_feature_count,
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
    # HermiteKANLayer constructs an evenly spaced grid for every edge. Arithmetic
    # indexing avoids materializing a num_splines x grid_points x batch mask.
    grid_step = torch.clamp((grid[:, -1] - grid[:, 0]) / (num_grid_points - 1), min=1e-8)
    idx = torch.floor((x_eval - grid[:, [0]]) / grid_step[:, None]).to(torch.long)
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
        output_logit_anchor_feature_index: int | None = None,
        output_logit_residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.monotonic_vars = dict(monotonic_vars or {})
        self.output_upper_bound = float(output_upper_bound) if output_upper_bound is not None else None
        self.output_logit_anchor = float(output_logit_anchor) if output_logit_anchor is not None else None
        self.output_logit_anchor_feature_index = (
            int(output_logit_anchor_feature_index) if output_logit_anchor_feature_index is not None else None
        )
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
            if self.output_logit_anchor_feature_index is not None:
                anchor = features[:, self.output_logit_anchor_feature_index]
                fraction = torch.clamp(anchor / self.output_upper_bound, min=1e-6, max=1.0 - 1e-6)
                x = torch.logit(fraction).unsqueeze(-1) + self.output_logit_residual_scale * x
            elif self.output_logit_anchor is not None:
                x = self.output_logit_anchor + self.output_logit_residual_scale * x
            return (self.output_upper_bound * torch.sigmoid(x)).squeeze(-1)
        return torch.nn.functional.softplus(x).squeeze(-1)

    @torch.no_grad()
    def _calibrate_initial_output(self, target_order: float) -> None:
        """Set the final bias to a stable normalized order at a reference input."""
        if self.output_upper_bound is not None and self.output_logit_anchor_feature_index is not None:
            final_layer = self.layers[-1]
            final_layer.scale_base.zero_()
            final_layer.scale_spline.zero_()
            final_layer.coef.zero_()
            final_layer.bias.zero_()
            return
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
        elif self.output_logit_anchor is not None or self.output_logit_anchor_feature_index is not None:
            target_pre_activation = torch.zeros((), device=x.device)
        else:
            target_fraction = torch.tensor(target_order / self.output_upper_bound, device=x.device)
            target_fraction = torch.clamp(target_fraction, min=1e-6, max=1 - 1e-6)
            target_pre_activation = torch.logit(target_fraction)
        self.layers[-1].bias.add_(target_pre_activation - x.squeeze())

    @torch.no_grad()
    def initialize_constant_output(self, target_value: float) -> None:
        """Initialize a constant output without permanently disabling final splines.

        Zero edge scales make the initial policy exactly context independent. The
        spline coefficients remain initialized so scale gradients can open the
        spline path on the first optimization step; zeroing both multiplicative
        factors would leave that path permanently dead. Non-final edge scales
        receive independent positive magnitudes (and the required sign for
        decreasing inputs) so hidden units do not start as duplicate functions.
        """
        if target_value <= 0:
            raise ValueError("target_value must be positive")
        self._diversify_nonfinal_edge_scales()
        final_layer = self.layers[-1]
        final_layer.scale_base.zero_()
        final_layer.scale_spline.zero_()
        final_layer.bias.fill_(torch.log(torch.expm1(torch.tensor(float(target_value)))).item())
        self.apply_monotonic_constraints()

    @torch.no_grad()
    def initialize_open_output(
        self,
        target_value: float,
        reference_contexts: torch.Tensor | None = None,
    ) -> None:
        """Calibrate the output without cutting gradients to preceding layers."""
        if target_value <= 0:
            raise ValueError("target_value must be positive")
        self._diversify_nonfinal_edge_scales()
        final_layer = self.layers[-1]
        nominal_scale = 1.0 / max(final_layer.in_dim, 1)
        final_layer.scale_base.fill_(nominal_scale)
        final_layer.scale_spline.fill_(nominal_scale)
        final_layer.bias.zero_()
        self.apply_monotonic_constraints()

        if reference_contexts is None:
            reference_contexts = torch.zeros(
                1,
                self.layers[0].in_dim,
                dtype=final_layer.bias.dtype,
                device=final_layer.bias.device,
            )
        else:
            reference_contexts = reference_contexts.to(
                dtype=final_layer.bias.dtype,
                device=final_layer.bias.device,
            )
        pre_activation = reference_contexts
        for layer in self.layers:
            pre_activation = layer(pre_activation)
        reference_pre_activation = torch.median(pre_activation.squeeze(-1))
        target_pre_activation = torch.log(
            torch.expm1(
                torch.tensor(
                    float(target_value),
                    dtype=final_layer.bias.dtype,
                    device=final_layer.bias.device,
                )
            )
        )
        final_layer.bias.add_(target_pre_activation - reference_pre_activation)

    @torch.no_grad()
    def _diversify_nonfinal_edge_scales(self) -> None:
        """Break hidden-unit symmetry while preserving configured monotone signs."""
        for layer in self.layers[:-1]:
            nominal_scale = 1.0 / max(layer.in_dim, 1)
            base_magnitude = nominal_scale * (0.5 + torch.rand_like(layer.scale_base))
            spline_magnitude = nominal_scale * (0.5 + torch.rand_like(layer.scale_spline))
            layer.scale_base.copy_(base_magnitude)
            layer.scale_spline.copy_(spline_magnitude)

        if not self.layers:
            return
        first_layer = self.layers[0]
        first_scale_base = first_layer.scale_base.view(first_layer.out_dim, first_layer.in_dim)
        first_scale_spline = first_layer.scale_spline.view(first_layer.out_dim, first_layer.in_dim)
        for input_index, direction in self.monotonic_vars.items():
            first_scale_base[:, input_index].copy_(
                float(direction) * first_scale_base[:, input_index].abs()
            )
            first_scale_spline[:, input_index].abs_()

    @torch.no_grad()
    def apply_monotonic_constraints(self) -> None:
        """Project layer parameters to the configured monotone feasible set."""
        if not self.layers or not self.monotonic_vars:
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


def paper_raw_action_monotonic_vars(history_length: int) -> dict[int, int]:
    """Return hard directions for canonical raw paper-policy features."""
    history_length = int(history_length)
    return {
        0: -1,
        **{index: 1 for index in range(1, history_length + 1)},
        history_length + 1: 1,
        history_length + 2: -1,
        history_length + 3: -1,
        history_length + 4: 1,
    }


def make_paper_deep_grid7_raw_action_monokan_model(
    config: dict,
    metadata: dict | None = None,
) -> MonoKANPolicy:
    """Build the frozen 37->16->16->1 raw-action DeepGrid7 MonoKAN."""
    defaults = config.get("monokan", {})
    source = metadata or defaults
    history_length = int(source.get("history_length_H", config["history_length_H"]))
    model = MonoKANPolicy(
        input_dim=1 + history_length + 4,
        hidden_layers=2,
        hidden_units=16,
        grid=7,
        grid_range=tuple(source.get("grid_range", defaults.get("grid_range", [-2.0, 4.0]))),
        noise_scale=float(source.get("noise_scale", defaults.get("noise_scale", 0.01))),
        base_scale_init=_optional_float(source.get("base_scale_init", defaults.get("base_scale_init"))),
        spline_scale_init=_optional_float(source.get("spline_scale_init", defaults.get("spline_scale_init"))),
        monotonic_vars=paper_raw_action_monotonic_vars(history_length),
        initial_normalized_order=float(source.get("initial_normalized_order", 0.5)),
        output_upper_bound=None,
    )
    model.paper_action_parameterization = "raw"
    model.paper_history_feature_mode = "raw_lags"
    model.paper_roll_history = True
    model.paper_enforce_action_cap = False
    model.paper_scale_action_by_history_mean = False
    model.paper_architecture = "raw_action_deepgrid7"
    return model


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


def make_paper_direct_target_stock_kan_model(config: dict, metadata: dict | None = None) -> MonoKANPolicy:
    """Build an unconstrained KAN that directly predicts a positive target-stock ratio."""
    return _make_paper_direct_target_stock_kan_model(config, metadata=metadata, monotonic_vars={})


def make_paper_direct_target_stock_monokan_model(config: dict, metadata: dict | None = None) -> MonoKANPolicy:
    """Build a MonoKAN that directly predicts a positive context-monotone target-stock ratio."""
    return _make_paper_direct_target_stock_kan_model(config, metadata=metadata, monotonic_vars="context")


def make_paper_deep_grid7_direct_target_stock_kan_model(
    config: dict,
    metadata: dict | None = None,
) -> MonoKANPolicy:
    """Build the fixed 2x16, grid-7 KAN used by the capacity diagnostic."""
    source = dict(metadata or config.get("monokan", {}))
    source.update({"hidden_layers": 2, "hidden_units": 16, "grid": 7})
    return make_paper_direct_target_stock_kan_model(config, metadata=source)


def make_paper_deep_grid7_direct_target_stock_monokan_model(
    config: dict,
    metadata: dict | None = None,
) -> MonoKANPolicy:
    """Build the fixed 2x16, grid-7 MonoKAN used by the capacity diagnostic."""
    source = dict(metadata or config.get("monokan", {}))
    source.update({"hidden_layers": 2, "hidden_units": 16, "grid": 7})
    return make_paper_direct_target_stock_monokan_model(config, metadata=source)


def make_paper_selective_direct_target_stock_kan_model(config: dict, metadata: dict | None = None) -> MonoKANPolicy:
    """Build an unconstrained KAN with the explicit demand-level feature used by selective MonoKAN."""
    return _make_paper_direct_target_stock_kan_model(config, metadata=metadata, monotonic_vars="selective_context")


def make_paper_selective_direct_target_stock_monokan_model(
    config: dict,
    metadata: dict | None = None,
) -> MonoKANPolicy:
    """Constrain demand level and economics while leaving individual raw lags unconstrained."""
    return _make_paper_direct_target_stock_kan_model(config, metadata=metadata, monotonic_vars="selective_context_monotone")


def make_paper_selective_direct_target_stock_monokan_model_for_products(
    config: dict,
    products: Sequence[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
    metadata: dict | None = None,
) -> MonoKANPolicy:
    """Calibrate selective MonoKAN's open readout on observable training contexts."""
    source = dict(metadata or config.get("monokan", {}))
    initial_target = empirical_newsvendor_target_ratio(products, feature_stats=feature_stats)
    source["initial_normalized_target"] = initial_target
    model = make_paper_selective_direct_target_stock_monokan_model(config, metadata=source)
    contexts = torch.tensor(
        [
            build_paper_lost_sales_features(
                inventory=0.0,
                historical_demand=product.historical_demand,
                price=product.price,
                purchase_cost=product.purchase_cost,
                holding_cost=product.holding_cost,
                lost_sale_penalty=product.lost_sale_penalty,
                feature_stats=feature_stats,
                history_feature_mode=str(model.paper_history_feature_mode),
                include_demand_level_feature=True,
            )[1:]
            for product in products
        ],
        dtype=torch.float32,
    )
    model.initialize_open_output(initial_target, reference_contexts=contexts)
    model.paper_initialization_mode = "empirical_newsvendor_open_readout_selective_context"
    return model


def make_paper_direct_target_stock_monokan_model_for_products(
    config: dict,
    products: Sequence[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
    metadata: dict | None = None,
) -> MonoKANPolicy:
    """Build direct-target MonoKAN calibrated on observable training contexts."""
    source = dict(metadata or config.get("monokan", {}))
    initial_target = empirical_newsvendor_target_ratio(products, feature_stats=feature_stats)
    source["initial_normalized_target"] = initial_target
    model = make_paper_direct_target_stock_monokan_model(config, metadata=source)
    contexts = torch.tensor(
        [
            build_paper_lost_sales_features(
                inventory=0.0,
                historical_demand=product.historical_demand,
                price=product.price,
                purchase_cost=product.purchase_cost,
                holding_cost=product.holding_cost,
                lost_sale_penalty=product.lost_sale_penalty,
                feature_stats=feature_stats,
                history_feature_mode=str(model.paper_history_feature_mode),
                include_newsvendor_feature=False,
            )[1:]
            for product in products
        ],
        dtype=torch.float32,
    )
    model.initialize_open_output(initial_target, reference_contexts=contexts)
    model.paper_initialization_mode = "empirical_newsvendor_open_readout"
    return model


def make_paper_deep_grid7_direct_target_stock_monokan_model_for_products(
    config: dict,
    products: Sequence[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
    metadata: dict | None = None,
) -> MonoKANPolicy:
    """Calibrate the fixed DeepGrid7 MonoKAN on observable training contexts."""
    source = dict(metadata or config.get("monokan", {}))
    source.update({"hidden_layers": 2, "hidden_units": 16, "grid": 7})
    return make_paper_direct_target_stock_monokan_model_for_products(
        config,
        products,
        feature_stats,
        metadata=source,
    )


def _make_paper_direct_target_stock_kan_model(
    config: dict,
    monotonic_vars: dict[int, int] | str,
    metadata: dict | None = None,
) -> MonoKANPolicy:
    """Build a pure direct target-stock KAN without a persistent heuristic anchor or output cap."""
    source = metadata or config.get("monokan", {})
    history_length = int(source.get("history_length_H", config["history_length_H"]))
    history_feature_mode = str(
        source.get("paper_history_feature_mode", config.get("paper_history_feature_mode", "raw_lags"))
    )
    history_features = paper_history_feature_count(history_length, history_feature_mode)
    include_demand_level_feature = isinstance(monotonic_vars, str) and monotonic_vars in {
        "selective_context",
        "selective_context_monotone",
    }
    context_features = paper_context_feature_count(
        history_length,
        history_feature_mode,
        include_demand_level_feature=include_demand_level_feature,
    )
    if monotonic_vars == "context":
        resolved_monotonic_vars = _paper_context_monotonic_vars(history_features, history_feature_mode)
    elif monotonic_vars == "selective_context_monotone":
        resolved_monotonic_vars = _paper_selective_context_monotonic_vars(history_features)
    elif monotonic_vars == "selective_context":
        resolved_monotonic_vars = {}
    else:
        resolved_monotonic_vars = dict(monotonic_vars)
    initial_target = float(source.get("initial_normalized_target", 1.0))
    model = MonoKANPolicy(
        input_dim=context_features,
        hidden_layers=int(source.get("hidden_layers", config.get("monokan", {}).get("hidden_layers", 1))),
        hidden_units=int(source.get("hidden_units", config.get("monokan", {}).get("hidden_units", 16))),
        grid=int(source.get("grid", config.get("monokan", {}).get("grid", 5))),
        grid_range=tuple(source.get("grid_range", config.get("monokan", {}).get("grid_range", [-2.0, 4.0]))),
        noise_scale=float(source.get("noise_scale", config.get("monokan", {}).get("noise_scale", 0.01))),
        base_scale_init=_optional_float(source.get("base_scale_init", config.get("monokan", {}).get("base_scale_init"))),
        spline_scale_init=_optional_float(
            source.get("spline_scale_init", config.get("monokan", {}).get("spline_scale_init"))
        ),
        monotonic_vars=resolved_monotonic_vars,
        initial_normalized_order=initial_target,
        output_upper_bound=None,
    )
    if resolved_monotonic_vars:
        model.initialize_open_output(initial_target)
        initialization_mode = "empirical_newsvendor_open_readout_zero_context"
    else:
        model.initialize_constant_output(initial_target)
        initialization_mode = "global_empirical_newsvendor_median"
    model.paper_action_parameterization = "order_up_to"
    model.paper_scale_target_by_history_mean = True
    model.paper_history_feature_mode = history_feature_mode
    model.paper_include_newsvendor_feature = False
    model.paper_include_demand_level_feature = include_demand_level_feature
    model.paper_enforce_action_cap = False
    model.paper_initialization_mode = initialization_mode
    model.paper_roll_history = bool(source.get("roll_history", False))
    return model


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
    include_newsvendor_feature = bool(
        source.get("paper_include_newsvendor_feature", config.get("paper_include_newsvendor_feature", False))
    )
    history_features = paper_history_feature_count(history_length, history_feature_mode)
    context_features = paper_context_feature_count(history_length, history_feature_mode, include_newsvendor_feature)
    resolved_monotonic_vars = (
        _paper_context_monotonic_vars(history_features, history_feature_mode, include_newsvendor_feature)
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
        input_dim=context_features,
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
        output_logit_anchor_feature_index=context_features - 1 if use_logit_residual and include_newsvendor_feature else None,
        output_logit_residual_scale=float(source.get("base_stock_logit_residual_scale", 0.25)) if use_logit_residual else 1.0,
    )
    model.paper_action_parameterization = "order_up_to"
    model.paper_scale_target_by_history_mean = bool(source.get("scale_action_by_history_mean", True))
    model.paper_history_feature_mode = history_feature_mode
    model.paper_include_newsvendor_feature = include_newsvendor_feature
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
    include_newsvendor_feature = bool(
        source.get("paper_include_newsvendor_feature", config.get("paper_include_newsvendor_feature", False))
    )
    history_features = paper_history_feature_count(history_length, history_feature_mode)
    context_features = paper_context_feature_count(history_length, history_feature_mode, include_newsvendor_feature)
    resolved_monotonic_vars = (
        _paper_context_monotonic_vars(history_features, history_feature_mode, include_newsvendor_feature)
        if monotonic_vars == "context"
        else dict(monotonic_vars)
    )
    model = MonoKANPolicy(
        input_dim=context_features,
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
    model.paper_include_newsvendor_feature = include_newsvendor_feature
    return model


def _paper_context_monotonic_vars(
    history_length: int,
    history_feature_mode: str = "raw_lags",
    include_newsvendor_feature: bool = False,
) -> dict[int, int]:
    """Return structural directions for context-only target-stock prediction.

    Context features are historical demand followed by price, purchase cost,
    holding cost, and lost-sale penalty. Higher observed demand, selling price,
    and lost-sale penalty should not reduce the target stock; higher purchase
    and holding cost should not increase it.
    """
    if history_feature_mode == "summary":
        directions = {index: 1 for index in (0, 3, 4, 6, 7, 8)}
    elif history_feature_mode == "raw_lags_temporal":
        raw_history_length = int(history_length) - 5
        directions = {index: 1 for index in range(raw_history_length)}
        directions.update({raw_history_length + index: 1 for index in range(3)})
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
    if include_newsvendor_feature:
        directions[int(history_length) + 4] = 1
    return directions


def _paper_selective_context_monotonic_vars(history_features: int) -> dict[int, int]:
    """Constrain an explicit demand-level summary and economics, not individual demand lags."""
    demand_level_index = int(history_features)
    return {
        demand_level_index: 1,
        demand_level_index + 1: 1,
        demand_level_index + 2: -1,
        demand_level_index + 3: -1,
        demand_level_index + 4: 1,
    }


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
