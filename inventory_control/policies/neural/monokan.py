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


def _evaluate_hermite_curves(
    x_eval: torch.Tensor,
    grid: torch.Tensor,
    coef: torch.Tensor,
    uniform_grid: bool = True,
) -> torch.Tensor:
    """Evaluate batched cubic Hermite curves.

    Args:
        x_eval: Tensor with shape ``(num_splines, batch)``.
        grid: Tensor with shape ``(num_splines, num_grid_points)``.
        coef: Tensor with shape ``(num_splines, num_grid_points, 2)``.
        uniform_grid: Whether every edge grid is evenly spaced. Evenly spaced
            grids use arithmetic indexing; data-adaptive grids fall back to a
            binary search over knots.

    Returns:
        Tensor with shape ``(num_splines, batch)``.
    """
    num_grid_points = grid.shape[1]
    if uniform_grid:
        # HermiteKANLayer constructs an evenly spaced grid for every edge. Arithmetic
        # indexing avoids materializing a num_splines x grid_points x batch mask.
        grid_step = torch.clamp((grid[:, -1] - grid[:, 0]) / (num_grid_points - 1), min=1e-8)
        idx = torch.floor((x_eval - grid[:, [0]]) / grid_step[:, None]).to(torch.long)
    else:
        idx = torch.searchsorted(grid.contiguous(), x_eval.contiguous().detach(), right=True) - 1
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
        self.uniform_grid = True
        coef = torch.zeros(self.num_edges, self.num_grid_intervals + 1, 2)
        coef[:, :, 0] = noise_scale * torch.randn(self.num_edges, self.num_grid_intervals + 1)
        self.coef = nn.Parameter(coef)
        edge_scale = 1.0 / max(self.in_dim, 1)
        self.scale_base = nn.Parameter(torch.full((self.num_edges,), edge_scale if base_scale_init is None else float(base_scale_init)))
        self.scale_spline = nn.Parameter(
            torch.full((self.num_edges,), edge_scale if spline_scale_init is None else float(spline_scale_init))
        )
        self.bias = nn.Parameter(torch.zeros(self.out_dim))

    def _refresh_grid_uniformity(self, tolerance: float = 1e-5) -> None:
        """Record whether every edge grid is still evenly spaced.

        Uniformity is derived from the grid buffer rather than stored, so
        checkpoints written before adaptive grids existed still load.
        """
        steps = self.grid[:, 1:] - self.grid[:, :-1]
        span = torch.clamp(self.grid[:, [-1]] - self.grid[:, [0]], min=1e-8)
        self.uniform_grid = bool(
            ((steps - span / self.num_grid_intervals).abs() / span <= tolerance).all().item()
        )

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        super()._load_from_state_dict(*args, **kwargs)
        self._refresh_grid_uniformity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return transformed layer activations."""
        batch_size = x.shape[0]
        expanded = x.unsqueeze(1).expand(batch_size, self.out_dim, self.in_dim)
        edge_inputs = expanded.reshape(batch_size, self.num_edges).transpose(0, 1)
        spline_values = _evaluate_hermite_curves(
            edge_inputs, self.grid, self.coef, uniform_grid=self.uniform_grid
        ).transpose(0, 1)
        base_values = _positive_elu(edge_inputs).transpose(0, 1)
        edge_values = self.scale_base * base_values + self.scale_spline * spline_values
        return edge_values.reshape(batch_size, self.out_dim, self.in_dim).sum(dim=2) + self.bias

    @torch.no_grad()
    def adapt_grid_to_samples(self, x: torch.Tensor, grid_eps: float = 0.1, min_span: float = 1e-3) -> None:
        """Move knots onto the empirical quantiles of the inputs each edge sees.

        This is the Hermite analogue of the grid update in Liu et al. (2024).
        Knot placement changes only where a spline is resolved, never the sign
        of its slope, so any monotone certification survives the update.
        """
        if not 0.0 <= float(grid_eps) <= 1.0:
            raise ValueError("grid_eps must be in [0, 1]")
        if x.shape[1] != self.in_dim:
            raise ValueError(f"expected {self.in_dim} input features, got {x.shape[1]}")
        probabilities = torch.linspace(0.0, 1.0, steps=self.num_grid_intervals + 1, device=x.device, dtype=x.dtype)
        # Edge e = out_index * in_dim + in_index sees input feature in_index.
        per_feature = torch.quantile(x, probabilities, dim=0).transpose(0, 1)
        uniform = torch.stack(
            [
                torch.linspace(
                    float(per_feature[index, 0]),
                    float(per_feature[index, -1]),
                    steps=self.num_grid_intervals + 1,
                    device=x.device,
                    dtype=x.dtype,
                )
                for index in range(self.in_dim)
            ]
        )
        blended = float(grid_eps) * uniform + (1.0 - float(grid_eps)) * per_feature
        # Degenerate features (constant, or fewer distinct values than knots)
        # would collapse the grid; fall back to an even span around the median.
        span = blended[:, -1] - blended[:, 0]
        degenerate = span < min_span
        if bool(degenerate.any()):
            center = blended[:, self.num_grid_intervals // 2]
            fallback = center[:, None] + torch.linspace(
                -0.5, 0.5, steps=self.num_grid_intervals + 1, device=x.device, dtype=x.dtype
            )[None, :]
            blended = torch.where(degenerate[:, None], fallback, blended)
        # Repeated quantiles would collapse an interval; a strictly increasing
        # offset keeps every knot separated without moving them noticeably.
        offsets = min_span * torch.arange(self.num_grid_intervals + 1, device=x.device, dtype=x.dtype)
        blended = torch.cummax(blended, dim=1)[0] + offsets[None, :]
        self.grid.copy_(blended.repeat(self.out_dim, 1).to(self.grid.dtype))
        self._refresh_grid_uniformity()


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

    # Each interior knot derivative is shared by the two intervals meeting at
    # that knot, so both intervals' conditions must be combined before writing.
    # Applying them as separate `[:, :-1]` and `[:, 1:]` assignments lets the
    # second write restore a derivative the first had just corrected.
    edge_pad = torch.zeros_like(delta[:, :1], dtype=torch.bool)
    flatten_knot = torch.cat([zero_delta, edge_pad], dim=1) | torch.cat([edge_pad, zero_delta], dim=1)
    coef[:, :, 1] = torch.where(flatten_knot, torch.zeros_like(coef[:, :, 1]), coef[:, :, 1])

    if direction == 1:
        coef[:, :, 1] = torch.clamp(coef[:, :, 1], min=0.0)
    else:
        coef[:, :, 1] = torch.clamp(coef[:, :, 1], max=0.0)

    safe_delta = torch.where(zero_delta, torch.full_like(delta, float("inf")), delta)
    alpha = coef[:, :-1, 1] / safe_delta
    beta = coef[:, 1:, 1] / safe_delta
    norm = torch.sqrt(alpha**2 + beta**2)
    # Fritsch-Carlson: an interval is monotone once alpha^2 + beta^2 <= 9.
    # Shrinking a derivative can only reduce a neighbour's norm, so taking the
    # tighter of the two adjacent factors satisfies both intervals at once.
    tau = torch.where(norm > 3.0, 3.0 / torch.clamp(norm, min=1e-8), torch.ones_like(norm))
    unit = torch.ones_like(delta[:, :1])
    knot_scale = torch.minimum(torch.cat([unit, tau], dim=1), torch.cat([tau, unit], dim=1))
    coef[:, :, 1] = coef[:, :, 1] * knot_scale


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
        self.adaptive_grid = False
        self.adaptive_grid_eps = 0.0
        self.sorted_history_length = 0
        self.sorted_history_offset = 0
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

    def _prepare_inputs(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the input reparameterization the network is trained on.

        Sorting the demand-lag block (which starts at ``sorted_history_offset``,
        non-zero when inventory precedes the lags) replaces exchangeable raw lags with their
        order statistics, so each first-layer spline sees one quantile of the
        observed demand distribution instead of an arbitrary chronological
        position. Every order statistic is nondecreasing in every raw lag, so
        the certified direction in raw demand coordinates is preserved.
        """
        length = int(self.sorted_history_length)
        if length <= 0:
            return features
        start = int(self.sorted_history_offset)
        stop = start + length
        sorted_history = torch.sort(features[..., start:stop], dim=-1).values
        return torch.cat([features[..., :start], sorted_history, features[..., stop:]], dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return non-negative normalized order quantity."""
        x = self._prepare_inputs(features)
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
        pre_activation = self._prepare_inputs(reference_contexts)
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
    def adapt_grids_to_contexts(self, contexts: torch.Tensor, grid_eps: float = 0.1) -> None:
        """Place every layer's knots on the data it actually sees.

        The default grid spans a fixed ``[-2, 4]`` for every edge at every
        depth. Observed contexts are neither centered nor comparably scaled:
        normalized demand lags are right-skewed with mass in the first
        interval, and normalized holding cost and lost-sales penalty span a
        fraction of one interval, so their splines cannot bend at all. Layer
        activations drift right with depth for the same reason. Refitting knots
        to per-feature quantiles restores spline resolution where the data is.

        Monotonicity is unaffected: knots decide where a spline can bend, and
        the projection in :meth:`apply_monotonic_constraints` already works on
        arbitrary knot spacing, so the certified slope signs are untouched.
        """
        activations = self._prepare_inputs(
            contexts.to(dtype=self.layers[0].bias.dtype, device=self.layers[0].bias.device)
        )
        for layer in self.layers:
            layer.adapt_grid_to_samples(activations, grid_eps=grid_eps)
            activations = layer(activations)
        self.adaptive_grid = True
        self.adaptive_grid_eps = float(grid_eps)

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


class MonoKANSoftQuantileTargetPolicy(nn.Module):
    """Order-up-to policy whose target is a soft quantile of the observed demand window.

    The flat direct-target policies ask one network to produce ``S_t / m_0``,
    which is scale free, out of demand lags that are divided by a single global
    constant while product mean demand spans three orders of magnitude. A
    monotone network with an additive first layer cannot form that ratio, so it
    settles for a nearly constant ratio and inherits the estimation error of a
    policy that barely reads its own history.

    This decoder supplies the scale equivariance instead. Writing ``z`` for the
    ascending order statistics of the observed window and ``t_k`` for fixed
    positions in ``[0, 1]``,

    ``S_t = scale * ( z_1 + sum_k sigmoid((u - t_k) / tau) * (z_k - z_{k-1}) )``

    which is a convex combination of order statistics, i.e. a smoothed empirical
    quantile at learned level ``u``. Only ``u`` is learned by a monotone KAN, and
    only from product economics -- exactly the quantity the newsvendor structure
    says should drive the target.

    Certified directions follow from the decoder rather than from a constraint on
    a 32-dimensional network:

    - the sigmoid weights are non-negative and telescope to one, and every order
      statistic is nondecreasing in every raw lag, so ``S_t`` is nondecreasing in
      all raw demand lags;
    - the order statistics are sorted, so ``S_t`` increases with ``u``, and the
      monotone level network makes ``u`` increase in price and lost-sales penalty
      and decrease in purchase and holding cost;
    - the order quantity keeps ``q_t = max(S_t - I_t, 0)``, so it stays
      nonincreasing in current inventory.

    ``tau`` is learnable and controls how many order statistics are averaged, so
    the policy can trade the variance of a raw sample quantile against the bias
    of a smoother L-statistic instead of being fixed to either.
    """

    def __init__(
        self,
        history_length: int,
        economics_dim: int = 4,
        demand_scale: float = 1.0,
        hidden_layers: int = 2,
        hidden_units: int = 8,
        grid: int = 7,
        grid_range: tuple[float, float] = (-2.0, 4.0),
        noise_scale: float = 0.01,
        monotonic_vars: dict[int, int] | None = None,
        initial_level: float = 0.85,
        initial_temperature: float = 0.05,
        gain_mode: str = "none",
        level_network: str = "monokan",
        temperature_mode: str = "position",
    ) -> None:
        super().__init__()
        if not 0.0 < initial_level < 1.0:
            raise ValueError("initial_level must lie strictly inside (0, 1)")
        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        self.history_length = int(history_length)
        self.economics_dim = int(economics_dim)
        # The decoder is architecture neutral, so the level network can be swapped
        # to separate what the monotone constraint costs from what the decoder buys.
        if level_network not in {"monokan", "kan", "mlp"}:
            raise ValueError("level_network must be 'monokan', 'kan' or 'mlp'")
        self.level_network = str(level_network)
        if self.level_network == "mlp":
            layers: list[nn.Module] = []
            previous = self.economics_dim
            for _ in range(int(hidden_layers)):
                layers.append(nn.Linear(previous, int(hidden_units)))
                layers.append(nn.ELU())
                previous = int(hidden_units)
            layers.append(nn.Linear(previous, 1))
            self.level_net = nn.Sequential(*layers)
        else:
            self.level_net = MonoKANPolicy(
                input_dim=self.economics_dim,
                hidden_layers=int(hidden_layers),
                hidden_units=int(hidden_units),
                grid=int(grid),
                grid_range=grid_range,
                noise_scale=noise_scale,
                monotonic_vars=(monotonic_vars or {}) if self.level_network == "monokan" else {},
                initial_normalized_order=0.5,
            )
        self.register_buffer("demand_scale", torch.tensor(float(demand_scale)))
        self.register_buffer("positions", torch.linspace(0.0, 1.0, steps=self.history_length))
        # "position" measures the sigmoid width as a fraction of the window, so a
        # longer window smooths over proportionally more order statistics; "count"
        # measures it in order statistics, which is what transfers when the window
        # length differs between training and deployment.
        if temperature_mode not in {"position", "count"}:
            raise ValueError("temperature_mode must be 'position' or 'count'")
        self.temperature_mode = str(temperature_mode)
        # Parameters are stored pre-activation: `raw_` fields pass through softplus
        # to a positive value, `log_` fields through exp to a multiplicative factor.
        initial_log = torch.log(torch.expm1(torch.tensor(float(initial_temperature))))
        self.raw_temperature = nn.Parameter(initial_log)
        # A convex combination of order statistics can never exceed the largest
        # observed demand, but with 32 samples and a median critical fractile of
        # about 0.9 the optimal target sits above every observation for roughly a
        # fifth of products, so that ceiling binds. A strictly positive gain lifts
        # it. ``level`` grows the gain with the quantile level, which is where the
        # extrapolation is actually needed; because the gain is positive and
        # nondecreasing in the level, and the soft quantile is itself nonnegative
        # and nondecreasing in the level, every certified direction survives.
        if gain_mode not in {"none", "global", "level"}:
            raise ValueError("gain_mode must be 'none', 'global' or 'level'")
        self.gain_mode = str(gain_mode)
        if self.gain_mode == "global":
            self.log_gain_bias = nn.Parameter(torch.zeros(()))
        elif self.gain_mode == "level":
            self.log_gain_bias = nn.Parameter(torch.zeros(()))
            self.raw_gain_slope = nn.Parameter(torch.full((), -3.0))
        self.calibrate_level(initial_level)

    def _level_logit(self, economics: torch.Tensor) -> torch.Tensor:
        if isinstance(self.level_net, MonoKANPolicy):
            # Run the layers directly so the softplus readout is bypassed; this
            # decoder applies its own sigmoid to get a quantile level.
            x = economics
            for layer in self.level_net.layers:
                x = layer(x)
            return x.squeeze(-1)
        return self.level_net(economics).squeeze(-1)

    def _readout_bias(self) -> nn.Parameter:
        if isinstance(self.level_net, MonoKANPolicy):
            return self.level_net.layers[-1].bias
        for module in reversed(list(self.level_net)):
            if isinstance(module, nn.Linear):
                return module.bias
        raise ValueError("level network has no linear readout")  # pragma: no cover

    def soft_quantile_target(self, window: torch.Tensor, economics: torch.Tensor) -> torch.Tensor:
        """Return the target stock in raw demand units for an observation window.

        ``window`` may hold any number of normalized observations, so the same
        decoder serves the fixed 32-lag window and an expanding window that grows
        with every period actually observed.
        """
        if window.shape[-1] < 2:
            raise ValueError("the soft quantile needs at least two observations")
        order_statistics = torch.sort(window, dim=-1).values
        level = torch.sigmoid(self._level_logit(economics)).unsqueeze(-1)
        temperature = torch.clamp(torch.nn.functional.softplus(self.raw_temperature), min=1e-4)
        if self.temperature_mode == "count":
            temperature = temperature / max(window.shape[-1] - 1, 1)
        positions = torch.linspace(
            0.0, 1.0, steps=window.shape[-1], device=window.device, dtype=window.dtype
        )
        gaps = order_statistics[..., 1:] - order_statistics[..., :-1]
        weights = torch.sigmoid((level - positions[1:]) / temperature)
        target = order_statistics[..., 0] + (weights * gaps).sum(dim=-1)
        if self.gain_mode == "global":
            target = target * torch.exp(self.log_gain_bias)
        elif self.gain_mode == "level":
            slope = torch.nn.functional.softplus(self.raw_gain_slope)
            target = target * torch.exp(slope * level.squeeze(-1) + self.log_gain_bias)
        return target * self.demand_scale

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return the target stock level in raw demand units."""
        if features.shape[-1] != self.history_length + self.economics_dim:
            raise ValueError(
                f"expected {self.history_length + self.economics_dim} context features, got {features.shape[-1]}"
            )
        return self.soft_quantile_target(
            features[..., : self.history_length], features[..., self.history_length :]
        )

    @torch.no_grad()
    def predicted_level(self, features: torch.Tensor) -> torch.Tensor:
        """Return the learned quantile level, for interpretability reporting."""
        return torch.sigmoid(self._level_logit(features[..., self.history_length :]))

    @torch.no_grad()
    def calibrate_level(self, target_level: float, reference_economics: torch.Tensor | None = None) -> None:
        """Shift the readout bias so the initial level sits at ``target_level``."""
        bias = self._readout_bias()
        if reference_economics is None:
            reference_economics = torch.zeros(1, self.economics_dim, dtype=bias.dtype, device=bias.device)
        reference_economics = reference_economics.to(dtype=bias.dtype, device=bias.device)
        current = torch.median(self._level_logit(reference_economics))
        target = torch.logit(torch.tensor(float(target_level), dtype=bias.dtype, device=bias.device))
        bias.add_(target - current)

    @torch.no_grad()
    def apply_monotonic_constraints(self) -> None:
        """Project the level network back into its monotone feasible set."""
        if isinstance(self.level_net, MonoKANPolicy):
            self.level_net.apply_monotonic_constraints()

    @torch.no_grad()
    def adapt_grids_to_contexts(self, contexts: torch.Tensor, grid_eps: float = 0.1) -> None:
        """Fit the level network's knots to the observed economics."""
        if isinstance(self.level_net, MonoKANPolicy):
            self.level_net.adapt_grids_to_contexts(contexts[..., self.history_length :], grid_eps=grid_eps)


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
    if bool(source.get("sort_history_lags", config.get("monokan", {}).get("sort_history_lags", False))):
        if str(model.paper_history_feature_mode) != "raw_lags":
            raise ValueError("sort_history_lags requires the raw_lags history feature mode")
        model.sorted_history_length = int(
            source.get("history_length_H", config["history_length_H"])
        )
    if bool(source.get("adaptive_grid", config.get("monokan", {}).get("adaptive_grid", False))):
        model.adapt_grids_to_contexts(
            contexts,
            grid_eps=float(source.get("adaptive_grid_eps", config.get("monokan", {}).get("adaptive_grid_eps", 0.1))),
        )
        model.apply_monotonic_constraints()
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
    context_features = paper_context_feature_count(history_length, history_feature_mode)
    if monotonic_vars == "context":
        resolved_monotonic_vars = _paper_context_monotonic_vars(history_features, history_feature_mode)
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
    model.paper_include_demand_level_feature = False
    model.paper_enforce_action_cap = False
    model.paper_initialization_mode = initialization_mode
    model.paper_roll_history = bool(source.get("roll_history", False))
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
    spline_values = _evaluate_hermite_curves(
        x_eval, selected_grid, layer.coef[edge_indices], uniform_grid=layer.uniform_grid
    )
    base_values = _positive_elu(x_eval)
    values = (
        layer.scale_base[edge_indices].unsqueeze(1) * base_values
        + layer.scale_spline[edge_indices].unsqueeze(1) * spline_values
    )
    return x.detach().cpu(), values.detach().cpu()


def make_paper_soft_quantile_target_stock_monokan_model_for_products(
    config: dict,
    products: Sequence[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
    metadata: dict | None = None,
) -> MonoKANSoftQuantileTargetPolicy:
    """Build the soft-quantile MonoKAN target policy calibrated on training contexts."""
    source = dict(metadata or config.get("monokan", {}))
    history_length = int(source.get("history_length_H", config["history_length_H"]))
    history_feature_mode = str(config.get("paper_history_feature_mode", "raw_lags"))
    if history_feature_mode != "raw_lags":
        raise ValueError("the soft-quantile target policy requires the raw_lags history feature mode")
    economics_dim = paper_context_feature_count(history_length, history_feature_mode) - history_length
    model = MonoKANSoftQuantileTargetPolicy(
        history_length=history_length,
        economics_dim=economics_dim,
        demand_scale=float(feature_stats.demand_scale),
        hidden_layers=int(source.get("level_hidden_layers", 2)),
        hidden_units=int(source.get("level_hidden_units", 8)),
        grid=int(source.get("grid", 7)),
        grid_range=tuple(source.get("grid_range", [-2.0, 4.0])),
        noise_scale=float(source.get("noise_scale", 0.01)),
        monotonic_vars=_paper_economics_monotonic_vars(),
        initial_level=float(source.get("initial_quantile_level", 0.85)),
        initial_temperature=float(source.get("initial_temperature", 0.05)),
        gain_mode=str(source.get("gain_mode", "none")),
        level_network=str(source.get("level_network", "monokan")),
        temperature_mode=str(source.get("temperature_mode", "position")),
    )
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
                history_feature_mode=history_feature_mode,
                include_newsvendor_feature=False,
            )[1:]
            for product in products
        ],
        dtype=torch.float32,
    )
    if bool(source.get("adaptive_grid", config.get("monokan", {}).get("adaptive_grid", False))):
        model.adapt_grids_to_contexts(
            contexts,
            grid_eps=float(source.get("adaptive_grid_eps", config.get("monokan", {}).get("adaptive_grid_eps", 0.1))),
        )
    model.apply_monotonic_constraints()
    model.calibrate_level(
        float(source.get("initial_quantile_level", 0.85)),
        reference_economics=contexts[:, history_length:],
    )
    model.paper_action_parameterization = "order_up_to"
    # The decoder already returns a raw-unit target, so the order-up-to path must
    # not rescale it by the product's historical mean a second time.
    model.paper_scale_target_by_history_mean = False
    model.paper_history_feature_mode = history_feature_mode
    model.paper_include_newsvendor_feature = False
    model.paper_include_demand_level_feature = False
    model.paper_enforce_action_cap = False
    # Re-estimating the target every period makes it jitter, and with
    # q_t = max(S_t - I_t, 0) that jitter is asymmetric: an upward move buys
    # stock, a downward move cannot sell it back. Under stationary demand a
    # target frozen from the initial history is therefore the better default;
    # a rolling target only pays for itself when demand actually drifts.
    model.paper_roll_history = bool(source.get("roll_history", True))
    model.paper_target_smoothing = str(source.get("target_smoothing", "none"))
    model.paper_window_mode = str(source.get("window_mode", "sliding"))
    model.paper_initialization_mode = "soft_quantile_level_calibration"
    return model


def _paper_economics_monotonic_vars() -> dict[int, int]:
    """Return certified directions for price, purchase cost, holding cost, penalty."""
    return {0: 1, 1: -1, 2: -1, 3: 1}


def _history_mean(historical_demand: Sequence[float]) -> float:
    """Return the mean observed demand used to scale inventory onto demand units."""
    values = list(historical_demand)
    return float(sum(values) / max(len(values), 1))


def _paper_raw_action_monotonic_vars(history_length: int) -> dict[int, int]:
    """Return structural directions for a direct-quantity ordering policy.

    Raw-action features are on-hand inventory, then the demand lags, then price,
    purchase cost, holding cost, and lost-sale penalty. Unlike the target-stock
    decoder, nothing here subtracts inventory from a target, so the direction
    that matters most - never order more because stock went up - has to be
    certified by the network itself.
    """
    directions = {0: -1}
    directions.update({1 + index: 1 for index in range(int(history_length))})
    directions.update(
        {
            int(history_length) + 1: 1,
            int(history_length) + 2: -1,
            int(history_length) + 3: -1,
            int(history_length) + 4: 1,
        }
    )
    return directions


def make_paper_raw_action_monokan_model_for_products(
    config: dict,
    products: Sequence[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
    metadata: dict | None = None,
) -> MonoKANPolicy:
    """Build a direct-quantity MonoKAN that emits the order itself.

    The target-stock decoder obtains ``q = max(S - I, 0)`` monotonicity in
    inventory from the subtraction, so every variant satisfies it and the
    certified constraint only binds on the economics. Taking inventory as an
    input and emitting the order directly is the setting where the constraint
    actually does work, so this factory is what the monotone comparison needs.
    """
    source = dict(metadata or config.get("monokan", {}))
    history_length = int(source.get("history_length_H", config["history_length_H"]))
    history_feature_mode = str(config.get("paper_history_feature_mode", "raw_lags"))
    if history_feature_mode != "raw_lags":
        raise ValueError("the raw-action policy requires the raw_lags history feature mode")
    initial_order = empirical_newsvendor_target_ratio(products, feature_stats=feature_stats)
    # Dropping the constraints leaves the same architecture. By default ("auto",
    # as in the reported runs) the two variants are initialised differently below:
    # an open readout for the certified model, a constant output otherwise.
    # "open" or "constant" forces one start for either variant, so the constraint
    # can be compared from an identical initialisation.
    monotone = bool(source.get("monotone", True))
    initialization = str(source.get("initialization", "auto"))
    if initialization not in {"auto", "open", "constant"}:
        raise ValueError("initialization must be 'auto', 'open' or 'constant'")
    open_readout = monotone if initialization == "auto" else initialization == "open"
    model = MonoKANPolicy(
        input_dim=1 + history_length + 4,
        hidden_layers=int(source.get("hidden_layers", 1)),
        hidden_units=int(source.get("hidden_units", 16)),
        grid=int(source.get("grid", 5)),
        grid_range=tuple(source.get("grid_range", [-2.0, 4.0])),
        noise_scale=float(source.get("noise_scale", 0.01)),
        monotonic_vars=_paper_raw_action_monotonic_vars(history_length) if monotone else {},
        initial_normalized_order=initial_order,
        output_upper_bound=None,
    )
    # Grids must span the inventory the policy will actually meet, so the
    # calibration contexts sweep stock levels instead of evaluating at zero.
    inventory_levels = torch.linspace(0.0, 2.0, 5).tolist()
    contexts = torch.tensor(
        [
            build_paper_lost_sales_features(
                inventory=float(inventory) * _history_mean(product.historical_demand),
                historical_demand=product.historical_demand,
                price=product.price,
                purchase_cost=product.purchase_cost,
                holding_cost=product.holding_cost,
                lost_sale_penalty=product.lost_sale_penalty,
                feature_stats=feature_stats,
                history_feature_mode=history_feature_mode,
                include_newsvendor_feature=False,
            )
            for product in products
            for inventory in inventory_levels
        ],
        dtype=torch.float32,
    )
    if bool(source.get("sort_history_lags", config.get("monokan", {}).get("sort_history_lags", False))):
        model.sorted_history_length = history_length
        model.sorted_history_offset = 1
    if bool(source.get("adaptive_grid", config.get("monokan", {}).get("adaptive_grid", False))):
        model.adapt_grids_to_contexts(
            contexts,
            grid_eps=float(source.get("adaptive_grid_eps", config.get("monokan", {}).get("adaptive_grid_eps", 0.1))),
        )
    model.apply_monotonic_constraints()
    if open_readout:
        model.initialize_open_output(initial_order, reference_contexts=contexts)
    else:
        model.initialize_constant_output(initial_order)
    model.paper_action_parameterization = "raw"
    model.paper_history_feature_mode = history_feature_mode
    model.paper_include_newsvendor_feature = False
    model.paper_include_demand_level_feature = False
    model.paper_enforce_action_cap = False
    model.paper_initialization_mode = (
        "empirical_newsvendor_open_readout" if open_readout else "global_empirical_newsvendor_median"
    )
    model.paper_roll_history = bool(source.get("roll_history", True))
    return model
