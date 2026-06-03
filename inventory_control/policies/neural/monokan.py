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
        coef[:, :, 1] = 1.0
        self.coef = nn.Parameter(coef)
        self.scale_base = nn.Parameter(torch.ones(self.num_edges) / max(self.in_dim, 1) ** 0.5)
        self.scale_spline = nn.Parameter(torch.ones(self.num_edges))
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
        monotonic_vars: dict[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.monotonic_vars = dict(monotonic_vars or {})
        widths = [int(input_dim)] + [int(hidden_units)] * int(hidden_layers) + [1]
        self.layers = nn.ModuleList(
            [
                HermiteKANLayer(widths[index], widths[index + 1], grid=grid, grid_range=grid_range)
                for index in range(len(widths) - 1)
            ]
        )
        self.apply_monotonic_constraints()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return non-negative normalized order quantity."""
        x = features
        for layer in self.layers:
            x = layer(x)
        return torch.nn.functional.softplus(x).squeeze(-1)

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
    return MonoKANPolicy(
        input_dim=1 + history_length + 4,
        hidden_layers=int(source.get("hidden_layers", config.get("monokan", {}).get("hidden_layers", 1))),
        hidden_units=int(source.get("hidden_units", config.get("monokan", {}).get("hidden_units", 16))),
        grid=int(source.get("grid", config.get("monokan", {}).get("grid", 5))),
        grid_range=tuple(source.get("grid_range", config.get("monokan", {}).get("grid_range", [-2.0, 4.0]))),
        monotonic_vars={0: -1},
    )


def monokan_inventory_violation_rate(
    model: nn.Module,
    fixed_features: Sequence[float],
    inventory_values: Sequence[float],
) -> float:
    """Return fraction of adjacent inventory pairs violating nonincreasing orders."""
    features = torch.tensor([list(fixed_features) for _ in inventory_values], dtype=torch.float32)
    features[:, 0] = torch.tensor(list(inventory_values), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        orders = model(features)
    if orders.numel() < 2:
        return 0.0
    violations = orders[1:] > orders[:-1] + 1e-6
    return float(violations.float().mean().item())
