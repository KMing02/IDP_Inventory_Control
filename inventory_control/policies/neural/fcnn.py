"""Fully connected neural policy for Maggiar-style lost-sales replication."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install neural dependencies with `python -m pip install -r requirements-neural.txt`.") from exc

from inventory_control.products import PaperLostSalesProduct


@dataclass(frozen=True)
class PaperFCNNFeatureStats:
    """Feature scales computed from training products only."""

    demand_scale: float
    price_scale: float


def compute_paper_fcnn_feature_stats(products: Sequence[PaperLostSalesProduct]) -> PaperFCNNFeatureStats:
    """Compute FCNN normalization scales from observable training data."""
    if not products:
        return PaperFCNNFeatureStats(demand_scale=1.0, price_scale=1.0)
    historical_values = [float(value) for product in products for value in product.historical_demand]
    prices = [float(product.price) for product in products]
    demand_scale = max(sum(historical_values) / len(historical_values), 1.0)
    price_scale = max(sum(prices) / len(prices), 1.0)
    return PaperFCNNFeatureStats(demand_scale=demand_scale, price_scale=price_scale)


class FCNNPolicy(nn.Module):
    """FCNN mapping observable features to non-negative order quantity."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: int = 2,
        hidden_units: int = 32,
        activation: str = "ELU",
    ) -> None:
        super().__init__()
        activation_cls = getattr(nn, activation)
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(previous_dim, hidden_units))
            layers.append(activation_cls())
            previous_dim = hidden_units
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return non-negative continuous order quantity."""
        return torch.nn.functional.softplus(self.network(features)).squeeze(-1)


def build_paper_lost_sales_features(
    inventory: float,
    historical_demand: Sequence[float],
    price: float,
    purchase_cost: float,
    holding_cost: float,
    lost_sale_penalty: float,
    feature_stats: PaperFCNNFeatureStats | None = None,
) -> list[float]:
    """Build observable FCNN features without hidden Gamma parameters."""
    history = [float(value) for value in historical_demand]
    if feature_stats is None:
        demand_scale = max(sum(history) / len(history), 1.0)
        price_scale = max(float(price), 1.0)
    else:
        demand_scale = feature_stats.demand_scale
        price_scale = feature_stats.price_scale
    return [
        float(inventory) / demand_scale,
        *[value / demand_scale for value in history],
        float(price) / price_scale,
        float(purchase_cost) / price_scale,
        float(holding_cost) / price_scale,
        float(lost_sale_penalty) / price_scale,
    ]


class PaperFCNNPolicyAdapter:
    """Adapter for evaluating FCNN policies in paper-style NumPy rollouts."""

    name = "FCNN"

    def __init__(
        self,
        model: FCNNPolicy,
        product: PaperLostSalesProduct,
        feature_stats: PaperFCNNFeatureStats | None = None,
        max_order_factor: float = 2.0,
        max_order_multiplier: float | None = None,
    ) -> None:
        self.model = model
        self.product = product
        self.feature_stats = feature_stats
        self.max_order_factor = max_order_factor if max_order_multiplier is None else max_order_multiplier

    def act(self, inventory: float, t: int, rng=None) -> float:
        """Return non-negative continuous order quantity."""
        features = torch.tensor(
            [
                build_paper_lost_sales_features(
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
        max_inventory_level = self.max_order_factor * historical_mean
        order_cap = max(max_inventory_level - float(inventory), 0.0)
        return min(order, order_cap)
