"""PyTorch neural inventory policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError("Install the neural extra with `python -m pip install -e '.[neural]'`.") from exc


class TorchMLPPolicy(nn.Module):
    """Unconstrained MLP policy for zero-lead-time inventory control."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (32, 32),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return non-negative continuous order quantities."""
        return torch.nn.functional.softplus(self.network(features)).squeeze(-1)


def lead_time_product_feature_vector(
    state: Sequence[float],
    historical_mean_demand: float,
    historical_std_demand: float,
    holding_cost: float,
    lost_sales_cost: float,
    unit_order_cost: float,
    lead_time: int,
    max_lead_time: int,
) -> list[float]:
    """Build observable lead-time product features for a neural policy."""
    demand_scale = max(historical_mean_demand, 1.0)
    on_hand = float(state[0])
    pipeline = [float(value) for value in state[1:]]
    padded_pipeline = pipeline[:max_lead_time] + [0.0] * max(max_lead_time - len(pipeline), 0)
    return [
        on_hand / demand_scale,
        *[value / demand_scale for value in padded_pipeline],
        historical_mean_demand / demand_scale,
        historical_std_demand / demand_scale,
        holding_cost,
        lost_sales_cost,
        unit_order_cost,
        float(lead_time) / max(float(max_lead_time), 1.0),
    ]


@dataclass
class TorchPolicyAdapter:
    """Adapter that lets a trained Torch policy run in the true simulator."""

    model: TorchMLPPolicy
    demand_scale: float
    holding_cost: float
    lost_sales_cost: float
    unit_order_cost: float
    round_action: bool = True
    name: str = "TorchMLPPolicy"

    def reset(self, seed: int | None = None) -> None:
        """Reset policy state. The MLP policy is stateless."""

    def act(self, inventory: int | float, t: int, rng=None) -> int:
        """Return a simulator-compatible non-negative order quantity."""
        self.model.eval()
        inventory_value = float(inventory[0] if isinstance(inventory, list | tuple) else inventory)
        features = torch.tensor(
            [[
                inventory_value / self.demand_scale,
                self.holding_cost,
                self.lost_sales_cost,
                self.unit_order_cost,
            ]],
            dtype=torch.float32,
        )
        with torch.no_grad():
            action = float(self.model(features).item())
        if self.round_action:
            return max(int(round(action)), 0)
        return max(int(action), 0)


@dataclass
class LeadTimeTorchPolicyAdapter:
    """Adapter for evaluating product-level lead-time Torch policies."""

    model: TorchMLPPolicy
    historical_mean_demand: float
    historical_std_demand: float
    holding_cost: float
    lost_sales_cost: float
    unit_order_cost: float
    lead_time: int
    max_lead_time: int
    round_action: bool = True
    name: str = "LeadTimeTorchMLPPolicy"

    def reset(self, seed: int | None = None) -> None:
        """Reset policy state. The MLP policy is stateless."""

    def act(self, state: Sequence[float], t: int, rng=None) -> int:
        """Return a true-simulator compatible order quantity."""
        self.model.eval()
        features = torch.tensor(
            [
                lead_time_product_feature_vector(
                    state=state,
                    historical_mean_demand=self.historical_mean_demand,
                    historical_std_demand=self.historical_std_demand,
                    holding_cost=self.holding_cost,
                    lost_sales_cost=self.lost_sales_cost,
                    unit_order_cost=self.unit_order_cost,
                    lead_time=self.lead_time,
                    max_lead_time=self.max_lead_time,
                )
            ],
            dtype=torch.float32,
        )
        with torch.no_grad():
            action = float(self.model(features).item())
        if self.round_action:
            return max(int(round(action)), 0)
        return max(int(action), 0)
