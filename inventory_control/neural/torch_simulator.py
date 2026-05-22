"""Differentiable PyTorch simulator for zero-lead-time lost sales."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError("Install the neural extra with `python -m pip install -e '.[neural]'`.") from exc


@dataclass(frozen=True)
class DifferentiableLostSalesZeroLeadTimeConfig:
    """Configuration for differentiable zero-lead-time training."""

    horizon: int
    initial_inventory: float
    holding_cost: float
    lost_sales_cost: float
    unit_order_cost: float
    demand_lambda: float


class DifferentiableLostSalesZeroLeadTimeSimulator:
    """Batched differentiable zero-lead-time lost-sales simulator."""

    def __init__(self, config: DifferentiableLostSalesZeroLeadTimeConfig) -> None:
        self.config = config

    def sample_demand_paths(
        self,
        batch_size: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample integer demand paths as tensors."""
        rates = torch.full(
            (batch_size, self.config.horizon),
            float(self.config.demand_lambda),
            dtype=torch.float32,
        )
        return torch.poisson(rates, generator=generator)

    def rollout(self, policy, demand_paths: torch.Tensor) -> dict[str, torch.Tensor]:
        """Roll out a policy and return differentiable cost tensors."""
        batch_size = demand_paths.shape[0]
        inventory = torch.full(
            (batch_size,),
            float(self.config.initial_inventory),
            dtype=demand_paths.dtype,
            device=demand_paths.device,
        )
        total_cost = torch.zeros_like(inventory)
        orders: list[torch.Tensor] = []
        inventories: list[torch.Tensor] = []
        costs: list[torch.Tensor] = []

        demand_scale = max(float(self.config.demand_lambda), 1.0)
        cost_features = torch.tensor(
            [
                self.config.holding_cost,
                self.config.lost_sales_cost,
                self.config.unit_order_cost,
            ],
            dtype=demand_paths.dtype,
            device=demand_paths.device,
        ).expand(batch_size, -1)

        for t in range(self.config.horizon):
            features = torch.cat([(inventory / demand_scale).unsqueeze(-1), cost_features], dim=1)
            order_quantity = policy(features)
            demand = demand_paths[:, t]
            available = inventory + order_quantity
            sales = torch.minimum(available, demand)
            lost_sales = torch.relu(demand - available)
            ending_inventory = torch.relu(available - demand)
            period_cost = (
                self.config.holding_cost * ending_inventory
                + self.config.lost_sales_cost * lost_sales
                + self.config.unit_order_cost * order_quantity
            )
            total_cost = total_cost + period_cost
            inventory = ending_inventory
            orders.append(order_quantity)
            inventories.append(ending_inventory)
            costs.append(period_cost)

        return {
            "total_cost": total_cost,
            "orders": torch.stack(orders, dim=1),
            "ending_inventory": torch.stack(inventories, dim=1),
            "period_cost": torch.stack(costs, dim=1),
        }


@dataclass(frozen=True)
class DifferentiableLeadTimeProductBatch:
    """Tensor batch for product-level lead-time training."""

    initial_inventory: torch.Tensor
    historical_mean_demand: torch.Tensor
    historical_std_demand: torch.Tensor
    holding_cost: torch.Tensor
    lost_sales_cost: torch.Tensor
    unit_order_cost: torch.Tensor
    lead_time: torch.Tensor
    demand_paths: torch.Tensor


class DifferentiableLostSalesLeadTimeProductSimulator:
    """Batched differentiable lead-time simulator over heterogeneous products."""

    def __init__(self, horizon: int, max_lead_time: int) -> None:
        self.horizon = horizon
        self.max_lead_time = max_lead_time

    def rollout(self, policy, batch: DifferentiableLeadTimeProductBatch) -> dict[str, torch.Tensor]:
        """Roll out lead-time products with padded pipelines."""
        batch_size = batch.demand_paths.shape[0]
        inventory = batch.initial_inventory
        pipeline = torch.zeros(
            (batch_size, self.max_lead_time),
            dtype=batch.demand_paths.dtype,
            device=batch.demand_paths.device,
        )
        total_cost = torch.zeros_like(inventory)
        orders: list[torch.Tensor] = []
        costs: list[torch.Tensor] = []
        demand_scale = torch.clamp(batch.historical_mean_demand, min=1.0)

        for t in range(self.horizon):
            arrivals = pipeline[:, 0]
            features = torch.cat(
                [
                    (inventory / demand_scale).unsqueeze(-1),
                    pipeline / demand_scale.unsqueeze(-1),
                    (batch.historical_mean_demand / demand_scale).unsqueeze(-1),
                    (batch.historical_std_demand / demand_scale).unsqueeze(-1),
                    batch.holding_cost.unsqueeze(-1),
                    batch.lost_sales_cost.unsqueeze(-1),
                    batch.unit_order_cost.unsqueeze(-1),
                    (batch.lead_time / max(float(self.max_lead_time), 1.0)).unsqueeze(-1),
                ],
                dim=1,
            )
            order_quantity = policy(features)
            demand = batch.demand_paths[:, t]
            available = inventory + arrivals
            lost_sales = torch.relu(demand - available)
            ending_inventory = torch.relu(available - demand)
            period_cost = (
                batch.holding_cost * ending_inventory
                + batch.lost_sales_cost * lost_sales
                + batch.unit_order_cost * order_quantity
            )
            insert_mask = torch.nn.functional.one_hot(
                (batch.lead_time.long() - 1).clamp(min=0, max=self.max_lead_time - 1),
                num_classes=self.max_lead_time,
            ).to(dtype=batch.demand_paths.dtype, device=batch.demand_paths.device)
            shifted_pipeline = torch.cat([pipeline[:, 1:], torch.zeros_like(pipeline[:, :1])], dim=1)
            pipeline = shifted_pipeline + insert_mask * order_quantity.unsqueeze(-1)
            inventory = ending_inventory
            total_cost = total_cost + period_cost
            orders.append(order_quantity)
            costs.append(period_cost)

        return {
            "total_cost": total_cost,
            "orders": torch.stack(orders, dim=1),
            "period_cost": torch.stack(costs, dim=1),
        }


def make_lead_time_product_batch(
    products: Sequence,
    horizon: int,
    generator: torch.Generator | None = None,
) -> DifferentiableLeadTimeProductBatch:
    """Create a tensor batch from lead-time products using historical-demand bootstraps."""
    demand_paths = []
    for product in products:
        historical = torch.tensor(product.historical_demand, dtype=torch.float32)
        indices = torch.randint(
            low=0,
            high=len(product.historical_demand),
            size=(horizon,),
            generator=generator,
        )
        demand_paths.append(historical[indices])

    return DifferentiableLeadTimeProductBatch(
        initial_inventory=torch.tensor([p.initial_inventory for p in products], dtype=torch.float32),
        historical_mean_demand=torch.tensor(
            [float(sum(p.historical_demand) / len(p.historical_demand)) for p in products],
            dtype=torch.float32,
        ),
        historical_std_demand=torch.tensor(
            [
                float(torch.tensor(p.historical_demand, dtype=torch.float32).std(unbiased=False).item())
                for p in products
            ],
            dtype=torch.float32,
        ),
        holding_cost=torch.tensor([p.holding_cost for p in products], dtype=torch.float32),
        lost_sales_cost=torch.tensor([p.lost_sales_cost for p in products], dtype=torch.float32),
        unit_order_cost=torch.tensor([p.unit_order_cost for p in products], dtype=torch.float32),
        lead_time=torch.tensor([p.lead_time or 1 for p in products], dtype=torch.float32),
        demand_paths=torch.stack(demand_paths, dim=0),
    )
