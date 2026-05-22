"""DirectBackprop-style training loops for neural inventory policies."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError("Install the neural extra with `python -m pip install -e '.[neural]'`.") from exc

from inventory_control.neural.policies import TorchMLPPolicy
from inventory_control.neural.torch_simulator import (
    DifferentiableLostSalesLeadTimeProductSimulator,
    DifferentiableLostSalesZeroLeadTimeSimulator,
    make_lead_time_product_batch,
)


@dataclass
class TrainingResult:
    """Result of a neural policy training run."""

    policy: TorchMLPPolicy
    losses: list[float]


def train_zero_lead_time_policy(
    simulator: DifferentiableLostSalesZeroLeadTimeSimulator,
    policy: TorchMLPPolicy,
    n_steps: int = 200,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> TrainingResult:
    """Train an unconstrained MLP by backpropagating through the simulator."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    losses: list[float] = []

    for _ in range(n_steps):
        demand_paths = simulator.sample_demand_paths(batch_size=batch_size, generator=generator)
        rollout = simulator.rollout(policy, demand_paths)
        loss = rollout["total_cost"].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    return TrainingResult(policy=policy, losses=losses)


def train_lead_time_product_policy(
    products,
    policy: TorchMLPPolicy,
    horizon: int,
    max_lead_time: int,
    n_steps: int = 200,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> TrainingResult:
    """Train one lead-time policy across many products using historical demand."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    simulator = DifferentiableLostSalesLeadTimeProductSimulator(
        horizon=horizon,
        max_lead_time=max_lead_time,
    )
    losses: list[float] = []

    for _ in range(n_steps):
        indices = torch.randint(
            low=0,
            high=len(products),
            size=(batch_size,),
            generator=generator,
        ).tolist()
        product_batch = [products[index] for index in indices]
        batch = make_lead_time_product_batch(product_batch, horizon=horizon, generator=generator)
        rollout = simulator.rollout(policy, batch)
        loss = rollout["total_cost"].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    return TrainingResult(policy=policy, losses=losses)
