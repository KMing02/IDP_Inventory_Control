"""DirectBackprop training for paper-style lost-sales replication."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import copy
import time

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install neural dependencies with `python -m pip install -r requirements-neural.txt`.") from exc

from torch import nn

from inventory_control.policies.neural.fcnn import FCNNPolicy, PaperFCNNFeatureStats
from inventory_control.paper_lost_sales import historical_demand_base_stock_policy
from inventory_control.products import PaperLostSalesProduct


PAPER_FCNN_MAX_ORDER_FACTOR = 2.0


@dataclass
class DirectBackpropResult:
    """Training result for DirectBackprop FCNN."""

    model: nn.Module
    training_losses: list[float]
    mean_total_rewards: list[float]
    validation_rewards: list[float] | None = None
    validation_epochs: list[int] | None = None
    best_validation_reward: float | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    training_log: list[dict[str, float | int]] | None = None
    monotonicity_penalties: list[float] | None = None
    elapsed_seconds: float = 0.0

    @property
    def losses(self) -> list[float]:
        """Backward-compatible alias for training losses."""
        return self.training_losses


def make_paper_lost_sales_training_batch(
    products: list[PaperLostSalesProduct],
    horizon: int,
    generator: torch.Generator,
    feature_stats: PaperFCNNFeatureStats | None = None,
) -> dict[str, torch.Tensor]:
    """Build a tensor batch using only historical demand observations."""
    demand_paths = []
    histories = []
    demand_scale = 1.0
    price_scale = 1.0
    if feature_stats is not None:
        demand_scale = feature_stats.demand_scale
        price_scale = feature_stats.price_scale
    for product in products:
        history = torch.tensor(product.historical_demand, dtype=torch.float32)
        if product.future_demand is not None and len(product.future_demand) >= horizon:
            demand_paths.append(torch.tensor(product.future_demand[:horizon], dtype=torch.float32))
        else:
            indices = torch.randint(0, len(product.historical_demand), (horizon,), generator=generator)
            demand_paths.append(history[indices])
        scale = torch.tensor(demand_scale, dtype=torch.float32) if feature_stats is not None else torch.clamp(history.mean(), min=1.0)
        histories.append(history / scale)
    return {
        "initial_inventory": torch.tensor(
            [
                float(torch.rand((), generator=generator).item()) * 2.0 * max(float(p.historical_demand[-1]), 1.0)
                for p in products
            ],
            dtype=torch.float32,
        ),
        "history": torch.stack(histories, dim=0),
        "feature_demand_scale": torch.full((len(products),), float(demand_scale), dtype=torch.float32)
        if feature_stats is not None
        else torch.tensor(
            [max(float(sum(p.historical_demand) / len(p.historical_demand)), 1.0) for p in products],
            dtype=torch.float32,
        ),
        "feature_price_scale": torch.full((len(products),), float(price_scale), dtype=torch.float32)
        if feature_stats is not None
        else torch.tensor([max(float(p.price), 1.0) for p in products], dtype=torch.float32),
        "historical_mean_demand": torch.tensor(
            [float(sum(p.historical_demand) / len(p.historical_demand)) for p in products],
            dtype=torch.float32,
        ),
        "price": torch.tensor([p.price for p in products], dtype=torch.float32),
        "purchase_cost": torch.tensor([p.purchase_cost for p in products], dtype=torch.float32),
        "holding_cost": torch.tensor([p.holding_cost for p in products], dtype=torch.float32),
        "lost_sale_penalty": torch.tensor([p.lost_sale_penalty for p in products], dtype=torch.float32),
        "historical_base_stock_level": torch.tensor(
            [historical_demand_base_stock_policy(p).S for p in products],
            dtype=torch.float32,
        ),
        "demand_paths": torch.stack(demand_paths, dim=0),
    }


def paper_lost_sales_fcnn_order(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    inventory: torch.Tensor,
    max_order_factor: float = PAPER_FCNN_MAX_ORDER_FACTOR,
    scale_action_by_history_mean: bool = False,
) -> torch.Tensor:
    """Return capped FCNN orders for a batch and inventory vector."""
    demand_scale = torch.clamp(batch["feature_demand_scale"], min=1.0)
    price_scale = torch.clamp(batch["feature_price_scale"], min=1.0)
    economics = torch.stack(
        [
            batch["price"] / price_scale,
            batch["purchase_cost"] / price_scale,
            batch["holding_cost"] / price_scale,
            batch["lost_sale_penalty"] / price_scale,
        ],
        dim=1,
    )
    features = torch.cat([(inventory / demand_scale).unsqueeze(-1), batch["history"], economics], dim=1)
    max_inventory_level = float(max_order_factor) * torch.clamp(batch["historical_mean_demand"], min=1.0)
    order_cap = torch.relu(max_inventory_level - inventory)
    order = model(features)
    if scale_action_by_history_mean:
        order = order * torch.clamp(batch["historical_mean_demand"], min=1.0)
    return torch.minimum(order, order_cap)


def paper_lost_sales_features(
    batch: dict[str, torch.Tensor],
    inventory: torch.Tensor,
) -> torch.Tensor:
    """Build the normalized model inputs for an inventory vector."""
    demand_scale = torch.clamp(batch["feature_demand_scale"], min=1.0)
    price_scale = torch.clamp(batch["feature_price_scale"], min=1.0)
    economics = torch.stack(
        [
            batch["price"] / price_scale,
            batch["purchase_cost"] / price_scale,
            batch["holding_cost"] / price_scale,
            batch["lost_sale_penalty"] / price_scale,
        ],
        dim=1,
    )
    return torch.cat([(inventory / demand_scale).unsqueeze(-1), batch["history"], economics], dim=1)


def paper_inventory_monotonicity_penalty(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    generator: torch.Generator,
    samples_per_product: int = 1,
    max_inventory_factor: float = 4.0,
    scale_action_by_history_mean: bool = False,
) -> torch.Tensor:
    """Penalize raw order increases between paired low/high inventory states."""
    if samples_per_product <= 0:
        return torch.zeros((), dtype=batch["initial_inventory"].dtype)
    batch_size = batch["initial_inventory"].shape[0]
    max_inventory = float(max_inventory_factor) * torch.clamp(batch["historical_mean_demand"], min=1.0)
    penalties = []
    for _ in range(samples_per_product):
        first = torch.rand(batch_size, generator=generator)
        second = torch.rand(batch_size, generator=generator)
        inventory_low = torch.minimum(first, second) * max_inventory
        inventory_high = torch.maximum(first, second) * max_inventory
        low_order = model(paper_lost_sales_features(batch, inventory_low))
        high_order = model(paper_lost_sales_features(batch, inventory_high))
        if scale_action_by_history_mean:
            action_scale = torch.clamp(batch["historical_mean_demand"], min=1.0)
            low_order = low_order * action_scale
            high_order = high_order * action_scale
        penalties.append(torch.mean(torch.relu(high_order - low_order) ** 2))
    return torch.stack(penalties).mean()


def paper_reward_normalization_scale(
    batch: dict[str, torch.Tensor],
    horizon: int,
) -> torch.Tensor:
    """Return a detached positive scale for comparable reward gradients."""
    per_product_scale = (
        torch.clamp(batch["price"], min=1.0)
        * torch.clamp(batch["historical_mean_demand"], min=1.0)
        * max(int(horizon), 1)
    )
    return torch.clamp(per_product_scale.mean(), min=1.0).detach()


def paper_lost_sales_fcnn_rollout(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    max_order_factor: float = PAPER_FCNN_MAX_ORDER_FACTOR,
    scale_action_by_history_mean: bool = False,
) -> torch.Tensor:
    """Return total reward per product for FCNN training."""
    inventory = batch["initial_inventory"]
    demand_paths = batch["demand_paths"]
    horizon = demand_paths.shape[1]
    total_reward = torch.zeros_like(inventory)
    for t in range(horizon):
        order = paper_lost_sales_fcnn_order(
            model,
            batch,
            inventory,
            max_order_factor=max_order_factor,
            scale_action_by_history_mean=scale_action_by_history_mean,
        )
        demand = demand_paths[:, t]
        available = inventory + order
        sales = torch.minimum(demand, available)
        lost_sales = torch.relu(demand - available)
        ending_inventory = torch.relu(available - demand)
        reward = (
            batch["price"] * sales
            - batch["purchase_cost"] * order
            - batch["lost_sale_penalty"] * lost_sales
            - batch["holding_cost"] * ending_inventory
        )
        total_reward = total_reward + reward
        inventory = ending_inventory
    return total_reward


def historical_base_stock_imitation_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    generator: torch.Generator,
    samples_per_product: int = 4,
    max_order_factor: float = PAPER_FCNN_MAX_ORDER_FACTOR,
    scale_action_by_history_mean: bool = False,
) -> torch.Tensor:
    """Supervised warm-start loss matching historical base-stock actions."""
    batch_size = batch["initial_inventory"].shape[0]
    max_inventory_level = float(max_order_factor) * torch.clamp(batch["historical_mean_demand"], min=1.0)
    repeated_losses = []
    for _ in range(samples_per_product):
        random_fraction = torch.rand(batch_size, generator=generator)
        inventory = random_fraction * max_inventory_level
        predicted = paper_lost_sales_fcnn_order(
            model,
            batch,
            inventory,
            max_order_factor=max_order_factor,
            scale_action_by_history_mean=scale_action_by_history_mean,
        )
        target = torch.relu(batch["historical_base_stock_level"] - inventory)
        target = torch.minimum(target, torch.relu(max_inventory_level - inventory))
        repeated_losses.append(torch.mean((predicted - target) ** 2))
    return torch.stack(repeated_losses).mean()


def train_paper_lost_sales_fcnn(
    products: list[PaperLostSalesProduct],
    model: nn.Module,
    horizon: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    imitation_epochs: int = 0,
    validation_products: list[PaperLostSalesProduct] | None = None,
    validation_interval: int = 0,
    validation_callback: Callable[[int, nn.Module], float] | None = None,
    feature_stats: PaperFCNNFeatureStats | None = None,
    max_order_factor: float = PAPER_FCNN_MAX_ORDER_FACTOR,
    scale_action_by_history_mean: bool = False,
    normalize_reward_loss: bool = False,
    monotonicity_penalty_weight: float = 0.0,
    monotonicity_samples_per_product: int = 1,
) -> DirectBackpropResult:
    """Train FCNN with DirectBackprop-style negative reward minimization."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    penalty_generator = torch.Generator().manual_seed(seed + 1_000_003)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    training_losses: list[float] = []
    mean_total_rewards: list[float] = []
    validation_rewards: list[float] = []
    validation_epochs: list[int] = []
    training_log: list[dict[str, float | int]] = []
    monotonicity_penalties: list[float] = []
    best_validation_reward: float | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    total_epochs = imitation_epochs + epochs
    start_time = time.perf_counter()
    for epoch in range(total_epochs):
        indices = torch.randint(0, len(products), (batch_size,), generator=generator).tolist()
        batch_products = [products[index] for index in indices]
        batch = make_paper_lost_sales_training_batch(batch_products, horizon, generator, feature_stats=feature_stats)
        if epoch < imitation_epochs:
            mean_total_reward = paper_lost_sales_fcnn_rollout(
                model,
                batch,
                max_order_factor=max_order_factor,
                scale_action_by_history_mean=scale_action_by_history_mean,
            ).mean()
            primary_loss = historical_base_stock_imitation_loss(
                model,
                batch,
                generator,
                max_order_factor=max_order_factor,
                scale_action_by_history_mean=scale_action_by_history_mean,
            )
        else:
            mean_total_reward = paper_lost_sales_fcnn_rollout(
                model,
                batch,
                max_order_factor=max_order_factor,
                scale_action_by_history_mean=scale_action_by_history_mean,
            ).mean()
            primary_loss = -mean_total_reward
            if normalize_reward_loss:
                primary_loss = primary_loss / paper_reward_normalization_scale(batch, horizon)
        if monotonicity_penalty_weight > 0:
            monotonicity_penalty = paper_inventory_monotonicity_penalty(
                model,
                batch,
                penalty_generator,
                samples_per_product=monotonicity_samples_per_product,
                max_inventory_factor=max_order_factor,
                scale_action_by_history_mean=scale_action_by_history_mean,
            )
        else:
            monotonicity_penalty = torch.zeros((), dtype=primary_loss.dtype, device=primary_loss.device)
        loss = primary_loss + float(monotonicity_penalty_weight) * monotonicity_penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if hasattr(model, "apply_monotonic_constraints"):
            model.apply_monotonic_constraints()
        training_losses.append(float(loss.detach().item()))
        mean_total_rewards.append(float(mean_total_reward.detach().item()))
        monotonicity_penalties.append(float(monotonicity_penalty.detach().item()))
        log_row: dict[str, float | int] = {
            "epoch": epoch + 1,
            "phase": 0 if epoch < imitation_epochs else 1,
            "training_loss": float(loss.detach().item()),
            "primary_loss": float(primary_loss.detach().item()),
            "training_mean_total_reward": float(mean_total_reward.detach().item()),
            "monotonicity_penalty": float(monotonicity_penalty.detach().item()),
            "monotonicity_penalty_weight": float(monotonicity_penalty_weight),
            "elapsed_seconds": float(time.perf_counter() - start_time),
        }
        if validation_products and validation_interval and (epoch + 1) % validation_interval == 0:
            if validation_callback is not None:
                validation_reward = validation_callback(epoch + 1, model)
            else:
                validation_batch = make_paper_lost_sales_training_batch(
                    validation_products,
                    horizon,
                    generator,
                    feature_stats=feature_stats,
                )
                with torch.no_grad():
                    validation_reward_tensor = paper_lost_sales_fcnn_rollout(
                        model,
                        validation_batch,
                        max_order_factor=max_order_factor,
                        scale_action_by_history_mean=scale_action_by_history_mean,
                    ).mean()
                validation_reward = float(validation_reward_tensor.item())
            validation_rewards.append(float(validation_reward))
            validation_epochs.append(epoch + 1)
            log_row["validation_mean_total_reward"] = float(validation_reward)
            if best_validation_reward is None or validation_reward > best_validation_reward:
                best_validation_reward = float(validation_reward)
                best_state_dict = copy.deepcopy(model.state_dict())
        training_log.append(log_row)
    elapsed_seconds = float(time.perf_counter() - start_time)
    return DirectBackpropResult(
        model=model,
        training_losses=training_losses,
        mean_total_rewards=mean_total_rewards,
        validation_rewards=validation_rewards or None,
        validation_epochs=validation_epochs or None,
        best_validation_reward=best_validation_reward,
        best_state_dict=best_state_dict,
        training_log=training_log,
        monotonicity_penalties=monotonicity_penalties,
        elapsed_seconds=elapsed_seconds,
    )
