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

from inventory_control.policies.neural.fcnn import FCNNPolicy, PaperFCNNFeatureStats, infer_model_device
from inventory_control.paper_lost_sales import non_omniscient_gamma_base_stock_policy
from inventory_control.paper_lost_sales import historical_empirical_quantile, paper_critical_quantile
from inventory_control.products import PaperLostSalesProduct


PAPER_FCNN_MAX_ORDER_FACTOR = 2.0


def default_torch_device() -> torch.device:
    """Pick the best available device for neural training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move a tensor batch to the selected device."""
    return {key: value.to(device) for key, value in batch.items()}


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
    best_imitation_loss: float | None = None
    best_imitation_epoch: int | None = None
    best_imitation_state_dict: dict[str, torch.Tensor] | None = None
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
    demand_mode: str = "fixed_future",
    residual_anchor: str = "gamma",
) -> dict[str, torch.Tensor]:
    """Build a tensor batch using only historical demand observations."""
    if demand_mode not in {"fixed_future", "resample"}:
        raise ValueError(f"Unsupported training demand mode: {demand_mode}")
    if residual_anchor not in {"gamma", "empirical"}:
        raise ValueError(f"Unsupported residual anchor: {residual_anchor}")
    demand_paths = []
    histories = []
    demand_scale = 1.0
    price_scale = 1.0
    if feature_stats is not None:
        demand_scale = feature_stats.demand_scale
        price_scale = feature_stats.price_scale
    for product in products:
        history = torch.tensor(product.historical_demand, dtype=torch.float32)
        if demand_mode == "fixed_future" and product.future_demand is not None and len(product.future_demand) >= horizon:
            demand_paths.append(torch.tensor(product.future_demand[:horizon], dtype=torch.float32))
        else:
            demand_paths.append(
                torch.tensor(product.sample_demand(_torch_seeded_numpy_rng(generator), size=horizon), dtype=torch.float32)
            )
        scale = torch.tensor(demand_scale, dtype=torch.float32) if feature_stats is not None else torch.clamp(history.mean(), min=1.0)
        histories.append(history / scale)
    return {
        "initial_inventory": torch.tensor(
            [
                float(torch.rand((), generator=generator).item())
                * 2.0
                * max(float(sum(p.historical_demand) / len(p.historical_demand)), 1.0)
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
            [_residual_anchor_level(p, residual_anchor) for p in products],
            dtype=torch.float32,
        ),
        "demand_paths": torch.stack(demand_paths, dim=0),
    }


def _torch_seeded_numpy_rng(generator: torch.Generator):
    import numpy as np

    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
    return np.random.default_rng(seed)


def _residual_anchor_level(product: PaperLostSalesProduct, residual_anchor: str) -> float:
    if residual_anchor == "gamma":
        return non_omniscient_gamma_base_stock_policy(product).S
    if residual_anchor == "empirical":
        return historical_empirical_quantile(product, paper_critical_quantile(product))
    raise ValueError(f"Unsupported residual anchor: {residual_anchor}")


def paper_lost_sales_raw_order(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    inventory: torch.Tensor,
    scale_action_by_history_mean: bool = False,
) -> torch.Tensor:
    """Return uncapped policy orders for a batch and inventory vector."""
    return paper_lost_sales_uncapped_order(
        model,
        batch,
        inventory,
        scale_action_by_history_mean=scale_action_by_history_mean,
    )


def paper_lost_sales_fcnn_order(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    inventory: torch.Tensor,
    max_order_factor: float = PAPER_FCNN_MAX_ORDER_FACTOR,
    scale_action_by_history_mean: bool = False,
) -> torch.Tensor:
    """Return capped policy orders for a batch and inventory vector."""
    max_inventory_level = float(max_order_factor) * torch.clamp(batch["historical_mean_demand"], min=1.0)
    order_cap = torch.relu(max_inventory_level - inventory)
    order = paper_lost_sales_raw_order(
        model,
        batch,
        inventory,
        scale_action_by_history_mean=scale_action_by_history_mean,
    )
    return torch.minimum(order, order_cap)


def paper_lost_sales_policy_order(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    inventory: torch.Tensor,
    max_order_factor: float = PAPER_FCNN_MAX_ORDER_FACTOR,
    scale_action_by_history_mean: bool = False,
) -> torch.Tensor:
    """Architecture-neutral alias for the shared capped paper-policy decoder."""
    return paper_lost_sales_fcnn_order(
        model,
        batch,
        inventory,
        max_order_factor=max_order_factor,
        scale_action_by_history_mean=scale_action_by_history_mean,
    )


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


def paper_lost_sales_context_features(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Build normalized context features without current inventory."""
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
    return torch.cat([batch["history"], economics], dim=1)


def paper_lost_sales_uncapped_order(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    inventory: torch.Tensor,
    scale_action_by_history_mean: bool = False,
) -> torch.Tensor:
    """Decode the model's uncapped purchase quantity for supported parameterizations."""
    parameterization = getattr(model, "paper_action_parameterization", "raw")
    if parameterization in {"order_up_to", "residual_order_up_to"}:
        target = paper_lost_sales_target_level(model, batch)
        return torch.relu(target - inventory)

    historical_mean = torch.clamp(batch["historical_mean_demand"], min=1.0)
    features = paper_lost_sales_features(batch, inventory)
    order = model(features)
    if scale_action_by_history_mean:
        order = order * historical_mean
    return order


def paper_lost_sales_target_level(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return target stock level for context-only action parameterizations."""
    parameterization = getattr(model, "paper_action_parameterization", "raw")
    historical_mean = torch.clamp(batch["historical_mean_demand"], min=1.0)
    if parameterization == "order_up_to":
        target_factor = model(paper_lost_sales_context_features(batch))
        if bool(getattr(model, "paper_scale_target_by_history_mean", True)):
            return target_factor * historical_mean
        return target_factor
    if parameterization == "residual_order_up_to":
        residual_factor = model(paper_lost_sales_context_features(batch))
        residual_center = float(getattr(model, "paper_residual_center", 0.0))
        residual_scale = float(getattr(model, "paper_residual_scale", 0.5))
        return batch["historical_base_stock_level"] + residual_scale * (residual_factor - residual_center) * historical_mean
    raise ValueError(f"Model parameterization {parameterization!r} does not predict a context target level")


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
        first = torch.rand(batch_size, generator=generator).to(batch["initial_inventory"].device)
        second = torch.rand(batch_size, generator=generator).to(batch["initial_inventory"].device)
        inventory_low = torch.minimum(first, second) * max_inventory
        inventory_high = torch.maximum(first, second) * max_inventory
        low_order = paper_lost_sales_uncapped_order(
            model,
            batch,
            inventory_low,
            scale_action_by_history_mean=scale_action_by_history_mean,
        )
        high_order = paper_lost_sales_uncapped_order(
            model,
            batch,
            inventory_high,
            scale_action_by_history_mean=scale_action_by_history_mean,
        )
        penalties.append(torch.mean(torch.relu(high_order - low_order) ** 2))
    return torch.stack(penalties).mean()


def paper_action_cap_excess_penalty(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    generator: torch.Generator,
    samples_per_product: int = 1,
    max_order_factor: float = PAPER_FCNN_MAX_ORDER_FACTOR,
    scale_action_by_history_mean: bool = False,
) -> torch.Tensor:
    """Penalize raw orders that exceed the deployed physical action cap."""
    if samples_per_product <= 0:
        return torch.zeros((), dtype=batch["initial_inventory"].dtype, device=batch["initial_inventory"].device)
    batch_size = batch["initial_inventory"].shape[0]
    historical_mean = torch.clamp(batch["historical_mean_demand"], min=1.0)
    max_inventory = float(max_order_factor) * historical_mean
    penalties = []
    for _ in range(samples_per_product):
        inventory = torch.rand(batch_size, generator=generator).to(batch["initial_inventory"].device) * max_inventory
        raw_order = paper_lost_sales_raw_order(
            model,
            batch,
            inventory,
            scale_action_by_history_mean=scale_action_by_history_mean,
        )
        cap = torch.relu(max_inventory - inventory)
        normalized_excess = torch.relu(raw_order - cap) / historical_mean
        penalties.append(torch.mean(normalized_excess**2))
    return torch.stack(penalties).mean()


def paper_target_stock_penalty(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    generator: torch.Generator,
    samples_per_product: int = 1,
    max_inventory_factor: float = PAPER_FCNN_MAX_ORDER_FACTOR,
    scale_action_by_history_mean: bool = False,
) -> torch.Tensor:
    """Penalize deviations of implied order-up-to levels from historical base stock."""
    if samples_per_product <= 0:
        return torch.zeros((), dtype=batch["initial_inventory"].dtype, device=batch["initial_inventory"].device)
    batch_size = batch["initial_inventory"].shape[0]
    historical_mean = torch.clamp(batch["historical_mean_demand"], min=1.0)
    max_inventory = float(max_inventory_factor) * historical_mean
    target_level = batch["historical_base_stock_level"]
    penalties = []
    for _ in range(samples_per_product):
        random_fraction = torch.rand(batch_size, generator=generator).to(batch["initial_inventory"].device)
        active_upper = torch.minimum(max_inventory, torch.clamp(target_level, min=0.0))
        inventory = random_fraction * active_upper
        order = paper_lost_sales_raw_order(
            model,
            batch,
            inventory,
            scale_action_by_history_mean=scale_action_by_history_mean,
        )
        implied_level = inventory + order
        normalized_error = (implied_level - target_level) / historical_mean
        penalties.append(torch.mean(normalized_error**2))
    return torch.stack(penalties).mean()


def paper_active_slope_penalty(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    generator: torch.Generator,
    samples_per_product: int = 1,
    max_inventory_factor: float = PAPER_FCNN_MAX_ORDER_FACTOR,
    scale_action_by_history_mean: bool = False,
    step_fraction: float = 0.05,
) -> torch.Tensor:
    """Penalize active-region inventory/order slope away from minus one."""
    if samples_per_product <= 0:
        return torch.zeros((), dtype=batch["initial_inventory"].dtype, device=batch["initial_inventory"].device)
    batch_size = batch["initial_inventory"].shape[0]
    historical_mean = torch.clamp(batch["historical_mean_demand"], min=1.0)
    max_inventory = float(max_inventory_factor) * historical_mean
    target_level = torch.clamp(batch["historical_base_stock_level"], min=0.0)
    active_upper = torch.minimum(max_inventory, target_level)
    step = torch.clamp(float(step_fraction) * historical_mean, min=1e-3)
    usable_upper = torch.relu(active_upper - step)
    penalties = []
    for _ in range(samples_per_product):
        random_fraction = torch.rand(batch_size, generator=generator).to(batch["initial_inventory"].device)
        inventory_low = random_fraction * usable_upper
        inventory_high = inventory_low + step
        low_order = paper_lost_sales_raw_order(
            model,
            batch,
            inventory_low,
            scale_action_by_history_mean=scale_action_by_history_mean,
        )
        high_order = paper_lost_sales_raw_order(
            model,
            batch,
            inventory_high,
            scale_action_by_history_mean=scale_action_by_history_mean,
        )
        slope = (high_order - low_order) / step
        active_mask = (active_upper > step).float()
        squared_error = ((slope + 1.0) ** 2) * active_mask
        penalties.append(squared_error.sum() / torch.clamp(active_mask.sum(), min=1.0))
    return torch.stack(penalties).mean()


def paper_named_parameter_l2_penalty(model: nn.Module, name_fragments: tuple[str, ...]) -> torch.Tensor:
    """Return mean squared parameter value for parameters matching any name fragment."""
    penalties = [
        parameter.square().mean()
        for name, parameter in model.named_parameters()
        if any(fragment in name for fragment in name_fragments)
    ]
    if not penalties:
        device = infer_model_device(model)
        return torch.zeros((), dtype=torch.float32, device=device)
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
    include_terminal_salvage: bool = True,
) -> torch.Tensor:
    """Return total reward per product for FCNN training."""
    inventory = batch["initial_inventory"]
    demand_paths = batch["demand_paths"]
    horizon = demand_paths.shape[1]
    total_reward = torch.zeros_like(inventory)
    parameterization = getattr(model, "paper_action_parameterization", "raw")
    target_level = (
        paper_lost_sales_target_level(model, batch)
        if parameterization in {"order_up_to", "residual_order_up_to"}
        else None
    )
    max_inventory_level = float(max_order_factor) * torch.clamp(batch["historical_mean_demand"], min=1.0)
    for t in range(horizon):
        if target_level is None:
            order = paper_lost_sales_fcnn_order(
                model,
                batch,
                inventory,
                max_order_factor=max_order_factor,
                scale_action_by_history_mean=scale_action_by_history_mean,
            )
        else:
            order = torch.minimum(torch.relu(target_level - inventory), torch.relu(max_inventory_level - inventory))
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
    if include_terminal_salvage:
        total_reward = total_reward + batch["purchase_cost"] * inventory
    return total_reward


def paper_lost_sales_policy_rollout(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    max_order_factor: float = PAPER_FCNN_MAX_ORDER_FACTOR,
    scale_action_by_history_mean: bool = False,
    include_terminal_salvage: bool = True,
) -> torch.Tensor:
    """Architecture-neutral alias for the shared differentiable paper rollout."""
    return paper_lost_sales_fcnn_rollout(
        model,
        batch,
        max_order_factor=max_order_factor,
        scale_action_by_history_mean=scale_action_by_history_mean,
        include_terminal_salvage=include_terminal_salvage,
    )


def historical_base_stock_imitation_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    generator: torch.Generator,
    samples_per_product: int = 4,
    max_order_factor: float = PAPER_FCNN_MAX_ORDER_FACTOR,
    scale_action_by_history_mean: bool = False,
    clip_predictions: bool = True,
) -> torch.Tensor:
    """Supervised warm-start loss matching historical base-stock actions."""
    batch_size = batch["initial_inventory"].shape[0]
    max_inventory_level = float(max_order_factor) * torch.clamp(batch["historical_mean_demand"], min=1.0)
    repeated_losses = []
    for _ in range(samples_per_product):
        random_fraction = torch.rand(batch_size, generator=generator).to(batch["initial_inventory"].device)
        inventory = random_fraction * max_inventory_level
        if clip_predictions:
            predicted = paper_lost_sales_fcnn_order(
                model,
                batch,
                inventory,
                max_order_factor=max_order_factor,
                scale_action_by_history_mean=scale_action_by_history_mean,
            )
        else:
            predicted = paper_lost_sales_raw_order(
                model,
                batch,
                inventory,
                scale_action_by_history_mean=scale_action_by_history_mean,
            )
        target = torch.relu(batch["historical_base_stock_level"] - inventory)
        target = torch.minimum(target, torch.relu(max_inventory_level - inventory))
        repeated_losses.append(torch.mean((predicted - target) ** 2))
    return torch.stack(repeated_losses).mean()


@torch.no_grad()
def paper_order_diagnostics(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    max_order_factor: float,
    scale_action_by_history_mean: bool,
    inventory_fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75),
) -> dict[str, float]:
    """Measure raw-order calibration on fixed fractions of the observable cap."""
    batch = move_batch_to_device(batch, infer_model_device(model))
    max_inventory = float(max_order_factor) * torch.clamp(batch["historical_mean_demand"], min=1.0)
    raw_orders = []
    deployed_orders = []
    caps = []
    targets = []
    for fraction in inventory_fractions:
        inventory = float(fraction) * max_inventory
        raw = paper_lost_sales_raw_order(
            model,
            batch,
            inventory,
            scale_action_by_history_mean=scale_action_by_history_mean,
        )
        cap = torch.relu(max_inventory - inventory)
        target = torch.minimum(torch.relu(batch["historical_base_stock_level"] - inventory), cap)
        raw_orders.append(raw)
        deployed_orders.append(torch.minimum(raw, cap))
        caps.append(cap)
        targets.append(target)
    raw = torch.cat(raw_orders)
    deployed = torch.cat(deployed_orders)
    cap = torch.cat(caps)
    target = torch.cat(targets)
    errors = raw - target
    return {
        "raw_order_mean": float(raw.mean().item()),
        "raw_order_min": float(raw.min().item()),
        "raw_order_max": float(raw.max().item()),
        "deployed_order_mean": float(deployed.mean().item()),
        "cap_hit_rate": float(torch.mean((raw >= cap - 1e-6).float()).item()),
        "raw_imitation_mae": float(torch.mean(torch.abs(errors)).item()),
        "raw_imitation_rmse": float(torch.sqrt(torch.mean(errors**2)).item()),
    }


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
    cap_excess_penalty_weight: float = 0.0,
    cap_excess_samples_per_product: int = 1,
    target_stock_penalty_weight: float = 0.0,
    target_stock_samples_per_product: int = 1,
    active_slope_penalty_weight: float = 0.0,
    active_slope_samples_per_product: int = 1,
    spline_l2_weight: float = 0.0,
    scale_l2_weight: float = 0.0,
    training_demand_mode: str = "fixed_future",
    residual_anchor: str = "gamma",
    include_terminal_salvage: bool = True,
    clip_imitation_predictions: bool = True,
    device: torch.device | str | None = None,
) -> DirectBackpropResult:
    """Train FCNN with DirectBackprop-style negative reward minimization."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    penalty_generator = torch.Generator().manual_seed(seed + 1_000_003)
    cap_excess_generator = torch.Generator().manual_seed(seed + 2_000_003)
    target_stock_generator = torch.Generator().manual_seed(seed + 3_000_003)
    active_slope_generator = torch.Generator().manual_seed(seed + 4_000_003)
    chosen_device = torch.device(device) if device is not None else default_torch_device()
    model = model.to(chosen_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    training_losses: list[float] = []
    mean_total_rewards: list[float] = []
    validation_rewards: list[float] = []
    validation_epochs: list[int] = []
    training_log: list[dict[str, float | int]] = []
    monotonicity_penalties: list[float] = []
    best_validation_reward: float | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_imitation_loss: float | None = None
    best_imitation_epoch: int | None = None
    best_imitation_state_dict: dict[str, torch.Tensor] | None = None
    total_epochs = imitation_epochs + epochs
    start_time = time.perf_counter()
    for epoch in range(total_epochs):
        indices = torch.randint(0, len(products), (batch_size,), generator=generator).tolist()
        batch_products = [products[index] for index in indices]
        batch = move_batch_to_device(
            make_paper_lost_sales_training_batch(
                batch_products,
                horizon,
                generator,
                feature_stats=feature_stats,
                demand_mode=training_demand_mode,
                residual_anchor=residual_anchor,
            ),
            infer_model_device(model),
        )
        if epoch < imitation_epochs:
            mean_total_reward = paper_lost_sales_fcnn_rollout(
                model,
                batch,
                max_order_factor=max_order_factor,
                scale_action_by_history_mean=scale_action_by_history_mean,
                include_terminal_salvage=include_terminal_salvage,
            ).mean()
            primary_loss = historical_base_stock_imitation_loss(
                model,
                batch,
                generator,
                max_order_factor=max_order_factor,
                scale_action_by_history_mean=scale_action_by_history_mean,
                clip_predictions=clip_imitation_predictions,
            )
        else:
            mean_total_reward = paper_lost_sales_fcnn_rollout(
                model,
                batch,
                max_order_factor=max_order_factor,
                scale_action_by_history_mean=scale_action_by_history_mean,
                include_terminal_salvage=include_terminal_salvage,
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
        if cap_excess_penalty_weight > 0:
            cap_excess_penalty = paper_action_cap_excess_penalty(
                model,
                batch,
                cap_excess_generator,
                samples_per_product=cap_excess_samples_per_product,
                max_order_factor=max_order_factor,
                scale_action_by_history_mean=scale_action_by_history_mean,
            )
        else:
            cap_excess_penalty = torch.zeros((), dtype=primary_loss.dtype, device=primary_loss.device)
        if target_stock_penalty_weight > 0:
            target_stock_penalty = paper_target_stock_penalty(
                model,
                batch,
                target_stock_generator,
                samples_per_product=target_stock_samples_per_product,
                max_inventory_factor=max_order_factor,
                scale_action_by_history_mean=scale_action_by_history_mean,
            )
        else:
            target_stock_penalty = torch.zeros((), dtype=primary_loss.dtype, device=primary_loss.device)
        if active_slope_penalty_weight > 0:
            active_slope_penalty = paper_active_slope_penalty(
                model,
                batch,
                active_slope_generator,
                samples_per_product=active_slope_samples_per_product,
                max_inventory_factor=max_order_factor,
                scale_action_by_history_mean=scale_action_by_history_mean,
            )
        else:
            active_slope_penalty = torch.zeros((), dtype=primary_loss.dtype, device=primary_loss.device)
        if spline_l2_weight > 0:
            spline_l2_penalty = paper_named_parameter_l2_penalty(model, ("coef",)).to(primary_loss.device)
        else:
            spline_l2_penalty = torch.zeros((), dtype=primary_loss.dtype, device=primary_loss.device)
        if scale_l2_weight > 0:
            scale_l2_penalty = paper_named_parameter_l2_penalty(model, ("scale_base", "scale_spline")).to(primary_loss.device)
        else:
            scale_l2_penalty = torch.zeros((), dtype=primary_loss.dtype, device=primary_loss.device)
        loss = (
            primary_loss
            + float(monotonicity_penalty_weight) * monotonicity_penalty
            + float(cap_excess_penalty_weight) * cap_excess_penalty
            + float(target_stock_penalty_weight) * target_stock_penalty
            + float(active_slope_penalty_weight) * active_slope_penalty
            + float(spline_l2_weight) * spline_l2_penalty
            + float(scale_l2_weight) * scale_l2_penalty
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if hasattr(model, "apply_monotonic_constraints"):
            model.apply_monotonic_constraints()
        if epoch < imitation_epochs:
            current_imitation_loss = float(primary_loss.detach().item())
            if best_imitation_loss is None or current_imitation_loss < best_imitation_loss:
                best_imitation_loss = current_imitation_loss
                best_imitation_epoch = epoch + 1
                best_imitation_state_dict = copy.deepcopy(model.state_dict())
        order_diagnostics = paper_order_diagnostics(
            model,
            batch,
            max_order_factor=max_order_factor,
            scale_action_by_history_mean=scale_action_by_history_mean,
        )
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
            "cap_excess_penalty": float(cap_excess_penalty.detach().item()),
            "cap_excess_penalty_weight": float(cap_excess_penalty_weight),
            "target_stock_penalty": float(target_stock_penalty.detach().item()),
            "target_stock_penalty_weight": float(target_stock_penalty_weight),
            "active_slope_penalty": float(active_slope_penalty.detach().item()),
            "active_slope_penalty_weight": float(active_slope_penalty_weight),
            "spline_l2_penalty": float(spline_l2_penalty.detach().item()),
            "spline_l2_weight": float(spline_l2_weight),
            "scale_l2_penalty": float(scale_l2_penalty.detach().item()),
            "scale_l2_weight": float(scale_l2_weight),
            "elapsed_seconds": float(time.perf_counter() - start_time),
            **order_diagnostics,
        }
        should_validate = bool(
            validation_products
            and validation_interval
            and ((epoch + 1) % validation_interval == 0 or epoch + 1 == total_epochs)
        )
        if should_validate:
            if validation_callback is not None:
                validation_reward = validation_callback(epoch + 1, model)
            else:
                validation_batch = make_paper_lost_sales_training_batch(
                    validation_products,
                    horizon,
                    generator,
                    feature_stats=feature_stats,
                    demand_mode="fixed_future",
                    residual_anchor=residual_anchor,
                )
                validation_batch = move_batch_to_device(validation_batch, infer_model_device(model))
                with torch.no_grad():
                    validation_reward_tensor = paper_lost_sales_fcnn_rollout(
                        model,
                        validation_batch,
                        max_order_factor=max_order_factor,
                        scale_action_by_history_mean=scale_action_by_history_mean,
                        include_terminal_salvage=include_terminal_salvage,
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
        best_imitation_loss=best_imitation_loss,
        best_imitation_epoch=best_imitation_epoch,
        best_imitation_state_dict=best_imitation_state_dict,
        training_log=training_log,
        monotonicity_penalties=monotonicity_penalties,
        elapsed_seconds=elapsed_seconds,
    )


def train_paper_lost_sales_policy(*args, **kwargs) -> DirectBackpropResult:
    """Architecture-neutral alias for DirectBackprop paper lost-sales training."""
    return train_paper_lost_sales_fcnn(*args, **kwargs)
