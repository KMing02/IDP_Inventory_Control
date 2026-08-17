"""DirectBackprop training for paper-style lost-sales replication."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import copy
import time
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install neural dependencies with `python -m pip install -r requirements-neural.txt`.") from exc

from torch import nn

from inventory_control.policies.neural.fcnn import FCNNPolicy, PaperFCNNFeatureStats, infer_model_device
from inventory_control.paper_lost_sales import non_omniscient_gamma_base_stock_policy
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


def _parameter_gradient_norm(parameters: list[nn.Parameter]) -> float:
    squared = sum(
        float(torch.sum(parameter.grad.detach() ** 2).item())
        for parameter in parameters
        if parameter.grad is not None
    )
    return squared**0.5


def _parameter_update_norm(
    parameters: list[nn.Parameter],
    before: list[torch.Tensor],
) -> float:
    squared = sum(
        float(torch.sum((parameter.detach() - original) ** 2).item())
        for parameter, original in zip(parameters, before)
    )
    return squared**0.5


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
    best_imitation_validation_mae: float | None = None
    best_imitation_validation_rmse: float | None = None
    imitation_selection_metric: str | None = None
    imitation_end_state_dict: dict[str, torch.Tensor] | None = None
    imitation_epochs_completed: int = 0
    imitation_early_stopped: bool = False
    optimizer_reset_after_imitation: bool = False
    training_log: list[dict[str, float | int | str]] | None = None
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
            [non_omniscient_gamma_base_stock_policy(p).S for p in products],
            dtype=torch.float32,
        ),
        "demand_paths": torch.stack(demand_paths, dim=0),
    }


def paper_lost_sales_raw_order(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    inventory: torch.Tensor,
    scale_action_by_history_mean: bool = False,
) -> torch.Tensor:
    """Return uncapped policy orders for a batch and inventory vector."""
    action_parameterization = getattr(model, "paper_action_parameterization", "raw")
    if action_parameterization == "residual_order_up_to":
        return torch.relu(paper_lost_sales_target_level(model, batch) - inventory)
    if action_parameterization == "target_stock":
        return torch.relu(paper_lost_sales_target_stock_level(model, batch) - inventory)
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
    order = model(features)
    if scale_action_by_history_mean:
        order = order * torch.clamp(batch["historical_mean_demand"], min=1.0)
    return order


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
    """Build normalized history/economic context without current inventory."""
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


def paper_lost_sales_target_stock_level(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Decode a context-only normalized target-stock prediction.

    The model predicts the dimensionless positive multiple ``s_theta``. The
    physical target level is ``historical_mean_demand * s_theta``. No
    product-specific analytical anchor participates in this decoder.
    """
    if getattr(model, "paper_action_parameterization", "raw") != "target_stock":
        raise ValueError("model does not use target_stock parameterization")
    if getattr(model, "paper_target_stock_scale", "historical_mean_demand") != "historical_mean_demand":
        raise ValueError("target-stock policies must scale by historical_mean_demand")
    historical_mean = torch.clamp(batch["historical_mean_demand"], min=1.0)
    normalized_target = model(paper_lost_sales_context_features(batch))
    return historical_mean * normalized_target


def paper_lost_sales_target_level(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Decode the residual policy's context-only target-stock level."""
    if getattr(model, "paper_action_parameterization", "raw") != "residual_order_up_to":
        raise ValueError("model does not use residual_order_up_to parameterization")
    if getattr(model, "paper_residual_anchor", "gamma") != "gamma":
        raise ValueError("the initial residual ablation requires the historical Gamma anchor")
    historical_mean = torch.clamp(batch["historical_mean_demand"], min=1.0)
    residual_output = model(paper_lost_sales_context_features(batch))
    center = float(getattr(model, "paper_residual_center", 1.0))
    scale = float(getattr(model, "paper_residual_scale", 0.5))
    return batch["historical_base_stock_level"] + scale * (residual_output - center) * historical_mean


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
    include_terminal_salvage: bool = True,
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
    if include_terminal_salvage:
        total_reward = total_reward + batch["purchase_cost"] * inventory
    return total_reward


def historical_base_stock_imitation_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    generator: torch.Generator,
    samples_per_product: int = 4,
    max_order_factor: float = PAPER_FCNN_MAX_ORDER_FACTOR,
    scale_action_by_history_mean: bool = False,
    clip_predictions: bool = True,
) -> torch.Tensor:
    """Supervised warm-start loss with semantically matched action targets.

    ``clip_predictions=True`` compares deployed orders with deployment-capped
    base-stock targets. ``clip_predictions=False`` compares uncapped raw orders
    with uncapped base-stock targets.
    """
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
        raw_target = torch.relu(batch["historical_base_stock_level"] - inventory)
        if clip_predictions:
            target = torch.minimum(raw_target, torch.relu(max_inventory_level - inventory))
        else:
            target = raw_target
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
    """Measure consistent raw and deployed imitation calibration.

    Raw orders are compared with uncapped base-stock targets. Deployed orders
    are compared with targets clipped by the same state-dependent action cap.
    """
    max_inventory = float(max_order_factor) * torch.clamp(batch["historical_mean_demand"], min=1.0)
    raw_orders = []
    deployed_orders = []
    caps = []
    raw_targets = []
    deployed_targets = []
    for fraction in inventory_fractions:
        inventory = float(fraction) * max_inventory
        raw = paper_lost_sales_raw_order(
            model,
            batch,
            inventory,
            scale_action_by_history_mean=scale_action_by_history_mean,
        )
        cap = torch.relu(max_inventory - inventory)
        raw_target = torch.relu(batch["historical_base_stock_level"] - inventory)
        deployed_target = torch.minimum(raw_target, cap)
        raw_orders.append(raw)
        deployed_orders.append(torch.minimum(raw, cap))
        caps.append(cap)
        raw_targets.append(raw_target)
        deployed_targets.append(deployed_target)
    raw = torch.cat(raw_orders)
    deployed = torch.cat(deployed_orders)
    cap = torch.cat(caps)
    raw_target = torch.cat(raw_targets)
    deployed_target = torch.cat(deployed_targets)
    raw_errors = raw - raw_target
    deployed_errors = deployed - deployed_target
    return {
        "raw_order_mean": float(raw.mean().item()),
        "raw_order_min": float(raw.min().item()),
        "raw_order_max": float(raw.max().item()),
        "deployed_order_mean": float(deployed.mean().item()),
        "cap_hit_rate": float(torch.mean((raw >= cap - 1e-6).float()).item()),
        "raw_imitation_mae": float(torch.mean(torch.abs(raw_errors)).item()),
        "raw_imitation_rmse": float(torch.sqrt(torch.mean(raw_errors**2)).item()),
        "deployed_imitation_mae": float(torch.mean(torch.abs(deployed_errors)).item()),
        "deployed_imitation_rmse": float(torch.sqrt(torch.mean(deployed_errors**2)).item()),
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
    include_terminal_salvage: bool = True,
    clip_imitation_predictions: bool = True,
    imitation_early_stopping: bool = False,
    imitation_max_epochs: int | None = None,
    imitation_patience: int = 10,
    imitation_min_delta: float = 0.0,
    track_imitation_validation: bool = False,
    reset_optimizer_after_imitation: bool = False,
    adam_kwargs: dict[str, Any] | None = None,
    device: torch.device | str | None = None,
) -> DirectBackpropResult:
    """Train a neural policy with imitation followed by DirectBackprop.

    Fixed imitation remains the default. When ``imitation_early_stopping`` is
    enabled, ``imitation_max_epochs`` is the upper bound and a deterministic
    validation-grid RMSE controls patience. The validation-best imitation
    checkpoint is restored before reward training. Adam can optionally be
    recreated at that boundary without resetting model weights.
    """
    if imitation_epochs < 0 or epochs < 0:
        raise ValueError("imitation_epochs and epochs must be nonnegative")
    if imitation_early_stopping:
        if not validation_products:
            raise ValueError("imitation early stopping requires validation_products")
        if imitation_max_epochs is None:
            imitation_limit = int(imitation_epochs)
        else:
            imitation_limit = int(imitation_max_epochs)
        if imitation_limit < 1:
            raise ValueError("imitation_max_epochs must be positive when early stopping is enabled")
        if imitation_patience < 1:
            raise ValueError("imitation_patience must be positive")
        if imitation_min_delta < 0.0:
            raise ValueError("imitation_min_delta must be nonnegative")
    else:
        imitation_limit = int(imitation_epochs)

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    penalty_generator = torch.Generator().manual_seed(seed + 1_000_003)
    chosen_device = torch.device(device) if device is not None else default_torch_device()
    model = model.to(chosen_device)
    optimizer_parameters = dict(adam_kwargs or {})

    def make_optimizer() -> torch.optim.Adam:
        return torch.optim.Adam(model.parameters(), lr=learning_rate, **optimizer_parameters)

    optimizer = make_optimizer()
    training_losses: list[float] = []
    mean_total_rewards: list[float] = []
    validation_rewards: list[float] = []
    validation_epochs: list[int] = []
    training_log: list[dict[str, float | int | str]] = []
    monotonicity_penalties: list[float] = []
    best_validation_reward: float | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_imitation_loss: float | None = None
    best_imitation_epoch: int | None = None
    best_imitation_state_dict: dict[str, torch.Tensor] | None = None
    best_imitation_validation_mae: float | None = None
    best_imitation_validation_rmse: float | None = None
    imitation_end_state_dict: dict[str, torch.Tensor] | None = None
    imitation_epochs_completed = 0
    imitation_early_stopped = False
    optimizer_reset_performed = False
    imitation_selection_metric = (
        "deployed_imitation_rmse" if clip_imitation_predictions else "raw_imitation_rmse"
    )
    patience_reference: float | None = None
    epochs_without_improvement = 0
    imitation_validation_batch: dict[str, torch.Tensor] | None = None
    if (imitation_early_stopping or track_imitation_validation) and validation_products:
        validation_generator = torch.Generator().manual_seed(seed + 2_000_003)
        imitation_validation_batch = move_batch_to_device(
            make_paper_lost_sales_training_batch(
                validation_products,
                horizon,
                validation_generator,
                feature_stats=feature_stats,
            ),
            infer_model_device(model),
        )
    start_time = time.perf_counter()

    def train_epoch(
        epoch_number: int, imitation_phase: bool
    ) -> tuple[dict[str, float | int | str], float]:
        nonlocal optimizer
        indices = torch.randint(0, len(products), (batch_size,), generator=generator).tolist()
        batch_products = [products[index] for index in indices]
        batch = move_batch_to_device(
            make_paper_lost_sales_training_batch(batch_products, horizon, generator, feature_stats=feature_stats),
            infer_model_device(model),
        )
        if imitation_phase:
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
        loss = primary_loss + float(monotonicity_penalty_weight) * monotonicity_penalty
        track_residual_gradients = bool(
            getattr(model, "paper_track_residual_gradient_diagnostics", False)
        )
        residual_diagnostics: dict[str, float] = {}
        if track_residual_gradients:
            backbone_parameters = list(model.backbone.parameters())
            readout_parameters = list(model.residual_readout.parameters())
            backbone_before = [parameter.detach().clone() for parameter in backbone_parameters]
            readout_before = [parameter.detach().clone() for parameter in readout_parameters]
        optimizer.zero_grad()
        loss.backward()
        if track_residual_gradients:
            residual_diagnostics.update(
                {
                    "residual_backbone_gradient_norm": _parameter_gradient_norm(backbone_parameters),
                    "residual_readout_gradient_norm": _parameter_gradient_norm(readout_parameters),
                }
            )
        optimizer.step()
        if track_residual_gradients:
            residual_diagnostics.update(
                {
                    "residual_backbone_update_norm": _parameter_update_norm(
                        backbone_parameters, backbone_before
                    ),
                    "residual_readout_update_norm": _parameter_update_norm(
                        readout_parameters, readout_before
                    ),
                    "residual_readout_weight_norm": float(
                        torch.linalg.vector_norm(model.residual_readout.weight.detach()).item()
                    ),
                }
            )
        if hasattr(model, "apply_monotonic_constraints"):
            model.apply_monotonic_constraints()
        order_diagnostics = paper_order_diagnostics(
            model,
            batch,
            max_order_factor=max_order_factor,
            scale_action_by_history_mean=scale_action_by_history_mean,
        )
        training_losses.append(float(loss.detach().item()))
        mean_total_rewards.append(float(mean_total_reward.detach().item()))
        monotonicity_penalties.append(float(monotonicity_penalty.detach().item()))
        log_row: dict[str, float | int | str] = {
            "epoch": epoch_number,
            "phase": 0 if imitation_phase else 1,
            "training_loss": float(loss.detach().item()),
            "primary_loss": float(primary_loss.detach().item()),
            "training_mean_total_reward": float(mean_total_reward.detach().item()),
            "monotonicity_penalty": float(monotonicity_penalty.detach().item()),
            "monotonicity_penalty_weight": float(monotonicity_penalty_weight),
            "elapsed_seconds": float(time.perf_counter() - start_time),
            **order_diagnostics,
            **residual_diagnostics,
        }

        return log_row, float(primary_loss.detach().item())

    def validate_reward(
        epoch_number: int, log_row: dict[str, float | int | str], is_final: bool
    ) -> None:
        nonlocal best_validation_reward, best_state_dict
        should_validate = bool(
            validation_products
            and validation_interval
            and (epoch_number % validation_interval == 0 or is_final)
        )
        if should_validate:
            if validation_callback is not None:
                validation_reward = validation_callback(epoch_number, model)
            else:
                validation_batch = make_paper_lost_sales_training_batch(
                    validation_products,
                    horizon,
                    generator,
                    feature_stats=feature_stats,
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
            validation_epochs.append(epoch_number)
            log_row["validation_mean_total_reward"] = float(validation_reward)
            if best_validation_reward is None or validation_reward > best_validation_reward:
                best_validation_reward = float(validation_reward)
                best_state_dict = copy.deepcopy(model.state_dict())

    global_epoch = 0
    for imitation_index in range(imitation_limit):
        global_epoch += 1
        log_row, current_imitation_loss = train_epoch(global_epoch, imitation_phase=True)
        imitation_epochs_completed += 1
        validation_mae: float | None = None
        validation_rmse: float | None = None
        if imitation_validation_batch is not None:
            validation_diagnostics = paper_order_diagnostics(
                model,
                imitation_validation_batch,
                max_order_factor=max_order_factor,
                scale_action_by_history_mean=scale_action_by_history_mean,
                inventory_fractions=(0.0, 0.25, 0.5, 0.75, 1.0),
            )
            selection_prefix = "deployed" if clip_imitation_predictions else "raw"
            validation_mae = float(validation_diagnostics[f"{selection_prefix}_imitation_mae"])
            validation_rmse = float(validation_diagnostics[f"{selection_prefix}_imitation_rmse"])
            log_row["imitation_validation_raw_mae"] = float(
                validation_diagnostics["raw_imitation_mae"]
            )
            log_row["imitation_validation_raw_rmse"] = float(
                validation_diagnostics["raw_imitation_rmse"]
            )
            log_row["imitation_validation_deployed_mae"] = float(
                validation_diagnostics["deployed_imitation_mae"]
            )
            log_row["imitation_validation_deployed_rmse"] = float(
                validation_diagnostics["deployed_imitation_rmse"]
            )
            log_row["imitation_checkpoint_metric"] = imitation_selection_metric
            log_row["imitation_validation_mae"] = validation_mae
            log_row["imitation_validation_rmse"] = validation_rmse
            zero_inventory = torch.zeros_like(imitation_validation_batch["initial_inventory"])
            zero_raw_order = paper_lost_sales_raw_order(
                model,
                imitation_validation_batch,
                zero_inventory,
                scale_action_by_history_mean=scale_action_by_history_mean,
            )
            log_row["imitation_validation_zero_inventory_raw_order_mean"] = float(
                zero_raw_order.mean().item()
            )
        if imitation_early_stopping:
            if validation_mae is None or validation_rmse is None:  # pragma: no cover
                raise RuntimeError("imitation validation diagnostics were not computed")
            if best_imitation_validation_rmse is None or validation_rmse < best_imitation_validation_rmse:
                best_imitation_validation_mae = validation_mae
                best_imitation_validation_rmse = validation_rmse
                best_imitation_loss = validation_rmse
                best_imitation_epoch = global_epoch
                best_imitation_state_dict = copy.deepcopy(model.state_dict())
            if patience_reference is None or validation_rmse < patience_reference - float(imitation_min_delta):
                patience_reference = validation_rmse
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            imitation_early_stopped = epochs_without_improvement >= int(imitation_patience)
            log_row["imitation_epochs_without_improvement"] = epochs_without_improvement
            log_row["imitation_early_stop_triggered"] = int(imitation_early_stopped)
        else:
            if best_imitation_loss is None or current_imitation_loss < best_imitation_loss:
                best_imitation_loss = current_imitation_loss
                best_imitation_epoch = global_epoch
                best_imitation_state_dict = copy.deepcopy(model.state_dict())

        is_final = epochs == 0 and (imitation_early_stopped or imitation_index + 1 == imitation_limit)
        validate_reward(global_epoch, log_row, is_final=is_final)
        training_log.append(log_row)

        if imitation_early_stopped:
            break

    if imitation_early_stopping and best_imitation_state_dict is not None:
        model.load_state_dict(best_imitation_state_dict)
    if imitation_epochs_completed > 0:
        imitation_end_state_dict = copy.deepcopy(model.state_dict())
    if reset_optimizer_after_imitation and imitation_epochs_completed > 0 and epochs > 0:
        optimizer = make_optimizer()
        optimizer_reset_performed = True

    for reward_index in range(int(epochs)):
        global_epoch += 1
        log_row, _ = train_epoch(global_epoch, imitation_phase=False)
        log_row["optimizer_reset_after_imitation"] = int(optimizer_reset_performed)
        validate_reward(global_epoch, log_row, is_final=reward_index + 1 == int(epochs))
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
        best_imitation_validation_mae=best_imitation_validation_mae,
        best_imitation_validation_rmse=best_imitation_validation_rmse,
        imitation_selection_metric=(imitation_selection_metric if imitation_early_stopping else None),
        imitation_end_state_dict=imitation_end_state_dict,
        imitation_epochs_completed=imitation_epochs_completed,
        imitation_early_stopped=imitation_early_stopped,
        optimizer_reset_after_imitation=optimizer_reset_performed,
        training_log=training_log,
        monotonicity_penalties=monotonicity_penalties,
        elapsed_seconds=elapsed_seconds,
    )
