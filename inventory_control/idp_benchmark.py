"""Unified benchmark for the monotonic neural inventory-control IDP."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install neural dependencies with `python -m pip install -e '.[dev]'`.") from exc

from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.paper_evaluation import paper_evaluation_seed_start, require_final_test_confirmation
from inventory_control.paper_lost_sales import (
    BaseStockOrderPolicy,
    evaluate_paper_lost_sales_policy,
    historical_demand_base_stock_policy,
    non_omniscient_gamma_base_stock_policy,
    omniscient_gamma_base_stock_policy,
    paper_evaluation_demand_path,
)
from inventory_control.policies.neural.fcnn import (
    PaperFCNNFeatureStats,
    compute_paper_fcnn_feature_stats,
    infer_model_device,
    make_paper_fcnn_model,
    make_paper_order_up_to_fcnn_model,
    make_paper_residual_order_up_to_fcnn_model,
)
from inventory_control.policies.neural.constrained_mlp import (
    make_constrained_monotone_mlp_model,
    make_paper_residual_order_up_to_constrained_mlp_model,
    make_positive_weight_mlp_model,
)
from inventory_control.policies.neural.monokan import (
    MonoKANPolicy,
    make_paper_monokan_model,
    make_paper_order_up_to_monokan_model,
    make_paper_residual_order_up_to_kan_model,
    make_paper_residual_order_up_to_monokan_model,
    sample_monokan_first_layer_splines,
)
from inventory_control.policies.neural.registry import (
    DEFAULT_STRUCTURE_POLICY_NAMES,
    NEURAL_POLICY_SPECS,
    TRAINABLE_POLICY_NAMES,
    checkpoint_filenames,
)
from inventory_control.products import PaperLostSalesProduct, PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import (
    DirectBackpropResult,
    make_paper_lost_sales_training_batch,
    paper_lost_sales_context_features,
    paper_lost_sales_raw_order,
    paper_lost_sales_target_level,
    train_paper_lost_sales_policy,
)


POLICY_ORDER = [spec.name for spec in sorted(NEURAL_POLICY_SPECS, key=lambda spec: spec.order)] + [
    "NonOmniscientOptimal",
    "OmniscientOptimal",
    "HistoricalDemandBaseStock",
]

TRAINABLE_POLICIES = TRAINABLE_POLICY_NAMES
DEFAULT_TRAINABLE_POLICIES = DEFAULT_STRUCTURE_POLICY_NAMES


def _anchor_policy(product: PaperLostSalesProduct, residual_anchor: str) -> BaseStockOrderPolicy:
    if residual_anchor == "gamma":
        return historical_demand_base_stock_policy(product, fit_gamma=True)
    if residual_anchor == "empirical":
        return historical_demand_base_stock_policy(product, fit_gamma=False)
    raise ValueError(f"Unsupported residual_anchor: {residual_anchor}")


@dataclass(frozen=True)
class IDPBenchmarkArtifacts:
    """Files and in-memory tables produced by one benchmark run."""

    output_dir: Path
    summary: pd.DataFrame
    operational_metrics: pd.DataFrame
    predictions: pd.DataFrame
    training_log: pd.DataFrame
    spline_values: pd.DataFrame
    target_structure: pd.DataFrame
    spline_certification: pd.DataFrame
    sample_efficiency: pd.DataFrame | None = None


class _TorchPaperPolicyAdapter:
    """Apply one trained torch model using the common IDP deployment rule."""

    def __init__(
        self,
        name: str,
        model: nn.Module,
        product: PaperLostSalesProduct,
        feature_stats: PaperFCNNFeatureStats,
        max_order_factor: float,
        scale_action_by_history_mean: bool,
    ) -> None:
        self.name = name
        self.model = model
        self.product = product
        self.feature_stats = feature_stats
        self.max_order_factor = max_order_factor
        self.scale_action_by_history_mean = scale_action_by_history_mean

    def act(self, inventory: float, t: int, rng=None) -> float:
        """Return the deployed, state-capped order quantity."""
        raw, deployed = _model_orders(
            self.model,
            self.product,
            self.feature_stats,
            np.asarray([inventory], dtype=float),
            self.max_order_factor,
            self.scale_action_by_history_mean,
        )
        del raw
        return float(deployed[0])


def _profile_settings(config: dict[str, Any], profile: str) -> dict[str, Any]:
    if profile not in {"compact", "full", "paper_replication"}:
        raise ValueError("profile must be 'compact', 'full', or 'paper_replication'")
    benchmark = config.get("idp_benchmark")
    if benchmark is None:
        raise ValueError("paper lost-sales config is missing idp_benchmark settings")
    settings = {**benchmark, **benchmark[profile]}
    for field in ("learning_rate", "include_terminal_salvage"):
        if field not in settings and field in config:
            settings[field] = config[field]
    settings["profile"] = profile
    if profile in {"compact", "paper_replication"}:
        settings["penalty_weights"] = benchmark[profile].get("penalty_weights", [1.0])
    return settings


def _model_config(config: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Return a config with matched depth/width for IDP neural models."""
    monokan_config = config.get("monokan", {})
    return {
        **config,
        "fcnn": {
            **config["fcnn"],
            "history_encoder": "causal_cnn",
            "hidden_layers": int(config["fcnn"]["hidden_layers"]),
            "hidden_units": int(config["fcnn"]["hidden_units"]),
            "cnn_channels": int(config["fcnn"].get("cnn_channels", 8)),
            "cnn_dilations": list(config["fcnn"].get("cnn_dilations", [1, 2, 4, 8, 16])),
            "activation": str(config["fcnn"]["activation"]),
        },
        "monokan": {
            **monokan_config,
            "hidden_layers": int(monokan_config.get("hidden_layers", settings["hidden_layers"])),
            "hidden_units": int(monokan_config.get("hidden_units", settings["hidden_units"])),
            "max_order_factor": float(monokan_config.get("max_order_factor", settings["max_order_factor"])),
            "scale_action_by_history_mean": bool(
                monokan_config.get("scale_action_by_history_mean", settings["scale_action_by_history_mean"])
            ),
        },
        "positive_weight_mlp": {
            **config.get("positive_weight_mlp", {}),
            "hidden_layers": int(settings["hidden_layers"]),
            "hidden_units": int(settings["hidden_units"]),
            "activation": str(settings.get("activation", config["fcnn"]["activation"])),
            "mixed_activation": False,
        },
        "constrained_monotone_mlp": {
            **config.get("constrained_monotone_mlp", {}),
            "hidden_layers": int(settings["hidden_layers"]),
            "hidden_units": int(settings["hidden_units"]),
            "activation": str(settings.get("activation", config["fcnn"]["activation"])),
            "mixed_activation": True,
        },
    }


def _model_orders(
    model: nn.Module,
    product: PaperLostSalesProduct,
    feature_stats: PaperFCNNFeatureStats,
    inventories: np.ndarray,
    max_order_factor: float,
    scale_action_by_history_mean: bool,
) -> tuple[np.ndarray, np.ndarray]:
    device = infer_model_device(model)
    history = torch.tensor(
        [[float(value) / feature_stats.demand_scale for value in product.historical_demand] for _ in inventories],
        dtype=torch.float32,
        device=device,
    )
    historical_mean = max(float(np.mean(product.historical_demand)), 1.0)
    batch = {
        "history": history,
        "feature_demand_scale": torch.full((len(inventories),), feature_stats.demand_scale, dtype=torch.float32, device=device),
        "feature_price_scale": torch.full((len(inventories),), feature_stats.price_scale, dtype=torch.float32, device=device),
        "historical_mean_demand": torch.full((len(inventories),), historical_mean, dtype=torch.float32, device=device),
        "historical_base_stock_level": torch.full(
            (len(inventories),),
            float(_anchor_policy(product, str(getattr(model, "paper_residual_anchor", "gamma"))).S),
            dtype=torch.float32,
            device=device,
        ),
        "price": torch.full((len(inventories),), float(product.price), dtype=torch.float32, device=device),
        "purchase_cost": torch.full((len(inventories),), float(product.purchase_cost), dtype=torch.float32, device=device),
        "holding_cost": torch.full((len(inventories),), float(product.holding_cost), dtype=torch.float32, device=device),
        "lost_sale_penalty": torch.full((len(inventories),), float(product.lost_sale_penalty), dtype=torch.float32, device=device),
    }
    inventory_t = torch.tensor(inventories, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        raw = paper_lost_sales_raw_order(
            model,
            batch,
            inventory_t,
            scale_action_by_history_mean=scale_action_by_history_mean,
        )
        raw = raw.detach().cpu().numpy().astype(float)
    cap = np.maximum(float(max_order_factor) * historical_mean - inventories, 0.0)
    return raw, np.minimum(raw, cap)


def _batched_torch_policy_metrics(
    products: list[PaperLostSalesProduct],
    model: nn.Module,
    feature_stats: PaperFCNNFeatureStats,
    settings: dict[str, Any],
    split: str,
) -> pd.DataFrame:
    """Evaluate one torch policy over products with batched per-period forward passes."""
    if not products:
        return pd.DataFrame()
    horizon = int(settings["evaluation_horizon"])
    burn_in = int(settings["burn_in"])
    seed_start = paper_evaluation_seed_start(split)
    device = infer_model_device(model)
    histories = torch.tensor(
        [
            [
                value / feature_stats.demand_scale
                for value in product.historical_demand
            ]
            for product in products
        ],
        dtype=torch.float32,
        device=device,
    )
    historical_mean = np.asarray(
        [max(float(np.mean(product.historical_demand)), 1.0) for product in products],
        dtype=float,
    )
    historical_mean_t = torch.tensor(historical_mean, dtype=torch.float32, device=device)
    price = torch.tensor([product.price for product in products], dtype=torch.float32, device=device)
    purchase_cost = torch.tensor([product.purchase_cost for product in products], dtype=torch.float32, device=device)
    holding_cost = torch.tensor([product.holding_cost for product in products], dtype=torch.float32, device=device)
    lost_sale_penalty = torch.tensor([product.lost_sale_penalty for product in products], dtype=torch.float32, device=device)
    historical_base_stock = torch.tensor(
        [_anchor_policy(product, str(getattr(model, "paper_residual_anchor", settings.get("residual_anchor", "gamma")))).S for product in products],
        dtype=torch.float32,
        device=device,
    )
    batch = {
        "history": histories,
        "feature_demand_scale": torch.full((len(products),), feature_stats.demand_scale, dtype=torch.float32, device=device),
        "feature_price_scale": torch.full((len(products),), feature_stats.price_scale, dtype=torch.float32, device=device),
        "historical_mean_demand": historical_mean_t,
        "historical_base_stock_level": historical_base_stock,
        "price": price,
        "purchase_cost": purchase_cost,
        "holding_cost": holding_cost,
        "lost_sale_penalty": lost_sale_penalty,
    }
    demand_paths = torch.tensor(
        np.asarray(
            [
            paper_evaluation_demand_path(product, horizon=horizon, seed=seed_start + index)
            for index, product in enumerate(products)
            ],
            dtype=float,
        ),
        dtype=torch.float32,
        device=device,
    )
    inventory = torch.tensor([float(product.initial_inventory) for product in products], dtype=torch.float32, device=device)
    max_inventory_level = float(settings["max_order_factor"]) * historical_mean_t
    total_reward = torch.zeros(len(products), dtype=torch.float32, device=device)
    total_sales = torch.zeros_like(total_reward)
    total_demand = torch.zeros_like(total_reward)
    total_lost_sales = torch.zeros_like(total_reward)
    total_order = torch.zeros_like(total_reward)
    total_inventory = torch.zeros_like(total_reward)
    total_revenue = torch.zeros_like(total_reward)
    total_purchase = torch.zeros_like(total_reward)
    total_lost_penalty = torch.zeros_like(total_reward)
    total_holding = torch.zeros_like(total_reward)
    min_order = torch.full_like(total_reward, float("inf"))
    max_order = torch.zeros_like(total_reward)
    counted = max(horizon - burn_in, 1)

    model.eval()
    with torch.no_grad():
        parameterization = getattr(model, "paper_action_parameterization", "raw")
        target_level = (
            paper_lost_sales_target_level(model, batch)
            if parameterization in {"order_up_to", "residual_order_up_to"}
            else None
        )
        for t in range(horizon):
            if target_level is None:
                raw_order = paper_lost_sales_raw_order(
                    model,
                    batch,
                    inventory,
                    scale_action_by_history_mean=bool(settings["scale_action_by_history_mean"]),
                )
            else:
                raw_order = torch.relu(target_level - inventory)
            order_cap = torch.relu(max_inventory_level - inventory)
            order = torch.minimum(raw_order, order_cap)
            demand = demand_paths[:, t]
            available = inventory + order
            sales = torch.minimum(demand, available)
            lost_sales = torch.relu(demand - available)
            ending_inventory = torch.relu(available - demand)
            revenue = price * sales
            purchase = purchase_cost * order
            lost_penalty = lost_sale_penalty * lost_sales
            holding = holding_cost * ending_inventory
            reward = revenue - purchase - lost_penalty - holding
            if t >= burn_in:
                total_reward += reward
                total_sales += sales
                total_demand += demand
                total_lost_sales += lost_sales
                total_order += order
                min_order = torch.minimum(min_order, order)
                max_order = torch.maximum(max_order, order)
                total_inventory += ending_inventory
                total_revenue += revenue
                total_purchase += purchase
                total_lost_penalty += lost_penalty
                total_holding += holding
            inventory = ending_inventory

    rows = []
    values = {
        "mean_total_reward": total_reward.detach().cpu().numpy(),
        "service_level": torch.where(total_demand > 0, total_sales / total_demand, torch.ones_like(total_demand)).detach().cpu().numpy(),
        "lost_sales_rate": torch.where(total_demand > 0, total_lost_sales / total_demand, torch.zeros_like(total_demand)).detach().cpu().numpy(),
        "mean_order_quantity": (total_order / counted).detach().cpu().numpy(),
        "min_order_quantity": torch.where(torch.isinf(min_order), torch.zeros_like(min_order), min_order).detach().cpu().numpy(),
        "max_order_quantity": max_order.detach().cpu().numpy(),
        "mean_inventory": (total_inventory / counted).detach().cpu().numpy(),
        "mean_holding_cost": (total_holding / counted).detach().cpu().numpy(),
        "mean_purchase_cost": (total_purchase / counted).detach().cpu().numpy(),
        "mean_sales_revenue": (total_revenue / counted).detach().cpu().numpy(),
        "mean_lost_sales_penalty": (total_lost_penalty / counted).detach().cpu().numpy(),
    }
    for index, product in enumerate(products):
        row = {
            "product_id": product.product_id,
            "policy_name": "",
            "evaluation_split": split,
            "evaluation_seed": seed_start + index,
        }
        row.update({name: float(array[index]) for name, array in values.items()})
        row["mean_reward_per_period"] = float(row["mean_total_reward"] / counted)
        rows.append(row)
    return pd.DataFrame(rows)


def _mean_validation_reward(
    config: dict[str, Any],
    settings: dict[str, Any],
    products: list[PaperLostSalesProduct],
    model: nn.Module,
    name: str,
    feature_stats: PaperFCNNFeatureStats,
    split: str,
) -> float:
    del config, name
    metrics = _batched_torch_policy_metrics(products, model, feature_stats, settings, split)
    return float(metrics["mean_total_reward"].mean()) if not metrics.empty else 0.0


def _train_model(
    config: dict[str, Any],
    settings: dict[str, Any],
    products: list[PaperLostSalesProduct],
    validation_products: list[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
    model: nn.Module,
    name: str,
    split: str,
    penalty_weight: float = 0.0,
    clip_imitation_predictions: bool = True,
    progress: Callable[[str], None] | None = None,
) -> DirectBackpropResult:
    def validation_callback(epoch: int, current_model: nn.Module) -> float:
        validation_reward = _mean_validation_reward(
            config,
            settings,
            validation_products,
            current_model,
            name,
            feature_stats,
            split,
        )
        if progress is not None:
            progress(f"{name}: validation epoch={epoch} mean_total_reward={validation_reward:.3f}")
        return validation_reward

    if progress is not None:
        progress(
            f"{name}: start training products={len(products)} batch_size={min(int(settings['batch_size']), len(products))} "
            f"horizon={int(settings['training_horizon'])} epochs={int(settings['imitation_epochs']) + int(settings['reward_epochs'])}"
        )
    result = train_paper_lost_sales_policy(
        products,
        model,
        horizon=int(settings["training_horizon"]),
        batch_size=min(int(settings["batch_size"]), len(products)),
        epochs=int(settings["reward_epochs"]),
        learning_rate=float(settings.get("learning_rate", config["learning_rate"])),
        seed=int(settings["seed"]),
        imitation_epochs=int(settings["imitation_epochs"]),
        validation_products=validation_products,
        validation_interval=int(settings["validation_interval"]),
        validation_callback=validation_callback,
        feature_stats=feature_stats,
        max_order_factor=float(settings["max_order_factor"]),
        scale_action_by_history_mean=bool(settings["scale_action_by_history_mean"]),
        normalize_reward_loss=bool(settings["normalize_reward_loss"]),
        monotonicity_penalty_weight=float(penalty_weight),
        monotonicity_samples_per_product=int(settings["monotonicity_samples_per_product"]),
        cap_excess_penalty_weight=float(settings.get("cap_excess_penalty_weight", 0.0)),
        cap_excess_samples_per_product=int(settings.get("cap_excess_samples_per_product", 1)),
        target_stock_penalty_weight=float(settings.get("target_stock_penalty_weight", 0.0)),
        target_stock_samples_per_product=int(settings.get("target_stock_samples_per_product", 1)),
        active_slope_penalty_weight=float(settings.get("active_slope_penalty_weight", 0.0)),
        active_slope_samples_per_product=int(settings.get("active_slope_samples_per_product", 1)),
        spline_l2_weight=float(settings.get("spline_l2_weight", 0.0)),
        scale_l2_weight=float(settings.get("scale_l2_weight", 0.0)),
        training_demand_mode=str(settings.get("training_demand_mode", "fixed_future")),
        residual_anchor=str(settings.get("residual_anchor", "gamma")),
        include_terminal_salvage=bool(settings.get("include_terminal_salvage", False)),
        clip_imitation_predictions=clip_imitation_predictions,
    )
    if result.best_state_dict is not None:
        model.load_state_dict(result.best_state_dict)
    if progress is not None:
        best = result.best_validation_reward
        suffix = f" best_validation={best:.3f}" if best is not None else ""
        progress(f"{name}: finished training_seconds={result.elapsed_seconds:.3f}{suffix}")
    return result


def _oracle_levels(
    products: list[PaperLostSalesProduct],
    oracle_mc_samples: int,
    seed: int,
) -> dict[str, float]:
    del oracle_mc_samples, seed
    return {
        product.product_id: omniscient_gamma_base_stock_policy(product).S
        for product in products
    }


def _prediction_table(
    products: list[PaperLostSalesProduct],
    models: dict[str, nn.Module],
    feature_stats: PaperFCNNFeatureStats,
    settings: dict[str, Any],
    oracle_levels: dict[str, float],
    model_action_settings: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for product in products:
        historical_mean = max(float(np.mean(product.historical_demand)), 1.0)
        max_inventory = float(settings["max_order_factor"]) * historical_mean
        inventories = np.linspace(0.0, max_inventory, int(settings["inventory_states"]))
        oracle_order = np.minimum(
            np.maximum(float(oracle_levels[product.product_id]) - inventories, 0.0),
            np.maximum(max_inventory - inventories, 0.0),
        )
        non_omniscient_level = non_omniscient_gamma_base_stock_policy(product).S
        empirical_level = historical_demand_base_stock_policy(product).S
        policy_orders: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "OmniscientOptimal": (oracle_order, oracle_order),
            "NonOmniscientOptimal": (
                np.maximum(float(non_omniscient_level) - inventories, 0.0),
                np.minimum(
                    np.maximum(float(non_omniscient_level) - inventories, 0.0),
                    np.maximum(max_inventory - inventories, 0.0),
                ),
            ),
            "HistoricalDemandBaseStock": (
                np.maximum(float(empirical_level) - inventories, 0.0),
                np.minimum(
                    np.maximum(float(empirical_level) - inventories, 0.0),
                    np.maximum(max_inventory - inventories, 0.0),
                ),
            ),
        }
        for name, model in models.items():
            action_settings = (model_action_settings or {}).get(name, settings)
            policy_orders[name] = _model_orders(
                model,
                product,
                feature_stats,
                inventories,
                float(action_settings["max_order_factor"]),
                bool(action_settings["scale_action_by_history_mean"]),
            )
        for policy_name, (raw_orders, deployed_orders) in policy_orders.items():
            for state_index, inventory in enumerate(inventories):
                deployed = float(deployed_orders[state_index])
                target = float(oracle_order[state_index])
                rows.append(
                    {
                        "product_id": product.product_id,
                        "policy_name": policy_name,
                        "state_index": state_index,
                        "inventory": float(inventory),
                        "inventory_normalized": float(inventory / historical_mean),
                        "historical_mean_demand": historical_mean,
                        "oracle_order": target,
                        "raw_order": float(raw_orders[state_index]),
                        "deployed_order": deployed,
                        "error": deployed - target,
                        "order_up_to_level": float(deployed + inventory),
                        "omniscient_order_up_to_level": float(oracle_levels[product.product_id]),
                        "true_coefficient_of_variation": float(product.true_coefficient_of_variation),
                    }
                )
    return pd.DataFrame(rows)


def _prediction_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy_name, policy_rows in predictions.groupby("policy_name", sort=False):
        errors = policy_rows["error"].to_numpy(dtype=float)
        zero_inventory_rows = policy_rows.loc[policy_rows["state_index"] == 0]
        order_up_to_errors = (
            zero_inventory_rows["order_up_to_level"].to_numpy(dtype=float)
            - zero_inventory_rows["omniscient_order_up_to_level"].to_numpy(dtype=float)
        )
        raw_violations: list[float] = []
        deployed_violations: list[float] = []
        for _, product_rows in policy_rows.groupby("product_id", sort=False):
            ordered = product_rows.sort_values("state_index")
            raw_violations.extend(np.diff(ordered["raw_order"].to_numpy(dtype=float)).tolist())
            deployed_violations.extend(np.diff(ordered["deployed_order"].to_numpy(dtype=float)).tolist())
        raw_increases = np.maximum(np.asarray(raw_violations, dtype=float), 0.0)
        deployed_increases = np.maximum(np.asarray(deployed_violations, dtype=float), 0.0)
        active_slope_deviation, inactive_slope_deviation = _slope_deviations(policy_rows)
        rows.append(
            {
                "policy_name": policy_name,
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "order_up_to_mae": float(np.mean(np.abs(order_up_to_errors))),
                "order_up_to_bias": float(np.mean(order_up_to_errors)),
                "order_up_to_rmse": float(np.sqrt(np.mean(order_up_to_errors**2))),
                "raw_order_mean": float(policy_rows["raw_order"].mean()),
                "deployed_order_mean": float(policy_rows["deployed_order"].mean()),
                "cap_hit_rate": float(
                    np.mean(
                        policy_rows["raw_order"].to_numpy(dtype=float)
                        >= policy_rows["deployed_order"].to_numpy(dtype=float) + 1e-6
                    )
                ),
                "raw_violation_rate": float(np.mean(raw_increases > 1e-6)),
                "deployed_violation_rate": float(np.mean(deployed_increases > 1e-6)),
                "raw_max_violation": float(raw_increases.max(initial=0.0)),
                "deployed_max_violation": float(deployed_increases.max(initial=0.0)),
                "mean_slope_deviation_from_minus_one": active_slope_deviation,
                "active_slope_deviation_from_minus_one": active_slope_deviation,
                "inactive_slope_deviation_from_zero": inactive_slope_deviation,
            }
        )
    return pd.DataFrame(rows)


def _target_level_from_context(model: nn.Module, batch: dict[str, torch.Tensor], context: torch.Tensor) -> torch.Tensor:
    """Decode a target-stock level from prebuilt context features."""
    parameterization = getattr(model, "paper_action_parameterization", "raw")
    historical_mean = torch.clamp(batch["historical_mean_demand"], min=1.0)
    if parameterization == "order_up_to":
        target_factor = model(context)
        if bool(getattr(model, "paper_scale_target_by_history_mean", True)):
            return target_factor * historical_mean
        return target_factor
    if parameterization == "residual_order_up_to":
        residual_factor = model(context)
        residual_center = float(getattr(model, "paper_residual_center", 0.0))
        residual_scale = float(getattr(model, "paper_residual_scale", 0.5))
        return batch["historical_base_stock_level"] + residual_scale * (residual_factor - residual_center) * historical_mean
    raise ValueError(f"Model parameterization {parameterization!r} does not predict a context target level")


def _context_feature_specs(history_length: int) -> list[dict[str, Any]]:
    return [
        {"feature": "demand_level", "direction": 1, "indices": list(range(history_length)), "step": 0.05},
        {"feature": "price", "direction": 1, "indices": [history_length], "step": 0.02},
        {"feature": "purchase_cost", "direction": -1, "indices": [history_length + 1], "step": 0.02},
        {"feature": "holding_cost", "direction": -1, "indices": [history_length + 2], "step": 0.02},
        {"feature": "lost_sale_penalty", "direction": 1, "indices": [history_length + 3], "step": 0.02},
    ]


def _target_structure_diagnostics(
    products: list[PaperLostSalesProduct],
    models: dict[str, nn.Module],
    feature_stats: PaperFCNNFeatureStats,
    settings: dict[str, Any],
) -> pd.DataFrame:
    """Measure target-stock comparative statics for context-only policies."""
    eligible = {
        name: model
        for name, model in models.items()
        if getattr(model, "paper_action_parameterization", "raw") in {"order_up_to", "residual_order_up_to"}
    }
    if not products or not eligible:
        return pd.DataFrame()
    device = infer_model_device(next(iter(eligible.values())))
    batch = make_paper_lost_sales_training_batch(
        products,
        horizon=1,
        generator=torch.Generator().manual_seed(int(settings["seed"]) + 77_000),
        feature_stats=feature_stats,
        residual_anchor=str(settings.get("residual_anchor", "gamma")),
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    history_length = int(batch["history"].shape[1])
    specs = _context_feature_specs(history_length)
    rows: list[dict[str, float | str | int]] = []
    for name, model in eligible.items():
        model.to(device)
        model.eval()
        with torch.no_grad():
            context = paper_lost_sales_context_features(batch)
            base_target = _target_level_from_context(model, batch, context)
            anchor = batch["historical_base_stock_level"]
            historical_mean = torch.clamp(batch["historical_mean_demand"], min=1.0)
            correction = base_target - anchor
            correction_abs = torch.abs(correction)
            correction_threshold = 0.1 * historical_mean
        for spec in specs:
            context_hi = context.detach().clone()
            context_hi[:, spec["indices"]] += float(spec["step"])
            with torch.no_grad():
                hi_target = _target_level_from_context(model, batch, context_hi)
            diff = hi_target - base_target
            wrong_direction = torch.clamp(-float(spec["direction"]) * diff, min=0.0)
            context_grad = context.detach().clone().requires_grad_(True)
            grad_target = _target_level_from_context(model, batch, context_grad)
            selected = torch.autograd.grad(grad_target.sum(), context_grad, retain_graph=False)[0][:, spec["indices"]]
            directional_grad = float(spec["direction"]) * selected.sum(dim=1)
            rows.append(
                {
                    "policy_name": name,
                    "feature": str(spec["feature"]),
                    "expected_direction": int(spec["direction"]),
                    "finite_diff_step": float(spec["step"]),
                    "finite_diff_violation_rate": float(torch.mean((wrong_direction > 1e-6).float()).item()),
                    "finite_diff_max_wrong_direction": float(wrong_direction.max().item()),
                    "autograd_min_directional_derivative": float(directional_grad.min().item()),
                    "autograd_mean_directional_derivative": float(directional_grad.mean().item()),
                    "autograd_violation_rate": float(torch.mean((directional_grad < -1e-6).float()).item()),
                    "target_stock_mean": float(base_target.mean().item()),
                    "residual_correction_abs_mean": float(correction_abs.mean().item()),
                    "residual_correction_abs_p90": float(torch.quantile(correction_abs, 0.9).item()),
                    "residual_correction_large_rate": float(torch.mean((correction_abs > correction_threshold).float()).item()),
                    "products": len(products),
                }
            )
    return pd.DataFrame(rows)


def _monokan_spline_certification(models: dict[str, nn.Module], samples: int = 200) -> pd.DataFrame:
    """Certify sampled first-layer edge directions for constrained MonoKAN inputs."""
    rows: list[dict[str, float | str | int]] = []
    for name, model in models.items():
        if not isinstance(model, MonoKANPolicy) or not model.monotonic_vars:
            continue
        for feature_index, direction in sorted(model.monotonic_vars.items()):
            x_values, edge_values = sample_monokan_first_layer_splines(model, feature_index=feature_index, samples=samples)
            del x_values
            increments = torch.diff(edge_values, dim=1)
            wrong_direction = torch.clamp(-float(direction) * increments, min=0.0)
            rows.append(
                {
                    "policy_name": name,
                    "feature_index": int(feature_index),
                    "expected_direction": int(direction),
                    "edges": int(edge_values.shape[0]),
                    "intervals": int(increments.numel()),
                    "violating_interval_rate": float(torch.mean((wrong_direction > 1e-6).float()).item()),
                    "max_wrong_direction_increment": float(wrong_direction.max().item()),
                    "domain_min": float(model.layers[0].grid[:, 0].min().item()),
                    "domain_max": float(model.layers[0].grid[:, -1].max().item()),
                }
            )
    return pd.DataFrame(rows)


def _mean_slope_deviation(policy_rows: pd.DataFrame) -> float:
    """Return active-region slope deviation retained for compatibility."""
    active_deviation, _ = _slope_deviations(policy_rows)
    return active_deviation


def _slope_deviations(policy_rows: pd.DataFrame) -> tuple[float, float]:
    """Return active-region minus-one and inactive-region zero slope deviations."""
    active_deviations: list[float] = []
    inactive_deviations: list[float] = []
    for _, product_rows in policy_rows.groupby("product_id", sort=False):
        ordered = product_rows.sort_values("state_index")
        inventory = ordered["inventory"].to_numpy(dtype=float)
        orders = ordered["deployed_order"].to_numpy(dtype=float)
        oracle_orders = ordered["oracle_order"].to_numpy(dtype=float)
        if len(inventory) < 2:
            continue
        slopes = np.diff(orders) / np.maximum(np.diff(inventory), 1e-12)
        active = oracle_orders[:-1] > 1e-6
        if np.any(active):
            active_deviations.extend(np.abs(slopes[active] + 1.0).tolist())
        inactive = ~active
        if np.any(inactive):
            inactive_deviations.extend(np.abs(slopes[inactive]).tolist())
    active_mean = float(np.mean(active_deviations)) if active_deviations else 0.0
    inactive_mean = float(np.mean(inactive_deviations)) if inactive_deviations else 0.0
    return active_mean, inactive_mean


def _operational_table(
    products: list[PaperLostSalesProduct],
    models: dict[str, nn.Module],
    feature_stats: PaperFCNNFeatureStats,
    settings: dict[str, Any],
    oracle_levels: dict[str, float],
    split: str,
    model_action_settings: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    rows = []
    seed_start = paper_evaluation_seed_start(split)
    for index, product in enumerate(products):
        policies = [
            non_omniscient_gamma_base_stock_policy(product),
            BaseStockOrderPolicy(
                S=float(oracle_levels[product.product_id]),
                name="OmniscientOptimal",
            ),
            historical_demand_base_stock_policy(product),
        ]
        for policy in policies:
            metrics = evaluate_paper_lost_sales_policy(
                product,
                policy,
                horizon=int(settings["evaluation_horizon"]),
                n_episodes=1,
                seed=seed_start + index,
                burn_in=int(settings["burn_in"]),
            )
            rows.append(
                {
                    "product_id": product.product_id,
                    "policy_name": policy.name,
                    "evaluation_split": split,
                    "evaluation_seed": seed_start + index,
                    **metrics,
                }
            )
    for name, model in models.items():
        action_settings = (model_action_settings or {}).get(name, settings)
        neural_rows = _batched_torch_policy_metrics(
            products,
            model,
            feature_stats,
            action_settings,
            split,
        )
        if not neural_rows.empty:
            neural_rows["policy_name"] = name
            rows.extend(neural_rows.to_dict("records"))
    return pd.DataFrame(rows)


def _select_penalty_candidate(
    candidates: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    compliant = [candidate for candidate in candidates if candidate["raw_violation_rate"] <= threshold]
    if compliant:
        return max(compliant, key=lambda candidate: candidate["validation_reward"])
    return min(
        candidates,
        key=lambda candidate: (candidate["raw_violation_rate"], -candidate["validation_reward"]),
    )


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _normalize_requested_models(models: list[str] | tuple[str, ...] | set[str] | None) -> set[str]:
    if models is None:
        return set(DEFAULT_TRAINABLE_POLICIES)
    requested = {model.strip() for model in models if model.strip()}
    unknown = requested - TRAINABLE_POLICIES
    if unknown:
        supported = ", ".join(sorted(TRAINABLE_POLICIES))
        raise ValueError(f"unknown trainable model(s): {', '.join(sorted(unknown))}; supported: {supported}")
    if not requested:
        raise ValueError("at least one trainable model must be selected")
    return requested


def run_idp_benchmark(
    config_path: str | Path | None = None,
    profile: str = "compact",
    mode: str = "structure-ls",
    split: str = "validation",
    final_test: bool = False,
    output_dir: str | Path | None = None,
    models: list[str] | tuple[str, ...] | set[str] | None = None,
    progress: bool = False,
    diagnostics: bool = True,
) -> IDPBenchmarkArtifacts:
    """Train, evaluate, and save the unified IDP benchmark."""
    require_final_test_confirmation(split, final_test)
    if mode not in {"replicate-ls", "structure-ls", "sample-efficiency"}:
        raise ValueError("mode must be 'replicate-ls', 'structure-ls', or 'sample-efficiency'")
    requested_models = _normalize_requested_models(models)
    progress_callback = print if progress else None
    config = load_paper_lost_sales_config(config_path)
    settings = _profile_settings(config, profile)
    model_config = _model_config(config, settings)
    destination = Path(output_dir or f"outputs/idp_benchmark/{mode}_{profile}_{split}")

    generator = PaperLostSalesProductGenerator(config)
    train_products = generator.generate(int(settings["training_products"]), split="train")
    evaluation_products = generator.generate(int(settings["validation_products"]), split=split)
    feature_stats = compute_paper_fcnn_feature_stats(train_products)

    oracle_levels = _oracle_levels(
        evaluation_products,
        int(settings["oracle_mc_samples"]),
        int(settings["seed"]) + 10_000,
    )
    trained_models: dict[str, nn.Module] = {}
    model_action_settings: dict[str, dict[str, Any]] = {}
    paper_fcnn_result: DirectBackpropResult | None = None
    paper_fcnn: nn.Module | None = None
    shared_initial_state: dict[str, torch.Tensor] | None = None
    if "PaperFCNN" in requested_models or "PenaltyMLP" in requested_models:
        torch.manual_seed(int(settings["seed"]))
        paper_fcnn = make_paper_fcnn_model(model_config)
        shared_initial_state = copy.deepcopy(paper_fcnn.state_dict())
    if "PaperFCNN" in requested_models:
        paper_fcnn_result = _train_model(
            config,
            settings,
            train_products,
            evaluation_products,
            feature_stats,
            paper_fcnn,
            "PaperFCNN",
            split,
            progress=progress_callback,
        )
        trained_models["PaperFCNN"] = paper_fcnn
    penalty_candidates: list[dict[str, Any]] = []
    penalty_logs: list[pd.DataFrame] = []
    selected_penalty: dict[str, Any] | None = None
    monokan_result: DirectBackpropResult | None = None
    monokan: nn.Module | None = None
    positive_weight_result: DirectBackpropResult | None = None
    positive_weight_mlp: nn.Module | None = None
    constrained_mlp_result: DirectBackpropResult | None = None
    constrained_mlp: nn.Module | None = None
    extra_results: dict[str, DirectBackpropResult] = {}
    extra_models: dict[str, nn.Module] = {}
    sample_efficiency = pd.DataFrame()
    if mode == "sample-efficiency":
        sample_efficiency = _run_sample_efficiency(
            config,
            settings,
            model_config,
            split,
            destination,
        )
    if mode == "structure-ls":
        if "PaperFCNNOrderUpTo" in requested_models:
            torch.manual_seed(int(settings["seed"]) + 201)
            model = make_paper_order_up_to_fcnn_model(model_config)
            result = _train_model(
                config,
                settings,
                train_products,
                evaluation_products,
                feature_stats,
                model,
                "PaperFCNNOrderUpTo",
                split,
                progress=progress_callback,
            )
            trained_models["PaperFCNNOrderUpTo"] = model
            extra_models["PaperFCNNOrderUpTo"] = model
            extra_results["PaperFCNNOrderUpTo"] = result
        if "PaperFCNNResidualOrderUpTo" in requested_models:
            torch.manual_seed(int(settings["seed"]) + 202)
            model = make_paper_residual_order_up_to_fcnn_model(model_config)
            fcnn_residual_settings = {
                **settings,
                "residual_anchor": str(model_config["fcnn"].get("residual_anchor", settings.get("residual_anchor", "gamma"))),
            }
            result = _train_model(
                config,
                fcnn_residual_settings,
                train_products,
                evaluation_products,
                feature_stats,
                model,
                "PaperFCNNResidualOrderUpTo",
                split,
                progress=progress_callback,
            )
            trained_models["PaperFCNNResidualOrderUpTo"] = model
            extra_models["PaperFCNNResidualOrderUpTo"] = model
            extra_results["PaperFCNNResidualOrderUpTo"] = result
        if "ConstrainedMLPResidualOrderUpTo" in requested_models:
            torch.manual_seed(int(settings["seed"]) + 203)
            model = make_paper_residual_order_up_to_constrained_mlp_model(model_config)
            constrained_residual_settings = {
                **settings,
                "residual_anchor": str(
                    model_config["constrained_monotone_mlp"].get(
                        "residual_anchor",
                        model_config["fcnn"].get("residual_anchor", settings.get("residual_anchor", "gamma")),
                    )
                ),
            }
            result = _train_model(
                config,
                constrained_residual_settings,
                train_products,
                evaluation_products,
                feature_stats,
                model,
                "ConstrainedMLPResidualOrderUpTo",
                split,
                progress=progress_callback,
            )
            trained_models["ConstrainedMLPResidualOrderUpTo"] = model
            model_action_settings["ConstrainedMLPResidualOrderUpTo"] = constrained_residual_settings
            extra_models["ConstrainedMLPResidualOrderUpTo"] = model
            extra_results["ConstrainedMLPResidualOrderUpTo"] = result
        if "PenaltyMLP" in requested_models:
            if shared_initial_state is None:
                raise RuntimeError("PenaltyMLP requires an initialized PaperFCNN state")
            for weight in settings["penalty_weights"]:
                penalty_model = make_paper_fcnn_model(model_config)
                penalty_model.load_state_dict(shared_initial_state)
                penalty_kind = str(settings.get("structural_penalty_kind", "target_stock"))
                if penalty_kind == "active_slope":
                    penalty_settings = {**settings, "active_slope_penalty_weight": float(weight)}
                elif penalty_kind == "target_stock":
                    penalty_settings = {**settings, "target_stock_penalty_weight": float(weight)}
                else:
                    raise ValueError(f"Unsupported structural_penalty_kind: {penalty_kind}")
                result = _train_model(
                    config,
                    penalty_settings,
                    train_products,
                    evaluation_products,
                    feature_stats,
                    penalty_model,
                    f"PenaltyMLP(lambda={float(weight):g})",
                    split,
                    penalty_weight=0.0,
                    progress=progress_callback,
                )
                candidate_predictions = _prediction_table(
                    evaluation_products,
                    {"PenaltyMLP": penalty_model},
                    feature_stats,
                    settings,
                    oracle_levels,
                )
                penalty_prediction_summary = _prediction_summary(candidate_predictions)
                penalty_diagnostics = penalty_prediction_summary.loc[
                    penalty_prediction_summary["policy_name"] == "PenaltyMLP"
                ].iloc[0]
                validation_reward = _mean_validation_reward(
                    config,
                    settings,
                    evaluation_products,
                    penalty_model,
                    "PenaltyMLP",
                    feature_stats,
                    split,
                )
                penalty_candidates.append(
                    {
                        "weight": float(weight),
                        "model": penalty_model,
                        "result": result,
                        "validation_reward": validation_reward,
                        "raw_violation_rate": float(penalty_diagnostics["raw_violation_rate"]),
                    }
                )
                candidate_log = pd.DataFrame(result.training_log or [])
                candidate_log["policy_name"] = f"PenaltyMLP(lambda={float(weight):g})"
                candidate_log["penalty_weight"] = float(weight)
                penalty_logs.append(candidate_log)
            selected_penalty = _select_penalty_candidate(
                penalty_candidates,
                float(settings["raw_violation_threshold"]),
            )
            trained_models["PenaltyMLP"] = selected_penalty["model"]
        if "PositiveWeightMLP" in requested_models:
            torch.manual_seed(int(settings["seed"]) + 101)
            positive_weight_mlp = make_positive_weight_mlp_model(model_config)
            positive_weight_result = _train_model(
                config,
                settings,
                train_products,
                evaluation_products,
                feature_stats,
                positive_weight_mlp,
                "PositiveWeightMLP",
                split,
                progress=progress_callback,
            )
            trained_models["PositiveWeightMLP"] = positive_weight_mlp
        if "ConstrainedMonoMLP" in requested_models:
            torch.manual_seed(int(settings["seed"]) + 102)
            constrained_mlp = make_constrained_monotone_mlp_model(model_config)
            constrained_mlp_result = _train_model(
                config,
                settings,
                train_products,
                evaluation_products,
                feature_stats,
                constrained_mlp,
                "ConstrainedMonoMLP",
                split,
                progress=progress_callback,
            )
            trained_models["ConstrainedMonoMLP"] = constrained_mlp
        if "MonoKAN" in requested_models:
            torch.manual_seed(int(settings["seed"]))
            monokan = make_paper_monokan_model(model_config)
            monokan_settings = {
                **settings,
                "max_order_factor": float(model_config["monokan"]["max_order_factor"]),
                "scale_action_by_history_mean": bool(model_config["monokan"]["scale_action_by_history_mean"]),
                "cap_excess_penalty_weight": float(model_config["monokan"].get("cap_excess_penalty_weight", 0.0)),
                "cap_excess_samples_per_product": int(model_config["monokan"].get("cap_excess_samples_per_product", 1)),
            }
            monokan_result = _train_model(
                config,
                monokan_settings,
                train_products,
                evaluation_products,
                feature_stats,
                monokan,
                "MonoKAN",
                split,
                clip_imitation_predictions=False,
                progress=progress_callback,
            )
            trained_models["MonoKAN"] = monokan
            model_action_settings["MonoKAN"] = monokan_settings
        if "MonoKANOrderUpTo" in requested_models:
            torch.manual_seed(int(settings["seed"]) + 301)
            model = make_paper_order_up_to_monokan_model(model_config)
            monokan_target_settings = {
                **settings,
                "max_order_factor": float(model_config["monokan"]["max_order_factor"]),
                "scale_action_by_history_mean": bool(model_config["monokan"]["scale_action_by_history_mean"]),
                "residual_anchor": str(model_config["monokan"].get("residual_anchor", settings.get("residual_anchor", "gamma"))),
            }
            result = _train_model(
                config,
                monokan_target_settings,
                train_products,
                evaluation_products,
                feature_stats,
                model,
                "MonoKANOrderUpTo",
                split,
                progress=progress_callback,
            )
            trained_models["MonoKANOrderUpTo"] = model
            model_action_settings["MonoKANOrderUpTo"] = monokan_target_settings
            extra_models["MonoKANOrderUpTo"] = model
            extra_results["MonoKANOrderUpTo"] = result
        if "MonoKANResidualOrderUpTo" in requested_models:
            torch.manual_seed(int(settings["seed"]) + 302)
            model = make_paper_residual_order_up_to_monokan_model(model_config)
            monokan_target_settings = {
                **settings,
                "max_order_factor": float(model_config["monokan"]["max_order_factor"]),
                "scale_action_by_history_mean": bool(model_config["monokan"]["scale_action_by_history_mean"]),
                "residual_anchor": str(model_config["monokan"].get("residual_anchor", settings.get("residual_anchor", "gamma"))),
            }
            result = _train_model(
                config,
                monokan_target_settings,
                train_products,
                evaluation_products,
                feature_stats,
                model,
                "MonoKANResidualOrderUpTo",
                split,
                progress=progress_callback,
            )
            trained_models["MonoKANResidualOrderUpTo"] = model
            model_action_settings["MonoKANResidualOrderUpTo"] = monokan_target_settings
            extra_models["MonoKANResidualOrderUpTo"] = model
            extra_results["MonoKANResidualOrderUpTo"] = result
        if "KANResidualOrderUpTo" in requested_models:
            torch.manual_seed(int(settings["seed"]) + 303)
            model = make_paper_residual_order_up_to_kan_model(model_config)
            kan_target_settings = {
                **settings,
                "max_order_factor": float(model_config["monokan"]["max_order_factor"]),
                "scale_action_by_history_mean": bool(model_config["monokan"]["scale_action_by_history_mean"]),
                "residual_anchor": str(model_config["monokan"].get("residual_anchor", settings.get("residual_anchor", "gamma"))),
            }
            result = _train_model(
                config,
                kan_target_settings,
                train_products,
                evaluation_products,
                feature_stats,
                model,
                "KANResidualOrderUpTo",
                split,
                progress=progress_callback,
            )
            trained_models["KANResidualOrderUpTo"] = model
            model_action_settings["KANResidualOrderUpTo"] = kan_target_settings
            extra_models["KANResidualOrderUpTo"] = model
            extra_results["KANResidualOrderUpTo"] = result

    operational = _operational_table(
        evaluation_products,
        trained_models,
        feature_stats,
        settings,
        oracle_levels,
        split,
        model_action_settings=model_action_settings,
    )
    operational_summary = (
        operational.groupby("policy_name", as_index=False)
        .agg(
            mean_total_reward=("mean_total_reward", "mean"),
            std_total_reward=("mean_total_reward", "std"),
            mean_reward_per_period=("mean_reward_per_period", "mean"),
            service_level=("service_level", "mean"),
            lost_sales_rate=("lost_sales_rate", "mean"),
            mean_order_quantity=("mean_order_quantity", "mean"),
            mean_inventory=("mean_inventory", "mean"),
        )
    )
    if diagnostics:
        predictions = _prediction_table(
            evaluation_products,
            trained_models,
            feature_stats,
            settings,
            oracle_levels,
            model_action_settings=model_action_settings,
        )
        prediction_summary = _prediction_summary(predictions)
        summary = operational_summary.merge(prediction_summary, on="policy_name", how="left")
    else:
        predictions = pd.DataFrame()
        prediction_summary = pd.DataFrame()
        summary = operational_summary
    training_rows: list[dict[str, float | str]] = []
    if paper_fcnn is not None and paper_fcnn_result is not None:
        training_rows.append(
            {
                "policy_name": "PaperFCNN",
                "parameter_count": _parameter_count(paper_fcnn),
                "training_seconds": paper_fcnn_result.elapsed_seconds,
                "selected_penalty_weight": np.nan,
            }
        )
    if selected_penalty is not None:
        training_rows.append(
            {
                "policy_name": "PenaltyMLP",
                "parameter_count": _parameter_count(selected_penalty["model"]),
                "training_seconds": selected_penalty["result"].elapsed_seconds,
                "selected_penalty_weight": selected_penalty["weight"],
            }
        )
    if positive_weight_mlp is not None and positive_weight_result is not None:
        training_rows.append(
            {
                "policy_name": "PositiveWeightMLP",
                "parameter_count": _parameter_count(positive_weight_mlp),
                "training_seconds": positive_weight_result.elapsed_seconds,
                "selected_penalty_weight": np.nan,
            }
        )
    if constrained_mlp is not None and constrained_mlp_result is not None:
        training_rows.append(
            {
                "policy_name": "ConstrainedMonoMLP",
                "parameter_count": _parameter_count(constrained_mlp),
                "training_seconds": constrained_mlp_result.elapsed_seconds,
                "selected_penalty_weight": np.nan,
            }
        )
    if monokan is not None and monokan_result is not None:
        training_rows.append(
            {
                "policy_name": "MonoKAN",
                "parameter_count": _parameter_count(monokan),
                "training_seconds": monokan_result.elapsed_seconds,
                "selected_penalty_weight": np.nan,
            }
        )
    for name, model in extra_models.items():
        result = extra_results[name]
        training_rows.append(
            {
                "policy_name": name,
                "parameter_count": _parameter_count(model),
                "training_seconds": result.elapsed_seconds,
                "selected_penalty_weight": np.nan,
            }
        )
    training_info = pd.DataFrame(training_rows)
    summary = summary.merge(training_info, on="policy_name", how="left")
    order_map = {name: index for index, name in enumerate(POLICY_ORDER)}
    summary["_order"] = summary["policy_name"].map(order_map)
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    logs = []
    for name, result in [("PaperFCNN", paper_fcnn_result)]:
        if result is None:
            continue
        frame = pd.DataFrame(result.training_log or [])
        frame["policy_name"] = name
        frame["penalty_weight"] = 0.0
        logs.append(frame)
    if monokan_result is not None:
        frame = pd.DataFrame(monokan_result.training_log or [])
        frame["policy_name"] = "MonoKAN"
        frame["penalty_weight"] = 0.0
        logs.append(frame)
    if positive_weight_result is not None:
        frame = pd.DataFrame(positive_weight_result.training_log or [])
        frame["policy_name"] = "PositiveWeightMLP"
        frame["penalty_weight"] = 0.0
        logs.append(frame)
    if constrained_mlp_result is not None:
        frame = pd.DataFrame(constrained_mlp_result.training_log or [])
        frame["policy_name"] = "ConstrainedMonoMLP"
        frame["penalty_weight"] = 0.0
        logs.append(frame)
    for name, result in extra_results.items():
        frame = pd.DataFrame(result.training_log or [])
        frame["policy_name"] = name
        frame["penalty_weight"] = 0.0
        logs.append(frame)
    logs.extend(penalty_logs)
    training_log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    if training_log.empty:
        training_log["selected"] = pd.Series(dtype=bool)
    elif selected_penalty is None:
        training_log["selected"] = True
    else:
        training_log["selected"] = ~training_log["policy_name"].str.startswith("PenaltyMLP") | (
            training_log["penalty_weight"] == float(selected_penalty["weight"])
        )

    spline_rows = []
    if diagnostics:
        for name, model in trained_models.items():
            if not isinstance(model, MonoKANPolicy) or not model.monotonic_vars:
                continue
            for feature_index in sorted(model.monotonic_vars):
                spline_x, spline_values_tensor = sample_monokan_first_layer_splines(
                    model,
                    feature_index=feature_index,
                    samples=100,
                )
                for unit_index, values in enumerate(spline_values_tensor.numpy()):
                    for x_value, spline_value in zip(spline_x.numpy(), values):
                        spline_rows.append(
                            {
                                "policy_name": name,
                                "feature_index": int(feature_index),
                                "hidden_unit": unit_index,
                                "feature_value": float(x_value),
                                "edge_value": float(spline_value),
                            }
                        )
    spline_values = pd.DataFrame(spline_rows)
    diagnostic_products = evaluation_products[: int(settings.get("order_up_to_products", len(evaluation_products)))]
    target_structure = (
        _target_structure_diagnostics(diagnostic_products, trained_models, feature_stats, settings)
        if diagnostics
        else pd.DataFrame()
    )
    spline_certification = _monokan_spline_certification(trained_models) if diagnostics else pd.DataFrame()

    destination.mkdir(parents=True, exist_ok=True)
    model_dir = destination / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    if paper_fcnn is not None:
        torch.save(paper_fcnn.state_dict(), model_dir / "paper_fcnn.pt")
    if selected_penalty is not None:
        torch.save(selected_penalty["model"].state_dict(), model_dir / "penalty_mlp.pt")
    if positive_weight_mlp is not None:
        torch.save(positive_weight_mlp.state_dict(), model_dir / "positive_weight_mlp.pt")
    if constrained_mlp is not None:
        torch.save(constrained_mlp.state_dict(), model_dir / "constrained_monotone_mlp.pt")
    if monokan is not None:
        torch.save(monokan.state_dict(), model_dir / "monokan.pt")
    model_filenames = checkpoint_filenames()
    for name, model in extra_models.items():
        torch.save(model.state_dict(), model_dir / model_filenames[name])
    summary.to_csv(destination / "summary.csv", index=False)
    table3 = _paper_table3(summary)
    table3.to_csv(destination / "paper_table3_replication.csv", index=False)
    operational.to_csv(destination / "operational_metrics.csv", index=False)
    predictions.to_csv(destination / "predictions.csv", index=False)
    training_log.to_csv(destination / "training_log.csv", index=False)
    spline_values.to_csv(destination / "monokan_splines.csv", index=False)
    target_structure.to_csv(destination / "target_structure_diagnostics.csv", index=False)
    spline_certification.to_csv(destination / "monokan_spline_certification.csv", index=False)
    if not sample_efficiency.empty:
        sample_efficiency.to_csv(destination / "sample_efficiency.csv", index=False)
    if diagnostics:
        _save_policy_structure_plots(destination, predictions, mode)
    metadata = {
        "mode": mode,
        "profile": profile,
        "split": split,
        "requested_models": sorted(requested_models),
        "diagnostics": diagnostics,
        "settings": settings,
        "feature_stats": {
            "demand_scale": feature_stats.demand_scale,
            "price_scale": feature_stats.price_scale,
        },
        "selected_penalty_weight": selected_penalty["weight"] if selected_penalty is not None else None,
        "penalty_candidates": [
            {
                "weight": candidate["weight"],
                "validation_reward": candidate["validation_reward"],
                "raw_violation_rate": candidate["raw_violation_rate"],
            }
            for candidate in penalty_candidates
        ],
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return IDPBenchmarkArtifacts(
        output_dir=destination,
        summary=summary,
        operational_metrics=operational,
        predictions=predictions,
        training_log=training_log,
        spline_values=spline_values,
        target_structure=target_structure,
        spline_certification=spline_certification,
        sample_efficiency=sample_efficiency if not sample_efficiency.empty else None,
    )


def _paper_table3(summary: pd.DataFrame) -> pd.DataFrame:
    """Create a paper-style reward table with gaps to omniscient optimal."""
    reward_column = "mean_reward_per_period"
    omniscient = summary.loc[summary["policy_name"] == "OmniscientOptimal", reward_column]
    reference = float(omniscient.iloc[0]) if not omniscient.empty else float(summary[reward_column].max())
    table = summary[["policy_name", reward_column]].copy()
    table = table.rename(columns={reward_column: "average_reward"})
    table["gap_to_omniscient_pct"] = 100.0 * (table["average_reward"] / reference - 1.0)
    return table


def _save_policy_structure_plots(destination: Path, predictions: pd.DataFrame, mode: str) -> None:
    """Save paper-aligned structural diagnostic plots."""
    plots_dir = destination / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if predictions.empty:
        return
    product_ids = predictions["product_id"].drop_duplicates().head(3).tolist()
    subset = predictions[predictions["product_id"].isin(product_ids)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for (policy_name, product_id), rows in subset.groupby(["policy_name", "product_id"], sort=False):
        if policy_name not in {
            "PaperFCNN",
            "PositiveWeightMLP",
            "ConstrainedMonoMLP",
            "MonoKAN",
            "OmniscientOptimal",
            "NonOmniscientOptimal",
        }:
            continue
        label = f"{policy_name} {product_id}"
        rows.sort_values("inventory").plot(x="inventory", y="deployed_order", ax=ax, label=label, alpha=0.8)
    ax.set_title("Purchase quantity vs inventory")
    ax.set_xlabel("Inventory level")
    ax.set_ylabel("Purchase quantity")
    ax.grid(alpha=0.3)
    fig.savefig(plots_dir / "figure2_policy_curves.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    zero_inventory = predictions.loc[predictions["state_index"] == 0].copy()
    if not zero_inventory.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        for policy_name, rows in zero_inventory.groupby("policy_name", sort=False):
            if policy_name not in {"PaperFCNN", "PositiveWeightMLP", "ConstrainedMonoMLP", "MonoKAN", "NonOmniscientOptimal"}:
                continue
            ax.scatter(
                rows["omniscient_order_up_to_level"],
                rows["order_up_to_level"],
                c=rows["true_coefficient_of_variation"],
                cmap="viridis",
                label=policy_name,
                alpha=0.7,
            )
        max_level = float(
            max(zero_inventory["omniscient_order_up_to_level"].max(), zero_inventory["order_up_to_level"].max())
        )
        ax.plot([0, max_level], [0, max_level], color="black", linewidth=1)
        ax.set_title("Learned vs omniscient order-up-to level")
        ax.set_xlabel("Omniscient order-up-to level")
        ax.set_ylabel("Learned order-up-to level")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.savefig(plots_dir / "figure3_order_up_to_scatter.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    if mode == "structure-ls":
        diagnostics = _prediction_summary(predictions)
        fig, ax = plt.subplots(figsize=(7, 4))
        diagnostics.set_index("policy_name")["raw_violation_rate"].plot.bar(ax=ax)
        ax.set_title("Raw monotonicity violation rate")
        ax.set_ylabel("Fraction of adjacent inventory pairs")
        ax.grid(axis="y", alpha=0.3)
        fig.savefig(plots_dir / "monotonicity_violations.png", bbox_inches="tight", dpi=150)
        plt.close(fig)


def _run_sample_efficiency(
    config: dict[str, Any],
    settings: dict[str, Any],
    model_config: dict[str, Any],
    split: str,
    destination: Path,
) -> pd.DataFrame:
    """Run a compact paper-style N/H sample-efficiency grid."""
    benchmark = config.get("idp_benchmark", {})
    sample_config = benchmark.get("sample_efficiency", {})
    rows = []
    for history_window in sample_config.get("history_windows", [config["history_length_H"]]):
        grid_config = copy.deepcopy(config)
        grid_config["history_length_H"] = int(history_window)
        generator = PaperLostSalesProductGenerator(grid_config)
        for n_products in sample_config.get("training_products_grid", [settings["training_products"]]):
            local_settings = dict(settings)
            local_settings["training_products"] = min(int(n_products), int(settings["training_products"]))
            local_settings["validation_products"] = int(settings["validation_products"])
            train_products = generator.generate(int(local_settings["training_products"]), split="train")
            eval_products = generator.generate(int(local_settings["validation_products"]), split=split)
            feature_stats = compute_paper_fcnn_feature_stats(train_products)
            local_model_config = copy.deepcopy(model_config)
            local_model_config["history_length_H"] = int(history_window)
            torch.manual_seed(int(settings["seed"]))
            paper_fcnn = make_paper_fcnn_model(local_model_config)
            result = _train_model(
                grid_config,
                local_settings,
                train_products,
                eval_products,
                feature_stats,
                paper_fcnn,
                "PaperFCNN",
                split,
            )
            reward = _mean_validation_reward(
                grid_config,
                local_settings,
                eval_products,
                paper_fcnn,
                "PaperFCNN",
                feature_stats,
                split,
            )
            rows.append(
                {
                    "history_window": int(history_window),
                    "training_products": int(local_settings["training_products"]),
                    "policy_name": "PaperFCNN",
                    "mean_total_reward": reward,
                    "training_seconds": result.elapsed_seconds,
                }
            )
    destination.mkdir(parents=True, exist_ok=True)
    return pd.DataFrame(rows)
