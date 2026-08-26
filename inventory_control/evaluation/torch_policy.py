"""Batched torch-policy evaluation for final paper lost-sales experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from inventory_control.paper_evaluation import paper_evaluation_seed_start
from inventory_control.paper_lost_sales import historical_demand_base_stock_policy, paper_evaluation_demand_path
from inventory_control.policies.neural.fcnn import PaperFCNNFeatureStats, infer_model_device
from inventory_control.products import PaperLostSalesProduct
from inventory_control.training.direct_backprop import (
    paper_lost_sales_raw_order,
    paper_lost_sales_target_level,
    paper_rolling_target_levels,
)


def _historical_anchor_level(product: PaperLostSalesProduct, residual_anchor: str) -> float:
    if residual_anchor == "gamma":
        return float(historical_demand_base_stock_policy(product, fit_gamma=True).S)
    if residual_anchor == "empirical":
        return float(historical_demand_base_stock_policy(product, fit_gamma=False).S)
    raise ValueError(f"Unsupported residual_anchor: {residual_anchor}")


def batched_torch_policy_metrics(
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
        [[value / feature_stats.demand_scale for value in product.historical_demand] for product in products],
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
        [
            _historical_anchor_level(
                product,
                str(getattr(model, "paper_residual_anchor", settings.get("residual_anchor", "gamma"))),
            )
            for product in products
        ],
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
            [paper_evaluation_demand_path(product, horizon=horizon, seed=seed_start + index) for index, product in enumerate(products)],
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
        roll_history = bool(getattr(model, "paper_roll_history", False))
        rollout_batch = dict(batch) if roll_history else batch
        uses_target_level = parameterization in {"order_up_to", "residual_order_up_to"}
        rolling_target_levels = paper_rolling_target_levels(model, batch, demand_paths) if uses_target_level and roll_history else None
        target_level = paper_lost_sales_target_level(model, rollout_batch) if uses_target_level and not roll_history else None
        for t in range(horizon):
            if uses_target_level:
                current_target_level = rolling_target_levels[:, t] if rolling_target_levels is not None else target_level
                assert current_target_level is not None
                raw_order = torch.relu(current_target_level - inventory)
            else:
                raw_order = paper_lost_sales_raw_order(
                    model,
                    rollout_batch,
                    inventory,
                    scale_action_by_history_mean=bool(settings["scale_action_by_history_mean"]),
                )
            order = torch.minimum(raw_order, torch.relu(max_inventory_level - inventory)) if bool(settings.get("enforce_action_cap", True)) else raw_order
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
            if roll_history and not uses_target_level:
                observed = demand / torch.clamp(batch["feature_demand_scale"], min=1.0)
                rollout_batch["history"] = torch.cat([rollout_batch["history"][:, 1:], observed.unsqueeze(1)], dim=1)

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
    rows = []
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
