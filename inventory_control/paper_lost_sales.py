"""Maggiar-style no-lead-time lost-sales utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.stats import gamma as scipy_gamma

from inventory_control.products import PaperLostSalesProduct


class ScalarOrderPolicy(Protocol):
    """Protocol for policies returning a scalar order quantity."""

    name: str

    def act(self, inventory: float, t: int, rng=None) -> float:
        """Return order quantity."""


def paper_evaluation_demand_path(
    product: PaperLostSalesProduct,
    horizon: int,
    seed: int,
) -> np.ndarray:
    """Return the reproducible demand path used for one evaluation episode."""
    rng = np.random.default_rng(seed)
    return _paper_evaluation_demand_path(product, horizon, rng)


def _paper_evaluation_demand_path(
    product: PaperLostSalesProduct,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if product.future_demand is not None and len(product.future_demand) >= horizon:
        return np.asarray(product.future_demand[:horizon], dtype=float)
    return np.asarray(product.sample_demand(rng, size=horizon), dtype=float)


def paper_lost_sales_step(
    inventory: float,
    order_quantity: float,
    demand: float,
    price: float,
    purchase_cost: float,
    holding_cost: float,
    lost_sale_penalty: float,
) -> tuple[float, float, dict[str, float]]:
    """Apply paper-style lost-sales reward for one period."""
    order_quantity = max(float(order_quantity), 0.0)
    available = inventory + order_quantity
    sales = min(demand, available)
    lost_sales = max(demand - available, 0.0)
    ending_inventory = max(available - demand, 0.0)
    revenue = price * sales
    purchase = purchase_cost * order_quantity
    lost_penalty = lost_sale_penalty * lost_sales
    holding = holding_cost * ending_inventory
    reward = revenue - purchase - lost_penalty - holding
    info = {
        "demand": float(demand),
        "sales": float(sales),
        "lost_sales": float(lost_sales),
        "ending_inventory": float(ending_inventory),
        "order_quantity": float(order_quantity),
        "revenue": float(revenue),
        "purchase_cost": float(purchase),
        "lost_sales_penalty": float(lost_penalty),
        "holding_cost": float(holding),
        "reward": float(reward),
    }
    return ending_inventory, float(reward), info


def evaluate_paper_lost_sales_policy(
    product: PaperLostSalesProduct,
    policy: ScalarOrderPolicy,
    horizon: int,
    n_episodes: int,
    seed: int,
    burn_in: int = 0,
) -> dict[str, float]:
    """Evaluate a scalar-order policy over ``horizon`` total periods.

    Periods before ``burn_in`` are simulated but excluded from reported rewards
    and aggregate statistics, matching the paper's 520-period evaluation with
    the first 20 periods discarded.
    """
    rng = np.random.default_rng(seed)
    total_rewards: list[float] = []
    total_sales = 0.0
    total_demand = 0.0
    total_lost_sales = 0.0
    total_order_quantity = 0.0
    min_order_quantity = float("inf")
    max_order_quantity = 0.0
    total_ending_inventory = 0.0
    total_revenue = 0.0
    total_purchase_cost = 0.0
    total_lost_sales_penalty = 0.0
    total_holding_cost = 0.0
    counted_periods = 0
    for _ in range(n_episodes):
        inventory = float(product.initial_inventory)
        episode_reward = 0.0
        demands = _paper_evaluation_demand_path(product, horizon, rng)
        for t, demand in enumerate(demands):
            order = policy.act(inventory, t, rng=rng)
            inventory, reward, info = paper_lost_sales_step(
                inventory,
                order,
                float(demand),
                product.price,
                product.purchase_cost,
                product.holding_cost,
                product.lost_sale_penalty,
            )
            if t >= burn_in:
                episode_reward += reward
                total_sales += info["sales"]
                total_demand += info["demand"]
                total_lost_sales += info["lost_sales"]
                total_order_quantity += info["order_quantity"]
                min_order_quantity = min(min_order_quantity, info["order_quantity"])
                max_order_quantity = max(max_order_quantity, info["order_quantity"])
                total_ending_inventory += info["ending_inventory"]
                total_revenue += info["revenue"]
                total_purchase_cost += info["purchase_cost"]
                total_lost_sales_penalty += info["lost_sales_penalty"]
                total_holding_cost += info["holding_cost"]
                counted_periods += 1
        total_rewards.append(episode_reward)
    denominator = max(counted_periods, 1)
    return {
        "mean_total_reward": float(np.mean(total_rewards)),
        "mean_reward_per_period": float(np.mean(total_rewards) / max(horizon - burn_in, 1)),
        "service_level": float(total_sales / total_demand) if total_demand > 0 else 1.0,
        "lost_sales_rate": float(total_lost_sales / total_demand) if total_demand > 0 else 0.0,
        "mean_order_quantity": float(total_order_quantity / denominator),
        "min_order_quantity": float(0.0 if min_order_quantity == float("inf") else min_order_quantity),
        "max_order_quantity": float(max_order_quantity),
        "mean_inventory": float(total_ending_inventory / denominator),
        "mean_holding_cost": float(total_holding_cost / denominator),
        "mean_purchase_cost": float(total_purchase_cost / denominator),
        "mean_sales_revenue": float(total_revenue / denominator),
        "mean_lost_sales_penalty": float(total_lost_sales_penalty / denominator),
    }


@dataclass
class BaseStockOrderPolicy:
    """Order-up-to policy for paper-style no-lead-time lost sales."""

    S: float
    name: str

    def act(self, inventory: float, t: int, rng=None) -> float:
        """Order enough to raise inventory position to ``S``."""
        return max(float(self.S) - float(inventory), 0.0)


def paper_critical_quantile(product: PaperLostSalesProduct) -> float:
    """Return critical fractile for paper-style reward economics."""
    numerator = product.price + product.lost_sale_penalty - product.purchase_cost
    denominator = numerator + product.holding_cost
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def gamma_quantile_monte_carlo(
    product: PaperLostSalesProduct,
    quantile: float,
    n_samples: int,
    seed: int,
) -> float:
    """Approximate Gamma demand quantile using hidden true parameters."""
    rng = np.random.default_rng(seed)
    samples = product.sample_demand(rng, size=n_samples)
    return float(np.quantile(samples, quantile))


def gamma_quantile_exact(shape: float, scale: float, quantile: float) -> float:
    """Return the exact Gamma quantile used by the paper baselines."""
    if shape <= 0 or scale <= 0:
        return 0.0
    return float(scipy_gamma.ppf(float(np.clip(quantile, 0.0, 1.0)), a=float(shape), scale=float(scale)))


def historical_empirical_quantile(product: PaperLostSalesProduct, quantile: float) -> float:
    """Estimate demand quantile from historical demand only."""
    return float(np.quantile(product.historical_demand, quantile))


def historical_gamma_quantile(product: PaperLostSalesProduct, quantile: float) -> float:
    """Estimate a Gamma quantile from historical mean/CV, without true parameters."""
    history = np.asarray(product.historical_demand, dtype=float)
    mean = float(history.mean())
    std = float(history.std(ddof=0))
    if mean <= 0:
        return 0.0
    cv = max(std / mean, 1e-6)
    shape = 1.0 / (cv**2)
    scale = mean / shape
    return gamma_quantile_exact(shape, scale, quantile)


def omniscient_gamma_base_stock_policy(product: PaperLostSalesProduct) -> BaseStockOrderPolicy:
    """Create the paper's omniscient optimal LS policy from true Gamma parameters."""
    quantile = paper_critical_quantile(product)
    stock_level = gamma_quantile_exact(product.gamma_shape, product.gamma_scale, quantile)
    return BaseStockOrderPolicy(S=stock_level, name="OmniscientOptimal")


def non_omniscient_gamma_base_stock_policy(product: PaperLostSalesProduct) -> BaseStockOrderPolicy:
    """Create the paper's non-omniscient Gamma fit base-stock benchmark."""
    quantile = paper_critical_quantile(product)
    return BaseStockOrderPolicy(S=historical_gamma_quantile(product, quantile), name="NonOmniscientOptimal")


def oracle_gamma_base_stock_policy(
    product: PaperLostSalesProduct,
    n_samples: int,
    seed: int,
) -> BaseStockOrderPolicy:
    """Backward-compatible alias for the exact omniscient Gamma policy."""
    del n_samples, seed
    policy = omniscient_gamma_base_stock_policy(product)
    return BaseStockOrderPolicy(S=policy.S, name="OracleGammaBaseStock")


def historical_demand_base_stock_policy(
    product: PaperLostSalesProduct,
    fit_gamma: bool = False,
) -> BaseStockOrderPolicy:
    """Create non-omniscient base-stock policy from historical demand only."""
    quantile = paper_critical_quantile(product)
    stock_level = (
        historical_gamma_quantile(product, quantile)
        if fit_gamma
        else historical_empirical_quantile(product, quantile)
    )
    name = "NonOmniscientOptimal" if fit_gamma else "HistoricalDemandBaseStock"
    return BaseStockOrderPolicy(S=stock_level, name=name)


def monotonicity_violation_rate(
    policy,
    feature_builder,
    inventory_low: np.ndarray,
    inventory_high: np.ndarray,
) -> float:
    """Report how often order increases when inventory increases."""
    low_orders = np.asarray([policy(feature_builder(value)) for value in inventory_low], dtype=float)
    high_orders = np.asarray([policy(feature_builder(value)) for value in inventory_high], dtype=float)
    return float(np.mean(high_orders > low_orders + 1e-8))
