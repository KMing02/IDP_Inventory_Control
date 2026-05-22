"""Plotting utilities for baseline inventory experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inventory_control.policies.base import Policy


def _save_or_return(fig: plt.Figure, save_path: str | Path | None) -> plt.Figure:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)
    return fig


def plot_cost_per_period(trace: pd.DataFrame, save_path: str | Path | None = None) -> plt.Figure:
    """Plot total cost per period for one episode."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(trace["period"], trace["total_cost"], marker="o")
    ax.set_xlabel("Period")
    ax.set_ylabel("Cost")
    ax.set_title("Cost per Period")
    ax.grid(True, alpha=0.3)
    return _save_or_return(fig, save_path)


def plot_inventory_trajectory(trace: pd.DataFrame, save_path: str | Path | None = None) -> plt.Figure:
    """Plot starting inventory by period for one episode."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(trace["period"], trace["state"], where="post", label="start inventory")
    ax.step(trace["period"], trace["ending_inventory"], where="post", label="ending inventory")
    ax.set_xlabel("Period")
    ax.set_ylabel("Inventory")
    ax.set_title("Inventory Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save_or_return(fig, save_path)


def plot_order_quantity_trajectory(trace: pd.DataFrame, save_path: str | Path | None = None) -> plt.Figure:
    """Plot order quantity per period for one episode."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(trace["period"], trace["order_quantity"])
    ax.set_xlabel("Period")
    ax.set_ylabel("Order Quantity")
    ax.set_title("Order Quantity Trajectory")
    ax.grid(True, axis="y", alpha=0.3)
    return _save_or_return(fig, save_path)


def plot_policy_curve(
    policy: Policy,
    max_inventory: int,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot order quantity as a function of inventory level."""
    inventories = np.arange(max_inventory + 1)
    rng = np.random.default_rng(0)
    orders = [policy.act(int(inventory), t=0, rng=rng) for inventory in inventories]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(inventories, orders, marker="o")
    ax.set_xlabel("Inventory Level")
    ax.set_ylabel("Order Quantity")
    ax.set_title(f"Policy Curve: {policy.name}")
    ax.grid(True, alpha=0.3)
    return _save_or_return(fig, save_path)
