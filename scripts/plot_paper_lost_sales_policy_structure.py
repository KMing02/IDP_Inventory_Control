"""Plot paper lost-sales FCNN policy structure and monotonicity diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs/.cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import torch

from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.paper_lost_sales import monotonicity_violation_rate
from inventory_control.policies.neural.fcnn import (
    PaperFCNNFeatureStats,
    build_paper_lost_sales_features,
    compute_paper_fcnn_feature_stats,
    make_paper_fcnn_model,
)
from inventory_control.products import PaperLostSalesProductGenerator


def main() -> None:
    """Plot order quantity against inventory for one representative product."""
    config = load_paper_lost_sales_config()
    model_path = Path("outputs/models/paper_lost_sales_fcnn.pt")
    metadata_path = Path("outputs/models/paper_lost_sales_fcnn_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    model = make_paper_fcnn_model(config, metadata=metadata)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    generator = PaperLostSalesProductGenerator(config)
    n_training_products = min(int(config["training_products"]), int(config["runtime"]["fcnn_train_products"]))
    if "feature_demand_scale" in metadata and "feature_price_scale" in metadata:
        feature_stats = PaperFCNNFeatureStats(
            demand_scale=float(metadata["feature_demand_scale"]),
            price_scale=float(metadata["feature_price_scale"]),
        )
    else:
        feature_stats = compute_paper_fcnn_feature_stats(generator.generate(n_training_products, split="train"))
    product = generator.generate(1, split="test")[0]
    max_order_factor = float(metadata.get("max_order_factor", config["fcnn"].get("max_order_factor", 2.0)))
    scale_action_by_history_mean = bool(
        metadata.get("scale_action_by_history_mean", config["fcnn"].get("scale_action_by_history_mean", False))
    )

    def feature_builder(inventory: float):
        return build_paper_lost_sales_features(
            inventory,
            product.historical_demand,
            product.price,
            product.purchase_cost,
            product.holding_cost,
            product.lost_sale_penalty,
            feature_stats=feature_stats,
        )

    def policy(features):
        with torch.no_grad():
            tensor = torch.tensor([features], dtype=torch.float32)
            order = float(model(tensor).item())
        inventory = features[0] * feature_stats.demand_scale
        historical_mean = max(sum(product.historical_demand) / len(product.historical_demand), 1.0)
        if scale_action_by_history_mean:
            order *= historical_mean
        max_inventory_level = max_order_factor * historical_mean
        return min(order, max(max_inventory_level - inventory, 0.0))

    inventories = np.linspace(0, max(product.historical_demand) * 3, 100)
    orders = [policy(feature_builder(value)) for value in inventories]
    low = inventories[:-1]
    high = inventories[1:]
    violation_rate = monotonicity_violation_rate(policy, feature_builder, low, high)

    output_path = Path("outputs/plots/paper_lost_sales_policy_structure.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(inventories, orders)
    ax.set_xlabel("Inventory level")
    ax.set_ylabel("Order quantity")
    ax.set_title(f"Paper LS FCNN Policy Structure; violation rate={violation_rate:.3f}")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"monotonicity_violation_rate={violation_rate:.6f}")
    print(f"saved_plot={output_path}")


if __name__ == "__main__":
    main()
