"""Plot controlled 2D policy surfaces for saved residual order-up-to models."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from inventory_control.idp_benchmark import _model_config, _profile_settings, load_paper_lost_sales_config
from inventory_control.paper_lost_sales import non_omniscient_gamma_base_stock_policy
from inventory_control.policies.neural.fcnn import compute_paper_fcnn_feature_stats
from inventory_control.policies.neural.registry import (
    RESIDUAL_COMPARISON_POLICY_NAMES,
    checkpoint_filenames,
    policy_factories,
)
from inventory_control.products import PaperLostSalesProduct, PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import paper_lost_sales_policy_order, paper_lost_sales_target_level


def _batch_for_product(product: PaperLostSalesProduct, feature_stats, residual_anchor: str) -> dict[str, torch.Tensor]:
    history = torch.tensor(product.historical_demand, dtype=torch.float32)
    historical_mean = max(float(history.mean().item()), 1.0)
    return {
        "history": (history / float(feature_stats.demand_scale)).unsqueeze(0),
        "feature_demand_scale": torch.tensor([float(feature_stats.demand_scale)], dtype=torch.float32),
        "feature_price_scale": torch.tensor([float(feature_stats.price_scale)], dtype=torch.float32),
        "historical_mean_demand": torch.tensor([historical_mean], dtype=torch.float32),
        "price": torch.tensor([product.price], dtype=torch.float32),
        "purchase_cost": torch.tensor([product.purchase_cost], dtype=torch.float32),
        "holding_cost": torch.tensor([product.holding_cost], dtype=torch.float32),
        "lost_sale_penalty": torch.tensor([product.lost_sale_penalty], dtype=torch.float32),
        "historical_base_stock_level": torch.tensor(
            [non_omniscient_gamma_base_stock_policy(product).S if residual_anchor == "gamma" else historical_mean],
            dtype=torch.float32,
        ),
    }


def _scaled_history_product(product: PaperLostSalesProduct, multiplier: float) -> PaperLostSalesProduct:
    scaled_history = [max(int(round(value * multiplier)), 0) for value in product.historical_demand]
    return dataclasses.replace(product, historical_demand=scaled_history)


def _parse_policy_names(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one policy name is required")
    return names


def _load_models(benchmark_dir: Path, model_config: dict, policy_names: list[str]) -> dict[str, torch.nn.Module]:
    models = {}
    factories = policy_factories(policy_names)
    filenames = checkpoint_filenames()
    for policy_name in policy_names:
        model_path = benchmark_dir / "models" / filenames[policy_name]
        if not model_path.exists():
            raise FileNotFoundError(f"missing checkpoint for {policy_name}: {model_path}")
        factory = factories[policy_name]
        model = factory(model_config)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        models[policy_name] = model
    if not models:
        raise ValueError(f"No supported residual model checkpoints found under {benchmark_dir / 'models'}")
    return models


def _surface(
    model: torch.nn.Module,
    product: PaperLostSalesProduct,
    feature_stats,
    residual_anchor: str,
    inventory_grid: np.ndarray,
    multiplier_grid: np.ndarray,
    max_order_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    deployed = np.zeros((len(multiplier_grid), len(inventory_grid)), dtype=float)
    target = np.zeros_like(deployed)
    with torch.no_grad():
        for row, multiplier in enumerate(multiplier_grid):
            scaled_product = _scaled_history_product(product, float(multiplier))
            batch = _batch_for_product(scaled_product, feature_stats, residual_anchor)
            target[row, :] = float(paper_lost_sales_target_level(model, batch).item())
            for col, inventory in enumerate(inventory_grid):
                inventory_tensor = torch.tensor([float(inventory)], dtype=torch.float32)
                deployed[row, col] = float(
                    paper_lost_sales_policy_order(
                        model,
                        batch,
                        inventory_tensor,
                        max_order_factor=max_order_factor,
                        scale_action_by_history_mean=True,
                    ).item()
                )
    return deployed, target


def _plot_surface(
    data: dict[str, np.ndarray],
    inventory_grid: np.ndarray,
    multiplier_grid: np.ndarray,
    title: str,
    colorbar_label: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(data), figsize=(6 * len(data), 5), squeeze=False)
    for ax, (policy_name, values) in zip(axes[0], data.items()):
        levels = 16
        contour = ax.contourf(inventory_grid, multiplier_grid, values, levels=levels, cmap="viridis")
        lines = ax.contour(inventory_grid, multiplier_grid, values, levels=levels, colors="black", linewidths=0.45, alpha=0.7)
        ax.clabel(lines, inline=True, fontsize=7, fmt="%.0f")
        fig.colorbar(contour, ax=ax, label=colorbar_label)
        ax.set_title(policy_name)
        ax.set_xlabel("inventory")
        ax.set_ylabel("historical-demand multiplier")
        ax.grid(alpha=0.2)
    fig.suptitle(title)
    fig.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/paper_lost_sales_fcnn_alignment_medium.yaml"))
    parser.add_argument("--profile", default="compact", choices=["compact", "full", "paper_replication"])
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--product-index", type=int, default=0)
    parser.add_argument("--inventory-points", type=int, default=81)
    parser.add_argument("--multiplier-points", type=int, default=61)
    parser.add_argument("--min-multiplier", type=float, default=0.5)
    parser.add_argument("--max-multiplier", type=float, default=1.5)
    parser.add_argument(
        "--models",
        default=",".join(RESIDUAL_COMPARISON_POLICY_NAMES),
        help="Comma-separated residual order-up-to policies with saved checkpoints.",
    )
    args = parser.parse_args()

    config = load_paper_lost_sales_config(args.config)
    settings = _profile_settings(config, args.profile)
    metadata_path = args.benchmark_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        settings = {**settings, **metadata.get("settings", {})}
    model_config = _model_config(config, settings)

    generator = PaperLostSalesProductGenerator(config)
    train_products = generator.generate(int(settings["training_products"]), split="train")
    eval_products = generator.generate(max(int(args.product_index) + 1, 1), split=args.split)
    product = eval_products[int(args.product_index)]
    feature_stats = compute_paper_fcnn_feature_stats(train_products)
    policy_names = _parse_policy_names(args.models)
    models = _load_models(args.benchmark_dir, model_config, policy_names)

    historical_mean = max(float(np.mean(product.historical_demand)), 1.0)
    inventory_grid = np.linspace(0.0, float(settings["max_order_factor"]) * historical_mean, int(args.inventory_points))
    multiplier_grid = np.linspace(float(args.min_multiplier), float(args.max_multiplier), int(args.multiplier_points))

    deployed_surfaces = {}
    target_surfaces = {}
    for policy_name, model in models.items():
        deployed, target = _surface(
            model,
            product,
            feature_stats,
            str(settings.get("residual_anchor", "gamma")),
            inventory_grid,
            multiplier_grid,
            float(settings["max_order_factor"]),
        )
        deployed_surfaces[policy_name] = deployed
        target_surfaces[policy_name] = target

    plots_dir = args.benchmark_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metadata_path_out = plots_dir / "controlled_policy_surface_metadata.json"
    metadata_path_out.write_text(
        json.dumps(
            {
                "benchmark_dir": str(args.benchmark_dir),
                "config": str(args.config),
                "profile": args.profile,
                "split": args.split,
                "product_index": int(args.product_index),
                "product_id": product.product_id,
                "models": list(models),
                "inventory_points": int(args.inventory_points),
                "multiplier_points": int(args.multiplier_points),
                "min_multiplier": float(args.min_multiplier),
                "max_multiplier": float(args.max_multiplier),
                "output_dir": str(plots_dir),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    deployed_path = plots_dir / "controlled_inventory_history_multiplier_deployed_order.png"
    target_path = plots_dir / "controlled_inventory_history_multiplier_target_stock.png"
    _plot_surface(
        deployed_surfaces,
        inventory_grid,
        multiplier_grid,
        f"Controlled deployed-order surfaces for {product.product_id}",
        "deployed order",
        deployed_path,
    )
    _plot_surface(
        target_surfaces,
        inventory_grid,
        multiplier_grid,
        f"Controlled target-stock surfaces for {product.product_id}",
        "target stock",
        target_path,
    )
    print(f"saved_deployed_surface={deployed_path}")
    print(f"saved_target_surface={target_path}")
    print(f"saved_metadata={metadata_path_out}")


if __name__ == "__main__":
    main()
