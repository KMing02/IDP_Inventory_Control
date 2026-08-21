"""Checkpoint-only structural audit for direct target-stock policies."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from inventory_control.base_stock_output_experiment import _compact_settings, _make_model
from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.policies.neural.fcnn import compute_paper_fcnn_feature_stats, empirical_newsvendor_target_ratio
from inventory_control.policies.neural.monokan import MonoKANPolicy, _evaluate_hermite_curves
from inventory_control.policies.neural.registry import DIRECT_TARGET_STOCK_POLICY_NAMES, checkpoint_filenames
from inventory_control.products import PaperLostSalesProduct, PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import (
    make_paper_lost_sales_training_batch,
    paper_lost_sales_context_features,
    paper_lost_sales_target_level,
)


FEATURE_SPECS = (
    ("demand_level", 1),
    ("selling_price", 1),
    ("purchase_cost", -1),
    ("holding_cost", -1),
    ("lost_sales_penalty", 1),
)


@dataclass(frozen=True)
class DirectTargetAuditArtifacts:
    """Tables produced from persisted direct-target checkpoints."""

    output_dir: Path
    target_monotonicity: pd.DataFrame
    spline_certification: pd.DataFrame


def run_direct_target_structural_audit(
    config_path: str | Path,
    checkpoint_root: str | Path,
    output_dir: str | Path,
    models: Sequence[str] = DIRECT_TARGET_STOCK_POLICY_NAMES,
    products_per_domain: int = 100,
) -> DirectTargetAuditArtifacts:
    """Audit complete targets and constrained splines without retraining."""
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"audit output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    config = load_paper_lost_sales_config(config_path)
    settings = _compact_settings(config)
    generator = PaperLostSalesProductGenerator(config)
    train_products = generator.generate(settings["training_products"], split="train")
    validation_products = generator.generate(products_per_domain, split="validation")
    feature_stats = compute_paper_fcnn_feature_stats(train_products)
    initial_target_ratio = empirical_newsvendor_target_ratio(train_products, feature_stats)
    filenames = checkpoint_filenames()

    target_frames: list[pd.DataFrame] = []
    spline_frames: list[pd.DataFrame] = []
    for seed_dir in sorted(Path(checkpoint_root).glob("seed_*")):
        seed = int(seed_dir.name.split("_", 1)[1])
        for model_name in models:
            checkpoint_path = seed_dir / filenames[model_name]
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
            model = _make_model(model_name, config, train_products, feature_stats, initial_target_ratio)
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
            model.eval()
            target_frames.append(
                _target_monotonicity_rows(
                    model,
                    model_name,
                    seed,
                    validation_products,
                    feature_stats,
                    settings,
                )
            )
            if isinstance(model, MonoKANPolicy) and model.monotonic_vars:
                spline_frames.append(_spline_rows(model, model_name, seed))

    target = pd.concat(target_frames, ignore_index=True) if target_frames else pd.DataFrame()
    spline = pd.concat(spline_frames, ignore_index=True) if spline_frames else pd.DataFrame()
    target.to_csv(destination / "complete_target_monotonicity.csv", index=False)
    spline.to_csv(destination / "monokan_spline_certification.csv", index=False)
    metadata = {
        "config_path": str(config_path),
        "checkpoint_root": str(checkpoint_root),
        "models": list(models),
        "products_per_domain": int(products_per_domain),
        "finite_difference_complete_target": True,
        "spline_certification_scope": "first_layer_edges_and_linear_extrapolation",
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(destination, target, spline)
    return DirectTargetAuditArtifacts(destination, target, spline)


def _target_monotonicity_rows(
    model: torch.nn.Module,
    model_name: str,
    seed: int,
    products: list[PaperLostSalesProduct],
    feature_stats,
    settings: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature, direction in FEATURE_SPECS:
        for domain, domain_products in (
            ("in_domain", products),
            ("stress_extrapolation", [_scale_product(product, feature, 3.0) for product in products]),
        ):
            perturbed = [_scale_product(product, feature, 1.05) for product in domain_products]
            base_target = _target_levels(domain_products, model, feature_stats, settings, seed)
            perturbed_target = _target_levels(perturbed, model, feature_stats, settings, seed)
            directional_difference = float(direction) * (perturbed_target - base_target)
            wrong = torch.clamp(-directional_difference, min=0.0)
            derivatives = _directional_derivatives(
                domain_products,
                model,
                feature_stats,
                settings,
                seed,
                feature,
                direction,
            )
            rows.append(
                {
                    "run_seed": seed,
                    "policy_name": model_name,
                    "feature": feature,
                    "expected_direction": direction,
                    "domain": domain,
                    "finite_diff_relative_step": 0.05,
                    "finite_diff_violation_rate": float(torch.mean((wrong > 1e-6).float()).item()),
                    "finite_diff_max_wrong_direction": float(wrong.max().item()),
                    "finite_diff_mean_wrong_direction": float(wrong.mean().item()),
                    "autograd_wrong_sign_rate": float(np.mean(derivatives < -1e-6)),
                    "autograd_min_directional_derivative": float(np.min(derivatives)),
                    "autograd_max_directional_derivative": float(np.max(derivatives)),
                    "autograd_max_wrong_direction_derivative": float(max(0.0, -np.min(derivatives))),
                    "evaluated_states": len(domain_products),
                    "complete_target": True,
                }
            )
    return pd.DataFrame(rows)


def _target_levels(products, model, feature_stats, settings, seed: int) -> torch.Tensor:
    batch = make_paper_lost_sales_training_batch(
        products,
        horizon=1,
        generator=torch.Generator().manual_seed(seed + 77_000),
        feature_stats=feature_stats,
        history_feature_mode=str(getattr(model, "paper_history_feature_mode", "raw_lags")),
        include_newsvendor_feature=bool(getattr(model, "paper_include_newsvendor_feature", False)),
        include_demand_level_feature=bool(getattr(model, "paper_include_demand_level_feature", False)),
    )
    with torch.no_grad():
        return paper_lost_sales_target_level(model, batch).detach().cpu()


def _directional_derivatives(products, model, feature_stats, settings, seed, feature, direction) -> np.ndarray:
    batch = make_paper_lost_sales_training_batch(
        products,
        horizon=1,
        generator=torch.Generator().manual_seed(seed + 78_000),
        feature_stats=feature_stats,
        history_feature_mode=str(getattr(model, "paper_history_feature_mode", "raw_lags")),
        include_newsvendor_feature=bool(getattr(model, "paper_include_newsvendor_feature", False)),
        include_demand_level_feature=bool(getattr(model, "paper_include_demand_level_feature", False)),
    )
    context = paper_lost_sales_context_features(batch).detach().clone().requires_grad_(True)
    history_length = int(batch["history"].shape[1])
    factor = model(context)
    reconstructed_mean = torch.clamp(
        context[:, :history_length].mean(dim=1) * float(feature_stats.demand_scale),
        min=1.0,
    )
    target = factor * reconstructed_mean
    gradient = torch.autograd.grad(target.sum(), context)[0]
    has_demand_level = bool(getattr(model, "paper_include_demand_level_feature", False))
    if feature == "demand_level":
        selected = gradient[:, :history_length].sum(dim=1)
        if has_demand_level:
            selected = selected + gradient[:, history_length]
    else:
        economics_start = history_length + int(has_demand_level)
        offset = {"selling_price": 0, "purchase_cost": 1, "holding_cost": 2, "lost_sales_penalty": 3}[feature]
        selected = gradient[:, economics_start + offset]
    return (float(direction) * selected.detach()).cpu().numpy()


def _scale_product(product: PaperLostSalesProduct, feature: str, factor: float) -> PaperLostSalesProduct:
    if feature == "demand_level":
        return dataclasses.replace(
            product,
            historical_demand=[max(int(round(value * factor)), 0) for value in product.historical_demand],
        )
    field = {
        "selling_price": "price",
        "purchase_cost": "purchase_cost",
        "holding_cost": "holding_cost",
        "lost_sales_penalty": "lost_sale_penalty",
    }[feature]
    return dataclasses.replace(product, **{field: float(getattr(product, field)) * factor})


def _spline_rows(model: MonoKANPolicy, model_name: str, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    layer = model.layers[0]
    for feature_index, direction in sorted(model.monotonic_vars.items()):
        edges = torch.arange(layer.out_dim) * layer.in_dim + int(feature_index)
        grid = layer.grid[edges]
        domain_min = float(grid[:, 0].min().item())
        domain_max = float(grid[:, -1].max().item())
        inside = torch.linspace(domain_min, domain_max, 200).repeat(layer.out_dim, 1)
        left = torch.linspace(domain_min - 1.0, domain_min, 50).repeat(layer.out_dim, 1)
        right = torch.linspace(domain_max, domain_max + 1.0, 50).repeat(layer.out_dim, 1)
        for domain, points in (("spline_grid", inside), ("left_extrapolation", left), ("right_extrapolation", right)):
            values = _evaluate_hermite_curves(points, grid, layer.coef[edges])
            wrong = torch.clamp(-float(direction) * torch.diff(values, dim=1), min=0.0)
            rows.append(
                {
                    "run_seed": seed,
                    "policy_name": model_name,
                    "feature_index": int(feature_index),
                    "expected_direction": int(direction),
                    "domain": domain,
                    "edge_count": int(layer.out_dim),
                    "interval_count": int(wrong.numel()),
                    "violating_interval_fraction": float(torch.mean((wrong > 1e-6).float()).item()),
                    "max_wrong_direction_increment": float(wrong.max().item()),
                    "first_layer_only": True,
                    "complete_model_certified_from_this_alone": False,
                }
            )
    return pd.DataFrame(rows)


def _write_report(destination: Path, target: pd.DataFrame, spline: pd.DataFrame) -> None:
    grouped = target.groupby(["policy_name", "domain"], as_index=False).agg(
        mean_violation_rate=("finite_diff_violation_rate", "mean"),
        max_wrong_direction=("finite_diff_max_wrong_direction", "max"),
        mean_autograd_wrong_sign_rate=("autograd_wrong_sign_rate", "mean"),
    )
    report = "# Direct Target-Stock Structural Audit\n\n"
    report += "Finite differences evaluate the complete target stock. Spline rows audit constrained first-layer edges only.\n\n"
    report += "## Complete-Target Summary\n\n```text\n" + grouped.to_string(index=False) + "\n```\n"
    if not spline.empty:
        report += "\n## Spline Summary\n\n"
        report += f"Rows: {len(spline)}; maximum wrong increment: {spline['max_wrong_direction_increment'].max():.8g}.\n"
    (destination / "STRUCTURAL_AUDIT.md").write_text(report, encoding="utf-8")
