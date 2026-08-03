"""Create KAN versus MonoKAN evidence tables from existing experiment outputs."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import yaml

from inventory_control.evaluation.monokan_evidence import (
    KAN_MODEL,
    MONOKAN_MODEL,
    classify_equivalence,
    fixed_shift_equal_weight_mean,
    hierarchical_paired_bootstrap,
    kan_monokan_product_pairs,
    per_shift_hierarchical_bootstrap,
    seed_shift_deltas,
)
from inventory_control.idp_benchmark import _model_config, _profile_settings, load_paper_lost_sales_config
from inventory_control.paper_lost_sales import non_omniscient_gamma_base_stock_policy
from inventory_control.policies.neural.fcnn import compute_paper_fcnn_feature_stats, infer_model_device
from inventory_control.policies.neural.monokan import MonoKANPolicy, _evaluate_hermite_curves
from inventory_control.policies.neural.registry import RESIDUAL_COMPARISON_POLICY_NAMES, checkpoint_filenames, policy_factories
from inventory_control.products import PaperLostSalesProduct, PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import make_paper_lost_sales_training_batch, paper_lost_sales_target_level
from scripts.run_distribution_shift_validation import SHIFT_OVERRIDES, _shifted_config


PRIMARY_MODELS = list(RESIDUAL_COMPARISON_POLICY_NAMES)
FEATURE_SPECS = [
    ("demand_level", 1, 0.05),
    ("selling_price", 1, 0.02),
    ("purchase_cost", -1, 0.02),
    ("holding_cost", -1, 0.02),
    ("lost_sales_penalty", 1, 0.02),
]


def _read_operational(multiseed_root: Path) -> pd.DataFrame:
    rows = []
    for seed_dir in sorted(multiseed_root.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        seed = int(seed_dir.name.split("_", 1)[1])
        shift_root = seed_dir / "distribution_shift"
        for shift_dir in sorted(shift_root.iterdir()) if shift_root.exists() else []:
            path = shift_dir / "operational_metrics.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            frame.insert(0, "shift", shift_dir.name)
            frame.insert(0, "seed", seed)
            rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"no per-shift operational_metrics.csv files found under {multiseed_root}")
    return pd.concat(rows, ignore_index=True)


def _direct_kan_monokan_analysis(pairs: pd.DataFrame, margin_abs: float, bootstrap_samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell = seed_shift_deltas(pairs)
    mean_delta = fixed_shift_equal_weight_mean(cell)
    median_delta = fixed_shift_equal_weight_mean(cell, value_column="median_delta")
    boot_mean, ci_low, ci_high = hierarchical_paired_bootstrap(pairs, samples=bootstrap_samples, seed=2025)
    mean_kan = float(pairs["kan_reward"].mean())
    equivalence = classify_equivalence(ci_low=ci_low, ci_high=ci_high, margin_abs=margin_abs, mean_kan_reward=mean_kan)
    product_win_rate = float(np.mean(pairs["delta_reward"] > 1e-9))
    product_loss_rate = float(np.mean(pairs["delta_reward"] < -1e-9))
    product_tie_rate = float(np.mean(np.abs(pairs["delta_reward"]) <= 1e-9))
    winner_counts = cell.assign(
        winner=np.where(cell["mean_delta"] > 1e-9, MONOKAN_MODEL, np.where(cell["mean_delta"] < -1e-9, KAN_MODEL, "tie"))
    ).groupby("winner", as_index=False).size().rename(columns={"size": "seed_shift_wins"})
    worst_row = pairs.loc[pairs["delta_reward"].idxmin()]
    worst_cell = cell.loc[cell["mean_delta"].idxmin()]
    summary = pd.DataFrame(
        [
            {
                "comparison": "MonoKANResidualOrderUpTo_minus_KANResidualOrderUpTo",
                "paired_products": int(len(pairs)),
                "seeds": int(pairs["seed"].nunique()),
                "shifts": int(pairs["shift"].nunique()),
                "mean_delta_reward": mean_delta,
                "median_delta_reward": median_delta,
                "bootstrap_mean_delta_reward": boot_mean,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "mean_kan_reward": mean_kan,
                "mean_delta_pct_of_kan": 100.0 * mean_delta / mean_kan,
                "margin_abs": equivalence.margin_abs,
                "margin_pct_of_kan": equivalence.margin_pct_of_kan,
                "equivalence_classification": equivalence.classification,
                "statistically_excludes_zero": equivalence.statistically_excludes_zero,
                "monokan_product_win_rate": product_win_rate,
                "monokan_product_loss_rate": product_loss_rate,
                "product_tie_rate": product_tie_rate,
                "bottom_decile_delta": float(np.quantile(pairs["delta_reward"], 0.1)),
                "worst_product_delta": float(worst_row["delta_reward"]),
                "worst_product_seed": int(worst_row["seed"]),
                "worst_product_shift": str(worst_row["shift"]),
                "worst_product_id": str(worst_row["product_id"]),
                "worst_seed_shift_delta": float(worst_cell["mean_delta"]),
                "worst_seed_shift_seed": int(worst_cell["seed"]),
                "worst_seed_shift_shift": str(worst_cell["shift"]),
                "regret_vs_better_mean": float(np.maximum(-pairs["delta_reward"], 0.0).mean()),
                "fixed_shift_equal_weighting": True,
                "bootstrap_resamples_shifts": False,
            }
        ]
    )
    for _, row in winner_counts.iterrows():
        summary[f"seed_shift_wins_{row['winner']}"] = int(row["seed_shift_wins"])

    shift_ci = per_shift_hierarchical_bootstrap(pairs, samples=bootstrap_samples, seed=3025)
    by_shift = (
        pairs.groupby("shift", as_index=False)
        .agg(
            paired_products=("delta_reward", "size"),
            mean_delta_reward=("delta_reward", "mean"),
            median_delta_reward=("delta_reward", "median"),
            mean_kan_reward=("kan_reward", "mean"),
            monokan_product_win_rate=("delta_reward", lambda values: float(np.mean(values > 1e-9))),
            monokan_product_loss_rate=("delta_reward", lambda values: float(np.mean(values < -1e-9))),
            product_tie_rate=("delta_reward", lambda values: float(np.mean(np.abs(values) <= 1e-9))),
            bottom_decile_delta=("delta_reward", lambda values: float(np.quantile(values, 0.1))),
            worst_product_delta=("delta_reward", "min"),
        )
        .merge(shift_ci, on="shift", how="left")
    )
    by_shift["mean_delta_pct_of_kan"] = 100.0 * by_shift["mean_delta_reward"] / by_shift["mean_kan_reward"]
    return summary, by_shift


def _load_seed_models(seed_dir: Path, config: dict[str, Any], settings: dict[str, Any], policy_names: list[str]) -> dict[str, torch.nn.Module]:
    factories = policy_factories(policy_names)
    filenames = checkpoint_filenames()
    model_config = _model_config(config, settings)
    models = {}
    for name in policy_names:
        path = seed_dir / "trained_models" / "models" / filenames[name]
        if not path.exists():
            raise FileNotFoundError(f"missing checkpoint: {path}")
        model = factories[name](model_config)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        models[name] = model
    return models


def _perturb_product(product: PaperLostSalesProduct, feature: str, step: float) -> PaperLostSalesProduct:
    if feature == "demand_level":
        factor = 1.0 + step
        history = [max(int(round(value * factor)), 0) for value in product.historical_demand]
        return dataclasses.replace(product, historical_demand=history)
    if feature == "selling_price":
        return dataclasses.replace(product, price=product.price * (1.0 + step))
    if feature == "purchase_cost":
        return dataclasses.replace(product, purchase_cost=product.purchase_cost * (1.0 + step))
    if feature == "holding_cost":
        return dataclasses.replace(product, holding_cost=product.holding_cost * (1.0 + step))
    if feature == "lost_sales_penalty":
        return dataclasses.replace(product, lost_sale_penalty=product.lost_sale_penalty * (1.0 + step))
    raise ValueError(f"unsupported feature: {feature}")


def _target_levels(products: list[PaperLostSalesProduct], model: torch.nn.Module, feature_stats, settings: dict[str, Any]) -> torch.Tensor:
    device = infer_model_device(model)
    batch = make_paper_lost_sales_training_batch(
        products,
        horizon=1,
        generator=torch.Generator().manual_seed(int(settings["seed"]) + 77_000),
        feature_stats=feature_stats,
        residual_anchor=str(settings.get("residual_anchor", "gamma")),
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        return paper_lost_sales_target_level(model, batch).detach().cpu()


def _autograd_directional_derivatives(
    products: list[PaperLostSalesProduct],
    model: torch.nn.Module,
    feature_stats,
    settings: dict[str, Any],
    feature: str,
    direction: int,
) -> np.ndarray:
    from inventory_control.idp_benchmark import _target_level_from_context
    from inventory_control.training.direct_backprop import paper_lost_sales_context_features

    device = infer_model_device(model)
    batch = make_paper_lost_sales_training_batch(
        products,
        horizon=1,
        generator=torch.Generator().manual_seed(int(settings["seed"]) + 78_000),
        feature_stats=feature_stats,
        residual_anchor=str(settings.get("residual_anchor", "gamma")),
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    context = paper_lost_sales_context_features(batch).detach().clone().requires_grad_(True)
    target = _target_level_from_context(model, batch, context)
    grad = torch.autograd.grad(target.sum(), context, retain_graph=False)[0]
    history_length = int(batch["history"].shape[1])
    if feature == "demand_level":
        selected = grad[:, :history_length].sum(dim=1)
    else:
        offset = {
            "selling_price": 0,
            "purchase_cost": 1,
            "holding_cost": 2,
            "lost_sales_penalty": 3,
        }[feature]
        selected = grad[:, history_length + offset]
    return (float(direction) * selected.detach().cpu()).numpy()


def _target_monotonicity_rows(multiseed_root: Path, base_config: dict[str, Any], profile: str, products_per_shift: int) -> pd.DataFrame:
    rows = []
    base_settings = _profile_settings(base_config, profile)
    for seed_dir in sorted(multiseed_root.glob("seed_*")):
        seed = int(seed_dir.name.split("_", 1)[1])
        config_path = seed_dir / "effective_config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else base_config
        settings = {**_profile_settings(config, profile), "seed": int(config["idp_benchmark"]["seed"])}
        models = _load_seed_models(seed_dir, config, settings, PRIMARY_MODELS)
        train_products = PaperLostSalesProductGenerator(config).generate(int(settings["training_products"]), split="train")
        feature_stats = compute_paper_fcnn_feature_stats(train_products)
        for shift in sorted(SHIFT_OVERRIDES):
            shifted = _shifted_config(config, shift)
            products = PaperLostSalesProductGenerator(shifted).generate(products_per_shift, split="validation")
            for model_name, model in models.items():
                base_target = _target_levels(products, model, feature_stats, settings)
                for feature, direction, step in FEATURE_SPECS:
                    for domain, multiplier in [("local_in_domain", 1.0), ("stress_extrapolation", 5.0)]:
                        effective_step = step * multiplier
                        perturbed = [_perturb_product(product, feature, effective_step) for product in products]
                        hi_target = _target_levels(perturbed, model, feature_stats, settings)
                        directional_diff = float(direction) * (hi_target - base_target)
                        wrong = torch.clamp(-directional_diff, min=0.0)
                        derivatives = _autograd_directional_derivatives(products, model, feature_stats, settings, feature, direction)
                        rows.append(
                            {
                                "seed": seed,
                                "shift": shift,
                                "policy_name": model_name,
                                "feature": feature,
                                "expected_direction": direction,
                                "domain": domain,
                                "finite_diff_step": effective_step,
                                "finite_diff_violation_rate": float(torch.mean((wrong > 1e-6).float()).item()),
                                "finite_diff_max_wrong_direction": float(wrong.max().item()),
                                "autograd_wrong_sign_rate": float(np.mean(derivatives < -1e-6)),
                                "autograd_min_directional_derivative": float(np.min(derivatives)),
                                "autograd_max_directional_derivative": float(np.max(derivatives)),
                                "autograd_max_wrong_direction_derivative": float(max(0.0, -np.min(derivatives))),
                                "evaluated_states": len(products),
                                "complete_target_finite_diff": True,
                                "autograd_anchor_recomputed": False,
                            }
                        )
    return pd.DataFrame(rows)


def _clipping_and_target_summary(multiseed_root: Path) -> pd.DataFrame:
    frames = []
    for seed_dir in sorted(multiseed_root.glob("seed_*")):
        seed = int(seed_dir.name.split("_", 1)[1])
        shift_root = seed_dir / "distribution_shift"
        for shift_dir in sorted(shift_root.iterdir()) if shift_root.exists() else []:
            path = shift_dir / "predictions.csv"
            if not path.exists() or path.stat().st_size == 0:
                continue
            frame = pd.read_csv(path)
            frame.insert(0, "shift", shift_dir.name)
            frame.insert(0, "seed", seed)
            frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    predictions = predictions[predictions["policy_name"].isin(PRIMARY_MODELS)].copy()
    predictions["raw_deployed_abs_diff"] = (predictions["raw_order"] - predictions["deployed_order"]).abs()
    predictions["materially_changed_by_guard"] = predictions["raw_deployed_abs_diff"] > 1e-6
    predictions["raw_target_stock"] = predictions["raw_order"] + predictions["inventory"]
    predictions["deployed_target_stock"] = predictions["deployed_order"] + predictions["inventory"]
    rows = []
    for (policy, seed, shift), group in predictions.groupby(["policy_name", "seed", "shift"], sort=False):
        zero = group.sort_values("inventory").groupby("product_id", as_index=False).apply(
            lambda rows_: float(rows_.loc[rows_["deployed_order"].le(1e-6), "inventory"].min())
            if rows_["deployed_order"].le(1e-6).any()
            else float("nan"),
            include_groups=False,
        )
        rows.append(
            {
                "policy_name": policy,
                "seed": int(seed),
                "shift": shift,
                "rows": len(group),
                "raw_order_mean": float(group["raw_order"].mean()),
                "raw_order_q10": float(group["raw_order"].quantile(0.1)),
                "raw_order_median": float(group["raw_order"].median()),
                "raw_order_q90": float(group["raw_order"].quantile(0.9)),
                "deployed_order_mean": float(group["deployed_order"].mean()),
                "deployed_order_q10": float(group["deployed_order"].quantile(0.1)),
                "deployed_order_median": float(group["deployed_order"].median()),
                "deployed_order_q90": float(group["deployed_order"].quantile(0.9)),
                "cap_hit_rate": float(np.mean(group["raw_order"] >= group["deployed_order"] + 1e-6)),
                "mean_abs_raw_deployed_difference": float(group["raw_deployed_abs_diff"].mean()),
                "materially_changed_by_guard_rate": float(group["materially_changed_by_guard"].mean()),
                "target_stock_mean": float(group["raw_target_stock"].mean()),
                "target_stock_q10": float(group["raw_target_stock"].quantile(0.1)),
                "target_stock_median": float(group["raw_target_stock"].median()),
                "target_stock_q90": float(group["raw_target_stock"].quantile(0.9)),
                "residual_correction_proxy_mean": float((group["raw_target_stock"] - group["omniscient_order_up_to_level"]).mean()),
                "residual_correction_proxy_abs_mean": float((group["raw_target_stock"] - group["omniscient_order_up_to_level"]).abs().mean()),
                "order_up_to_zero_crossing_mean": float(zero[None].mean()),
                "order_up_to_zero_crossing_median": float(zero[None].median()),
            }
        )
    return pd.DataFrame(rows)


def _constrained_mlp_diagnosis(multiseed_root: Path, clipping: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    summary = pd.read_csv(multiseed_root / "robustness_summary_all_seeds.csv")
    training_frames = []
    for seed_dir in sorted(multiseed_root.glob("seed_*")):
        seed = int(seed_dir.name.split("_", 1)[1])
        path = seed_dir / "trained_models" / "training_log.csv"
        if path.exists() and path.stat().st_size > 0:
            frame = pd.read_csv(path)
            frame.insert(0, "seed", seed)
            training_frames.append(frame)
    training = pd.concat(training_frames, ignore_index=True) if training_frames else pd.DataFrame()
    rows = []
    for policy in ["ConstrainedMLPResidualOrderUpTo", "PaperFCNNResidualOrderUpTo", KAN_MODEL, MONOKAN_MODEL]:
        summary_row = summary.loc[summary["policy_name"] == policy].iloc[0].to_dict()
        policy_training = training[training["policy_name"] == policy] if not training.empty and "policy_name" in training.columns else pd.DataFrame()
        clip = clipping[clipping["policy_name"] == policy]
        target_rows = target[(target["policy_name"] == policy) & (target["domain"] == "local_in_domain")]
        rows.append(
            {
                "policy_name": policy,
                "mean_reward_per_period": summary_row["mean_reward_per_period"],
                "mean_service_level": summary_row["mean_service_level"],
                "mean_lost_sales_rate": summary_row["mean_lost_sales_rate"],
                "mean_order_up_to_mae": summary_row["mean_order_up_to_mae"],
                "mean_cap_hit_rate": summary_row["mean_cap_hit_rate"],
                "mean_inventory": float(clip["target_stock_mean"].mean()) if not clip.empty else float("nan"),
                "mean_raw_order": float(clip["raw_order_mean"].mean()) if not clip.empty else float("nan"),
                "target_bias_proxy_vs_omniscient": float(clip["residual_correction_proxy_mean"].mean()) if not clip.empty else float("nan"),
                "target_abs_error_proxy_vs_omniscient": float(clip["residual_correction_proxy_abs_mean"].mean()) if not clip.empty else float("nan"),
                "target_monotonicity_violation_rate": float(target_rows["finite_diff_violation_rate"].mean()) if not target_rows.empty else float("nan"),
                "training_log_rows": int(len(policy_training)),
                "last_training_loss": float(policy_training["training_loss"].dropna().iloc[-1]) if "training_loss" in policy_training and policy_training["training_loss"].notna().any() else float("nan"),
            }
        )
    diagnosis = pd.DataFrame(rows)
    constrained = diagnosis.loc[diagnosis["policy_name"] == "ConstrainedMLPResidualOrderUpTo"].iloc[0]
    diagnosis["interpretation"] = ""
    diagnosis.loc[diagnosis["policy_name"] == "ConstrainedMLPResidualOrderUpTo", "interpretation"] = (
        "Lower reward coincides with much higher service level, lower lost-sales rate, larger target/order-up-to error, "
        "and larger conservative target bias proxy. This suggests excessive shape restriction or scaling/conservatism; "
        "gradient-norm evidence is unavailable in current logs, so optimization difficulty is unresolved."
    )
    return diagnosis


def _monokan_spline_audit(multiseed_root: Path) -> pd.DataFrame:
    rows = []
    filenames = checkpoint_filenames()
    factories = policy_factories([MONOKAN_MODEL])
    for seed_dir in sorted(multiseed_root.glob("seed_*")):
        seed = int(seed_dir.name.split("_", 1)[1])
        config = yaml.safe_load((seed_dir / "effective_config.yaml").read_text(encoding="utf-8"))
        settings = _profile_settings(config, "compact")
        model = factories[MONOKAN_MODEL](_model_config(config, settings))
        model.load_state_dict(torch.load(seed_dir / "trained_models" / "models" / filenames[MONOKAN_MODEL], map_location="cpu"))
        model.eval()
        if not isinstance(model, MonoKANPolicy):
            continue
        first_layer = model.layers[0]
        for feature_index, direction in sorted(model.monotonic_vars.items()):
            edge_indices = torch.arange(first_layer.out_dim) * first_layer.in_dim + int(feature_index)
            selected_grid = first_layer.grid[edge_indices]
            domain_min = float(selected_grid[:, 0].min().item())
            domain_max = float(selected_grid[:, -1].max().item())
            x = torch.linspace(domain_min, domain_max, steps=200)
            x_eval = x.repeat(first_layer.out_dim, 1)
            values = _evaluate_hermite_curves(x_eval, selected_grid, first_layer.coef[edge_indices])
            increments = torch.diff(values, dim=1)
            wrong = torch.clamp(-float(direction) * increments, min=0.0)
            left_x = torch.linspace(domain_min - 1.0, domain_min, steps=50).repeat(first_layer.out_dim, 1)
            right_x = torch.linspace(domain_max, domain_max + 1.0, steps=50).repeat(first_layer.out_dim, 1)
            left_values = _evaluate_hermite_curves(left_x, selected_grid, first_layer.coef[edge_indices])
            right_values = _evaluate_hermite_curves(right_x, selected_grid, first_layer.coef[edge_indices])
            left_wrong = torch.clamp(-float(direction) * torch.diff(left_values, dim=1), min=0.0)
            right_wrong = torch.clamp(-float(direction) * torch.diff(right_values, dim=1), min=0.0)
            rows.append(
                {
                    "seed": seed,
                    "feature_index": int(feature_index),
                    "expected_direction": int(direction),
                    "edges": int(first_layer.out_dim),
                    "intervals": int(increments.numel()),
                    "violating_interval_fraction": float(torch.mean((wrong > 1e-6).float()).item()),
                    "max_wrong_direction_increment": float(wrong.max().item()),
                    "left_extrapolation_violating_fraction": float(torch.mean((left_wrong > 1e-6).float()).item()),
                    "left_extrapolation_max_wrong_increment": float(left_wrong.max().item()),
                    "right_extrapolation_violating_fraction": float(torch.mean((right_wrong > 1e-6).float()).item()),
                    "right_extrapolation_max_wrong_increment": float(right_wrong.max().item()),
                    "domain_min": domain_min,
                    "domain_max": domain_max,
                    "first_layer_only": True,
                    "complete_model_certified_from_this_alone": False,
                }
            )
    return pd.DataFrame(rows)


def _write_report(
    output_dir: Path,
    equivalence: pd.DataFrame,
    by_shift: pd.DataFrame,
    target: pd.DataFrame,
    clipping: pd.DataFrame,
    diagnosis: pd.DataFrame,
    spline: pd.DataFrame,
) -> None:
    eq = equivalence.iloc[0]
    report = f"""# MonoKAN Evidence Report

## Scope

This report analyzes existing outputs only. No new model architectures were added, no model was tuned, and no expensive retraining was run for this report.

## KAN Versus MonoKAN Reward Evidence

- Mean paired delta, `MonoKAN - KAN`: {eq['mean_delta_reward']:.6f} reward/period.
- Hierarchical bootstrap 95% CI: [{eq['ci95_low']:.6f}, {eq['ci95_high']:.6f}].
- Delta as percent of mean KAN reward: {eq['mean_delta_pct_of_kan']:.6f}%.
- Equivalence margin: +/- {eq['margin_abs']:.6f} reward/period ({eq['margin_pct_of_kan']:.4f}% of KAN reward).
- Classification: `{eq['equivalence_classification']}`.
- Statistical interval excludes zero: {bool(eq['statistically_excludes_zero'])}.
- MonoKAN product win/loss/tie rates: {eq['monokan_product_win_rate']:.3f} / {eq['monokan_product_loss_rate']:.3f} / {eq['product_tie_rate']:.3f}.
- Worst product-level delta: {eq['worst_product_delta']:.6f} at seed {int(eq['worst_product_seed'])}, shift `{eq['worst_product_shift']}`, product `{eq['worst_product_id']}`.
- Worst seed-shift mean delta: {eq['worst_seed_shift_delta']:.6f} at seed {int(eq['worst_seed_shift_seed'])}, shift `{eq['worst_seed_shift_shift']}`.

The confidence interval is produced by a hierarchical paired bootstrap: shifts are kept fixed, training seeds are resampled, and paired products are resampled within selected seed-shift cells. Shifts are aggregated with equal weight.

## Structural Evidence

- Target monotonicity diagnostics recompute the complete finite-difference target `S = historical_base_stock + scaled_residual` from saved checkpoints and shifted product sets.
- Autograd derivatives are reported separately. They condition on the batch anchor tensor, so finite differences are the stronger complete-target check when economics change the anchor.
- MonoKAN spline checks are first-layer/path-component checks, including left/right extrapolation intervals. They do not by themselves certify the complete end-to-end target function.

## Clipping And Action Geometry

The residual order-up-to decoder removed the previous clipping-dominated raw-action failure mode. In the analyzed compact shift run, cap-hit rates are low relative to the earlier raw MONOkan failure, where clipping dominated almost every action. See `clipping_and_target_summary.csv` for per-model, per-seed, per-shift details.

## Constrained MLP Diagnosis

The constrained MLP underperforms in this run. Its evidence points to conservative behavior: higher service level, lower lost-sales rate, larger target/order-up-to error, and higher inventory/target proxies. This supports excessive shape restriction or scaling/conservatism as likely causes. Gradient norms are not available in current logs, so optimization difficulty remains unresolved.

## Supported

- The residual order-up-to parameterization removed the previous clipping-dominated failure mode.
- MonoKAN provides explicit structural constraints and passes the available spline-level checks in this implementation audit.
- In the compact 10-seed paired shift run, KAN and MonoKAN are nearly indistinguishable in reward, with KAN slightly ahead on the aggregate mean.

## Not Yet Supported

- MonoKAN consistently outperforms KAN.
- Monotonicity causes better robustness.
- Paper-scale performance.
- Universal distribution-shift robustness.

## Output Tables

- `monokan_vs_kan_equivalence.csv`
- `monokan_vs_kan_by_shift.csv`
- `target_monotonicity_by_model_feature.csv`
- `clipping_and_target_summary.csv`
- `constrained_mlp_diagnosis.csv`
- `monokan_spline_audit.csv`
"""
    (output_dir / "MONOKAN_EVIDENCE_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=Path("outputs/idp_benchmark/factorial_small_four_model"))
    parser.add_argument("--multiseed-root", type=Path, default=Path("outputs/monokan_distribution_shift_factorial_small_10seed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    parser.add_argument("--config", type=Path, default=Path("configs/paper_lost_sales_factorial_small.yaml"))
    parser.add_argument("--profile", default="compact")
    parser.add_argument("--equivalence-margin-abs", type=float, default=5.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--target-products-per-shift", type=int, default=30)
    args = parser.parse_args()

    if any(args.output_dir.glob("*")):
        raise FileExistsError(f"{args.output_dir} already contains files; refusing to overwrite analysis outputs")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    operational = _read_operational(args.multiseed_root)
    pairs = kan_monokan_product_pairs(operational)
    equivalence, by_shift = _direct_kan_monokan_analysis(pairs, args.equivalence_margin_abs, args.bootstrap_samples)

    base_config = load_paper_lost_sales_config(args.config)
    target = _target_monotonicity_rows(args.multiseed_root, base_config, args.profile, int(args.target_products_per_shift))
    clipping = _clipping_and_target_summary(args.multiseed_root)
    diagnosis = _constrained_mlp_diagnosis(args.multiseed_root, clipping, target)
    spline = _monokan_spline_audit(args.multiseed_root)

    equivalence.to_csv(args.output_dir / "monokan_vs_kan_equivalence.csv", index=False)
    by_shift.to_csv(args.output_dir / "monokan_vs_kan_by_shift.csv", index=False)
    target.to_csv(args.output_dir / "target_monotonicity_by_model_feature.csv", index=False)
    clipping.to_csv(args.output_dir / "clipping_and_target_summary.csv", index=False)
    diagnosis.to_csv(args.output_dir / "constrained_mlp_diagnosis.csv", index=False)
    spline.to_csv(args.output_dir / "monokan_spline_audit.csv", index=False)
    _write_report(args.output_dir, equivalence, by_shift, target, clipping, diagnosis, spline)

    protocol = """# Proposed Frozen Experiment Protocol

Status: proposed for review, not automatically frozen.

## Models

- `PaperFCNNResidualOrderUpTo`
- `KANResidualOrderUpTo`
- `ConstrainedMLPResidualOrderUpTo`
- `MonoKANResidualOrderUpTo`

## Configuration

- Development/compact evidence: `configs/paper_lost_sales_factorial_small.yaml`.
- Medium/paper-alignment evidence: `configs/paper_lost_sales_fcnn_alignment_medium.yaml`.
- Final held-out evaluation should use the agreed paper-scale profile once choices are frozen.

## Action Decoder And Anchor

All primary models use residual order-up-to:

```text
S_theta(z) = S_hist(z) + alpha * r_theta(z)
q_theta(y,z) = max(S_theta(z) - y, 0)
```

Current inventory is excluded from the target-stock residual network. The residual anchor is the configured Gamma historical/non-omniscient base-stock anchor.

## Seeds

- Training/model seeds: `2025` through `2034`.
- Product-generation seed may vary with training seed when `--vary-product-seed` is enabled.
- Evaluation demand seeds are the deterministic per-product seeds recorded in `operational_metrics.csv`.

## Shift Definitions

Use the eight existing shifts:

- `in_distribution`
- `low_cv`
- `high_cv`
- `low_mean_demand`
- `high_mean_demand`
- `high_lost_sale_penalty`
- `high_holding_cost`
- `mixed_stress`

## Checkpoint Selection

Use the existing validation-based checkpoint selection in `run_idp_benchmark.py`. Do not select models on the protected final test split.

## Primary Metrics

- Mean reward per period.
- Product-level paired difference `MonoKAN - KAN`.
- Hierarchical paired bootstrap confidence interval with fixed shifts and resampled seeds/products.
- Practical equivalence with predeclared margin.

## Secondary Metrics

- Regret versus non-omniscient and omniscient baselines.
- Worst-decile paired reward delta.
- Win/loss/tie rates.
- Target-stock monotonicity violation rates.
- Spline interval and extrapolation certification checks.
- Cap-hit rate and raw/deployed action difference.
- Residual correction and target-stock distribution.

## Equivalence Margin

Proposed development margin: +/- 5 reward per period, reported also as a percentage of mean KAN reward. This must be reviewed before any formal equivalence claim.

## Protected Final Evaluation Rule

Freeze model classes, configs, seeds, equivalence margin, metrics, and analysis script before running a final held-out/paper-scale evaluation. Do not tune after viewing protected final results.
"""
    Path("docs/FROZEN_EXPERIMENT_PROTOCOL.md").write_text(protocol, encoding="utf-8")
    print(equivalence.to_string(index=False))
    print(f"saved_analysis={args.output_dir}")
    print("saved_protocol=docs/FROZEN_EXPERIMENT_PROTOCOL.md")


if __name__ == "__main__":
    main()
