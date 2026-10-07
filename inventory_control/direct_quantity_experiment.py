"""Direct-quantity comparison, the report's main design: the network decides the order itself.

Under the target-stock decoder ``q = max(S - I, 0)`` the rule "order less when stock is higher" holds for every model,
so a monotone network only certifies the economics. Here the network emits the order from on-hand stock, the last
32 demands and the four economic inputs (1 + 32 + 4 = 37 inputs), so the stock direction has to come from the model.
Three direct-quantity policies are compared under one protocol: MonoKAN, the same KAN without the certificate, and
the paper's FCNN.

Each (model, seed) run writes its own result file, so runs can execute in parallel processes;
:func:`summarize_runs` pools them.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from inventory_control.analysis import paired_difference, regret_pct
from inventory_control.base_stock_output_experiment import _compact_settings, _evaluate_baseline
from inventory_control.evaluation.torch_policy import batched_torch_policy_metrics
from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.paper_lost_sales import omniscient_gamma_base_stock_policy
from inventory_control.policies.neural.fcnn import (
    PaperFCNNFeatureStats,
    build_paper_lost_sales_features,
    compute_paper_fcnn_feature_stats,
    make_paper_fcnn_model,
)
from inventory_control.policies.neural.monokan import make_paper_raw_action_monokan_model_for_products
from inventory_control.products import PaperLostSalesProduct, PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import train_paper_lost_sales_fcnn

CONFIG = Path("configs/paper_lost_sales_rolling_direct_target_dev_1k_normalized.yaml")
# MonoKAN and KAN share this architecture; only ``monotone`` differs between them.
KAN_METADATA = {
    "hidden_layers": 1,
    "hidden_units": 8,
    "grid": 5,
    "adaptive_grid": True,
    "adaptive_grid_eps": 0.1,
    "sort_history_lags": True,
}
MODELS = ("monokan", "kan", "fcnn")
SEEDS = (2025, 2026, 2027)


@dataclass
class DirectQuantitySetup:
    """Config, products and feature statistics shared by every run of the comparison."""

    config: dict[str, Any]
    settings: dict[str, Any]
    train_products: list[PaperLostSalesProduct]
    validation_products: list[PaperLostSalesProduct]
    feature_stats: PaperFCNNFeatureStats


def load_setup(config_path: str | Path = CONFIG) -> DirectQuantitySetup:
    """Generate the training and validation products exactly as every run does."""
    config = load_paper_lost_sales_config(config_path)
    settings = _compact_settings(config)
    generator = PaperLostSalesProductGenerator(config)
    train_products = generator.generate(settings["training_products"], split="train")
    validation_products = generator.generate(settings["validation_products"], split="validation")
    return DirectQuantitySetup(
        config=config,
        settings=settings,
        train_products=train_products,
        validation_products=validation_products,
        feature_stats=compute_paper_fcnn_feature_stats(train_products),
    )


def build_model(name: str, setup: DirectQuantitySetup) -> nn.Module:
    """Build one of ``MODELS``: ``monokan`` (certified), ``kan`` (same network, no certificate) or ``fcnn``."""
    if name not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    if name == "fcnn":
        return make_paper_fcnn_model(setup.config)
    metadata = dict(KAN_METADATA, monotone=(name == "monokan"))
    return make_paper_raw_action_monokan_model_for_products(
        setup.config, setup.train_products, setup.feature_stats, metadata=metadata
    )


def omniscient_rewards(products: Sequence[PaperLostSalesProduct], settings: dict[str, Any]) -> np.ndarray:
    """Per-product reward of the omniscient base-stock policy on the shared evaluation paths."""
    return np.array([
        _evaluate_baseline(product, "OmniscientOptimal", settings, index, omniscient_gamma_base_stock_policy)[
            "mean_reward_per_period"
        ]
        for index, product in enumerate(products)
    ])


def _orders(model: nn.Module, product: PaperLostSalesProduct, stock, feature_stats) -> np.ndarray:
    """Orders in units for each on-hand stock level, everything else fixed at the product's history."""
    features = torch.tensor(
        [
            build_paper_lost_sales_features(
                inventory=float(level),
                historical_demand=product.historical_demand,
                price=product.price,
                purchase_cost=product.purchase_cost,
                holding_cost=product.holding_cost,
                lost_sale_penalty=product.lost_sale_penalty,
                feature_stats=feature_stats,
            )
            for level in stock
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        return model(features).numpy() * max(float(np.mean(product.historical_demand)), 1.0)


def inventory_response(model: nn.Module, products, feature_stats) -> dict[str, float]:
    """Sweep on-hand stock from 0 to 4 days of mean demand, everything else fixed.

    Two things matter. Sign: does the order ever rise when stock rises (a violation of the stock rule)? Strength:
    an order-up-to rule lowers the order one-for-one with stock, so the ideal slope is -1 and a slope near 0 means
    the policy ignores its inventory. The slope is read over the range a sensible policy visits (up to 1.5 days).
    """
    products_hit, bad_steps, steps, worst = 0, 0, 0, 0.0
    slopes = []
    for product in products:
        mean_demand = max(float(np.mean(product.historical_demand)), 1.0)
        grid = np.linspace(0.0, 4.0 * mean_demand, 25)
        orders = _orders(model, product, grid, feature_stats)
        rises = np.diff(orders)
        flagged = rises > 1e-4 * mean_demand
        bad_steps += int(flagged.sum())
        steps += len(rises)
        if flagged.any():
            products_hit += 1
            worst = max(worst, float(rises.max() / mean_demand))
        near = grid <= 1.5 * mean_demand
        slopes.append(float(np.polyfit(grid[near], orders[near], 1)[0]))
    return {
        "products_with_any_violation_pct": 100 * products_hit / len(products),
        "violating_steps_pct": 100 * bad_steps / steps,
        "worst_rise_in_mean_demand_units": worst,
        "mean_inventory_slope": float(np.mean(slopes)),
        "inventory_slope_p10": float(np.percentile(slopes, 10)),
        "inventory_slope_p90": float(np.percentile(slopes, 90)),
    }


def train_and_evaluate(name: str, seed: int, epochs: int, out_dir: str | Path,
                       config_path: str | Path = CONFIG) -> dict[str, Any]:
    """Train one model with one seed, save its checkpoint and result row, and return the row."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup = load_setup(config_path)
    settings = setup.settings
    torch.manual_seed(seed)
    model = build_model(name, setup)
    assert getattr(model, "paper_action_parameterization", None) == "raw", name
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    start = time.time()
    result = train_paper_lost_sales_fcnn(
        setup.train_products,
        model,
        horizon=settings["training_horizon"],
        batch_size=min(settings["batch_size"], len(setup.train_products)),
        epochs=epochs,
        learning_rate=settings["learning_rate"],
        seed=seed,
        imitation_epochs=0,
        validation_products=setup.validation_products,
        validation_interval=max(epochs // 8, 1),
        feature_stats=setup.feature_stats,
        max_order_factor=settings["max_order_factor"],
        normalize_reward_loss=False,
        training_demand_mode=settings["training_demand_mode"],
        history_feature_mode="raw_lags",
        include_terminal_salvage=settings["include_terminal_salvage"],
        initial_inventory_mode=settings["initial_inventory_mode"],
        enforce_action_cap=False,
        scale_action_by_history_mean=True,
        device="cpu",
    )
    if result.best_state_dict:
        model.load_state_dict(result.best_state_dict)
    model.eval()
    torch.save(model.state_dict(), out_dir / f"{name}_{seed}.pt")

    metrics = batched_torch_policy_metrics(
        setup.validation_products,
        model,
        setup.feature_stats,
        settings={**settings, "scale_action_by_history_mean": True},
        split="validation",
    )
    rewards = metrics["mean_reward_per_period"].to_numpy(dtype=float)
    row = {
        "model": name,
        "seed": seed,
        "epochs": epochs,
        "params": params,
        "train_seconds": round(time.time() - start),
        "reward_per_period": float(rewards.mean()),
        "regret_pct": regret_pct(rewards, omniscient_rewards(setup.validation_products, settings)),
        "per_product_reward": rewards.tolist(),
    }
    row.update(inventory_response(model, setup.validation_products, setup.feature_stats))
    (out_dir / f"{name}_{seed}.json").write_text(json.dumps(row), encoding="utf-8")
    return row


def summarize_runs(out_dir: str | Path) -> dict[str, Any]:
    """Pool the per-run result files of ``out_dir`` and write ``summary.json``.

    ``monokan_minus_kan`` is the price of the certificate: the identical architecture with and without it, paired
    over the shared products with rewards averaged over seeds.
    """
    out_dir = Path(out_dir)
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(out_dir.glob("*_*.json"))
        if path.stem.split("_")[0] in MODELS
    ]
    summary: dict[str, Any] = {}
    for name in MODELS:
        subset = [row for row in rows if row["model"] == name]
        if not subset:
            continue
        summary[name] = {
            "seeds": sorted(row["seed"] for row in subset),
            "epochs": subset[0]["epochs"],
            "params": subset[0]["params"],
            **{
                key: float(np.mean([row[key] for row in subset]))
                for key in (
                    "reward_per_period",
                    "regret_pct",
                    "products_with_any_violation_pct",
                    "violating_steps_pct",
                    "mean_inventory_slope",
                )
            },
            "reward_seed_std": float(np.std([row["reward_per_period"] for row in subset], ddof=1))
            if len(subset) > 1
            else 0.0,
            "violation_pct_by_seed": [row["products_with_any_violation_pct"] for row in subset],
            "worst_rise_in_mean_demand_units": max(row["worst_rise_in_mean_demand_units"] for row in subset),
        }

    def per_product(name: str) -> np.ndarray:
        return np.mean([row["per_product_reward"] for row in rows if row["model"] == name], axis=0)

    if "monokan" in summary and "kan" in summary:
        summary["monokan_minus_kan"] = paired_difference(per_product("monokan"), per_product("kan"))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def saved_seeds(run_dir: str | Path) -> list[int]:
    """Seeds with a saved checkpoint for every model of the comparison."""
    found = [{int(path.stem.split("_")[1]) for path in Path(run_dir).glob(f"{name}_*.pt")} for name in MODELS]
    return sorted(set.intersection(*found))


def load_trained(name: str, seed: int, run_dir: str | Path, setup: DirectQuantitySetup) -> nn.Module:
    """Rebuild a model of the comparison and load its saved weights."""
    torch.manual_seed(seed)
    model = build_model(name, setup)
    model.load_state_dict(torch.load(Path(run_dir) / f"{name}_{seed}.pt"))
    return model.eval()


def implied_order_up_to(run_dir: str | Path, setup: DirectQuantitySetup | None = None,
                        seeds: Sequence[int] | None = None) -> dict[str, list[dict[str, float]]]:
    """Interpretability and decision speed of the saved direct-quantity models.

    Interpretability is the test of Maggiar et al. (2025, Figure 3): a direct-quantity policy's order at zero stock is
    its implied order-up-to level, compared with the omniscient optimal level. ``raw_corr`` compares raw units, where
    product scale alone makes the match look good; ``scaled_corr`` compares per unit of mean demand, which removes
    scale and so tests whether the network learned the economics. Speed is the median CPU time of one decision and
    of 10,000 decisions at once.
    """
    setup = setup or load_setup()
    seeds = seeds or saved_seeds(run_dir)
    products = setup.validation_products
    mean_demand = np.array([max(float(np.mean(p.historical_demand)), 1.0) for p in products])
    optimal = np.array([omniscient_gamma_base_stock_policy(p).S for p in products])
    zero_stock = torch.tensor(
        [
            build_paper_lost_sales_features(
                inventory=0.0,
                historical_demand=p.historical_demand,
                price=p.price,
                purchase_cost=p.purchase_cost,
                holding_cost=p.holding_cost,
                lost_sale_penalty=p.lost_sale_penalty,
                feature_stats=setup.feature_stats,
            )
            for p in products
        ],
        dtype=torch.float32,
    )
    one, batch = zero_stock[:1], zero_stock.repeat(10_000 // len(products) + 1, 1)[:10_000]
    results: dict[str, list[dict[str, float]]] = {}
    for name in MODELS:
        per_seed = []
        for seed in seeds:
            model = load_trained(name, seed, run_dir, setup)
            with torch.no_grad():
                implied = model(zero_stock).numpy() * mean_demand
                for _ in range(20):
                    model(one)
                single, many = [], []
                for _ in range(300):
                    start = time.perf_counter()
                    model(one)
                    single.append(time.perf_counter() - start)
                for _ in range(15):
                    start = time.perf_counter()
                    model(batch)
                    many.append(time.perf_counter() - start)
            per_seed.append({
                "seed": seed,
                "raw_corr": float(np.corrcoef(implied, optimal)[0, 1]),
                "scaled_corr": float(np.corrcoef(implied / mean_demand, optimal / mean_demand)[0, 1]),
                "median_rel_error": float(np.median(np.abs(implied - optimal) / np.maximum(optimal, 1e-9))),
                "ms_one": float(np.median(single) * 1e3),
                "ms_10k": float(np.median(many) * 1e3),
                "params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            })
        results[name] = per_seed
    return results


def largest_violation_curves(run_dir: str | Path, setup: DirectQuantitySetup | None = None,
                             seeds: Sequence[int] | None = None) -> dict[str, Any]:
    """Order-versus-stock curves for the product where the KAN without the certificate breaks the stock rule most.

    Searches every seed and validation product on the same sweep as :func:`inventory_response`, then returns the KAN,
    MonoKAN and omniscient curves for that product on a fine grid. Stock and orders are in days of mean demand.
    """
    setup = setup or load_setup()
    best = None
    for seed in seeds or saved_seeds(run_dir):
        kan = load_trained("kan", seed, run_dir, setup)
        for index, product in enumerate(setup.validation_products):
            mean_demand = max(float(np.mean(product.historical_demand)), 1.0)
            orders = _orders(kan, product, np.linspace(0.0, 4.0 * mean_demand, 25), setup.feature_stats)
            rise = float(np.diff(orders).max() / mean_demand)
            if best is None or rise > best[0]:
                best = (rise, seed, index)
    rise, seed, index = best
    product = setup.validation_products[index]
    mean_demand = max(float(np.mean(product.historical_demand)), 1.0)
    stock = np.linspace(0.0, 4.0 * mean_demand, 161)
    perfect = omniscient_gamma_base_stock_policy(product)
    curves = {
        name: (_orders(load_trained(name, seed, run_dir, setup), product, stock, setup.feature_stats)
               / mean_demand).tolist()
        for name in ("kan", "monokan")
    }
    return {
        "seed": seed,
        "product_index": index,
        "mean_demand": mean_demand,
        "largest_rise_share": rise,
        "stock": (stock / mean_demand).tolist(),
        **curves,
        "perfect": [perfect.act(level, 0) / mean_demand for level in stock],
    }
