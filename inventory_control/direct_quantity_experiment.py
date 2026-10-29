"""Direct-quantity comparison, the report's main design: the network decides the order itself.

Under the target-stock decoder ``q = max(S - I, 0)`` the rule "order less when stock is higher" holds for every model,
so a monotone network only certifies the economics. Here the network emits the order from on-hand stock, the last
32 demands and the four economic inputs (1 + 32 + 4 = 37 inputs), so the stock direction has to come from the model.
MonoKAN and the same KAN without the certificate, each from two output initialisations, are compared with the
paper's FCNN under one protocol.

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
from inventory_control.paper_lost_sales import (
    historical_empirical_quantile,
    historical_gamma_quantile,
    omniscient_gamma_base_stock_policy,
    paper_critical_quantile,
)
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
# MonoKAN and KAN share this architecture. ``monotone`` switches the projection and also the output
# initialisation (open readout for MonoKAN, constant output for KAN; see the factory).
KAN_METADATA = {
    "hidden_layers": 1,
    "hidden_units": 8,
    "grid": 5,
    "adaptive_grid": True,
    "adaptive_grid_eps": 0.1,
    "sort_history_lags": True,
}
# ``monokan`` and ``kan`` are the arms as first run: MonoKAN with an open readout, the KAN with a constant
# initial output. ``kan_open`` and ``monokan_constant`` swap the initialisation, so every pairing below can also be
# made from an identical start (open with open, constant with constant).
MODELS = ("monokan", "kan", "kan_open", "monokan_constant", "fcnn")
# Pairs reported by :func:`summarize_runs` as ``<first>_minus_<second>``: certified minus unconstrained (as first run
# and from each identical start), then the effect of the start within each network (open minus constant).
PAIRS = (
    ("monokan", "kan"),
    ("monokan", "kan_open"),
    ("monokan_constant", "kan"),
    ("monokan", "monokan_constant"),
    ("kan_open", "kan"),
)
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
    """Build one of ``MODELS``: certified (``monokan*``), unconstrained (``kan*``) or the paper ``fcnn``."""
    if name not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    if name == "fcnn":
        return make_paper_fcnn_model(setup.config)
    metadata = dict(KAN_METADATA, monotone=name.startswith("monokan"))
    if name == "kan_open":
        metadata["initialization"] = "open"
    elif name == "monokan_constant":
        metadata["initialization"] = "constant"
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

    # Training does not cap the order, but evaluation does (the evaluator's default, used in every reported run):
    # stock after ordering is kept at most ``max_order_factor`` times the initial mean demand. :func:`cap_sensitivity`
    # re-evaluates the saved models without the cap.
    metrics = batched_torch_policy_metrics(
        setup.validation_products,
        model,
        setup.feature_stats,
        settings={**settings, "scale_action_by_history_mean": True, "enforce_action_cap": True},
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

    ``monokan_minus_kan`` is the profit gap between the certified and the unconstrained model as first run (same
    architecture, different output initialisation). ``monokan_minus_kan_open`` (both open) and
    ``monokan_constant_minus_kan`` (both constant) compare from an identical start, and
    ``monokan_minus_monokan_constant`` and ``kan_open_minus_kan`` measure the start within each network. All are
    paired over the shared products with rewards averaged over seeds.
    """
    out_dir = Path(out_dir)
    rows = [json.loads(path.read_text(encoding="utf-8")) for _, _, path in _run_files(out_dir, ".json")]
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

    for first, second in PAIRS:
        if first in summary and second in summary:
            summary[f"{first}_minus_{second}"] = paired_difference(per_product(first), per_product(second))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def _run_files(run_dir: str | Path, suffix: str) -> list[tuple[str, int, Path]]:
    """(model, seed, path) of every ``<model>_<seed><suffix>`` file of the comparison in ``run_dir``."""
    found = []
    for path in sorted(Path(run_dir).glob(f"*_*{suffix}")):
        model, _, seed = path.stem.rpartition("_")
        if model in MODELS and seed.isdigit():
            found.append((model, int(seed), path))
    return found


def saved_models(run_dir: str | Path) -> list[str]:
    """Models of the comparison with at least one saved checkpoint, in ``MODELS`` order."""
    present = {model for model, _, _ in _run_files(run_dir, ".pt")}
    return [model for model in MODELS if model in present]


def saved_seeds(run_dir: str | Path) -> list[int]:
    """Seeds with a saved checkpoint for every model that has any checkpoint in ``run_dir``."""
    runs = _run_files(run_dir, ".pt")
    found = [{seed for model, seed, _ in runs if model == name} for name in saved_models(run_dir)]
    return sorted(set.intersection(*found)) if found else []


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
    of 10,000 decisions at once. For reference, ``gamma_fit`` and ``empirical_quantile`` give the same match for the
    classical rules on the same 32-day history, whose order-up-to level is explicit.
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

    def level_match(implied: np.ndarray) -> dict[str, float]:
        return {
            "raw_corr": float(np.corrcoef(implied, optimal)[0, 1]),
            "scaled_corr": float(np.corrcoef(implied / mean_demand, optimal / mean_demand)[0, 1]),
            "median_rel_error": float(np.median(np.abs(implied - optimal) / np.maximum(optimal, 1e-9))),
        }

    results: dict[str, list[dict[str, float]]] = {}
    for name in saved_models(run_dir):
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
                **level_match(implied),
                "ms_one": float(np.median(single) * 1e3),
                "ms_10k": float(np.median(many) * 1e3),
                "params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            })
        results[name] = per_seed
    classical = {"gamma_fit": historical_gamma_quantile, "empirical_quantile": historical_empirical_quantile}
    for name, estimate in classical.items():
        results[name] = [level_match(np.array([estimate(p, paper_critical_quantile(p)) for p in products]))]
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


def cap_sensitivity(run_dir: str | Path, setup: DirectQuantitySetup | None = None) -> dict[str, Any]:
    """Re-evaluate the saved models without the order cap that :func:`train_and_evaluate` applies in evaluation.

    Returns each model's reward and regret without the cap and the pairings of :func:`summarize_runs`, so the effect
    of the cap on every reported comparison can be read off.
    """
    setup = setup or load_setup()
    settings = {**setup.settings, "scale_action_by_history_mean": True, "enforce_action_cap": False}
    omniscient = omniscient_rewards(setup.validation_products, setup.settings)
    rewards: dict[str, list[np.ndarray]] = {}
    for name, seed, _ in _run_files(run_dir, ".pt"):
        metrics = batched_torch_policy_metrics(
            setup.validation_products,
            load_trained(name, seed, run_dir, setup),
            setup.feature_stats,
            settings=settings,
            split="validation",
        )
        rewards.setdefault(name, []).append(metrics["mean_reward_per_period"].to_numpy(dtype=float))
    result: dict[str, Any] = {
        name: {
            "seeds": len(runs),
            "reward_per_period": float(np.mean(runs)),
            "regret_pct": float(np.mean([regret_pct(run, omniscient) for run in runs])),
        }
        for name, runs in rewards.items()
    }
    for first, second in PAIRS:
        if first in rewards and second in rewards:
            result[f"{first}_minus_{second}"] = paired_difference(
                np.mean(rewards[first], axis=0), np.mean(rewards[second], axis=0)
            )
    return result
