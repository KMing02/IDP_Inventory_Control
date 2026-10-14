import json
from pathlib import Path

import pytest
import torch
from torch import nn

from inventory_control.direct_quantity_experiment import KAN_METADATA, inventory_response, saved_seeds, summarize_runs
from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.policies.neural.fcnn import compute_paper_fcnn_feature_stats
from inventory_control.policies.neural.monokan import make_paper_raw_action_monokan_model_for_products
from inventory_control.products import PaperLostSalesProductGenerator


class StockReaction(nn.Module):
    """Toy direct-quantity policy: the order moves with on-hand stock (input 0) at a fixed slope."""

    def __init__(self, slope: float) -> None:
        super().__init__()
        self.slope = slope

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return 2.0 + self.slope * features[:, 0]


def _products():
    config = load_paper_lost_sales_config("configs/paper_lost_sales_debug.yaml")
    products = PaperLostSalesProductGenerator(config).generate(3, split="validation")
    return products, compute_paper_fcnn_feature_stats(products)


def test_inventory_response_flags_orders_that_rise_with_stock() -> None:
    products, feature_stats = _products()

    rising = inventory_response(StockReaction(0.5), products, feature_stats)
    falling = inventory_response(StockReaction(-0.5), products, feature_stats)

    assert rising["products_with_any_violation_pct"] == 100.0
    assert rising["violating_steps_pct"] == 100.0
    assert rising["worst_rise_in_mean_demand_units"] > 0.0
    assert falling["products_with_any_violation_pct"] == 0.0
    assert falling["worst_rise_in_mean_demand_units"] == 0.0
    assert falling["mean_inventory_slope"] < 0.0 < rising["mean_inventory_slope"]


def _row(model: str, seed: int, rewards: list[float], violation: float) -> dict:
    return {
        "model": model, "seed": seed, "epochs": 5, "params": 10, "train_seconds": 1,
        "reward_per_period": sum(rewards) / len(rewards), "regret_pct": 1.0, "per_product_reward": rewards,
        "products_with_any_violation_pct": violation, "violating_steps_pct": violation / 10,
        "worst_rise_in_mean_demand_units": violation / 100, "mean_inventory_slope": -0.5,
    }


def test_summary_pools_seeds_and_prices_the_certificate(tmp_path: Path) -> None:
    rows = [
        _row("monokan", 2025, [10.0, 20.0, 30.0], 0.0),
        _row("monokan", 2026, [12.0, 22.0, 32.0], 0.0),
        _row("kan", 2025, [13.0, 22.0, 34.0], 40.0),
        _row("kan", 2026, [13.0, 25.0, 35.0], 20.0),
    ]
    for row in rows:
        (tmp_path / f"{row['model']}_{row['seed']}.json").write_text(json.dumps(row), encoding="utf-8")

    summary = summarize_runs(tmp_path)

    assert summary["monokan"]["seeds"] == [2025, 2026]
    assert summary["kan"]["products_with_any_violation_pct"] == pytest.approx(30.0)
    assert summary["kan"]["violation_pct_by_seed"] == [40.0, 20.0]
    # Seed-averaged per product: MonoKAN (11, 21, 31) against KAN (13, 23.5, 34.5).
    assert summary["monokan_minus_kan"]["mean"] == pytest.approx(-8 / 3)
    assert (tmp_path / "summary.json").exists()


def test_kan_open_starts_like_monokan_while_the_reported_kan_starts_constant() -> None:
    config = load_paper_lost_sales_config("configs/paper_lost_sales_debug.yaml")
    products = PaperLostSalesProductGenerator(config).generate(6, split="train")
    stats = compute_paper_fcnn_feature_stats(products)

    def build(**extra):
        torch.manual_seed(0)
        return make_paper_raw_action_monokan_model_for_products(
            config, products, stats, metadata=dict(KAN_METADATA, **extra)
        )

    monokan, kan, kan_open = build(monotone=True), build(monotone=False), build(monotone=False, initialization="open")

    assert monokan.paper_initialization_mode == kan_open.paper_initialization_mode == "empirical_newsvendor_open_readout"
    assert kan.paper_initialization_mode == "global_empirical_newsvendor_median"
    assert float(kan.layers[-1].scale_base.detach().abs().sum()) == 0.0
    assert float(kan_open.layers[-1].scale_base.detach().abs().sum()) > 0.0
    assert not kan_open.monotonic_vars
    monokan_constant = build(monotone=True, initialization="constant")
    assert monokan_constant.paper_initialization_mode == "global_empirical_newsvendor_median"
    assert monokan_constant.monotonic_vars == monokan.monotonic_vars
    with pytest.raises(ValueError, match="initialization"):
        build(monotone=False, initialization="warm")


def test_kan_open_runs_are_kept_apart_from_kan_runs(tmp_path: Path) -> None:
    rows = [
        _row("monokan", 2025, [10.0, 20.0, 30.0], 0.0),
        _row("kan", 2025, [13.0, 22.0, 34.0], 40.0),
        _row("kan_open", 2025, [11.0, 21.0, 31.0], 10.0),
    ]
    for row in rows:
        (tmp_path / f"{row['model']}_{row['seed']}.json").write_text(json.dumps(row), encoding="utf-8")
        (tmp_path / f"{row['model']}_{row['seed']}.pt").write_bytes(b"")

    summary = summarize_runs(tmp_path)

    assert summary["kan"]["violation_pct_by_seed"] == [40.0]
    assert summary["kan_open"]["violation_pct_by_seed"] == [10.0]
    assert summary["monokan_minus_kan"]["mean"] == pytest.approx(-3.0)
    assert summary["monokan_minus_kan_open"]["mean"] == pytest.approx(-1.0)
    assert summary["kan_open_minus_kan"]["mean"] == pytest.approx(-2.0)
    assert saved_seeds(tmp_path) == [2025]

