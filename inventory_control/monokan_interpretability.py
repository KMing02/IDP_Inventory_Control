"""Partial-dependence reporting for certified monotone target-stock policies.

The project claims a planner can read a MonoKAN policy's logic directly. This
module produces that evidence: how the recommended target responds to each
driver in isolation, and -- for the soft-quantile policy -- what order-up-to
level the network learned as a function of product economics.

Every curve is also checked against its certified direction, so the figure
doubles as a visual audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from inventory_control.base_stock_output_experiment import _compact_settings, _make_model
from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.paper_lost_sales import paper_critical_quantile
from inventory_control.policies.neural.fcnn import (
    compute_paper_fcnn_feature_stats,
    empirical_newsvendor_target_ratio,
)
from inventory_control.policies.neural.monokan import MonoKANSoftQuantileTargetPolicy
from inventory_control.products import PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import (
    make_paper_lost_sales_training_batch,
    paper_lost_sales_context_features,
)


ECONOMICS_DRIVERS: tuple[tuple[str, int, int], ...] = (
    ("selling price", 0, 1),
    ("purchase cost", 1, -1),
    ("holding cost", 2, -1),
    ("lost-sale penalty", 3, 1),
)


@dataclass(frozen=True)
class InterpretabilityArtifacts:
    """Tables and figure produced from one trained checkpoint."""

    output_dir: Path
    partial_dependence: pd.DataFrame
    level_curve: pd.DataFrame | None
    figure_path: Path | None


def partial_dependence_table(
    model: torch.nn.Module,
    contexts: torch.Tensor,
    history_length: int,
    samples: int = 60,
) -> pd.DataFrame:
    """Sweep each driver across its observed range with the rest held fixed."""
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for name, offset, direction in ECONOMICS_DRIVERS:
            column = history_length + offset
            values = contexts[:, column]
            sweep = torch.linspace(
                float(np.percentile(values.numpy(), 1)), float(np.percentile(values.numpy(), 99)), samples
            )
            responses = []
            for value in sweep:
                probe = contexts.clone()
                probe[:, column] = value
                responses.append(float(model(probe).mean()))
            responses = np.asarray(responses)
            increments = np.diff(responses) * direction
            rows.extend(
                {
                    "driver": name,
                    "certified_direction": direction,
                    "feature_value": float(x),
                    "mean_target": float(y),
                    "direction_holds": bool((increments >= -1e-6).all()),
                    "response_range": float(responses.max() - responses.min()),
                }
                for x, y in zip(sweep.numpy(), responses)
            )
        # Demand level: scale every observed lag together.
        factors = torch.linspace(0.4, 2.5, samples)
        responses = []
        for factor in factors:
            probe = contexts.clone()
            probe[:, :history_length] = probe[:, :history_length] * factor
            responses.append(float(model(probe).mean()))
        responses = np.asarray(responses)
        rows.extend(
            {
                "driver": "demand level (all lags scaled)",
                "certified_direction": 1,
                "feature_value": float(x),
                "mean_target": float(y),
                "direction_holds": bool((np.diff(responses) >= -1e-6).all()),
                "response_range": float(responses.max() - responses.min()),
            }
            for x, y in zip(factors.numpy(), responses)
        )
    return pd.DataFrame(rows)


def level_curve_table(
    model: MonoKANSoftQuantileTargetPolicy,
    contexts: torch.Tensor,
    products: Sequence[Any],
) -> pd.DataFrame:
    """Compare the learned order-up-to level with the analytic critical fractile.

    The fractile is never supplied to the model, so agreement here is evidence
    that reward alone recovered the newsvendor economics.
    """
    with torch.no_grad():
        level = model.predicted_level(contexts).numpy()
    fractile = np.asarray([paper_critical_quantile(product) for product in products], dtype=float)
    return pd.DataFrame(
        {
            "product_id": [product.product_id for product in products],
            "learned_level": level,
            "analytic_critical_fractile": fractile,
            "difference": level - fractile,
        }
    )


def run_monokan_interpretability(
    config_path: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    model_name: str = "MonoKANSoftQuantileTargetStock",
    make_figure: bool = True,
) -> InterpretabilityArtifacts:
    """Produce partial-dependence evidence from one persisted checkpoint."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    config = load_paper_lost_sales_config(config_path)
    settings = _compact_settings(config)
    generator = PaperLostSalesProductGenerator(config)
    train_products = generator.generate(settings["training_products"], split="train")
    validation_products = generator.generate(settings["validation_products"], split="validation")
    feature_stats = compute_paper_fcnn_feature_stats(train_products)

    torch.manual_seed(settings["seed"])
    model = _make_model(
        model_name,
        config,
        train_products,
        feature_stats,
        empirical_newsvendor_target_ratio(train_products, feature_stats),
    )
    model.load_state_dict(torch.load(Path(checkpoint), map_location="cpu", weights_only=True))
    model.eval()

    batch = make_paper_lost_sales_training_batch(
        validation_products,
        settings["training_horizon"],
        torch.Generator().manual_seed(settings["seed"]),
        feature_stats=feature_stats,
        initial_inventory_mode=settings["initial_inventory_mode"],
    )
    contexts = paper_lost_sales_context_features(batch)
    history_length = int(config["history_length_H"])

    partial_dependence = partial_dependence_table(model, contexts, history_length)
    partial_dependence.to_csv(destination / "partial_dependence.csv", index=False)

    level_curve = None
    if isinstance(model, MonoKANSoftQuantileTargetPolicy):
        level_curve = level_curve_table(model, contexts, validation_products)
        level_curve.to_csv(destination / "learned_level.csv", index=False)

    figure_path = _write_figure(destination, partial_dependence, level_curve) if make_figure else None
    _write_report(destination, partial_dependence, level_curve)
    return InterpretabilityArtifacts(destination, partial_dependence, level_curve, figure_path)


def _write_figure(
    destination: Path,
    partial_dependence: pd.DataFrame,
    level_curve: pd.DataFrame | None,
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return None

    drivers = list(partial_dependence["driver"].unique())
    panels = len(drivers) + (1 if level_curve is not None else 0)
    figure, axes = plt.subplots(1, panels, figsize=(3.8 * panels, 3.8))
    axes = np.atleast_1d(axes)
    for axis, driver in zip(axes, drivers):
        frame = partial_dependence.loc[partial_dependence["driver"] == driver]
        axis.plot(frame["feature_value"], frame["mean_target"], lw=2.2)
        holds = bool(frame["direction_holds"].iloc[0])
        axis.set_title(f"{driver}\ncertified direction: {'HELD' if holds else 'VIOLATED'}", fontsize=9)
        axis.set_xlabel("normalized value")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("mean target stock")
    if level_curve is not None:
        axis = axes[-1]
        axis.scatter(
            level_curve["analytic_critical_fractile"], level_curve["learned_level"], s=12, alpha=0.55
        )
        axis.plot([0.3, 1.0], [0.3, 1.0], "k--", lw=1.2)
        correlation = float(
            np.corrcoef(level_curve["learned_level"], level_curve["analytic_critical_fractile"])[0, 1]
        )
        axis.set_title(f"learned level vs newsvendor fractile\ncorrelation {correlation:.3f}", fontsize=9)
        axis.set_xlabel("analytic critical fractile")
        axis.set_ylabel("learned level")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    path = destination / "interpretability.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def _write_report(
    destination: Path,
    partial_dependence: pd.DataFrame,
    level_curve: pd.DataFrame | None,
) -> None:
    summary = (
        partial_dependence.groupby("driver", sort=False)
        .agg(
            certified_direction=("certified_direction", "first"),
            direction_holds=("direction_holds", "first"),
            response_range=("response_range", "first"),
        )
        .reset_index()
    )
    report = "# MonoKAN Interpretability\n\n"
    report += "Each driver is swept across its 1st-99th percentile range with the\n"
    report += "remaining context held at the observed validation distribution.\n\n"
    report += "```text\n" + summary.to_string(index=False) + "\n```\n"
    if level_curve is not None:
        correlation = float(
            np.corrcoef(level_curve["learned_level"], level_curve["analytic_critical_fractile"])[0, 1]
        )
        report += "\n## Learned Order-Up-To Level\n\n"
        report += f"Correlation with the analytic critical fractile: `{correlation:.4f}`.\n"
        report += f"Mean signed difference: `{level_curve['difference'].mean():+.4f}`.\n\n"
        report += "The fractile is never an input feature, so this measures how much of\n"
        report += "the newsvendor structure the reward signal alone recovered.\n"
    (destination / "INTERPRETABILITY.md").write_text(report, encoding="utf-8")
