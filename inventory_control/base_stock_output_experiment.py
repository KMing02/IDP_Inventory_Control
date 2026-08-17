"""Direct base-stock-output experiments for paper-style lost sales."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install neural dependencies with `python -m pip install -e '.[dev]'`.") from exc

from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.paper_lost_sales import evaluate_paper_lost_sales_policy, historical_demand_base_stock_policy
from inventory_control.policies.neural.fcnn import (
    PaperFCNNFeatureStats,
    compute_paper_fcnn_feature_stats,
    empirical_newsvendor_target_ratio,
    make_paper_base_stock_output_fcnn_model_for_products,
)
from inventory_control.policies.neural.monokan import (
    make_paper_base_stock_output_monokan_model_for_products,
    make_paper_base_stock_output_kan_model,
)
from inventory_control.policies.neural.registry import BASE_STOCK_OUTPUT_POLICY_NAMES, POLICY_SPECS_BY_NAME
from inventory_control.products import PaperLostSalesProduct, PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import DirectBackpropResult, train_paper_lost_sales_fcnn


@dataclass(frozen=True)
class BaseStockOutputArtifacts:
    """Artifacts from one direct base-stock-output comparison."""

    output_dir: Path
    summary: pd.DataFrame
    training_log: pd.DataFrame
    metadata: dict[str, Any]


def run_base_stock_output_experiment(
    config_path: str | Path = "configs/paper_lost_sales_debug.yaml",
    output_dir: str | Path = "outputs/base_stock_output_smoke",
    models: tuple[str, ...] = BASE_STOCK_OUTPUT_POLICY_NAMES,
) -> BaseStockOutputArtifacts:
    """Run a no-imitation direct base-stock-output comparison."""
    unknown = sorted(set(models) - set(BASE_STOCK_OUTPUT_POLICY_NAMES))
    if unknown:
        raise ValueError(f"unsupported base-stock-output models: {unknown}")

    config = load_paper_lost_sales_config(config_path)
    settings = _compact_settings(config)
    destination = Path(output_dir)
    generator = PaperLostSalesProductGenerator(config)
    train_products = generator.generate(settings["training_products"], split="train")
    validation_products = generator.generate(settings["validation_products"], split="validation")
    feature_stats = compute_paper_fcnn_feature_stats(train_products)
    initial_target_ratio = empirical_newsvendor_target_ratio(train_products, feature_stats)

    summary_rows: list[dict[str, Any]] = []
    log_frames: list[pd.DataFrame] = []
    for model_name in models:
        spec = POLICY_SPECS_BY_NAME[model_name]
        torch.manual_seed(settings["seed"] + spec.seed_offset)
        model = _make_model(model_name, config, train_products, feature_stats, initial_target_ratio)
        result = train_paper_lost_sales_fcnn(
            train_products,
            model,
            horizon=settings["training_horizon"],
            batch_size=min(settings["batch_size"], len(train_products)),
            epochs=settings["reward_epochs"],
            learning_rate=settings["learning_rate"],
            seed=settings["seed"] + spec.seed_offset,
            imitation_epochs=0,
            validation_products=validation_products,
            validation_interval=settings["validation_interval"],
            feature_stats=feature_stats,
            max_order_factor=settings["max_order_factor"],
            normalize_reward_loss=settings["normalize_reward_loss"],
            training_demand_mode=settings["training_demand_mode"],
        )
        summary_rows.append(
            {
                "policy_name": model_name,
                **_evaluate_model(model, validation_products, feature_stats, settings),
                "training_epochs": settings["reward_epochs"],
                "imitation_epochs": 0,
                "initial_target_ratio": initial_target_ratio,
                "parameter_count": _parameter_count(model),
                "training_seconds": result.elapsed_seconds,
                "checkpoint_filename": spec.checkpoint_filename,
            }
        )
        log_frames.append(_training_log_frame(result, model_name))

    historical = [_evaluate_historical(product, settings, index) for index, product in enumerate(validation_products)]
    summary_rows.append({"policy_name": "HistoricalDemandBaseStock", **_mean_metrics(historical)})
    summary = pd.DataFrame(summary_rows)
    training_log = pd.concat(log_frames, ignore_index=True) if log_frames else pd.DataFrame()
    metadata = {
        "config_path": str(config_path),
        "settings": settings,
        "models": list(models),
        "action_parameterization": "NN(z)->S_theta, q=max(S_theta-inventory,0)",
        "initialization": "final output initialized near empirical newsvendor target ratio",
        "imitation_pretraining": False,
    }

    destination.mkdir(parents=True, exist_ok=True)
    summary.to_csv(destination / "summary.csv", index=False)
    training_log.to_csv(destination / "training_log.csv", index=False)
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return BaseStockOutputArtifacts(destination, summary, training_log, metadata)


def _compact_settings(config: dict[str, Any]) -> dict[str, Any]:
    benchmark = config.get("idp_benchmark", {})
    compact = benchmark.get("compact", {})
    return {
        "seed": int(benchmark.get("seed", config.get("product_generation", {}).get("seed", 0))),
        "training_products": int(compact.get("training_products", config.get("training_products", 8))),
        "validation_products": int(compact.get("validation_products", config.get("test_products", 4))),
        "batch_size": int(compact.get("batch_size", config.get("batch_size", 4))),
        "training_horizon": int(compact.get("training_horizon", config.get("training_horizon_T", 8))),
        "reward_epochs": int(compact.get("reward_epochs", config.get("epochs", 2))),
        "validation_interval": int(compact.get("validation_interval", 0)),
        "evaluation_horizon": int(compact.get("evaluation_horizon", config.get("evaluation_horizon", 12))),
        "burn_in": int(compact.get("burn_in", config.get("burn_in", 0))),
        "max_order_factor": float(benchmark.get("max_order_factor", config.get("fcnn", {}).get("max_order_factor", 4.0))),
        "normalize_reward_loss": bool(benchmark.get("normalize_reward_loss", True)),
        "training_demand_mode": str(benchmark.get("training_demand_mode", "fixed_future")),
        "learning_rate": float(config.get("learning_rate", 1e-3)),
    }


def _make_model(
    name: str,
    config: dict[str, Any],
    train_products: list[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
    initial_target_ratio: float,
) -> nn.Module:
    if name == "PaperFCNNBaseStockOrderUpTo":
        return make_paper_base_stock_output_fcnn_model_for_products(config, train_products, feature_stats=feature_stats)
    if name == "KANBaseStockOrderUpTo":
        metadata = dict(config.get("monokan", {}))
        metadata["initial_normalized_target"] = initial_target_ratio
        return make_paper_base_stock_output_kan_model(config, metadata=metadata)
    if name == "MonoKANBaseStockOrderUpTo":
        return make_paper_base_stock_output_monokan_model_for_products(config, train_products, feature_stats=feature_stats)
    raise ValueError(f"unknown direct base-stock model: {name}")


def _evaluate_model(
    model: nn.Module,
    products: list[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
    settings: dict[str, Any],
) -> dict[str, float]:
    rows = []
    for index, product in enumerate(products):
        policy = _TorchBaseStockPolicyAdapter(model, product, feature_stats, settings["max_order_factor"])
        rows.append(
            evaluate_paper_lost_sales_policy(
                product,
                policy,
                horizon=settings["evaluation_horizon"],
                n_episodes=1,
                seed=settings["seed"] + 10_000 + index,
                burn_in=settings["burn_in"],
            )
        )
    return _mean_metrics(rows)


class _TorchBaseStockPolicyAdapter:
    """Evaluate a context-target torch model through the shared order-up-to decoder."""

    def __init__(
        self,
        model: nn.Module,
        product: PaperLostSalesProduct,
        feature_stats: PaperFCNNFeatureStats,
        max_order_factor: float,
    ) -> None:
        self.name = getattr(model, "name", "NeuralBaseStockOrderUpTo")
        self.model = model
        self.product = product
        self.feature_stats = feature_stats
        self.max_order_factor = float(max_order_factor)

    def act(self, inventory: float, t: int, rng=None) -> float:
        from inventory_control.policies.neural.fcnn import build_paper_lost_sales_features, infer_model_device

        features = torch.tensor(
            [
                build_paper_lost_sales_features(
                    inventory=inventory,
                    historical_demand=self.product.historical_demand,
                    price=self.product.price,
                    purchase_cost=self.product.purchase_cost,
                    holding_cost=self.product.holding_cost,
                    lost_sale_penalty=self.product.lost_sale_penalty,
                    feature_stats=self.feature_stats,
                )
            ],
            dtype=torch.float32,
            device=infer_model_device(self.model),
        )
        self.model.eval()
        with torch.no_grad():
            target_factor = float(self.model(features[:, 1:]).item())
        historical_mean = max(sum(self.product.historical_demand) / len(self.product.historical_demand), 1.0)
        target_stock = target_factor * historical_mean
        max_inventory_level = self.max_order_factor * historical_mean
        return min(max(target_stock - float(inventory), 0.0), max(max_inventory_level - float(inventory), 0.0))


def _evaluate_historical(
    product: PaperLostSalesProduct,
    settings: dict[str, Any],
    index: int,
) -> dict[str, float]:
    return evaluate_paper_lost_sales_policy(
        product,
        historical_demand_base_stock_policy(product),
        horizon=settings["evaluation_horizon"],
        n_episodes=1,
        seed=settings["seed"] + 10_000 + index,
        burn_in=settings["burn_in"],
    )


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    return {
        "mean_reward_per_period": float(frame["mean_reward_per_period"].mean()),
        "service_level": float(frame["service_level"].mean()),
        "lost_sales_rate": float(frame["lost_sales_rate"].mean()),
        "mean_order_quantity": float(frame["mean_order_quantity"].mean()),
        "mean_inventory": float(frame["mean_inventory"].mean()),
    }


def _training_log_frame(result: DirectBackpropResult, model_name: str) -> pd.DataFrame:
    frame = pd.DataFrame(result.training_log or [])
    frame["policy_name"] = model_name
    return frame


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
