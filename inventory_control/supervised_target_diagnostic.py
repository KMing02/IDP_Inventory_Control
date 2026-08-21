"""Supervised identification diagnostic for direct target-stock policies.

This module deliberately does not participate in policy pretraining. It asks
whether each architecture can represent the observable empirical newsvendor
mapping before reward-only DirectBackprop optimization is considered.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import math
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install neural dependencies with `python -m pip install -e '.[dev]'`.") from exc

from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.policies.neural.fcnn import (
    PaperFCNNFeatureStats,
    build_paper_lost_sales_features,
    compute_paper_fcnn_feature_stats,
    empirical_newsvendor_target_ratio,
    empirical_newsvendor_target_ratio_from_features,
    make_paper_direct_target_stock_fcnn_model,
    paper_history_feature_count,
)
from inventory_control.policies.neural.monokan import (
    MonoKANPolicy,
    make_paper_direct_target_stock_kan_model,
    make_paper_direct_target_stock_monokan_model_for_products,
)
from inventory_control.products import PaperLostSalesProduct, PaperLostSalesProductGenerator


SUPERVISED_TARGET_MODEL_NAMES = (
    "PaperFCNNDirectTargetStock",
    "KANDirectTargetStock",
    "MonoKANDirectTargetStock",
    "KANDirectTargetStockDeepGrid7",
    "MonoKANDirectTargetStockDeepGrid7",
    "KANDirectTargetStockDeepGrid10",
    "MonoKANDirectTargetStockDeepGrid10",
)


@dataclass(frozen=True)
class SupervisedTargetDataset:
    """Observable contexts and empirical-newsvendor targets for one split."""

    product_ids: tuple[str, ...]
    contexts: torch.Tensor
    target_ratios: torch.Tensor
    target_stocks: torch.Tensor
    historical_means: torch.Tensor


@dataclass(frozen=True)
class SupervisedTargetArtifacts:
    """Files and tables produced by one supervised identification run."""

    output_dir: Path
    summary: pd.DataFrame
    predictions: pd.DataFrame
    training_log: pd.DataFrame
    sensitivities: pd.DataFrame
    metadata: dict[str, Any]


def build_supervised_target_dataset(
    products: Sequence[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
    history_feature_mode: str,
) -> SupervisedTargetDataset:
    """Build labels using only each product's observable history and economics."""
    contexts: list[list[float]] = []
    target_ratios: list[float] = []
    target_stocks: list[float] = []
    historical_means: list[float] = []
    for product in products:
        context = build_paper_lost_sales_features(
            inventory=0.0,
            historical_demand=product.historical_demand,
            price=product.price,
            purchase_cost=product.purchase_cost,
            holding_cost=product.holding_cost,
            lost_sale_penalty=product.lost_sale_penalty,
            feature_stats=feature_stats,
            history_feature_mode=history_feature_mode,
            include_newsvendor_feature=False,
        )[1:]
        target_ratio = empirical_newsvendor_target_ratio_from_features(
            product.historical_demand,
            product.price,
            product.purchase_cost,
            product.holding_cost,
            product.lost_sale_penalty,
        )
        historical_mean = max(sum(product.historical_demand) / len(product.historical_demand), 1.0)
        contexts.append(context)
        target_ratios.append(target_ratio)
        target_stocks.append(target_ratio * historical_mean)
        historical_means.append(historical_mean)
    return SupervisedTargetDataset(
        product_ids=tuple(product.product_id for product in products),
        contexts=torch.tensor(contexts, dtype=torch.float32),
        target_ratios=torch.tensor(target_ratios, dtype=torch.float32),
        target_stocks=torch.tensor(target_stocks, dtype=torch.float32),
        historical_means=torch.tensor(historical_means, dtype=torch.float32),
    )


def run_supervised_target_diagnostic(
    config_path: str | Path = "configs/paper_lost_sales_base_stock_small.yaml",
    output_dir: str | Path = "outputs/supervised_target_diagnostic/medium",
    models: tuple[str, ...] = SUPERVISED_TARGET_MODEL_NAMES,
    *,
    epochs: int = 500,
    batch_size: int | None = None,
    learning_rate: float = 1e-3,
    validation_interval: int = 10,
    training_products: int | None = None,
    validation_products: int | None = None,
    seed: int | None = None,
    device: str = "cpu",
) -> SupervisedTargetArtifacts:
    """Fit empirical targets and report representation/optimization diagnostics."""
    unknown = sorted(set(models) - set(SUPERVISED_TARGET_MODEL_NAMES))
    if unknown:
        raise ValueError(f"unsupported supervised target models: {unknown}")
    if epochs < 1 or validation_interval < 1:
        raise ValueError("epochs and validation_interval must be positive")

    config = load_paper_lost_sales_config(config_path)
    settings = _resolved_settings(
        config,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_interval=validation_interval,
        training_products=training_products,
        validation_products=validation_products,
        seed=seed,
        device=device,
    )
    generator = PaperLostSalesProductGenerator(config)
    train_products = generator.generate(settings["training_products"], split="train")
    validation_products_list = generator.generate(settings["validation_products"], split="validation")
    feature_stats = compute_paper_fcnn_feature_stats(train_products)
    train_data = build_supervised_target_dataset(train_products, feature_stats, settings["history_feature_mode"])
    validation_data = build_supervised_target_dataset(
        validation_products_list,
        feature_stats,
        settings["history_feature_mode"],
    )
    initial_target = empirical_newsvendor_target_ratio(train_products, feature_stats)

    summary_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    log_frames: list[pd.DataFrame] = []
    sensitivity_frames: list[pd.DataFrame] = []
    for model_index, model_name in enumerate(models):
        model_seed = settings["seed"] + model_index
        torch.manual_seed(model_seed)
        model = _make_diagnostic_model(
            model_name,
            config,
            initial_target,
            settings["history_feature_mode"],
            train_products,
            feature_stats,
        )
        result = _fit_supervised_target_model(
            model,
            train_data,
            validation_data,
            epochs=settings["epochs"],
            batch_size=settings["batch_size"],
            learning_rate=settings["learning_rate"],
            validation_interval=settings["validation_interval"],
            seed=model_seed,
            device=settings["device"],
        )
        train_predictions = _prediction_frame(model, model_name, "train", train_data, settings["device"])
        validation_predictions = _prediction_frame(
            model,
            model_name,
            "validation",
            validation_data,
            settings["device"],
        )
        predictions = pd.concat([train_predictions, validation_predictions], ignore_index=True)
        prediction_frames.append(predictions)
        summary_frames.append(
            _summary_frame(
                model,
                model_name,
                predictions,
                train_data,
                validation_data,
                settings["device"],
                result["best_epoch"],
            )
        )
        log = pd.DataFrame(result["training_log"])
        log.insert(0, "model", model_name)
        log_frames.append(log)
        sensitivity_frames.append(
            _sensitivity_frame(model, model_name, validation_data, settings["history_feature_mode"], settings["device"])
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary = pd.concat(summary_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    training_log = pd.concat(log_frames, ignore_index=True)
    sensitivities = pd.concat(sensitivity_frames, ignore_index=True)
    metadata = {
        "diagnostic_only": True,
        "used_for_policy_pretraining": False,
        "label": "empirical_newsvendor_target_stock_from_observable_history_and_economics",
        "prediction_target": "target_stock_ratio_to_product_historical_mean",
        "future_demand_used": False,
        "test_split_used": False,
        "config_path": str(config_path),
        "models": list(models),
        "settings": settings,
        "feature_stats": {
            "demand_scale": feature_stats.demand_scale,
            "price_scale": feature_stats.price_scale,
        },
        "expanded_kan": {"hidden_layers": 2, "hidden_units": 16, "grids": [7, 10]},
    }
    summary.to_csv(destination / "summary.csv", index=False)
    predictions.to_csv(destination / "predictions.csv", index=False)
    training_log.to_csv(destination / "training_log.csv", index=False)
    sensitivities.to_csv(destination / "feature_sensitivities.csv", index=False)
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return SupervisedTargetArtifacts(destination, summary, predictions, training_log, sensitivities, metadata)


def _resolved_settings(
    config: dict[str, Any],
    **overrides: Any,
) -> dict[str, Any]:
    benchmark = config.get("idp_benchmark", {})
    compact = benchmark.get("compact", {})
    return {
        "training_products": int(overrides["training_products"] or compact.get("training_products", config.get("training_products", 300))),
        "validation_products": int(overrides["validation_products"] or compact.get("validation_products", config.get("test_products", 80))),
        "epochs": int(overrides["epochs"]),
        "batch_size": int(overrides["batch_size"] or compact.get("batch_size", config.get("batch_size", 64))),
        "learning_rate": float(overrides["learning_rate"]),
        "validation_interval": int(overrides["validation_interval"]),
        "seed": int(overrides["seed"] if overrides["seed"] is not None else benchmark.get("seed", 2025)),
        "device": str(overrides["device"]),
        "history_feature_mode": str(config.get("paper_history_feature_mode", "raw_lags")),
    }


def _make_diagnostic_model(
    name: str,
    config: dict[str, Any],
    initial_target: float,
    history_feature_mode: str,
    train_products: Sequence[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
) -> nn.Module:
    metadata: dict[str, Any] = {
        "initial_normalized_target": initial_target,
        "paper_history_feature_mode": history_feature_mode,
    }
    if "DeepGrid" in name:
        metadata.update({"hidden_layers": 2, "hidden_units": 16})
        metadata["grid"] = 7 if name.endswith("Grid7") else 10
    if name == "PaperFCNNDirectTargetStock":
        return make_paper_direct_target_stock_fcnn_model(config, metadata=metadata)
    if name.startswith("KAN"):
        return make_paper_direct_target_stock_kan_model(config, metadata=metadata)
    return make_paper_direct_target_stock_monokan_model_for_products(
        config,
        train_products,
        feature_stats,
        metadata=metadata,
    )


def _fit_supervised_target_model(
    model: nn.Module,
    train_data: SupervisedTargetDataset,
    validation_data: SupervisedTargetDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_interval: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    model.to(device)
    train_contexts = train_data.contexts.to(device)
    train_targets = train_data.target_ratios.to(device)
    validation_contexts = validation_data.contexts.to(device)
    validation_targets = validation_data.target_ratios.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    training_log: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_contexts), generator=generator)
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size].to(device)
            optimizer.zero_grad()
            prediction = model(train_contexts[indices])
            loss = torch.mean((prediction - train_targets[indices]) ** 2)
            loss.backward()
            optimizer.step()
            apply_constraints = getattr(model, "apply_monotonic_constraints", None)
            if apply_constraints is not None:
                apply_constraints()

        if epoch == 1 or epoch % validation_interval == 0 or epoch == epochs:
            train_loss = _mse(model, train_contexts, train_targets)
            validation_loss = _mse(model, validation_contexts, validation_targets)
            training_log.append(
                {
                    "epoch": epoch,
                    "train_mse": train_loss,
                    "validation_mse": validation_loss,
                }
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation_loss,
        "training_log": training_log,
    }


def _mse(model: nn.Module, contexts: torch.Tensor, targets: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float(torch.mean((model(contexts) - targets) ** 2).item())


def _prediction_frame(
    model: nn.Module,
    model_name: str,
    split: str,
    data: SupervisedTargetDataset,
    device: str,
) -> pd.DataFrame:
    model.eval()
    with torch.no_grad():
        predicted_ratios = model(data.contexts.to(device)).cpu()
    return pd.DataFrame(
        {
            "model": model_name,
            "split": split,
            "product_id": data.product_ids,
            "target_ratio": data.target_ratios.numpy(),
            "predicted_target_ratio": predicted_ratios.numpy(),
            "target_stock": data.target_stocks.numpy(),
            "predicted_target_stock": (predicted_ratios * data.historical_means).numpy(),
            "historical_mean_demand": data.historical_means.numpy(),
        }
    ).assign(
        target_ratio_error=lambda frame: frame["predicted_target_ratio"] - frame["target_ratio"],
        target_stock_error=lambda frame: frame["predicted_target_stock"] - frame["target_stock"],
    )


def _summary_frame(
    model: nn.Module,
    model_name: str,
    predictions: pd.DataFrame,
    train_data: SupervisedTargetDataset,
    validation_data: SupervisedTargetDataset,
    device: str,
    best_epoch: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, data in (("train", train_data), ("validation", validation_data)):
        frame = predictions[predictions["split"] == split]
        errors = frame["target_ratio_error"]
        target = frame["target_ratio"]
        residual_sum = float((errors**2).sum())
        total_sum = float(((target - target.mean()) ** 2).sum())
        hidden_rank, hidden_correlation = _hidden_diagnostics(model, data.contexts.to(device))
        rows.append(
            {
                "model": model_name,
                "split": split,
                "target_ratio_mae": float(errors.abs().mean()),
                "target_ratio_rmse": math.sqrt(float((errors**2).mean())),
                "target_ratio_r2": 1.0 - residual_sum / total_sum if total_sum > 0 else float("nan"),
                "target_ratio_mean": float(target.mean()),
                "target_ratio_std": float(target.std(ddof=0)),
                "predicted_ratio_mean": float(frame["predicted_target_ratio"].mean()),
                "predicted_ratio_std": float(frame["predicted_target_ratio"].std(ddof=0)),
                "target_stock_mae": float(frame["target_stock_error"].abs().mean()),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                "hidden_effective_rank": hidden_rank,
                "hidden_mean_abs_pairwise_correlation": hidden_correlation,
                "final_spline_scale_share": _final_spline_scale_share(model),
                "best_validation_epoch": best_epoch,
            }
        )
    return pd.DataFrame(rows)


def _hidden_diagnostics(model: nn.Module, contexts: torch.Tensor) -> tuple[float, float]:
    if not isinstance(model, MonoKANPolicy) or len(model.layers) < 2:
        return float("nan"), float("nan")
    model.eval()
    with torch.no_grad():
        hidden = model.layers[0](contexts)
    centered = hidden - hidden.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    squared = singular_values.square()
    probabilities = squared / torch.clamp(squared.sum(), min=1e-12)
    effective_rank = float(torch.exp(-(probabilities * torch.log(torch.clamp(probabilities, min=1e-12))).sum()).item())
    if hidden.shape[1] < 2:
        return effective_rank, float("nan")
    normalized = centered / torch.clamp(centered.norm(dim=0, keepdim=True), min=1e-12)
    correlation = normalized.transpose(0, 1) @ normalized
    mask = ~torch.eye(correlation.shape[0], dtype=torch.bool, device=correlation.device)
    return effective_rank, float(correlation[mask].abs().mean().item())


def _final_spline_scale_share(model: nn.Module) -> float:
    if not isinstance(model, MonoKANPolicy):
        return float("nan")
    final_layer = model.layers[-1]
    spline = float(final_layer.scale_spline.detach().abs().sum().item())
    base = float(final_layer.scale_base.detach().abs().sum().item())
    return spline / max(spline + base, 1e-12)


def _sensitivity_frame(
    model: nn.Module,
    model_name: str,
    data: SupervisedTargetDataset,
    history_feature_mode: str,
    device: str,
) -> pd.DataFrame:
    if history_feature_mode != "summary":
        return pd.DataFrame()
    history_features = paper_history_feature_count(len(data.contexts[0]), history_feature_mode)
    feature_directions = {
        "demand_level": (0, 1),
        "selling_price": (history_features, 1),
        "purchase_cost": (history_features + 1, -1),
        "holding_cost": (history_features + 2, -1),
        "lost_sales_penalty": (history_features + 3, 1),
    }
    contexts = data.contexts.to(device).detach().requires_grad_(True)
    model.eval()
    output = model(contexts)
    derivatives = torch.autograd.grad(output.sum(), contexts)[0]
    rows = []
    for feature, (index, direction) in feature_directions.items():
        values = derivatives[:, index].detach().cpu()
        wrong = values * direction < -1e-7
        rows.append(
            {
                "model": model_name,
                "split": "validation",
                "feature": feature,
                "expected_direction": "increasing" if direction == 1 else "decreasing",
                "wrong_sign_rate": float(wrong.float().mean().item()),
                "derivative_min": float(values.min().item()),
                "derivative_max": float(values.max().item()),
                "evaluated_products": len(values),
            }
        )
    return pd.DataFrame(rows)
