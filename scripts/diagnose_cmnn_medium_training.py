"""Instrument the medium seed-0 CMNN run without changing benchmark logic."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from inventory_control.idp_benchmark import (
    _mean_validation_reward,
    _model_config,
    _profile_settings,
)
from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.policies.neural.constrained_mlp import (
    ConstrainedMonotoneMLP,
    MonotoneActivation,
    make_constrained_monotone_mlp_model,
    make_positive_weight_mlp_model,
)
from inventory_control.policies.neural.fcnn import (
    build_paper_lost_sales_features,
    compute_paper_fcnn_feature_stats,
    make_paper_fcnn_model,
)
from inventory_control.products import PaperLostSalesProduct, PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import (
    historical_base_stock_imitation_loss,
    make_paper_lost_sales_training_batch,
    move_batch_to_device,
    paper_lost_sales_fcnn_rollout,
    paper_lost_sales_features,
    paper_reward_normalization_scale,
)


FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _tensor_stats(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().reshape(-1).float().cpu()
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
    }


def _parameter_group_norm(parameters: list[torch.Tensor]) -> float:
    squared = torch.zeros(())
    for parameter in parameters:
        if parameter.grad is not None:
            squared = squared + parameter.grad.detach().cpu().pow(2).sum()
    return float(torch.sqrt(squared).item())


def _fixed_feature_grid(
    products: list[PaperLostSalesProduct],
    feature_stats,
    max_order_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    features = []
    historical_means = []
    caps = []
    metadata = []
    for product in products:
        historical_mean = max(float(np.mean(product.historical_demand)), 1.0)
        maximum_inventory = float(max_order_factor) * historical_mean
        for fraction in FRACTIONS:
            inventory = fraction * maximum_inventory
            features.append(
                build_paper_lost_sales_features(
                    inventory=inventory,
                    historical_demand=product.historical_demand,
                    price=product.price,
                    purchase_cost=product.purchase_cost,
                    holding_cost=product.holding_cost,
                    lost_sale_penalty=product.lost_sale_penalty,
                    feature_stats=feature_stats,
                )
            )
            historical_means.append(historical_mean)
            caps.append(max(maximum_inventory - inventory, 0.0))
            metadata.append(
                {
                    "product_id": product.product_id,
                    "inventory_fraction": fraction,
                    "inventory": inventory,
                    "historical_mean_demand": historical_mean,
                }
            )
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(historical_means, dtype=torch.float32),
        torch.tensor(caps, dtype=torch.float32),
        metadata,
    )


@torch.no_grad()
def _fixed_raw_order_stats(
    model: nn.Module,
    features: torch.Tensor,
    historical_means: torch.Tensor,
) -> dict[str, float]:
    raw_orders = model(features) * historical_means
    return {
        "fixed_raw_min": float(raw_orders.min().item()),
        "fixed_raw_median": float(raw_orders.median().item()),
        "fixed_raw_mean": float(raw_orders.mean().item()),
        "fixed_raw_max": float(raw_orders.max().item()),
    }


@torch.no_grad()
def _activation_stats(
    model: ConstrainedMonotoneMLP,
    features: torch.Tensor,
    epoch: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    values = features
    hidden_layer = 0
    for layer in model.network:
        if isinstance(layer, MonotoneActivation):
            hidden_layer += 1
            if layer.activation_counts is None:
                values = layer(values)
                continue
            start = 0
            for group_name, count in zip(("convex", "concave", "bounded"), layer.activation_counts):
                stop = start + count
                group = values[..., start:stop]
                stats = _tensor_stats(group)
                row: dict[str, float | int | str] = {
                    "epoch": epoch,
                    "hidden_layer": hidden_layer,
                    "group": group_name,
                    "neuron_count": count,
                    "preactivation_mean": stats["mean"],
                    "preactivation_std": stats["std"],
                    "preactivation_min": stats["min"],
                    "preactivation_max": stats["max"],
                    "left_saturated_proportion": np.nan,
                    "central_active_proportion": np.nan,
                    "right_saturated_proportion": np.nan,
                }
                if group_name == "bounded":
                    row.update(
                        {
                            "left_saturated_proportion": float((group < -1.0).float().mean().item()),
                            "central_active_proportion": float(
                                ((group >= -1.0) & (group <= 1.0)).float().mean().item()
                            ),
                            "right_saturated_proportion": float((group > 1.0).float().mean().item()),
                        }
                    )
                rows.append(row)
                start = stop
        values = layer(values)
    return rows


def _bounded_gradient_rows(
    captured: dict[int, list[torch.Tensor]],
    activations: list[MonotoneActivation],
    epoch: int,
) -> list[dict[str, float | int]]:
    rows = []
    for hidden_layer, activation in enumerate(activations, start=1):
        if activation.activation_counts is None:
            continue
        bounded_start = activation.activation_counts[0] + activation.activation_counts[1]
        gradients = [
            tensor.grad[..., bounded_start:].detach().reshape(-1).cpu()
            for tensor in captured[id(activation)]
            if tensor.grad is not None
        ]
        flat = torch.cat(gradients) if gradients else torch.empty(0)
        rows.append(
            {
                "epoch": epoch,
                "hidden_layer": hidden_layer,
                "bounded_gradient_observations": int(flat.numel()),
                "bounded_gradient_l2": float(torch.linalg.vector_norm(flat).item()) if flat.numel() else 0.0,
                "bounded_gradient_mean_abs": float(flat.abs().mean().item()) if flat.numel() else 0.0,
                "bounded_gradient_max_abs": float(flat.abs().max().item()) if flat.numel() else 0.0,
                "bounded_gradient_zero_proportion": (
                    float((flat.abs() <= 1e-12).float().mean().item()) if flat.numel() else 1.0
                ),
            }
        )
    return rows


def _instrument_cmnn_training(
    config: dict[str, Any],
    settings: dict[str, Any],
    model_config: dict[str, Any],
    train_products: list[PaperLostSalesProduct],
    validation_products: list[PaperLostSalesProduct],
    feature_stats,
    fixed_features: torch.Tensor,
    fixed_historical_means: torch.Tensor,
) -> tuple[ConstrainedMonotoneMLP, dict[str, torch.Tensor], list[dict], list[dict], list[dict], dict[str, Any]]:
    seed = int(settings["seed"])
    torch.manual_seed(seed + 102)
    model = make_constrained_monotone_mlp_model(model_config)
    initial_state = copy.deepcopy(model.state_dict())
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    generator = torch.Generator().manual_seed(seed)
    total_epochs = int(settings["imitation_epochs"]) + int(settings["reward_epochs"])
    max_order_factor = float(settings["max_order_factor"])
    scale_actions = bool(settings["scale_action_by_history_mean"])
    epoch_rows: list[dict] = []
    activation_rows = _activation_stats(model, fixed_features, epoch=0)
    bounded_gradient_rows: list[dict] = []
    first_layer = model.network[0]
    final_layer = model.network[-1]
    initial_stats = _fixed_raw_order_stats(model, fixed_features, fixed_historical_means)
    epoch_rows.append(
        {
            "epoch": 0,
            "phase": "initial",
            "training_loss": np.nan,
            "training_mean_total_reward": np.nan,
            **initial_stats,
            "first_layer_gradient_norm": np.nan,
            "final_layer_gradient_norm": np.nan,
            "final_layer_bias": float(final_layer.bias.detach().mean().item()),
            "mean_effective_first_inventory_weight": float(
                model.effective_first_weight()[:, model.monotonic_feature].mean().item()
            ),
            "mean_downstream_effective_weight": float(
                torch.cat([weight.detach().reshape(-1) for weight in model.effective_downstream_weights()])
                .mean()
                .item()
            ),
            "validation_mean_total_reward": np.nan,
        }
    )
    best_validation_reward: float | None = None
    best_state: dict[str, torch.Tensor] | None = None

    for epoch_index in range(total_epochs):
        epoch = epoch_index + 1
        indices = torch.randint(0, len(train_products), (int(settings["batch_size"]),), generator=generator).tolist()
        batch_products = [train_products[index] for index in indices]
        batch = move_batch_to_device(
            make_paper_lost_sales_training_batch(
                batch_products,
                int(settings["training_horizon"]),
                generator,
                feature_stats=feature_stats,
            ),
            torch.device("cpu"),
        )
        captured: dict[int, list[torch.Tensor]] = {}
        activations = [layer for layer in model.network if isinstance(layer, MonotoneActivation)]
        handles = []
        for activation in activations:
            captured[id(activation)] = []

            def capture_input(module, inputs, output, *, storage=captured[id(activation)]):
                del module, output
                preactivation = inputs[0]
                preactivation.retain_grad()
                storage.append(preactivation)

            handles.append(activation.register_forward_hook(capture_input))

        mean_total_reward = paper_lost_sales_fcnn_rollout(
            model,
            batch,
            max_order_factor=max_order_factor,
            scale_action_by_history_mean=scale_actions,
            include_terminal_salvage=True,
        ).mean()
        if epoch_index < int(settings["imitation_epochs"]):
            primary_loss = historical_base_stock_imitation_loss(
                model,
                batch,
                generator,
                max_order_factor=max_order_factor,
                scale_action_by_history_mean=scale_actions,
                clip_predictions=True,
            )
            phase = "imitation"
        else:
            primary_loss = -mean_total_reward
            if bool(settings["normalize_reward_loss"]):
                primary_loss = primary_loss / paper_reward_normalization_scale(
                    batch,
                    int(settings["training_horizon"]),
                )
            phase = "reward"
        optimizer.zero_grad()
        primary_loss.backward()
        first_gradient_norm = _parameter_group_norm(list(first_layer.parameters()))
        final_gradient_norm = _parameter_group_norm(list(final_layer.parameters()))
        bounded_gradient_rows.extend(_bounded_gradient_rows(captured, activations, epoch))
        for handle in handles:
            handle.remove()
        optimizer.step()

        validation_reward = np.nan
        if epoch % int(settings["validation_interval"]) == 0 or epoch == total_epochs:
            validation_reward = _mean_validation_reward(
                config,
                settings,
                validation_products,
                model,
                "ConstrainedMonoMLPDiagnostic",
                feature_stats,
                "validation",
            )
            if best_validation_reward is None or validation_reward > best_validation_reward:
                best_validation_reward = float(validation_reward)
                best_state = copy.deepcopy(model.state_dict())

        fixed_stats = _fixed_raw_order_stats(model, fixed_features, fixed_historical_means)
        epoch_rows.append(
            {
                "epoch": epoch,
                "phase": phase,
                "training_loss": float(primary_loss.detach().item()),
                "training_mean_total_reward": float(mean_total_reward.detach().item()),
                **fixed_stats,
                "first_layer_gradient_norm": first_gradient_norm,
                "final_layer_gradient_norm": final_gradient_norm,
                "final_layer_bias": float(final_layer.bias.detach().mean().item()),
                "mean_effective_first_inventory_weight": float(
                    model.effective_first_weight()[:, model.monotonic_feature].mean().item()
                ),
                "mean_downstream_effective_weight": float(
                    torch.cat([weight.detach().reshape(-1) for weight in model.effective_downstream_weights()])
                    .mean()
                    .item()
                ),
                "validation_mean_total_reward": validation_reward,
            }
        )
        activation_rows.extend(_activation_stats(model, fixed_features, epoch=epoch))

    final_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    state_information = {
        "best_validation_reward": best_validation_reward,
        "best_state": best_state,
        "final_state": final_state,
    }
    return model, initial_state, epoch_rows, activation_rows, bounded_gradient_rows, state_information


def _penalty_with_violation_count(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    generator: torch.Generator,
    samples_per_product: int,
    max_inventory_factor: float,
    scale_action_by_history_mean: bool,
) -> tuple[torch.Tensor, int, int]:
    batch_size = batch["initial_inventory"].shape[0]
    max_inventory = float(max_inventory_factor) * torch.clamp(batch["historical_mean_demand"], min=1.0)
    penalties = []
    violating_pairs = 0
    for _ in range(samples_per_product):
        first = torch.rand(batch_size, generator=generator)
        second = torch.rand(batch_size, generator=generator)
        inventory_low = torch.minimum(first, second) * max_inventory
        inventory_high = torch.maximum(first, second) * max_inventory
        low_order = model(paper_lost_sales_features(batch, inventory_low))
        high_order = model(paper_lost_sales_features(batch, inventory_high))
        if scale_action_by_history_mean:
            action_scale = torch.clamp(batch["historical_mean_demand"], min=1.0)
            low_order = low_order * action_scale
            high_order = high_order * action_scale
        differences = high_order - low_order
        violating_pairs += int((differences > 0.0).sum().item())
        penalties.append(torch.mean(torch.relu(differences) ** 2))
    return torch.stack(penalties).mean(), violating_pairs, batch_size * samples_per_product


def _instrument_penalty_training(
    config: dict[str, Any],
    settings: dict[str, Any],
    model_config: dict[str, Any],
    train_products: list[PaperLostSalesProduct],
    feature_stats,
) -> list[dict[str, float | int | str]]:
    seed = int(settings["seed"])
    torch.manual_seed(seed)
    initial_model = make_paper_fcnn_model(model_config)
    initial_state = copy.deepcopy(initial_model.state_dict())
    model = make_paper_fcnn_model(model_config)
    model.load_state_dict(initial_state)
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    generator = torch.Generator().manual_seed(seed)
    penalty_generator = torch.Generator().manual_seed(seed + 1_000_003)
    rows = []
    total_epochs = int(settings["imitation_epochs"]) + int(settings["reward_epochs"])
    for epoch_index in range(total_epochs):
        indices = torch.randint(0, len(train_products), (int(settings["batch_size"]),), generator=generator).tolist()
        batch_products = [train_products[index] for index in indices]
        batch = make_paper_lost_sales_training_batch(
            batch_products,
            int(settings["training_horizon"]),
            generator,
            feature_stats=feature_stats,
        )
        mean_total_reward = paper_lost_sales_fcnn_rollout(
            model,
            batch,
            max_order_factor=float(settings["max_order_factor"]),
            scale_action_by_history_mean=bool(settings["scale_action_by_history_mean"]),
            include_terminal_salvage=True,
        ).mean()
        if epoch_index < int(settings["imitation_epochs"]):
            primary_loss = historical_base_stock_imitation_loss(
                model,
                batch,
                generator,
                max_order_factor=float(settings["max_order_factor"]),
                scale_action_by_history_mean=bool(settings["scale_action_by_history_mean"]),
                clip_predictions=True,
            )
            phase = "imitation"
        else:
            primary_loss = -mean_total_reward
            if bool(settings["normalize_reward_loss"]):
                primary_loss = primary_loss / paper_reward_normalization_scale(
                    batch,
                    int(settings["training_horizon"]),
                )
            phase = "reward"
        penalty, violating_pairs, sampled_pairs = _penalty_with_violation_count(
            model,
            batch,
            penalty_generator,
            int(settings["monotonicity_samples_per_product"]),
            float(settings["max_order_factor"]),
            bool(settings["scale_action_by_history_mean"]),
        )
        loss = primary_loss + 10.0 * penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rows.append(
            {
                "epoch": epoch_index + 1,
                "phase": phase,
                "monotonicity_penalty": float(penalty.detach().item()),
                "violating_pairs": violating_pairs,
                "sampled_pairs": sampled_pairs,
                "violation_rate": violating_pairs / sampled_pairs,
            }
        )
    return rows


def _feature_scale_rows(
    products: list[PaperLostSalesProduct],
    feature_stats,
    max_order_factor: float,
) -> list[dict[str, float | str]]:
    inventory = []
    history = []
    price = []
    purchase = []
    holding = []
    penalty = []
    for product in products:
        historical_mean = max(float(np.mean(product.historical_demand)), 1.0)
        inventory.extend([fraction * max_order_factor * historical_mean for fraction in FRACTIONS])
        history.extend(product.historical_demand)
        price.append(product.price)
        purchase.append(product.purchase_cost)
        holding.append(product.holding_cost)
        penalty.append(product.lost_sale_penalty)
    specifications = [
        ("inventory", inventory, feature_stats.demand_scale),
        ("demand_history", history, feature_stats.demand_scale),
        ("price", price, feature_stats.price_scale),
        ("purchase_cost", purchase, feature_stats.price_scale),
        ("holding_cost", holding, feature_stats.price_scale),
        ("lost_sales_penalty", penalty, feature_stats.price_scale),
    ]
    rows = []
    for feature, raw_values, scale in specifications:
        raw = np.asarray(raw_values, dtype=float)
        normalized = raw / float(scale)
        rows.append(
            {
                "feature": feature,
                "normalization_scale": float(scale),
                "raw_min": float(raw.min()),
                "raw_max": float(raw.max()),
                "raw_std": float(raw.std()),
                "normalized_min": float(normalized.min()),
                "normalized_max": float(normalized.max()),
                "normalized_std": float(normalized.std()),
            }
        )
    return rows


@torch.no_grad()
def _policy_grid_rows(
    policy_name: str,
    model: nn.Module,
    features: torch.Tensor,
    historical_means: torch.Tensor,
    caps: torch.Tensor,
    metadata: list[dict[str, Any]],
) -> list[dict[str, float | str]]:
    raw = model(features) * historical_means
    deployed = torch.minimum(raw, caps)
    rows = []
    for index, information in enumerate(metadata):
        rows.append(
            {
                "policy_name": policy_name,
                **information,
                "raw_order": float(raw[index].item()),
                "deployed_order": float(deployed[index].item()),
            }
        )
    return rows


def _configuration_comparison(
    compact_config: dict[str, Any],
    compact_settings: dict[str, Any],
    medium_config: dict[str, Any],
    medium_settings: dict[str, Any],
    compact_demand_scale: float,
    medium_demand_scale: float,
) -> list[dict[str, Any]]:
    comparisons = [
        ("training_products", compact_settings["training_products"], medium_settings["training_products"]),
        ("validation_products", compact_settings["validation_products"], medium_settings["validation_products"]),
        ("batch_size", compact_settings["batch_size"], medium_settings["batch_size"]),
        ("imitation_epochs", compact_settings["imitation_epochs"], medium_settings["imitation_epochs"]),
        ("reward_epochs", compact_settings["reward_epochs"], medium_settings["reward_epochs"]),
        (
            "total_training_iterations",
            compact_settings["imitation_epochs"] + compact_settings["reward_epochs"],
            medium_settings["imitation_epochs"] + medium_settings["reward_epochs"],
        ),
        ("learning_rate", compact_config["learning_rate"], medium_config["learning_rate"]),
        ("normalize_reward_loss", compact_settings["normalize_reward_loss"], medium_settings["normalize_reward_loss"]),
        ("hidden_layers", compact_settings["hidden_layers"], medium_settings["hidden_layers"]),
        ("hidden_units", compact_settings["hidden_units"], medium_settings["hidden_units"]),
        ("history_length", compact_config["history_length_H"], medium_config["history_length_H"]),
        ("empirical_global_demand_scale", compact_demand_scale, medium_demand_scale),
        (
            "scale_action_by_history_mean",
            compact_settings["scale_action_by_history_mean"],
            medium_settings["scale_action_by_history_mean"],
        ),
        ("max_order_factor", compact_settings["max_order_factor"], medium_settings["max_order_factor"]),
        ("base_seed", compact_settings["seed"], medium_settings["seed"]),
        ("cmnn_initialization_seed", compact_settings["seed"] + 102, medium_settings["seed"] + 102),
        ("product_generation_seed", compact_config["product_generation"]["seed"], medium_config["product_generation"]["seed"]),
        ("training_horizon", compact_settings["training_horizon"], medium_settings["training_horizon"]),
        ("evaluation_horizon", compact_settings["evaluation_horizon"], medium_settings["evaluation_horizon"]),
        ("burn_in", compact_settings["burn_in"], medium_settings["burn_in"]),
        ("validation_interval", compact_settings["validation_interval"], medium_settings["validation_interval"]),
        (
            "monotonicity_samples_per_product",
            compact_settings["monotonicity_samples_per_product"],
            medium_settings["monotonicity_samples_per_product"],
        ),
        ("activation", compact_settings["activation"], medium_settings["activation"]),
        (
            "activation_weights",
            compact_config["constrained_monotone_mlp"]["activation_weights"],
            medium_config["constrained_monotone_mlp"]["activation_weights"],
        ),
    ]
    return [
        {"setting": setting, "compact": compact_value, "medium": medium_value}
        for setting, compact_value, medium_value in comparisons
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/paper_lost_sales_medium_seed0.yaml"))
    parser.add_argument("--compact-config", type=Path, default=Path("configs/paper_lost_sales.yaml"))
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=Path("outputs/idp_benchmark/cmnn_medium_seed0_validation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/diagnostics/cmnn_medium_seed0"),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing diagnostics: {args.output_dir}")

    config = load_paper_lost_sales_config(args.config)
    settings = _profile_settings(config, "full")
    model_config = _model_config(config, settings)
    generator = PaperLostSalesProductGenerator(config)
    train_products = generator.generate(int(settings["training_products"]), "train")
    validation_products = generator.generate(int(settings["validation_products"]), "validation")
    feature_stats = compute_paper_fcnn_feature_stats(train_products)
    fixed_features, fixed_means, fixed_caps, fixed_metadata = _fixed_feature_grid(
        validation_products,
        feature_stats,
        float(settings["max_order_factor"]),
    )

    instrumented_cmnn, initial_state, epoch_rows, activation_rows, bounded_gradient_rows, state_info = (
        _instrument_cmnn_training(
            config,
            settings,
            model_config,
            train_products,
            validation_products,
            feature_stats,
            fixed_features,
            fixed_means,
        )
    )
    penalty_rows = _instrument_penalty_training(
        config,
        settings,
        model_config,
        train_products,
        feature_stats,
    )

    torch.manual_seed(int(settings["seed"]) + 102)
    untrained_cmnn = make_constrained_monotone_mlp_model(model_config)
    untrained_cmnn.load_state_dict(initial_state)
    saved_cmnn = make_constrained_monotone_mlp_model(model_config)
    saved_cmnn.load_state_dict(
        torch.load(args.benchmark_output / "models" / "constrained_monotone_mlp.pt", map_location="cpu")
    )
    positive = make_positive_weight_mlp_model(model_config)
    positive.load_state_dict(torch.load(args.benchmark_output / "models" / "positive_weight_mlp.pt", map_location="cpu"))
    paper = make_paper_fcnn_model(model_config)
    paper.load_state_dict(torch.load(args.benchmark_output / "models" / "paper_fcnn.pt", map_location="cpu"))
    policy_rows = []
    for name, model in [
        ("UntrainedCMNN", untrained_cmnn),
        ("InstrumentedTrainedCMNN", instrumented_cmnn),
        ("SavedTrainedCMNN", saved_cmnn),
        ("PositiveWeightMLP", positive),
        ("PaperFCNN", paper),
    ]:
        policy_rows.extend(_policy_grid_rows(name, model, fixed_features, fixed_means, fixed_caps, fixed_metadata))

    compact_config = load_paper_lost_sales_config(args.compact_config)
    compact_settings = _profile_settings(compact_config, "compact")
    compact_products = PaperLostSalesProductGenerator(compact_config).generate(
        int(compact_settings["training_products"]),
        "train",
    )
    compact_feature_stats = compute_paper_fcnn_feature_stats(compact_products)
    configuration_rows = _configuration_comparison(
        compact_config,
        compact_settings,
        config,
        settings,
        compact_feature_stats.demand_scale,
        feature_stats.demand_scale,
    )
    feature_rows = _feature_scale_rows(train_products, feature_stats, float(settings["max_order_factor"]))

    saved_state = torch.load(
        args.benchmark_output / "models" / "constrained_monotone_mlp.pt",
        map_location="cpu",
    )
    reproduction_max_difference = max(
        float((instrumented_cmnn.state_dict()[key] - saved_state[key]).abs().max().item())
        for key in saved_state
    )
    penalty_frame = pd.DataFrame(penalty_rows)
    bounded_frame = pd.DataFrame(bounded_gradient_rows)
    summary = {
        "base_seed": int(settings["seed"]),
        "cmnn_initialization_seed": int(settings["seed"]) + 102,
        "feature_demand_scale": feature_stats.demand_scale,
        "feature_price_scale": feature_stats.price_scale,
        "bounded_region_definition": {
            "left_saturated": "x < -1",
            "central_active": "-1 <= x <= 1",
            "right_saturated": "x > 1",
        },
        "best_validation_reward": state_info["best_validation_reward"],
        "reproduced_checkpoint_max_abs_difference": reproduction_max_difference,
        "penalty_min": float(penalty_frame["monotonicity_penalty"].min()),
        "penalty_mean": float(penalty_frame["monotonicity_penalty"].mean()),
        "penalty_max": float(penalty_frame["monotonicity_penalty"].max()),
        "penalty_total_violating_pairs": int(penalty_frame["violating_pairs"].sum()),
        "penalty_total_sampled_pairs": int(penalty_frame["sampled_pairs"].sum()),
        "bounded_gradient_mean_abs_by_layer": {
            str(layer): float(rows["bounded_gradient_mean_abs"].mean())
            for layer, rows in bounded_frame.groupby("hidden_layer")
        },
        "bounded_gradient_zero_proportion_by_layer": {
            str(layer): float(rows["bounded_gradient_zero_proportion"].mean())
            for layer, rows in bounded_frame.groupby("hidden_layer")
        },
    }

    args.output_dir.mkdir(parents=True)
    pd.DataFrame(configuration_rows).to_csv(args.output_dir / "configuration_comparison.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(args.output_dir / "cmnn_epoch_diagnostics.csv", index=False)
    pd.DataFrame(activation_rows).to_csv(args.output_dir / "cmnn_activation_preactivations.csv", index=False)
    bounded_frame.to_csv(args.output_dir / "cmnn_bounded_gradients.csv", index=False)
    pd.DataFrame(penalty_rows).to_csv(args.output_dir / "penalty_mlp_diagnostics.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(args.output_dir / "feature_scale_summary.csv", index=False)
    policy_frame = pd.DataFrame(policy_rows)
    policy_frame.to_csv(args.output_dir / "fixed_grid_policy_orders.csv", index=False)
    (
        policy_frame.groupby(["policy_name", "inventory_fraction"], as_index=False)
        .agg(
            raw_order_min=("raw_order", "min"),
            raw_order_median=("raw_order", "median"),
            raw_order_mean=("raw_order", "mean"),
            raw_order_max=("raw_order", "max"),
            deployed_order_mean=("deployed_order", "mean"),
        )
        .to_csv(args.output_dir / "fixed_grid_policy_order_summary.csv", index=False)
    )
    (args.output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"saved_diagnostics={args.output_dir}")


if __name__ == "__main__":
    main()
