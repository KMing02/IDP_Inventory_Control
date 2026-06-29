"""Unified benchmark for the monotonic neural inventory-control IDP."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install neural dependencies with `python -m pip install -e '.[dev]'`.") from exc

from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.paper_evaluation import paper_evaluation_seed_start, require_final_test_confirmation
from inventory_control.paper_lost_sales import (
    BaseStockOrderPolicy,
    evaluate_paper_lost_sales_policy,
    historical_demand_base_stock_policy,
    non_omniscient_gamma_base_stock_policy,
    omniscient_gamma_base_stock_policy,
)
from inventory_control.policies.neural.fcnn import (
    PaperFCNNFeatureStats,
    build_paper_lost_sales_features,
    compute_paper_fcnn_feature_stats,
    infer_model_device,
    make_paper_fcnn_model,
)
from inventory_control.policies.neural.constrained_mlp import (
    make_constrained_monotone_mlp_model,
    make_positive_weight_mlp_model,
)
from inventory_control.policies.neural.monokan import (
    make_paper_monokan_model,
    sample_monokan_first_layer_splines,
)
from inventory_control.products import PaperLostSalesProduct, PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import DirectBackpropResult, train_paper_lost_sales_fcnn


POLICY_ORDER = [
    "PaperFCNN",
    "PositiveWeightMLP",
    "ConstrainedMonoMLP",
    "MonoKAN",
    "PenaltyMLP",
    "NonOmniscientOptimal",
    "OmniscientOptimal",
    "HistoricalDemandBaseStock",
]


@dataclass(frozen=True)
class IDPBenchmarkArtifacts:
    """Files and in-memory tables produced by one benchmark run."""

    output_dir: Path
    summary: pd.DataFrame
    operational_metrics: pd.DataFrame
    predictions: pd.DataFrame
    training_log: pd.DataFrame
    spline_values: pd.DataFrame
    sample_efficiency: pd.DataFrame | None = None


class _TorchPaperPolicyAdapter:
    """Apply one trained torch model using the common IDP deployment rule."""

    def __init__(
        self,
        name: str,
        model: nn.Module,
        product: PaperLostSalesProduct,
        feature_stats: PaperFCNNFeatureStats,
        max_order_factor: float,
        scale_action_by_history_mean: bool,
    ) -> None:
        self.name = name
        self.model = model
        self.product = product
        self.feature_stats = feature_stats
        self.max_order_factor = max_order_factor
        self.scale_action_by_history_mean = scale_action_by_history_mean

    def act(self, inventory: float, t: int, rng=None) -> float:
        """Return the deployed, state-capped order quantity."""
        raw, deployed = _model_orders(
            self.model,
            self.product,
            self.feature_stats,
            np.asarray([inventory], dtype=float),
            self.max_order_factor,
            self.scale_action_by_history_mean,
        )
        del raw
        return float(deployed[0])


def _profile_settings(config: dict[str, Any], profile: str) -> dict[str, Any]:
    if profile not in {"compact", "full", "paper_replication"}:
        raise ValueError("profile must be 'compact', 'full', or 'paper_replication'")
    benchmark = config.get("idp_benchmark")
    if benchmark is None:
        raise ValueError("paper lost-sales config is missing idp_benchmark settings")
    settings = {**benchmark, **benchmark[profile]}
    settings["profile"] = profile
    if profile in {"compact", "paper_replication"}:
        settings["penalty_weights"] = benchmark[profile].get("penalty_weights", [1.0])
    return settings


def _model_config(config: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Return a config with matched depth/width for IDP neural models."""
    return {
        **config,
        "fcnn": {
            **config["fcnn"],
            "history_encoder": "causal_cnn",
            "hidden_layers": int(config["fcnn"]["hidden_layers"]),
            "hidden_units": int(config["fcnn"]["hidden_units"]),
            "cnn_channels": int(config["fcnn"].get("cnn_channels", 8)),
            "cnn_dilations": list(config["fcnn"].get("cnn_dilations", [1, 2, 4, 8, 16])),
            "activation": str(config["fcnn"]["activation"]),
        },
        "monokan": {
            **config.get("monokan", {}),
            "hidden_layers": int(settings["hidden_layers"]),
            "hidden_units": int(settings["hidden_units"]),
            "max_order_factor": float(settings["max_order_factor"]),
            "scale_action_by_history_mean": bool(settings["scale_action_by_history_mean"]),
        },
        "positive_weight_mlp": {
            **config.get("positive_weight_mlp", {}),
            "hidden_layers": int(settings["hidden_layers"]),
            "hidden_units": int(settings["hidden_units"]),
            "activation": str(settings.get("activation", config["fcnn"]["activation"])),
            "mixed_activation": False,
        },
        "constrained_monotone_mlp": {
            **config.get("constrained_monotone_mlp", {}),
            "hidden_layers": int(settings["hidden_layers"]),
            "hidden_units": int(settings["hidden_units"]),
            "activation": str(settings.get("activation", config["fcnn"]["activation"])),
            "mixed_activation": True,
        },
    }


def _model_orders(
    model: nn.Module,
    product: PaperLostSalesProduct,
    feature_stats: PaperFCNNFeatureStats,
    inventories: np.ndarray,
    max_order_factor: float,
    scale_action_by_history_mean: bool,
) -> tuple[np.ndarray, np.ndarray]:
    features = [
        build_paper_lost_sales_features(
            inventory=float(inventory),
            historical_demand=product.historical_demand,
            price=product.price,
            purchase_cost=product.purchase_cost,
            holding_cost=product.holding_cost,
            lost_sale_penalty=product.lost_sale_penalty,
            feature_stats=feature_stats,
        )
        for inventory in inventories
    ]
    model.eval()
    with torch.no_grad():
        raw = (
            model(torch.tensor(features, dtype=torch.float32, device=infer_model_device(model)))
            .detach()
            .cpu()
            .numpy()
            .astype(float)
        )
    historical_mean = max(float(np.mean(product.historical_demand)), 1.0)
    if scale_action_by_history_mean:
        raw = raw * historical_mean
    cap = np.maximum(float(max_order_factor) * historical_mean - inventories, 0.0)
    return raw, np.minimum(raw, cap)


def _mean_validation_reward(
    config: dict[str, Any],
    settings: dict[str, Any],
    products: list[PaperLostSalesProduct],
    model: nn.Module,
    name: str,
    feature_stats: PaperFCNNFeatureStats,
    split: str,
) -> float:
    seed_start = paper_evaluation_seed_start(split)
    rewards = []
    for index, product in enumerate(products):
        policy = _TorchPaperPolicyAdapter(
            name,
            model,
            product,
            feature_stats,
            float(settings["max_order_factor"]),
            bool(settings["scale_action_by_history_mean"]),
        )
        rewards.append(
            evaluate_paper_lost_sales_policy(
                product,
                policy,
                horizon=int(settings["evaluation_horizon"]),
                n_episodes=1,
                seed=seed_start + index,
                burn_in=int(settings["burn_in"]),
            )["mean_total_reward"]
        )
    return float(np.mean(rewards))


def _train_model(
    config: dict[str, Any],
    settings: dict[str, Any],
    products: list[PaperLostSalesProduct],
    validation_products: list[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
    model: nn.Module,
    name: str,
    split: str,
    penalty_weight: float = 0.0,
    clip_imitation_predictions: bool = True,
) -> DirectBackpropResult:
    def validation_callback(epoch: int, current_model: nn.Module) -> float:
        del epoch
        return _mean_validation_reward(
            config,
            settings,
            validation_products,
            current_model,
            name,
            feature_stats,
            split,
        )

    result = train_paper_lost_sales_fcnn(
        products,
        model,
        horizon=int(settings["training_horizon"]),
        batch_size=min(int(settings["batch_size"]), len(products)),
        epochs=int(settings["reward_epochs"]),
        learning_rate=float(config["learning_rate"]),
        seed=int(settings["seed"]),
        imitation_epochs=int(settings["imitation_epochs"]),
        validation_products=validation_products,
        validation_interval=int(settings["validation_interval"]),
        validation_callback=validation_callback,
        feature_stats=feature_stats,
        max_order_factor=float(settings["max_order_factor"]),
        scale_action_by_history_mean=bool(settings["scale_action_by_history_mean"]),
        normalize_reward_loss=bool(settings["normalize_reward_loss"]),
        monotonicity_penalty_weight=float(penalty_weight),
        monotonicity_samples_per_product=int(settings["monotonicity_samples_per_product"]),
        clip_imitation_predictions=clip_imitation_predictions,
    )
    if result.best_state_dict is not None:
        model.load_state_dict(result.best_state_dict)
    return result


def _oracle_levels(
    products: list[PaperLostSalesProduct],
    oracle_mc_samples: int,
    seed: int,
) -> dict[str, float]:
    del oracle_mc_samples, seed
    return {
        product.product_id: omniscient_gamma_base_stock_policy(product).S
        for product in products
    }


def _prediction_table(
    products: list[PaperLostSalesProduct],
    models: dict[str, nn.Module],
    feature_stats: PaperFCNNFeatureStats,
    settings: dict[str, Any],
    oracle_levels: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for product in products:
        historical_mean = max(float(np.mean(product.historical_demand)), 1.0)
        max_inventory = float(settings["max_order_factor"]) * historical_mean
        inventories = np.linspace(0.0, max_inventory, int(settings["inventory_states"]))
        oracle_order = np.minimum(
            np.maximum(float(oracle_levels[product.product_id]) - inventories, 0.0),
            np.maximum(max_inventory - inventories, 0.0),
        )
        non_omniscient_level = non_omniscient_gamma_base_stock_policy(product).S
        empirical_level = historical_demand_base_stock_policy(product).S
        policy_orders: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "OmniscientOptimal": (oracle_order, oracle_order),
            "NonOmniscientOptimal": (
                np.maximum(float(non_omniscient_level) - inventories, 0.0),
                np.minimum(
                    np.maximum(float(non_omniscient_level) - inventories, 0.0),
                    np.maximum(max_inventory - inventories, 0.0),
                ),
            ),
            "HistoricalDemandBaseStock": (
                np.maximum(float(empirical_level) - inventories, 0.0),
                np.minimum(
                    np.maximum(float(empirical_level) - inventories, 0.0),
                    np.maximum(max_inventory - inventories, 0.0),
                ),
            ),
        }
        for name, model in models.items():
            policy_orders[name] = _model_orders(
                model,
                product,
                feature_stats,
                inventories,
                float(settings["max_order_factor"]),
                bool(settings["scale_action_by_history_mean"]),
            )
        for policy_name, (raw_orders, deployed_orders) in policy_orders.items():
            for state_index, inventory in enumerate(inventories):
                deployed = float(deployed_orders[state_index])
                target = float(oracle_order[state_index])
                rows.append(
                    {
                        "product_id": product.product_id,
                        "policy_name": policy_name,
                        "state_index": state_index,
                        "inventory": float(inventory),
                        "inventory_normalized": float(inventory / historical_mean),
                        "historical_mean_demand": historical_mean,
                        "oracle_order": target,
                        "raw_order": float(raw_orders[state_index]),
                        "deployed_order": deployed,
                        "error": deployed - target,
                        "order_up_to_level": float(deployed + inventory),
                        "omniscient_order_up_to_level": float(oracle_levels[product.product_id]),
                        "true_coefficient_of_variation": float(product.true_coefficient_of_variation),
                    }
                )
    return pd.DataFrame(rows)


def _prediction_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy_name, policy_rows in predictions.groupby("policy_name", sort=False):
        errors = policy_rows["error"].to_numpy(dtype=float)
        zero_inventory_rows = policy_rows.loc[policy_rows["state_index"] == 0]
        order_up_to_errors = (
            zero_inventory_rows["order_up_to_level"].to_numpy(dtype=float)
            - zero_inventory_rows["omniscient_order_up_to_level"].to_numpy(dtype=float)
        )
        raw_violations: list[float] = []
        deployed_violations: list[float] = []
        for _, product_rows in policy_rows.groupby("product_id", sort=False):
            ordered = product_rows.sort_values("state_index")
            raw_violations.extend(np.diff(ordered["raw_order"].to_numpy(dtype=float)).tolist())
            deployed_violations.extend(np.diff(ordered["deployed_order"].to_numpy(dtype=float)).tolist())
        raw_increases = np.maximum(np.asarray(raw_violations, dtype=float), 0.0)
        deployed_increases = np.maximum(np.asarray(deployed_violations, dtype=float), 0.0)
        active_slope_deviation, inactive_slope_deviation = _slope_deviations(policy_rows)
        rows.append(
            {
                "policy_name": policy_name,
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "order_up_to_mae": float(np.mean(np.abs(order_up_to_errors))),
                "order_up_to_rmse": float(np.sqrt(np.mean(order_up_to_errors**2))),
                "raw_violation_rate": float(np.mean(raw_increases > 1e-6)),
                "deployed_violation_rate": float(np.mean(deployed_increases > 1e-6)),
                "raw_max_violation": float(raw_increases.max(initial=0.0)),
                "deployed_max_violation": float(deployed_increases.max(initial=0.0)),
                "mean_slope_deviation_from_minus_one": active_slope_deviation,
                "active_slope_deviation_from_minus_one": active_slope_deviation,
                "inactive_slope_deviation_from_zero": inactive_slope_deviation,
            }
        )
    return pd.DataFrame(rows)


def _mean_slope_deviation(policy_rows: pd.DataFrame) -> float:
    """Return active-region slope deviation retained for compatibility."""
    active_deviation, _ = _slope_deviations(policy_rows)
    return active_deviation


def _slope_deviations(policy_rows: pd.DataFrame) -> tuple[float, float]:
    """Return active-region minus-one and inactive-region zero slope deviations."""
    active_deviations: list[float] = []
    inactive_deviations: list[float] = []
    for _, product_rows in policy_rows.groupby("product_id", sort=False):
        ordered = product_rows.sort_values("state_index")
        inventory = ordered["inventory"].to_numpy(dtype=float)
        orders = ordered["deployed_order"].to_numpy(dtype=float)
        oracle_orders = ordered["oracle_order"].to_numpy(dtype=float)
        if len(inventory) < 2:
            continue
        slopes = np.diff(orders) / np.maximum(np.diff(inventory), 1e-12)
        active = oracle_orders[:-1] > 1e-6
        if np.any(active):
            active_deviations.extend(np.abs(slopes[active] + 1.0).tolist())
        inactive = ~active
        if np.any(inactive):
            inactive_deviations.extend(np.abs(slopes[inactive]).tolist())
    active_mean = float(np.mean(active_deviations)) if active_deviations else 0.0
    inactive_mean = float(np.mean(inactive_deviations)) if inactive_deviations else 0.0
    return active_mean, inactive_mean


def _operational_table(
    products: list[PaperLostSalesProduct],
    models: dict[str, nn.Module],
    feature_stats: PaperFCNNFeatureStats,
    settings: dict[str, Any],
    oracle_levels: dict[str, float],
    split: str,
) -> pd.DataFrame:
    rows = []
    seed_start = paper_evaluation_seed_start(split)
    for index, product in enumerate(products):
        policies = [
            non_omniscient_gamma_base_stock_policy(product),
            BaseStockOrderPolicy(
                S=float(oracle_levels[product.product_id]),
                name="OmniscientOptimal",
            ),
            historical_demand_base_stock_policy(product),
        ]
        policies.extend(
            _TorchPaperPolicyAdapter(
                name,
                model,
                product,
                feature_stats,
                float(settings["max_order_factor"]),
                bool(settings["scale_action_by_history_mean"]),
            )
            for name, model in models.items()
        )
        for policy in policies:
            metrics = evaluate_paper_lost_sales_policy(
                product,
                policy,
                horizon=int(settings["evaluation_horizon"]),
                n_episodes=1,
                seed=seed_start + index,
                burn_in=int(settings["burn_in"]),
            )
            rows.append(
                {
                    "product_id": product.product_id,
                    "policy_name": policy.name,
                    "evaluation_split": split,
                    "evaluation_seed": seed_start + index,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _select_penalty_candidate(
    candidates: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    compliant = [candidate for candidate in candidates if candidate["raw_violation_rate"] <= threshold]
    if compliant:
        return max(compliant, key=lambda candidate: candidate["validation_reward"])
    return min(
        candidates,
        key=lambda candidate: (candidate["raw_violation_rate"], -candidate["validation_reward"]),
    )


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def run_idp_benchmark(
    config_path: str | Path | None = None,
    profile: str = "compact",
    mode: str = "structure-ls",
    split: str = "validation",
    final_test: bool = False,
    output_dir: str | Path | None = None,
) -> IDPBenchmarkArtifacts:
    """Train, evaluate, and save the unified IDP benchmark."""
    require_final_test_confirmation(split, final_test)
    if mode not in {"replicate-ls", "structure-ls", "sample-efficiency"}:
        raise ValueError("mode must be 'replicate-ls', 'structure-ls', or 'sample-efficiency'")
    config = load_paper_lost_sales_config(config_path)
    settings = _profile_settings(config, profile)
    model_config = _model_config(config, settings)
    destination = Path(output_dir or f"outputs/idp_benchmark/{mode}_{profile}_{split}")

    generator = PaperLostSalesProductGenerator(config)
    train_products = generator.generate(int(settings["training_products"]), split="train")
    evaluation_products = generator.generate(int(settings["validation_products"]), split=split)
    feature_stats = compute_paper_fcnn_feature_stats(train_products)

    torch.manual_seed(int(settings["seed"]))
    paper_fcnn = make_paper_fcnn_model(model_config)
    shared_initial_state = copy.deepcopy(paper_fcnn.state_dict())
    paper_fcnn_result = _train_model(
        config,
        settings,
        train_products,
        evaluation_products,
        feature_stats,
        paper_fcnn,
        "PaperFCNN",
        split,
    )

    oracle_levels = _oracle_levels(
        evaluation_products,
        int(settings["oracle_mc_samples"]),
        int(settings["seed"]) + 10_000,
    )
    models: dict[str, nn.Module] = {"PaperFCNN": paper_fcnn}
    penalty_candidates: list[dict[str, Any]] = []
    penalty_logs: list[pd.DataFrame] = []
    selected_penalty: dict[str, Any] | None = None
    monokan_result: DirectBackpropResult | None = None
    monokan: nn.Module | None = None
    positive_weight_result: DirectBackpropResult | None = None
    positive_weight_mlp: nn.Module | None = None
    constrained_mlp_result: DirectBackpropResult | None = None
    constrained_mlp: nn.Module | None = None
    sample_efficiency = pd.DataFrame()
    if mode == "sample-efficiency":
        sample_efficiency = _run_sample_efficiency(
            config,
            settings,
            model_config,
            split,
            destination,
        )
    if mode == "structure-ls":
        for weight in settings["penalty_weights"]:
            penalty_model = make_paper_fcnn_model(model_config)
            penalty_model.load_state_dict(shared_initial_state)
            result = _train_model(
                config,
                settings,
                train_products,
                evaluation_products,
                feature_stats,
                penalty_model,
                f"PenaltyMLP(lambda={float(weight):g})",
                split,
                penalty_weight=float(weight),
            )
            candidate_predictions = _prediction_table(
                evaluation_products,
                {"PenaltyMLP": penalty_model},
                feature_stats,
                settings,
                oracle_levels,
            )
            diagnostics = _prediction_summary(candidate_predictions)
            penalty_diagnostics = diagnostics.loc[diagnostics["policy_name"] == "PenaltyMLP"].iloc[0]
            validation_reward = _mean_validation_reward(
                config,
                settings,
                evaluation_products,
                penalty_model,
                "PenaltyMLP",
                feature_stats,
                split,
            )
            penalty_candidates.append(
                {
                    "weight": float(weight),
                    "model": penalty_model,
                    "result": result,
                    "validation_reward": validation_reward,
                    "raw_violation_rate": float(penalty_diagnostics["raw_violation_rate"]),
                }
            )
            candidate_log = pd.DataFrame(result.training_log or [])
            candidate_log["policy_name"] = f"PenaltyMLP(lambda={float(weight):g})"
            candidate_log["penalty_weight"] = float(weight)
            penalty_logs.append(candidate_log)
        selected_penalty = _select_penalty_candidate(
            penalty_candidates,
            float(settings["raw_violation_threshold"]),
        )
        torch.manual_seed(int(settings["seed"]) + 101)
        positive_weight_mlp = make_positive_weight_mlp_model(model_config)
        positive_weight_result = _train_model(
            config,
            settings,
            train_products,
            evaluation_products,
            feature_stats,
            positive_weight_mlp,
            "PositiveWeightMLP",
            split,
        )
        torch.manual_seed(int(settings["seed"]) + 102)
        constrained_mlp = make_constrained_monotone_mlp_model(model_config)
        constrained_mlp_result = _train_model(
            config,
            settings,
            train_products,
            evaluation_products,
            feature_stats,
            constrained_mlp,
            "ConstrainedMonoMLP",
            split,
        )
        torch.manual_seed(int(settings["seed"]))
        monokan = make_paper_monokan_model(model_config)
        monokan_result = _train_model(
            config,
            settings,
            train_products,
            evaluation_products,
            feature_stats,
            monokan,
            "MonoKAN",
            split,
            clip_imitation_predictions=False,
        )
        models["PenaltyMLP"] = selected_penalty["model"]
        models["PositiveWeightMLP"] = positive_weight_mlp
        models["ConstrainedMonoMLP"] = constrained_mlp
        models["MonoKAN"] = monokan

    predictions = _prediction_table(
        evaluation_products,
        models,
        feature_stats,
        settings,
        oracle_levels,
    )
    prediction_summary = _prediction_summary(predictions)
    operational = _operational_table(
        evaluation_products,
        models,
        feature_stats,
        settings,
        oracle_levels,
        split,
    )
    operational_summary = (
        operational.groupby("policy_name", as_index=False)
        .agg(
            mean_total_reward=("mean_total_reward", "mean"),
            std_total_reward=("mean_total_reward", "std"),
            mean_reward_per_period=("mean_reward_per_period", "mean"),
            service_level=("service_level", "mean"),
            lost_sales_rate=("lost_sales_rate", "mean"),
            mean_order_quantity=("mean_order_quantity", "mean"),
            mean_inventory=("mean_inventory", "mean"),
        )
    )
    summary = operational_summary.merge(prediction_summary, on="policy_name", how="left")
    training_info = pd.DataFrame(
        [
            {
                "policy_name": "PaperFCNN",
                "parameter_count": _parameter_count(paper_fcnn),
                "training_seconds": paper_fcnn_result.elapsed_seconds,
                "selected_penalty_weight": np.nan,
            },
        ]
    )
    if selected_penalty is not None:
        training_info = pd.concat(
            [
                training_info,
                pd.DataFrame(
                    [
                        {
                            "policy_name": "PenaltyMLP",
                            "parameter_count": _parameter_count(selected_penalty["model"]),
                            "training_seconds": selected_penalty["result"].elapsed_seconds,
                            "selected_penalty_weight": selected_penalty["weight"],
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    if positive_weight_mlp is not None and positive_weight_result is not None:
        training_info = pd.concat(
            [
                training_info,
                pd.DataFrame(
                    [
                        {
                            "policy_name": "PositiveWeightMLP",
                            "parameter_count": _parameter_count(positive_weight_mlp),
                            "training_seconds": positive_weight_result.elapsed_seconds,
                            "selected_penalty_weight": np.nan,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    if constrained_mlp is not None and constrained_mlp_result is not None:
        training_info = pd.concat(
            [
                training_info,
                pd.DataFrame(
                    [
                        {
                            "policy_name": "ConstrainedMonoMLP",
                            "parameter_count": _parameter_count(constrained_mlp),
                            "training_seconds": constrained_mlp_result.elapsed_seconds,
                            "selected_penalty_weight": np.nan,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    if monokan is not None and monokan_result is not None:
        training_info = pd.concat(
            [
                training_info,
                pd.DataFrame(
                    [
                        {
                            "policy_name": "MonoKAN",
                            "parameter_count": _parameter_count(monokan),
                            "training_seconds": monokan_result.elapsed_seconds,
                            "selected_penalty_weight": np.nan,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    summary = summary.merge(training_info, on="policy_name", how="left")
    order_map = {name: index for index, name in enumerate(POLICY_ORDER)}
    summary["_order"] = summary["policy_name"].map(order_map)
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    logs = []
    for name, result in [
        ("PaperFCNN", paper_fcnn_result),
    ]:
        frame = pd.DataFrame(result.training_log or [])
        frame["policy_name"] = name
        frame["penalty_weight"] = 0.0
        logs.append(frame)
    if monokan_result is not None:
        frame = pd.DataFrame(monokan_result.training_log or [])
        frame["policy_name"] = "MonoKAN"
        frame["penalty_weight"] = 0.0
        logs.append(frame)
    if positive_weight_result is not None:
        frame = pd.DataFrame(positive_weight_result.training_log or [])
        frame["policy_name"] = "PositiveWeightMLP"
        frame["penalty_weight"] = 0.0
        logs.append(frame)
    if constrained_mlp_result is not None:
        frame = pd.DataFrame(constrained_mlp_result.training_log or [])
        frame["policy_name"] = "ConstrainedMonoMLP"
        frame["penalty_weight"] = 0.0
        logs.append(frame)
    logs.extend(penalty_logs)
    training_log = pd.concat(logs, ignore_index=True)
    if selected_penalty is None:
        training_log["selected"] = True
    else:
        training_log["selected"] = ~training_log["policy_name"].str.startswith("PenaltyMLP") | (
            training_log["penalty_weight"] == float(selected_penalty["weight"])
        )

    spline_rows = []
    if monokan is not None:
        spline_x, spline_values_tensor = sample_monokan_first_layer_splines(monokan, feature_index=0, samples=100)
        for unit_index, values in enumerate(spline_values_tensor.numpy()):
            for x_value, spline_value in zip(spline_x.numpy(), values):
                spline_rows.append(
                    {
                        "hidden_unit": unit_index,
                        "inventory_feature": float(x_value),
                        "edge_value": float(spline_value),
                    }
                )
    spline_values = pd.DataFrame(spline_rows)

    destination.mkdir(parents=True, exist_ok=True)
    model_dir = destination / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(paper_fcnn.state_dict(), model_dir / "paper_fcnn.pt")
    if selected_penalty is not None:
        torch.save(selected_penalty["model"].state_dict(), model_dir / "penalty_mlp.pt")
    if positive_weight_mlp is not None:
        torch.save(positive_weight_mlp.state_dict(), model_dir / "positive_weight_mlp.pt")
    if constrained_mlp is not None:
        torch.save(constrained_mlp.state_dict(), model_dir / "constrained_monotone_mlp.pt")
    if monokan is not None:
        torch.save(monokan.state_dict(), model_dir / "monokan.pt")
    summary.to_csv(destination / "summary.csv", index=False)
    table3 = _paper_table3(summary)
    table3.to_csv(destination / "paper_table3_replication.csv", index=False)
    operational.to_csv(destination / "operational_metrics.csv", index=False)
    predictions.to_csv(destination / "predictions.csv", index=False)
    training_log.to_csv(destination / "training_log.csv", index=False)
    spline_values.to_csv(destination / "monokan_splines.csv", index=False)
    if not sample_efficiency.empty:
        sample_efficiency.to_csv(destination / "sample_efficiency.csv", index=False)
    _save_policy_structure_plots(destination, predictions, mode)
    metadata = {
        "mode": mode,
        "profile": profile,
        "split": split,
        "settings": settings,
        "feature_stats": {
            "demand_scale": feature_stats.demand_scale,
            "price_scale": feature_stats.price_scale,
        },
        "selected_penalty_weight": selected_penalty["weight"] if selected_penalty is not None else None,
        "penalty_candidates": [
            {
                "weight": candidate["weight"],
                "validation_reward": candidate["validation_reward"],
                "raw_violation_rate": candidate["raw_violation_rate"],
            }
            for candidate in penalty_candidates
        ],
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return IDPBenchmarkArtifacts(
        output_dir=destination,
        summary=summary,
        operational_metrics=operational,
        predictions=predictions,
        training_log=training_log,
        spline_values=spline_values,
        sample_efficiency=sample_efficiency if not sample_efficiency.empty else None,
    )


def _paper_table3(summary: pd.DataFrame) -> pd.DataFrame:
    """Create a paper-style reward table with gaps to omniscient optimal."""
    reward_column = "mean_reward_per_period"
    omniscient = summary.loc[summary["policy_name"] == "OmniscientOptimal", reward_column]
    reference = float(omniscient.iloc[0]) if not omniscient.empty else float(summary[reward_column].max())
    table = summary[["policy_name", reward_column]].copy()
    table = table.rename(columns={reward_column: "average_reward"})
    table["gap_to_omniscient_pct"] = 100.0 * (table["average_reward"] / reference - 1.0)
    return table


def _save_policy_structure_plots(destination: Path, predictions: pd.DataFrame, mode: str) -> None:
    """Save paper-aligned structural diagnostic plots."""
    plots_dir = destination / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if predictions.empty:
        return
    product_ids = predictions["product_id"].drop_duplicates().head(3).tolist()
    subset = predictions[predictions["product_id"].isin(product_ids)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for (policy_name, product_id), rows in subset.groupby(["policy_name", "product_id"], sort=False):
        if policy_name not in {
            "PaperFCNN",
            "PositiveWeightMLP",
            "ConstrainedMonoMLP",
            "MonoKAN",
            "OmniscientOptimal",
            "NonOmniscientOptimal",
        }:
            continue
        label = f"{policy_name} {product_id}"
        rows.sort_values("inventory").plot(x="inventory", y="deployed_order", ax=ax, label=label, alpha=0.8)
    ax.set_title("Purchase quantity vs inventory")
    ax.set_xlabel("Inventory level")
    ax.set_ylabel("Purchase quantity")
    ax.grid(alpha=0.3)
    fig.savefig(plots_dir / "figure2_policy_curves.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    zero_inventory = predictions.loc[predictions["state_index"] == 0].copy()
    if not zero_inventory.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        for policy_name, rows in zero_inventory.groupby("policy_name", sort=False):
            if policy_name not in {"PaperFCNN", "PositiveWeightMLP", "ConstrainedMonoMLP", "MonoKAN", "NonOmniscientOptimal"}:
                continue
            ax.scatter(
                rows["omniscient_order_up_to_level"],
                rows["order_up_to_level"],
                c=rows["true_coefficient_of_variation"],
                cmap="viridis",
                label=policy_name,
                alpha=0.7,
            )
        max_level = float(
            max(zero_inventory["omniscient_order_up_to_level"].max(), zero_inventory["order_up_to_level"].max())
        )
        ax.plot([0, max_level], [0, max_level], color="black", linewidth=1)
        ax.set_title("Learned vs omniscient order-up-to level")
        ax.set_xlabel("Omniscient order-up-to level")
        ax.set_ylabel("Learned order-up-to level")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.savefig(plots_dir / "figure3_order_up_to_scatter.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    if mode == "structure-ls":
        diagnostics = _prediction_summary(predictions)
        fig, ax = plt.subplots(figsize=(7, 4))
        diagnostics.set_index("policy_name")["raw_violation_rate"].plot.bar(ax=ax)
        ax.set_title("Raw monotonicity violation rate")
        ax.set_ylabel("Fraction of adjacent inventory pairs")
        ax.grid(axis="y", alpha=0.3)
        fig.savefig(plots_dir / "monotonicity_violations.png", bbox_inches="tight", dpi=150)
        plt.close(fig)


def _run_sample_efficiency(
    config: dict[str, Any],
    settings: dict[str, Any],
    model_config: dict[str, Any],
    split: str,
    destination: Path,
) -> pd.DataFrame:
    """Run a compact paper-style N/H sample-efficiency grid."""
    benchmark = config.get("idp_benchmark", {})
    sample_config = benchmark.get("sample_efficiency", {})
    rows = []
    for history_window in sample_config.get("history_windows", [config["history_length_H"]]):
        grid_config = copy.deepcopy(config)
        grid_config["history_length_H"] = int(history_window)
        generator = PaperLostSalesProductGenerator(grid_config)
        for n_products in sample_config.get("training_products_grid", [settings["training_products"]]):
            local_settings = dict(settings)
            local_settings["training_products"] = min(int(n_products), int(settings["training_products"]))
            local_settings["validation_products"] = int(settings["validation_products"])
            train_products = generator.generate(int(local_settings["training_products"]), split="train")
            eval_products = generator.generate(int(local_settings["validation_products"]), split=split)
            feature_stats = compute_paper_fcnn_feature_stats(train_products)
            local_model_config = copy.deepcopy(model_config)
            local_model_config["history_length_H"] = int(history_window)
            torch.manual_seed(int(settings["seed"]))
            paper_fcnn = make_paper_fcnn_model(local_model_config)
            result = _train_model(
                grid_config,
                local_settings,
                train_products,
                eval_products,
                feature_stats,
                paper_fcnn,
                "PaperFCNN",
                split,
            )
            reward = _mean_validation_reward(
                grid_config,
                local_settings,
                eval_products,
                paper_fcnn,
                "PaperFCNN",
                feature_stats,
                split,
            )
            rows.append(
                {
                    "history_window": int(history_window),
                    "training_products": int(local_settings["training_products"]),
                    "policy_name": "PaperFCNN",
                    "mean_total_reward": reward,
                    "training_seconds": result.elapsed_seconds,
                }
            )
    destination.mkdir(parents=True, exist_ok=True)
    return pd.DataFrame(rows)
