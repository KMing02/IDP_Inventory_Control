"""Paper-aligned raw-action DeepGrid7 MonoKAN experiment."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd
import torch
import yaml

from inventory_control.base_stock_output_experiment import _product_metric_row
from inventory_control.idp_benchmark import _batched_torch_policy_metrics
from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.policies.neural.fcnn import PaperFCNNFeatureStats, compute_paper_fcnn_feature_stats
from inventory_control.policies.neural.monokan import (
    make_paper_deep_grid7_raw_action_monokan_model,
    paper_raw_action_monotonic_vars,
)
from inventory_control.products import PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import DirectBackpropResult, train_paper_lost_sales_fcnn


RAW_FCNN_CONFIG = Path("configs/paper_lost_sales_raw_fcnn_replication_5k.yaml")
RAW_MONOKAN_CONFIG = Path("configs/paper_lost_sales_raw_monokan_deepgrid7_5k.yaml")
POLICY_NAME = "MonoKANRawActionDeepGrid7"
CHECKPOINT_FILENAME = "monokan_raw_action_deepgrid7.pt"


def resolve_device(requested: str) -> torch.device:
    """Resolve a requested execution device without silently changing it."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def validate_raw_protocol(
    monokan_config: str | Path = RAW_MONOKAN_CONFIG,
    fcnn_config: str | Path = RAW_FCNN_CONFIG,
) -> dict[str, Any]:
    """Confirm that architecture is the only intended experimental difference."""
    monokan = load_paper_lost_sales_config(monokan_config)
    fcnn = load_paper_lost_sales_config(fcnn_config)
    monokan_signature = _protocol_signature(monokan)
    fcnn_signature = _protocol_signature(fcnn)
    if monokan_signature != fcnn_signature:
        raise ValueError(
            f"raw MonoKAN protocol differs from validated raw FCNN: "
            f"monokan={monokan_signature}, fcnn={fcnn_signature}"
        )
    architecture = monokan["monokan"]
    expected_architecture = {
        "policy_formulation": "raw_action",
        "architecture": "deepgrid7",
        "hidden_layers": 2,
        "hidden_units": 16,
        "grid": 7,
        "roll_history": True,
        "scale_action_by_history_mean": False,
        "imitation_epochs": 0,
    }
    mismatches = {
        key: (architecture.get(key), value)
        for key, value in expected_architecture.items()
        if architecture.get(key) != value
    }
    if mismatches:
        raise ValueError(f"raw DeepGrid7 architecture changed: {mismatches}")
    return {"protocol": monokan_signature, "architecture": expected_architecture}


def run_raw_monokan_seed(
    seed: int,
    output_dir: str | Path,
    device: str,
    checkpoint_raw_action: bool,
    config_path: str | Path = RAW_MONOKAN_CONFIG,
    raw_fcnn_result_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Train and evaluate one formal raw-action DeepGrid7 MonoKAN seed."""
    audit = validate_raw_protocol(config_path)
    chosen_device = resolve_device(device)
    config = load_paper_lost_sales_config(config_path)
    settings = _settings(config, seed, chosen_device)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    resolved_config_path = destination / "resolved_config.yaml"
    resolved_config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    raw_fcnn_dir = Path(raw_fcnn_result_dir or f"outputs/paper_raw_fcnn_replication_5k/seed_{seed}")
    _validate_raw_fcnn_result(raw_fcnn_dir, seed, settings)

    generator = PaperLostSalesProductGenerator(config)
    train_products = generator.generate(settings["training_products"], split="train")
    evaluation_products = generator.generate(settings["validation_products"], split="validation")
    feature_stats = (
        compute_paper_fcnn_feature_stats(train_products)
        if settings["normalize_observations"]
        else PaperFCNNFeatureStats(demand_scale=1.0, price_scale=1.0)
    )
    torch.manual_seed(seed)
    model = make_paper_deep_grid7_raw_action_monokan_model(config)

    def validation_callback(_epoch: int, current_model: torch.nn.Module) -> float:
        metrics = _batched_torch_policy_metrics(
            evaluation_products,
            current_model,
            feature_stats,
            settings,
            "validation",
        )
        return float(metrics["mean_total_reward"].mean())

    result = train_paper_lost_sales_fcnn(
        train_products,
        model,
        horizon=settings["training_horizon"],
        batch_size=settings["batch_size"],
        epochs=settings["reward_epochs"],
        learning_rate=settings["learning_rate"],
        seed=seed,
        imitation_epochs=0,
        validation_products=evaluation_products,
        validation_interval=settings["validation_interval"],
        validation_callback=validation_callback,
        feature_stats=feature_stats,
        max_order_factor=settings["max_order_factor"],
        scale_action_by_history_mean=False,
        normalize_reward_loss=False,
        training_demand_mode=settings["training_demand_mode"],
        include_terminal_salvage=True,
        initial_inventory_mode=settings["initial_inventory_mode"],
        enforce_action_cap=False,
        checkpoint_raw_action=checkpoint_raw_action,
        device=chosen_device,
    )
    final_state = copy.deepcopy(model.state_dict())
    selected_epoch = _selected_epoch(result)
    if result.best_state_dict is not None:
        model.load_state_dict(result.best_state_dict)

    model_dir = destination / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint = model_dir / "monokan_raw_action_deepgrid7_final.pt"
    selected_checkpoint = model_dir / CHECKPOINT_FILENAME
    torch.save(final_state, final_checkpoint)
    torch.save(model.state_dict(), selected_checkpoint)

    metrics = _batched_torch_policy_metrics(
        evaluation_products,
        model,
        feature_stats,
        settings,
        "validation",
    )
    product_metrics = pd.DataFrame(
        [
            _product_metric_row(
                metrics.iloc[index].to_dict(),
                POLICY_NAME,
                product,
                settings,
                index,
                model_seed=seed,
            )
            for index, product in enumerate(evaluation_products)
        ]
    )
    comparison, paired = _raw_fcnn_comparison(raw_fcnn_dir, product_metrics, seed)
    summary = _summary(comparison, result, selected_epoch, selected_checkpoint, final_checkpoint)
    training_log = pd.DataFrame(result.training_log or [])
    training_log["policy_name"] = POLICY_NAME

    summary_path = destination / "summary.csv"
    products_path = destination / "product_metrics.csv"
    comparison_path = destination / "raw_fcnn_comparison.csv"
    paired_path = destination / "raw_monokan_minus_raw_fcnn_paired.csv"
    training_log_path = destination / "training_log.csv"
    summary.to_csv(summary_path, index=False)
    product_metrics.to_csv(products_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    paired.to_csv(paired_path, index=False)
    training_log.to_csv(training_log_path, index=False)
    metadata = {
        "policy_name": POLICY_NAME,
        "policy_formulation": "q_t = MonoKAN(I_t, d_t-32:t-1, p, c, h, b)",
        "feature_order": [
            "inventory",
            *[f"demand_lag_{index}" for index in range(32)],
            "price",
            "purchase_cost",
            "holding_cost",
            "lost_sales_penalty",
        ],
        "monotonic_directions": paper_raw_action_monotonic_vars(32),
        "protocol_audit": audit,
        "config_path": str(config_path),
        "resolved_config_path": str(resolved_config_path),
        "resolved_settings": settings,
        "seed": seed,
        "device": str(chosen_device),
        "device_name": _device_name(chosen_device),
        "activation_checkpointing": bool(checkpoint_raw_action),
        "selected_checkpoint_epoch": selected_epoch,
        "final_epoch": settings["reward_epochs"],
        "selection_criterion": "maximum validation mean total reward",
        "validation_frequency": settings["validation_interval"],
        "selected_checkpoint": str(selected_checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "training_seconds": result.elapsed_seconds,
        "git_sha": _git_sha(),
        "raw_fcnn_reference_dir": str(raw_fcnn_dir),
    }
    metadata_path = destination / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "summary": summary_path,
        "product_metrics": products_path,
        "comparison": comparison_path,
        "paired": paired_path,
        "training_log": training_log_path,
        "metadata": metadata_path,
    }


def run_full_shape_benchmark(
    output_dir: str | Path,
    device: str = "mps",
    checkpoint_raw_action: bool = True,
    seed: int = 2025,
) -> dict[str, Any]:
    """Measure one real full-shape training epoch without formal evaluation."""
    audit = validate_raw_protocol()
    chosen_device = resolve_device(device)
    config = load_paper_lost_sales_config(RAW_MONOKAN_CONFIG)
    settings = _settings(config, seed, chosen_device)
    generator = PaperLostSalesProductGenerator(config)
    train_products = generator.generate(settings["training_products"], split="train")
    feature_stats = PaperFCNNFeatureStats(demand_scale=1.0, price_scale=1.0)
    torch.manual_seed(seed)
    model = make_paper_deep_grid7_raw_action_monokan_model(config)
    if chosen_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(chosen_device)
    result = train_paper_lost_sales_fcnn(
        train_products,
        model,
        horizon=settings["training_horizon"],
        batch_size=settings["batch_size"],
        epochs=1,
        learning_rate=settings["learning_rate"],
        seed=seed,
        imitation_epochs=0,
        validation_products=None,
        validation_interval=0,
        feature_stats=feature_stats,
        max_order_factor=settings["max_order_factor"],
        scale_action_by_history_mean=False,
        normalize_reward_loss=False,
        training_demand_mode=settings["training_demand_mode"],
        include_terminal_salvage=True,
        initial_inventory_mode=settings["initial_inventory_mode"],
        enforce_action_cap=False,
        checkpoint_raw_action=checkpoint_raw_action,
        device=chosen_device,
    )
    report: dict[str, Any] = {
        "device": str(chosen_device),
        "device_name": _device_name(chosen_device),
        "activation_checkpointing": bool(checkpoint_raw_action),
        "epochs_measured": 1,
        "seconds_per_epoch": result.elapsed_seconds,
        "estimated_training_hours_1000_epochs": result.elapsed_seconds * 1_000 / 3_600,
        "estimate_is_extrapolation": True,
        "protocol": audit["protocol"],
        "git_sha": _git_sha(),
    }
    if chosen_device.type == "cuda":
        report["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(chosen_device))
        report["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(chosen_device))
    elif chosen_device.type == "mps":
        report["mps_current_allocated_bytes"] = int(torch.mps.current_allocated_memory())
        if hasattr(torch.mps, "driver_allocated_memory"):
            report["mps_driver_allocated_bytes"] = int(torch.mps.driver_allocated_memory())
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "benchmark.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _settings(config: dict[str, Any], seed: int, device: torch.device) -> dict[str, Any]:
    benchmark = config["idp_benchmark"]
    compact = benchmark["compact"]
    return {
        **benchmark,
        **compact,
        "seed": int(seed),
        "product_generation_seed": int(config["product_generation"]["seed"]),
        "learning_rate": float(config["learning_rate"]),
        "device": str(device),
        "scale_action_by_history_mean": False,
        "enforce_action_cap": False,
        "paper_history_feature_mode": "raw_lags",
    }


def _protocol_signature(config: dict[str, Any]) -> dict[str, Any]:
    benchmark = config["idp_benchmark"]
    compact = benchmark["compact"]
    return {
        "history_length": int(config["history_length_H"]),
        "training_products": int(compact["training_products"]),
        "validation_products": int(compact["validation_products"]),
        "batch_size": int(compact["batch_size"]),
        "training_horizon": int(compact["training_horizon"]),
        "reward_epochs": int(compact["reward_epochs"]),
        "validation_interval": int(compact["validation_interval"]),
        "evaluation_horizon": int(compact["evaluation_horizon"]),
        "burn_in": int(compact["burn_in"]),
        "optimizer": str(config["optimizer"]),
        "learning_rate": float(config["learning_rate"]),
        "product_generation": config["product_generation"],
        "demand_distribution": str(config["demand_distribution"]),
        "include_terminal_salvage": bool(benchmark["include_terminal_salvage"]),
        "initial_inventory_mode": str(benchmark["initial_inventory_mode"]),
        "training_demand_mode": str(benchmark["training_demand_mode"]),
        "normalize_observations": bool(benchmark["normalize_observations"]),
        "normalize_reward_loss": bool(benchmark["normalize_reward_loss"]),
        "enforce_action_cap": bool(benchmark["enforce_action_cap"]),
        "scale_action_by_history_mean": bool(benchmark["scale_action_by_history_mean"]),
        "imitation_epochs": int(compact["imitation_epochs"]),
    }


def _selected_epoch(result: DirectBackpropResult) -> int:
    if not result.validation_rewards or not result.validation_epochs:
        return len(result.training_losses)
    index = max(range(len(result.validation_rewards)), key=result.validation_rewards.__getitem__)
    return int(result.validation_epochs[index])


def _raw_fcnn_comparison(
    raw_dir: Path,
    monokan: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(raw_dir / "operational_metrics.csv")
    raw = raw.loc[raw["policy_name"] == "PaperFCNN"].copy()
    keys = monokan[["product_id", "evaluation_seed", "demand_path_id", "demand_path_source"]]
    raw = raw.merge(keys, on=["product_id", "evaluation_seed"], validate="one_to_one")
    raw["run_seed"] = seed
    raw["model_seed"] = seed
    raw["product_generation_seed"] = 2025
    comparison = pd.concat([raw, monokan], ignore_index=True, sort=False)
    paired = raw[["product_id", "evaluation_seed", "mean_reward_per_period"]].merge(
        monokan[["product_id", "evaluation_seed", "mean_reward_per_period"]],
        on=["product_id", "evaluation_seed"],
        how="outer",
        suffixes=("_raw_fcnn", "_raw_monokan"),
        validate="one_to_one",
        indicator=True,
    )
    if not (paired["_merge"] == "both").all():
        raise ValueError("raw FCNN and raw MonoKAN evaluations are not fully paired")
    paired = paired.drop(columns="_merge")
    paired["raw_monokan_minus_raw_fcnn"] = (
        paired["mean_reward_per_period_raw_monokan"] - paired["mean_reward_per_period_raw_fcnn"]
    )
    paired["run_seed"] = seed
    return comparison, paired


def _summary(
    comparison: pd.DataFrame,
    result: DirectBackpropResult,
    selected_epoch: int,
    selected_checkpoint: Path,
    final_checkpoint: Path,
) -> pd.DataFrame:
    summary = (
        comparison.groupby("policy_name", as_index=False)
        .agg(
            mean_total_reward=("mean_total_reward", "mean"),
            mean_reward_per_period=("mean_reward_per_period", "mean"),
            product_reward_std=("mean_reward_per_period", "std"),
            products=("product_id", "count"),
            service_level=("service_level", "mean"),
            lost_sales_rate=("lost_sales_rate", "mean"),
            mean_order_quantity=("mean_order_quantity", "mean"),
            mean_inventory=("mean_inventory", "mean"),
        )
    )
    summary["product_reward_se"] = summary["product_reward_std"] / summary["products"].map(math.sqrt)
    learned = summary["policy_name"] == POLICY_NAME
    summary.loc[learned, "training_seconds"] = result.elapsed_seconds
    summary.loc[learned, "selected_checkpoint_epoch"] = selected_epoch
    summary.loc[learned, "final_epoch"] = len(result.training_losses)
    summary.loc[learned, "selected_checkpoint"] = str(selected_checkpoint)
    summary.loc[learned, "final_checkpoint"] = str(final_checkpoint)
    return summary


def _validate_raw_fcnn_result(raw_dir: Path, seed: int, settings: dict[str, Any]) -> None:
    for filename in ("metadata.json", "summary.csv", "operational_metrics.csv", "training_log.csv"):
        if not (raw_dir / filename).exists():
            raise FileNotFoundError(f"missing validated raw FCNN artifact: {raw_dir / filename}")
    metadata = json.loads((raw_dir / "metadata.json").read_text(encoding="utf-8"))
    if int(metadata["settings"]["seed"]) != seed:
        raise ValueError("raw FCNN result seed does not match requested MonoKAN seed")
    for key in (
        "training_products",
        "validation_products",
        "batch_size",
        "training_horizon",
        "reward_epochs",
        "validation_interval",
        "evaluation_horizon",
        "burn_in",
    ):
        if int(metadata["settings"][key]) != int(settings[key]):
            raise ValueError(f"raw FCNN result differs on {key}")


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Metal Performance Shaders"
    return "CPU"


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
