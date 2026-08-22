"""Matched rolling/static direct-target experiment orchestration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from inventory_control.base_stock_output_experiment import run_base_stock_output_experiment
from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.policies.neural.registry import POLICY_SPECS_BY_NAME
from inventory_control.products import PaperLostSalesProductGenerator


ROLLING_CONFIG = Path("configs/paper_lost_sales_rolling_direct_target_5k.yaml")
STATIC_CONFIG = Path("configs/paper_lost_sales_static_direct_target_5k.yaml")
RAW_CONFIG = Path("configs/paper_lost_sales_raw_fcnn_replication_5k.yaml")
DEV_ROLLING_CONFIG = Path("configs/paper_lost_sales_rolling_direct_target_dev_1k.yaml")
DEV_STATIC_CONFIG = Path("configs/paper_lost_sales_static_direct_target_dev_1k.yaml")
ROLLING_MODELS = ("PaperFCNNDirectTargetStock", "MonoKANDirectTargetStockDeepGrid7")
STATIC_MODELS = ("MonoKANDirectTargetStockDeepGrid7",)
BASELINE_NAMES = ("HistoricalDemandBaseStock", "NonOmniscientOptimal", "OmniscientOptimal")

POLICY_LABELS = {
    "PaperFCNN": "PaperStyleRollingRawActionFCNN",
    "PaperFCNNDirectTargetStock": "RollingPaperFCNNDirectTargetStock",
    "rolling:MonoKANDirectTargetStockDeepGrid7": "RollingMonoKANDirectTargetStockDeepGrid7",
    "static:MonoKANDirectTargetStockDeepGrid7": "StaticMonoKANDirectTargetStockDeepGrid7",
}

PAIRING_KEYS = (
    "run_seed",
    "product_generation_seed",
    "evaluation_seed",
    "demand_path_id",
    "product_id",
)


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit or automatic torch device without silent fallback."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false")
    return device


def frozen_protocol_signature(config_path: str | Path) -> dict[str, Any]:
    """Return fields that must match across the paper-aligned runs."""
    config = load_paper_lost_sales_config(config_path)
    compact = config["idp_benchmark"]["compact"]
    return {
        "history_length": int(config["history_length_H"]),
        "training_products": int(compact["training_products"]),
        "validation_products": int(compact["validation_products"]),
        "batch_size": int(compact["batch_size"]),
        "training_horizon": int(compact["training_horizon"]),
        "reward_epochs": int(compact["reward_epochs"]),
        "evaluation_horizon": int(compact["evaluation_horizon"]),
        "burn_in": int(compact["burn_in"]),
        "learning_rate": float(config["learning_rate"]),
        "optimizer": str(config["optimizer"]),
        "product_generation": config["product_generation"],
        "demand_distribution": config["demand_distribution"],
        "terminal_salvage": bool(config["idp_benchmark"]["include_terminal_salvage"]),
        "initial_inventory_mode": str(config["idp_benchmark"]["initial_inventory_mode"]),
        "training_demand_mode": str(config["idp_benchmark"]["training_demand_mode"]),
        "normalize_observations": bool(config["idp_benchmark"]["normalize_observations"]),
        "normalize_reward_loss": bool(config["idp_benchmark"]["normalize_reward_loss"]),
        "enforce_action_cap": bool(config["idp_benchmark"]["enforce_action_cap"]),
    }


def validate_frozen_protocol(
    rolling_config: str | Path = ROLLING_CONFIG,
    static_config: str | Path = STATIC_CONFIG,
    raw_config: str | Path = RAW_CONFIG,
) -> dict[str, Any]:
    """Fail if the rolling, static, and reusable raw-FCNN protocols diverge."""
    rolling = load_paper_lost_sales_config(rolling_config)
    static = load_paper_lost_sales_config(static_config)
    signatures = {
        "rolling": frozen_protocol_signature(rolling_config),
        "static": frozen_protocol_signature(static_config),
        "raw": frozen_protocol_signature(raw_config),
    }
    if not (signatures["rolling"] == signatures["static"] == signatures["raw"]):
        raise ValueError(f"matched protocol mismatch: {signatures}")
    expected = {
        "history_length": 32,
        "training_products": 5_000,
        "validation_products": 500,
        "batch_size": 2_500,
        "training_horizon": 100,
        "reward_epochs": 1_000,
        "evaluation_horizon": 520,
        "burn_in": 20,
        "enforce_action_cap": False,
        "terminal_salvage": True,
        "initial_inventory_mode": "last_historical_demand",
        "training_demand_mode": "fixed_future",
        "normalize_observations": False,
        "normalize_reward_loss": False,
    }
    mismatches = {
        key: (signatures["rolling"].get(key), value)
        for key, value in expected.items()
        if signatures["rolling"].get(key) != value
    }
    if mismatches:
        raise ValueError(f"frozen protocol values changed: {mismatches}")
    if rolling.get("paper_history_feature_mode") != "raw_lags":
        raise ValueError("rolling direct-target policy must use raw_lags")
    if bool(rolling.get("paper_include_newsvendor_feature", False)):
        raise ValueError("newsvendor target must not be an input feature")
    if not bool(rolling["monokan"].get("roll_history")) or bool(static["monokan"].get("roll_history")):
        raise ValueError("rolling/static MonoKAN configs must differ in rolling-history behavior")
    return signatures


def write_benchmark_config(
    destination: str | Path,
    epochs: int,
    source: str | Path = ROLLING_CONFIG,
) -> Path:
    """Write a full-shape timing config changing only the epoch count."""
    if not 1 <= epochs <= 5:
        raise ValueError("benchmark epochs must be between 1 and 5")
    config = load_paper_lost_sales_config(source)
    config["epochs"] = int(epochs)
    config["idp_benchmark"]["compact"]["reward_epochs"] = int(epochs)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def run_full_shape_benchmark(
    output_dir: str | Path,
    epochs: int,
    device: str,
    activation_checkpointing: bool,
    seed: int = 2025,
) -> dict[str, Any]:
    """Run 1-5 real full-shape epochs and persist timing/memory metadata."""
    validate_frozen_protocol()
    chosen_device = resolve_device(device)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    config_path = write_benchmark_config(destination / "benchmark_resolved_config.yaml", epochs)
    if chosen_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(chosen_device)
    artifacts = run_base_stock_output_experiment(
        config_path=config_path,
        output_dir=destination / "run",
        models=("MonoKANDirectTargetStockDeepGrid7",),
        device=str(chosen_device),
        seed=seed,
        activation_checkpointing=activation_checkpointing,
    )
    learned = artifacts.summary.loc[
        artifacts.summary["policy_name"] == "MonoKANDirectTargetStockDeepGrid7"
    ].iloc[0]
    training_seconds = float(learned["training_seconds"])
    seconds_per_epoch = training_seconds / epochs
    report: dict[str, Any] = {
        "device": str(chosen_device),
        "device_model": _device_name(chosen_device),
        "activation_checkpointing": bool(activation_checkpointing),
        "epochs_measured": epochs,
        "training_seconds": training_seconds,
        "seconds_per_epoch": seconds_per_epoch,
        "estimated_training_hours_1000_epochs": seconds_per_epoch * 1_000 / 3_600,
        "estimate_is_extrapolation": True,
        "config_path": str(config_path),
        "output_dir": str(destination / "run"),
    }
    if chosen_device.type == "cuda":
        report.update(
            {
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(chosen_device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(chosen_device)),
            }
        )
    elif chosen_device.type == "mps":
        report["mps_current_allocated_bytes"] = int(torch.mps.current_allocated_memory())
    (destination / "benchmark.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def run_matched_seed(
    seed: int,
    output_dir: str | Path,
    device: str,
    activation_checkpointing: bool,
    raw_result_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Run one matched training seed and save product-level paired outputs."""
    signatures = validate_frozen_protocol()
    chosen_device = resolve_device(device)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(raw_result_dir or f"outputs/paper_raw_fcnn_replication_5k/seed_{seed}")
    _validate_raw_result(raw_dir, seed)

    rolling = run_base_stock_output_experiment(
        config_path=ROLLING_CONFIG,
        output_dir=destination / "rolling",
        models=ROLLING_MODELS,
        device=str(chosen_device),
        seed=seed,
        activation_checkpointing=activation_checkpointing,
    )
    static = run_base_stock_output_experiment(
        config_path=STATIC_CONFIG,
        output_dir=destination / "static",
        models=STATIC_MODELS,
        device=str(chosen_device),
        seed=seed,
        activation_checkpointing=activation_checkpointing,
    )
    product_metrics = _combine_product_metrics(seed, raw_dir, rolling.product_metrics, static.product_metrics)
    summary = _combined_summary(product_metrics, raw_dir, rolling.summary, static.summary, destination)
    paired = _paired_monokan(product_metrics)

    product_path = destination / "matched_product_metrics.csv"
    summary_path = destination / "matched_summary.csv"
    paired_path = destination / "rolling_vs_static_monokan_paired.csv"
    product_metrics.to_csv(product_path, index=False)
    summary.to_csv(summary_path, index=False)
    paired.to_csv(paired_path, index=False)
    metadata = {
        "seed": seed,
        "device": str(chosen_device),
        "device_model": _device_name(chosen_device),
        "activation_checkpointing": bool(activation_checkpointing),
        "frozen_protocol": signatures["rolling"],
        "rolling_config": str(ROLLING_CONFIG),
        "static_config": str(STATIC_CONFIG),
        "reused_raw_config": str(RAW_CONFIG),
        "reused_raw_result": str(raw_dir),
        "rolling_models": list(ROLLING_MODELS),
        "static_models": list(STATIC_MODELS),
        "pairing_keys": list(PAIRING_KEYS),
        "paired_difference": "rolling_reward_per_period - static_reward_per_period",
    }
    metadata_path = destination / "matched_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "summary": summary_path,
        "product_metrics": product_path,
        "paired": paired_path,
        "metadata": metadata_path,
    }


def validate_small_development_protocol(
    rolling_config: str | Path = DEV_ROLLING_CONFIG,
    static_config: str | Path = DEV_STATIC_CONFIG,
) -> dict[str, Any]:
    """Validate the one-seed development protocol and rolling/static isolation."""
    rolling = load_paper_lost_sales_config(rolling_config)
    static = load_paper_lost_sales_config(static_config)
    rolling_copy = json.loads(json.dumps(rolling))
    static_copy = json.loads(json.dumps(static))
    for config in (rolling_copy, static_copy):
        config["fcnn"].pop("roll_history", None)
        config["monokan"].pop("roll_history", None)
    if rolling_copy != static_copy:
        raise ValueError("development rolling/static configs differ beyond roll_history")
    compact = rolling["idp_benchmark"]["compact"]
    expected = {
        "training_products": 1_000,
        "validation_products": 300,
        "batch_size": 512,
        "training_horizon": 100,
        "reward_epochs": 200,
        "validation_interval": 25,
        "evaluation_horizon": 520,
        "burn_in": 20,
    }
    mismatch = {key: (compact.get(key), value) for key, value in expected.items() if compact.get(key) != value}
    if mismatch:
        raise ValueError(f"development protocol changed: {mismatch}")
    if rolling.get("paper_include_newsvendor_feature", False):
        raise ValueError("development policies must not receive a newsvendor feature")
    benchmark = rolling["idp_benchmark"]
    if benchmark["initial_inventory_mode"] != "last_historical_demand":
        raise ValueError("training must sample I0 uniformly against the last observed demand")
    if benchmark["enforce_action_cap"]:
        raise ValueError("development direct-target policies must not use an action cap")
    return {
        **expected,
        "history_length": int(rolling["history_length_H"]),
        "learning_rate": float(rolling["learning_rate"]),
        "optimizer": str(rolling["optimizer"]),
        "training_initial_inventory": "Uniform(0, 2 * last_historical_demand)",
        "rolling": True,
        "static": False,
    }


def run_small_development_benchmark(
    output_dir: str | Path,
    device: str = "mps",
    seed: int = 2025,
) -> dict[str, Any]:
    """Run one real 1k/512 rolling DeepGrid7 training epoch."""
    protocol = validate_small_development_protocol()
    chosen_device = resolve_device(device)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    config = load_paper_lost_sales_config(DEV_ROLLING_CONFIG)
    config["epochs"] = 1
    config["idp_benchmark"]["compact"]["reward_epochs"] = 1
    config["idp_benchmark"]["compact"]["validation_interval"] = 1
    benchmark_config = destination / "benchmark_config.yaml"
    benchmark_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    artifacts = run_base_stock_output_experiment(
        config_path=benchmark_config,
        output_dir=destination / "run",
        models=("MonoKANDirectTargetStockDeepGrid7",),
        device=str(chosen_device),
        seed=seed,
        activation_checkpointing=True,
        shared_training_seed=True,
    )
    row = artifacts.summary.loc[
        artifacts.summary["policy_name"] == "MonoKANDirectTargetStockDeepGrid7"
    ].iloc[0]
    report = {
        "device": str(chosen_device),
        "device_model": _device_name(chosen_device),
        "seconds_per_epoch": float(row["training_seconds"]),
        "activation_checkpointing": True,
        "protocol": protocol,
        "output_dir": str(destination / "run"),
    }
    if chosen_device.type == "mps":
        report["mps_current_allocated_bytes"] = int(torch.mps.current_allocated_memory())
        if hasattr(torch.mps, "driver_allocated_memory"):
            report["mps_driver_allocated_bytes"] = int(torch.mps.driver_allocated_memory())
    (destination / "benchmark.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def run_small_matched_development(
    output_dir: str | Path,
    device: str = "mps",
    seed: int = 2025,
) -> dict[str, Path]:
    """Run the frozen one-seed 200-epoch rolling/static development comparison."""
    protocol = validate_small_development_protocol()
    chosen_device = resolve_device(device)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rolling = run_base_stock_output_experiment(
        config_path=DEV_ROLLING_CONFIG,
        output_dir=destination / "rolling",
        models=ROLLING_MODELS,
        device=str(chosen_device),
        seed=seed,
        activation_checkpointing=True,
        shared_training_seed=True,
    )
    static = run_base_stock_output_experiment(
        config_path=DEV_STATIC_CONFIG,
        output_dir=destination / "static",
        models=STATIC_MODELS,
        device=str(chosen_device),
        seed=seed,
        activation_checkpointing=True,
        shared_training_seed=True,
    )
    products = _development_product_metrics(rolling.product_metrics, static.product_metrics)
    paired = _paired_monokan(products)
    summary = _development_summary(products, rolling.summary, static.summary, destination)
    training_log = _development_training_log(rolling.training_log, static.training_log)

    summary_path = destination / "summary.csv"
    product_path = destination / "product_metrics.csv"
    paired_path = destination / "rolling_vs_static_monokan_paired.csv"
    log_path = destination / "training_log.csv"
    summary.to_csv(summary_path, index=False)
    products.to_csv(product_path, index=False)
    paired.to_csv(paired_path, index=False)
    training_log.to_csv(log_path, index=False)
    evaluation_products = PaperLostSalesProductGenerator(load_paper_lost_sales_config(DEV_ROLLING_CONFIG)).generate(
        protocol["validation_products"], split="validation"
    )
    metadata = {
        "evidence_label": "small matched development evidence, not final paper-scale results",
        "seed": seed,
        "device": str(chosen_device),
        "device_model": _device_name(chosen_device),
        "protocol": protocol,
        "training_seed": seed,
        "rolling_monokan_model_seed": seed + POLICY_SPECS_BY_NAME["MonoKANDirectTargetStockDeepGrid7"].seed_offset,
        "static_monokan_model_seed": seed + POLICY_SPECS_BY_NAME["MonoKANDirectTargetStockDeepGrid7"].seed_offset,
        "rolling_static_initialization_matched": True,
        "rolling_static_training_trajectories_matched": True,
        "evaluation_initial_inventory_min": min(float(product.initial_inventory) for product in evaluation_products),
        "evaluation_initial_inventory_max": max(float(product.initial_inventory) for product in evaluation_products),
        "checkpoint_selection": "maximum validation mean total reward every 25 epochs",
        "early_stopping": False,
    }
    metadata_path = destination / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "summary": summary_path,
        "product_metrics": product_path,
        "paired": paired_path,
        "training_log": log_path,
        "metadata": metadata_path,
    }


def _development_product_metrics(rolling: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    rolling = rolling.copy()
    rolling["source_policy_name"] = rolling["policy_name"]
    rolling["policy_name"] = rolling["policy_name"].map(
        lambda name: POLICY_LABELS.get(f"rolling:{name}", POLICY_LABELS.get(name, name))
    )
    static = static.loc[static["policy_name"] == "MonoKANDirectTargetStockDeepGrid7"].copy()
    static["source_policy_name"] = static["policy_name"]
    static["policy_name"] = POLICY_LABELS["static:MonoKANDirectTargetStockDeepGrid7"]
    combined = pd.concat([rolling, static], ignore_index=True, sort=False)
    if combined.duplicated([*PAIRING_KEYS, "policy_name"]).any():
        raise ValueError("development output contains duplicate product-policy rows")
    return combined


def _development_summary(
    products: pd.DataFrame,
    rolling: pd.DataFrame,
    static: pd.DataFrame,
    destination: Path,
) -> pd.DataFrame:
    summary = (
        products.groupby("policy_name", sort=False)
        .agg(
            mean_reward_per_period=("mean_reward_per_period", "mean"),
            product_reward_std=("mean_reward_per_period", "std"),
            products=("product_id", "count"),
            service_level=("service_level", "mean"),
            lost_sales_rate=("lost_sales_rate", "mean"),
            mean_order_quantity=("mean_order_quantity", "mean"),
            mean_inventory=("mean_inventory", "mean"),
        )
        .reset_index()
    )
    source_rows = []
    sources = (
        (rolling, "PaperFCNNDirectTargetStock", POLICY_LABELS["PaperFCNNDirectTargetStock"], destination / "rolling"),
        (rolling, "MonoKANDirectTargetStockDeepGrid7", POLICY_LABELS["rolling:MonoKANDirectTargetStockDeepGrid7"], destination / "rolling"),
        (static, "MonoKANDirectTargetStockDeepGrid7", POLICY_LABELS["static:MonoKANDirectTargetStockDeepGrid7"], destination / "static"),
    )
    for frame, source_name, label, source_dir in sources:
        row = frame.loc[frame["policy_name"] == source_name].iloc[0]
        source_rows.append(
            {
                "policy_name": label,
                "selected_checkpoint_epoch": int(row["selected_checkpoint_epoch"]),
                "final_checkpoint_epoch": int(row["final_checkpoint_epoch"]),
                "training_seconds": float(row["training_seconds"]),
                "selected_checkpoint": str(source_dir / row["checkpoint_filename"]),
                "final_checkpoint": str(source_dir / row["final_checkpoint_filename"]),
            }
        )
    return summary.merge(pd.DataFrame(source_rows), on="policy_name", how="left")


def _development_training_log(rolling: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    rolling = rolling.copy()
    rolling["policy_name"] = rolling["policy_name"].map(
        lambda name: POLICY_LABELS.get(f"rolling:{name}", POLICY_LABELS.get(name, name))
    )
    static = static.copy()
    static["policy_name"] = POLICY_LABELS["static:MonoKANDirectTargetStockDeepGrid7"]
    return pd.concat([rolling, static], ignore_index=True, sort=False)


def _validate_raw_result(raw_dir: Path, seed: int) -> None:
    required = (raw_dir / "operational_metrics.csv", raw_dir / "summary.csv", raw_dir / "metadata.json")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"raw-action FCNN artifacts are missing: {missing}")
    metadata = json.loads((raw_dir / "metadata.json").read_text(encoding="utf-8"))
    settings = metadata.get("settings", {})
    expected = frozen_protocol_signature(RAW_CONFIG)
    for key in (
        "training_products",
        "validation_products",
        "batch_size",
        "training_horizon",
        "reward_epochs",
        "evaluation_horizon",
        "burn_in",
        "initial_inventory_mode",
        "include_terminal_salvage",
        "normalize_observations",
        "normalize_reward_loss",
        "training_demand_mode",
    ):
        metadata_key = "include_terminal_salvage" if key == "terminal_salvage" else key
        expected_key = "terminal_salvage" if key == "include_terminal_salvage" else key
        if settings.get(metadata_key) != expected.get(expected_key):
            raise ValueError(f"raw result protocol mismatch for {key}: {settings.get(metadata_key)!r}")
    if int(settings.get("seed", -1)) != seed:
        raise ValueError(f"raw result seed mismatch: expected {seed}, got {settings.get('seed')}")
    rows = pd.read_csv(raw_dir / "operational_metrics.csv")
    raw = rows.loc[rows["policy_name"] == "PaperFCNN"]
    if len(raw) != expected["validation_products"]:
        raise ValueError(f"expected {expected['validation_products']} raw FCNN product rows, got {len(raw)}")
    if raw.duplicated(["product_id", "evaluation_seed"]).any():
        raise ValueError("raw FCNN output contains duplicate product/evaluation pairs")


def _combine_product_metrics(
    seed: int,
    raw_dir: Path,
    rolling: pd.DataFrame,
    static: pd.DataFrame,
) -> pd.DataFrame:
    raw_all = pd.read_csv(raw_dir / "operational_metrics.csv")
    _assert_baselines_match(raw_all, rolling, "raw-action")
    _assert_baselines_match(static, rolling, "static direct-target")
    reference = rolling.loc[rolling["policy_name"] == "HistoricalDemandBaseStock", [
        "product_id",
        "evaluation_seed",
        "demand_path_id",
        "demand_path_source",
        "product_generation_seed",
    ]].drop_duplicates()
    raw = raw_all.loc[raw_all["policy_name"] == "PaperFCNN"].merge(
        reference,
        on=["product_id", "evaluation_seed"],
        how="inner",
        validate="one_to_one",
    )
    if len(raw) != len(reference):
        raise ValueError("raw/direct evaluations do not contain identical product pairs")
    raw["run_seed"] = seed
    raw["model_seed"] = seed
    raw["source_policy_name"] = raw["policy_name"]
    raw["policy_name"] = POLICY_LABELS["PaperFCNN"]

    rolling = rolling.copy()
    rolling["source_policy_name"] = rolling["policy_name"]
    rolling["policy_name"] = rolling["policy_name"].map(
        lambda name: POLICY_LABELS.get(f"rolling:{name}", POLICY_LABELS.get(name, name))
    )
    static = static.loc[static["policy_name"] == "MonoKANDirectTargetStockDeepGrid7"].copy()
    static["source_policy_name"] = static["policy_name"]
    static["policy_name"] = POLICY_LABELS["static:MonoKANDirectTargetStockDeepGrid7"]
    combined = pd.concat([raw, rolling, static], ignore_index=True, sort=False)
    if combined.duplicated([*PAIRING_KEYS, "policy_name"]).any():
        raise ValueError("matched output contains duplicate policy/product pairs")
    return combined


def _assert_baselines_match(candidate: pd.DataFrame, reference: pd.DataFrame, label: str) -> None:
    """Use fixed baselines to verify identical products and evaluation paths."""
    key = ["policy_name", "product_id", "evaluation_seed"]
    value = "mean_reward_per_period"
    candidate_rows = candidate.loc[candidate["policy_name"].isin(BASELINE_NAMES), [*key, value]]
    reference_rows = reference.loc[reference["policy_name"].isin(BASELINE_NAMES), [*key, value]]
    joined = candidate_rows.merge(
        reference_rows,
        on=key,
        how="outer",
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
        indicator=True,
    )
    if not (joined["_merge"] == "both").all():
        raise ValueError(f"{label} baseline rows do not match the rolling evaluation")
    difference = (joined[f"{value}_candidate"] - joined[f"{value}_reference"]).abs()
    if not (difference <= 1e-6).all():
        raise ValueError(f"{label} baseline rewards differ from the rolling evaluation")


def _paired_monokan(product_metrics: pd.DataFrame) -> pd.DataFrame:
    rolling_name = POLICY_LABELS["rolling:MonoKANDirectTargetStockDeepGrid7"]
    static_name = POLICY_LABELS["static:MonoKANDirectTargetStockDeepGrid7"]
    columns = [*PAIRING_KEYS, "mean_reward_per_period"]
    rolling = product_metrics.loc[product_metrics["policy_name"] == rolling_name, columns].rename(
        columns={"mean_reward_per_period": "rolling_reward_per_period"}
    )
    static = product_metrics.loc[product_metrics["policy_name"] == static_name, columns].rename(
        columns={"mean_reward_per_period": "static_reward_per_period"}
    )
    paired = rolling.merge(static, on=list(PAIRING_KEYS), how="outer", validate="one_to_one", indicator=True)
    if not (paired["_merge"] == "both").all():
        raise ValueError("rolling/static MonoKAN evaluation has missing product pairs")
    paired = paired.drop(columns="_merge")
    paired["rolling_minus_static_reward_per_period"] = (
        paired["rolling_reward_per_period"] - paired["static_reward_per_period"]
    )
    return paired


def _combined_summary(
    products: pd.DataFrame,
    raw_dir: Path,
    rolling: pd.DataFrame,
    static: pd.DataFrame,
    destination: Path,
) -> pd.DataFrame:
    reward = products.groupby("policy_name", sort=False)["mean_reward_per_period"].agg(["mean", "std", "count"])
    reward = reward.rename(columns={"mean": "mean_reward_per_period", "std": "product_reward_std"}).reset_index()
    reward["product_reward_se"] = reward["product_reward_std"] / reward["count"].map(math.sqrt)

    source_rows: list[dict[str, Any]] = []
    raw_summary = pd.read_csv(raw_dir / "summary.csv")
    sources = (
        (rolling, "PaperFCNNDirectTargetStock", POLICY_LABELS["PaperFCNNDirectTargetStock"], destination / "rolling"),
        (
            rolling,
            "MonoKANDirectTargetStockDeepGrid7",
            POLICY_LABELS["rolling:MonoKANDirectTargetStockDeepGrid7"],
            destination / "rolling",
        ),
        (
            static,
            "MonoKANDirectTargetStockDeepGrid7",
            POLICY_LABELS["static:MonoKANDirectTargetStockDeepGrid7"],
            destination / "static",
        ),
    )
    source_rows.append(
        {
            "policy_name": POLICY_LABELS["PaperFCNN"],
            "source_policy_name": "PaperFCNN",
            "selected_checkpoint_epoch": _raw_selected_epoch(raw_dir),
            "final_epoch": 1_000,
            "training_seconds": raw_summary.loc[
                raw_summary["policy_name"] == "PaperFCNN", "training_seconds"
            ].iloc[0],
            "checkpoint_path": str(raw_dir / "models" / "paper_fcnn.pt"),
        }
    )
    for frame, source_name, label, source_dir in sources:
        row = frame.loc[frame["policy_name"] == source_name].iloc[0]
        checkpoint_name = row.get("checkpoint_filename")
        source_rows.append(
            {
                "policy_name": label,
                "source_policy_name": source_name,
                "selected_checkpoint_epoch": row.get("selected_checkpoint_epoch"),
                "final_epoch": 1_000,
                "training_seconds": row.get("training_seconds"),
                "checkpoint_path": str(source_dir / checkpoint_name) if isinstance(checkpoint_name, str) else None,
            }
        )
    for baseline in BASELINE_NAMES:
        source_rows.append(
            {
                "policy_name": baseline,
                "source_policy_name": baseline,
                "selected_checkpoint_epoch": None,
                "final_epoch": None,
                "training_seconds": None,
                "checkpoint_path": None,
            }
        )
    return reward.merge(pd.DataFrame(source_rows), on="policy_name", how="left", validate="one_to_one")


def _raw_selected_epoch(raw_dir: Path) -> int:
    log = pd.read_csv(raw_dir / "training_log.csv")
    fcnn = log.loc[log["policy_name"] == "PaperFCNN"]
    validation = fcnn.dropna(subset=["validation_mean_total_reward"])
    if validation.empty:
        return int(fcnn["epoch"].max())
    best_index = validation["validation_mean_total_reward"].idxmax()
    return int(validation.loc[best_index, "epoch"])


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Metal Performance Shaders"
    return "CPU"


def deep_grid7_checkpoint_filename() -> str:
    """Expose the frozen registry filename for runner regression tests."""
    filename = POLICY_SPECS_BY_NAME["MonoKANDirectTargetStockDeepGrid7"].checkpoint_filename
    assert filename is not None
    return filename
