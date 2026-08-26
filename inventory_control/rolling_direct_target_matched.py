"""Final matched rolling direct-target experiment orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from inventory_control.base_stock_output_experiment import run_base_stock_output_experiment
from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.policies.neural.registry import POLICY_SPECS_BY_NAME


ROLLING_CONFIG = Path("configs/paper_lost_sales_rolling_direct_target_5k.yaml")
RAW_CONFIG = Path("configs/paper_lost_sales_raw_fcnn_replication_5k.yaml")
DEV_ROLLING_CONFIG = Path("configs/paper_lost_sales_rolling_direct_target_dev_1k_normalized.yaml")
TEMPORAL_CONFIG = Path("configs/paper_lost_sales_rolling_direct_target_temporal_dev_1k_normalized.yaml")
ROLLING_MODELS = ("PaperFCNNDirectTargetStock", "MonoKANDirectTargetStockDeepGrid7")
BASELINE_NAMES = ("HistoricalDemandBaseStock", "NonOmniscientOptimal", "OmniscientOptimal")

POLICY_LABELS = {
    "PaperFCNN": "PaperStyleRollingRawActionFCNN",
    "PaperFCNNDirectTargetStock": "RollingPaperFCNNDirectTargetStock",
    "MonoKANDirectTargetStockDeepGrid7": "RollingMonoKANDirectTargetStockDeepGrid7",
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


def protocol_signature(config_path: str | Path) -> dict[str, Any]:
    """Return fields that must match across report-facing runs."""
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
        "history_feature_mode": str(config.get("paper_history_feature_mode", "raw_lags")),
    }


def validate_frozen_protocol(
    rolling_config: str | Path = ROLLING_CONFIG,
    raw_config: str | Path = RAW_CONFIG,
) -> dict[str, Any]:
    """Fail if the formal rolling direct-target and raw-FCNN anchor protocols diverge."""
    signatures = {"rolling": protocol_signature(rolling_config), "raw": protocol_signature(raw_config)}
    comparable_keys = set(signatures["rolling"]) - {"history_feature_mode"}
    mismatched = {
        key: (signatures["rolling"][key], signatures["raw"][key])
        for key in sorted(comparable_keys)
        if signatures["rolling"][key] != signatures["raw"][key]
    }
    if mismatched:
        raise ValueError(f"matched protocol mismatch: {mismatched}")
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
        "history_feature_mode": "raw_lags",
    }
    mismatches = {key: (signatures["rolling"].get(key), value) for key, value in expected.items() if signatures["rolling"].get(key) != value}
    if mismatches:
        raise ValueError(f"frozen protocol values changed: {mismatches}")
    rolling = load_paper_lost_sales_config(rolling_config)
    if bool(rolling.get("paper_include_newsvendor_feature", False)):
        raise ValueError("newsvendor target must not be an input feature")
    if not bool(rolling["fcnn"].get("roll_history")) or not bool(rolling["monokan"].get("roll_history")):
        raise ValueError("final target-stock policies must use rolling causal history")
    return signatures


def validate_development_protocol(config_path: str | Path = DEV_ROLLING_CONFIG) -> dict[str, Any]:
    """Validate the normalized 1k/200-epoch protocol behind the final three-seed evidence."""
    signature = protocol_signature(config_path)
    expected = {
        "history_length": 32,
        "training_products": 1_000,
        "validation_products": 300,
        "batch_size": 512,
        "training_horizon": 100,
        "reward_epochs": 200,
        "evaluation_horizon": 520,
        "burn_in": 20,
        "initial_inventory_mode": "last_historical_demand",
        "training_demand_mode": "fixed_future",
        "normalize_observations": True,
        "normalize_reward_loss": False,
        "enforce_action_cap": False,
    }
    mismatches = {key: (signature.get(key), value) for key, value in expected.items() if signature.get(key) != value}
    if mismatches:
        raise ValueError(f"development protocol changed: {mismatches}")
    config = load_paper_lost_sales_config(config_path)
    if not bool(config["fcnn"].get("roll_history")) or not bool(config["monokan"].get("roll_history")):
        raise ValueError("development direct-target policies must use rolling causal history")
    return signature


def validate_temporal_protocol(config_path: str | Path = TEMPORAL_CONFIG) -> dict[str, Any]:
    """Validate the temporal-feature ablation protocol."""
    signature = validate_development_protocol(config_path)
    if signature["history_feature_mode"] != "raw_lags_temporal":
        raise ValueError("temporal ablation must use raw_lags_temporal")
    return signature


def write_benchmark_config(destination: str | Path, epochs: int, source: str | Path = ROLLING_CONFIG) -> Path:
    """Write a timing config changing only the epoch count."""
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
    """Run 1-5 real full-shape epochs and persist timing metadata."""
    validate_frozen_protocol()
    chosen_device = resolve_device(device)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    config_path = write_benchmark_config(destination / "benchmark_resolved_config.yaml", epochs)
    artifacts = run_base_stock_output_experiment(
        config_path=config_path,
        output_dir=destination / "run",
        models=("MonoKANDirectTargetStockDeepGrid7",),
        device=str(chosen_device),
        seed=seed,
        activation_checkpointing=activation_checkpointing,
    )
    learned = artifacts.summary.loc[artifacts.summary["policy_name"] == "MonoKANDirectTargetStockDeepGrid7"].iloc[0]
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
    (destination / "benchmark.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def run_matched_seed(
    seed: int,
    output_dir: str | Path,
    device: str,
    activation_checkpointing: bool,
    raw_result_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Run one formal matched target-stock seed and combine it with the raw FCNN anchor."""
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
        shared_training_seed=True,
    )
    product_metrics = _combine_product_metrics(seed, raw_dir, rolling.product_metrics)
    summary = _combined_summary(product_metrics, raw_dir, rolling.summary, destination)
    paired = _paired_fcnn_monokan(product_metrics)
    product_path = destination / "matched_product_metrics.csv"
    summary_path = destination / "matched_summary.csv"
    paired_path = destination / "target_stock_fcnn_vs_monokan_paired.csv"
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
        "reused_raw_config": str(RAW_CONFIG),
        "reused_raw_result": str(raw_dir),
        "rolling_models": list(ROLLING_MODELS),
        "pairing_keys": list(PAIRING_KEYS),
        "paired_difference": "target_stock_fcnn_reward_per_period - monokan_reward_per_period",
    }
    metadata_path = destination / "matched_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {"summary": summary_path, "product_metrics": product_path, "paired": paired_path, "metadata": metadata_path}


def run_development_seed(
    output_dir: str | Path,
    device: str = "mps",
    seed: int = 2025,
    config_path: str | Path = DEV_ROLLING_CONFIG,
) -> dict[str, Path]:
    """Run one normalized 1k/200-epoch target-stock development seed."""
    protocol = validate_development_protocol(config_path)
    chosen_device = resolve_device(device)
    destination = Path(output_dir)
    artifacts = run_base_stock_output_experiment(
        config_path=config_path,
        output_dir=destination,
        models=ROLLING_MODELS,
        device=str(chosen_device),
        seed=seed,
        activation_checkpointing=True,
        shared_training_seed=True,
    )
    metadata_path = destination / "development_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "evidence_label": "normalized rolling direct-target development evidence",
                "seed": seed,
                "device": str(chosen_device),
                "device_model": _device_name(chosen_device),
                "protocol": protocol,
                "models": list(ROLLING_MODELS),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "summary": artifacts.output_dir / "summary.csv",
        "product_metrics": artifacts.output_dir / "product_metrics.csv",
        "training_log": artifacts.output_dir / "training_log.csv",
        "metadata": metadata_path,
    }


def _validate_raw_result(raw_dir: Path, seed: int) -> None:
    required = (raw_dir / "operational_metrics.csv", raw_dir / "summary.csv", raw_dir / "metadata.json")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"raw-action FCNN artifacts are missing: {missing}")
    metadata = json.loads((raw_dir / "metadata.json").read_text(encoding="utf-8"))
    settings = metadata.get("settings", {})
    expected = protocol_signature(RAW_CONFIG)
    for key in (
        "training_products",
        "validation_products",
        "batch_size",
        "training_horizon",
        "reward_epochs",
        "evaluation_horizon",
        "burn_in",
        "initial_inventory_mode",
        "normalize_observations",
        "normalize_reward_loss",
        "training_demand_mode",
    ):
        if settings.get(key) != expected.get(key):
            raise ValueError(f"raw result protocol mismatch for {key}: {settings.get(key)!r}")
    if bool(settings.get("include_terminal_salvage")) != bool(expected["terminal_salvage"]):
        raise ValueError("raw result terminal-salvage setting differs from the final protocol")
    if int(settings.get("seed", -1)) != seed:
        raise ValueError(f"raw result seed mismatch: expected {seed}, got {settings.get('seed')}")
    rows = pd.read_csv(raw_dir / "operational_metrics.csv")
    raw = rows.loc[rows["policy_name"] == "PaperFCNN"]
    if len(raw) != expected["validation_products"]:
        raise ValueError(f"expected {expected['validation_products']} raw FCNN product rows, got {len(raw)}")
    if raw.duplicated(["product_id", "evaluation_seed"]).any():
        raise ValueError("raw FCNN output contains duplicate product/evaluation pairs")


def _combine_product_metrics(seed: int, raw_dir: Path, rolling: pd.DataFrame) -> pd.DataFrame:
    raw_all = pd.read_csv(raw_dir / "operational_metrics.csv")
    _assert_baselines_match(raw_all, rolling, "raw-action")
    reference = rolling.loc[
        rolling["policy_name"] == "HistoricalDemandBaseStock",
        ["product_id", "evaluation_seed", "demand_path_id", "demand_path_source", "product_generation_seed"],
    ].drop_duplicates()
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
    rolling["policy_name"] = rolling["policy_name"].map(lambda name: POLICY_LABELS.get(name, name))
    combined = pd.concat([raw, rolling], ignore_index=True, sort=False)
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


def _paired_fcnn_monokan(product_metrics: pd.DataFrame) -> pd.DataFrame:
    fcnn_name = POLICY_LABELS["PaperFCNNDirectTargetStock"]
    monokan_name = POLICY_LABELS["MonoKANDirectTargetStockDeepGrid7"]
    columns = [*PAIRING_KEYS, "mean_reward_per_period"]
    fcnn = product_metrics.loc[product_metrics["policy_name"] == fcnn_name, columns].rename(
        columns={"mean_reward_per_period": "target_stock_fcnn_reward_per_period"}
    )
    monokan = product_metrics.loc[product_metrics["policy_name"] == monokan_name, columns].rename(
        columns={"mean_reward_per_period": "monokan_reward_per_period"}
    )
    paired = fcnn.merge(monokan, on=list(PAIRING_KEYS), how="outer", validate="one_to_one", indicator=True)
    if not (paired["_merge"] == "both").all():
        raise ValueError("target-stock FCNN/MonoKAN evaluation has missing product pairs")
    paired = paired.drop(columns="_merge")
    paired["fcnn_minus_monokan_reward_per_period"] = (
        paired["target_stock_fcnn_reward_per_period"] - paired["monokan_reward_per_period"]
    )
    return paired


def _combined_summary(products: pd.DataFrame, raw_dir: Path, rolling: pd.DataFrame, destination: Path) -> pd.DataFrame:
    reward = products.groupby("policy_name", sort=False)["mean_reward_per_period"].agg(["mean", "std", "count"])
    reward = reward.rename(columns={"mean": "mean_reward_per_period", "std": "product_reward_std"}).reset_index()
    raw_summary = pd.read_csv(raw_dir / "summary.csv")
    source_rows: list[dict[str, Any]] = [
        {
            "policy_name": POLICY_LABELS["PaperFCNN"],
            "source_policy_name": "PaperFCNN",
            "selected_checkpoint_epoch": _raw_selected_epoch(raw_dir),
            "final_epoch": 1_000,
            "training_seconds": raw_summary.loc[raw_summary["policy_name"] == "PaperFCNN", "training_seconds"].iloc[0],
            "checkpoint_path": str(raw_dir / "models" / "paper_fcnn.pt"),
        }
    ]
    for source_name in ROLLING_MODELS:
        row = rolling.loc[rolling["policy_name"] == source_name].iloc[0]
        checkpoint_name = row.get("checkpoint_filename")
        source_rows.append(
            {
                "policy_name": POLICY_LABELS[source_name],
                "source_policy_name": source_name,
                "selected_checkpoint_epoch": row.get("selected_checkpoint_epoch"),
                "final_epoch": 1_000,
                "training_seconds": row.get("training_seconds"),
                "checkpoint_path": str(destination / "rolling" / checkpoint_name) if isinstance(checkpoint_name, str) else None,
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
    """Expose the final registry filename for runner regression tests."""
    filename = POLICY_SPECS_BY_NAME["MonoKANDirectTargetStockDeepGrid7"].checkpoint_filename
    assert filename is not None
    return filename
