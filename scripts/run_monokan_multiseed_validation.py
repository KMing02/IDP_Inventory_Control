"""Run compact multi-seed validation for residual FCNN and MONOkan policies."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import pandas as pd

from inventory_control.policies.neural.registry import POLICY_SPECS_BY_NAME, RESIDUAL_COMPARISON_POLICY_NAMES

DEFAULT_MODELS = ",".join(RESIDUAL_COMPARISON_POLICY_NAMES)


def _parse_models(value: str) -> list[str]:
    models = [name.strip() for name in value.split(",") if name.strip()]
    if not models:
        raise ValueError("At least one model is required")
    unknown = [name for name in models if name not in POLICY_SPECS_BY_NAME]
    if unknown:
        raise ValueError(f"unknown model names: {unknown}")
    return models


def _write_seed_config(base_config: dict[str, Any], seed: int, directory: Path, vary_product_seed: bool) -> Path:
    config = copy.deepcopy(base_config)
    config["idp_benchmark"]["seed"] = int(seed)
    if vary_product_seed:
        config["product_generation"]["seed"] = int(seed)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_seed_{seed}.yaml",
        prefix="monokan_multiseed_",
        dir=directory,
        delete=False,
        encoding="utf-8",
    ) as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
        return Path(handle.name)


def _read_summary(summary_path: Path, seed: int) -> list[dict[str, Any]]:
    summary = pd.read_csv(summary_path)
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "seed": seed,
                "policy": row["policy_name"],
                "reward_per_period": float(row["mean_reward_per_period"]),
                "service_level": float(row["service_level"]),
                "lost_sales_rate": float(row["lost_sales_rate"]),
                "mean_inventory": float(row["mean_inventory"]),
                "training_seconds": float(row["training_seconds"]) if "training_seconds" in row and pd.notna(row["training_seconds"]) else None,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/paper_lost_sales_factorial_small.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/monokan_multiseed_validation"))
    parser.add_argument("--seeds", default="2025,2026,2027")
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--full-diagnostics", action="store_true")
    parser.add_argument(
        "--vary-product-seed",
        action="store_true",
        help="Also set product_generation.seed for each seed to test product-split robustness.",
    )
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    models = _parse_models(args.models)

    base_config = yaml.safe_load(args.config.read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": str(args.config),
        "output_root": str(args.output_root),
        "training_seeds": seeds,
        "models": models,
        "profile": "compact",
        "vary_product_seed": bool(args.vary_product_seed),
        "full_diagnostics": bool(args.full_diagnostics),
        "scenario": "compact_multiseed_validation",
        "evaluation_seed_source": "benchmark operational_metrics.csv evaluation_seed column",
    }
    (args.output_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_config = _write_seed_config(base_config, seed, args.output_root, args.vary_product_seed)
        output_dir = args.output_root / f"seed_{seed}"
        command = [
            sys.executable,
            "scripts/run_idp_benchmark.py",
            "--mode",
            "structure-ls",
            "--profile",
            "compact",
            "--config",
            str(seed_config),
            "--models",
            ",".join(models),
            "--output-dir",
            str(output_dir),
        ]
        if args.progress:
            command.append("--progress")
        if not args.full_diagnostics:
            command.append("--fast-diagnostics")
        print(f"=== seed {seed} ===", flush=True)
        try:
            subprocess.run(command, check=True)
        finally:
            seed_config.unlink(missing_ok=True)
        rows.extend(_read_summary(output_dir / "summary.csv", seed))

    per_seed = pd.DataFrame(rows)
    per_seed_path = args.output_root / "per_seed_summary.csv"
    per_seed.to_csv(per_seed_path, index=False)
    aggregate = (
        per_seed.groupby("policy", as_index=False)
        .agg(
            seeds=("seed", "count"),
            mean_reward_per_period=("reward_per_period", "mean"),
            std_reward_per_period=("reward_per_period", "std"),
            mean_service_level=("service_level", "mean"),
            mean_lost_sales_rate=("lost_sales_rate", "mean"),
            mean_inventory=("mean_inventory", "mean"),
            mean_training_seconds=("training_seconds", "mean"),
        )
        .sort_values("mean_reward_per_period", ascending=False)
    )
    aggregate_path = args.output_root / "aggregate_summary.csv"
    aggregate.to_csv(aggregate_path, index=False)
    print(aggregate.to_string(index=False))
    print(f"saved_per_seed={per_seed_path}")
    print(f"saved_aggregate={aggregate_path}")


if __name__ == "__main__":
    main()
