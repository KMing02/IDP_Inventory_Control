"""Run controlled MONOkan residual-order-up-to architecture diagnostics."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from inventory_control.policies.neural.registry import POLICY_SPECS_BY_NAME


def _parse_models(value: str) -> list[str]:
    models = [name.strip() for name in value.split(",") if name.strip()]
    if not models:
        raise ValueError("At least one model is required")
    unknown = [name for name in models if name not in POLICY_SPECS_BY_NAME]
    if unknown:
        raise ValueError(f"unknown model names: {unknown}")
    return models


def _set_compact_setting(config: dict[str, Any], key: str, value: Any) -> None:
    config["idp_benchmark"][key] = value
    config["idp_benchmark"]["compact"][key] = value


def _variant_config(base: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config.setdefault("monokan", {}).update(variant.get("monokan", {}))
    for key, value in variant.get("benchmark", {}).items():
        _set_compact_setting(config, key, value)
    return config


def _scale_width_variants() -> list[dict[str, Any]]:
    """Return a focused residual-scale and bottleneck-width calibration grid."""
    variants = []
    for hidden_units in (4, 8):
        for residual_scale in (0.08, 0.10, 0.12, 0.15, 0.18, 0.20):
            variants.append(
                {
                    "name": f"h{hidden_units}_scale{str(residual_scale).replace('.', '')}",
                    "monokan": {
                        "hidden_units": hidden_units,
                        "residual_scale": residual_scale,
                    },
                    "benchmark": {
                        "spline_l2_weight": 0.0,
                        "scale_l2_weight": 0.0,
                    },
                }
            )
    return variants


def _scale_refine_variants() -> list[dict[str, Any]]:
    """Return a narrower grid around the best observed practical MONOkan scale."""
    variants = []
    for residual_scale in (0.03, 0.05, 0.06, 0.07, 0.08, 0.09):
        variants.append(
            {
                "name": f"h4_scale{str(residual_scale).replace('.', '')}",
                "monokan": {
                    "hidden_units": 4,
                    "residual_scale": residual_scale,
                },
                "benchmark": {
                    "spline_l2_weight": 0.0,
                    "scale_l2_weight": 0.0,
                },
            }
        )
    return variants


def _variants(preset: str) -> list[dict[str, Any]]:
    if preset == "scale_width":
        return _scale_width_variants()
    if preset == "scale_refine":
        return _scale_refine_variants()
    raise ValueError(f"Unsupported sweep preset: {preset}")


def _read_policy_rows(summary_path: Path, variant_name: str, selected_models: list[str]) -> list[dict[str, Any]]:
    summary = pd.read_csv(summary_path)
    rows: list[dict[str, Any]] = []
    for policy in [*selected_models, "HistoricalDemandBaseStock", "NonOmniscientOptimal", "OmniscientOptimal"]:
        selected = summary.loc[summary.policy_name == policy]
        if selected.empty:
            continue
        row = selected.iloc[0]
        rows.append(
            {
                "variant": variant_name,
                "policy": policy,
                "reward_per_period": float(row.mean_reward_per_period),
                "order_up_to_mae": float(row.order_up_to_mae),
                "cap_hit_rate": float(row.cap_hit_rate),
                "mean_inventory": float(row.mean_inventory),
                "service_level": float(row.service_level),
                "training_seconds": float(row.training_seconds) if "training_seconds" in row and pd.notna(row.training_seconds) else None,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/paper_lost_sales_factorial_small.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/monokan_optimization_sweep"))
    parser.add_argument("--models", default="MonoKANResidualOrderUpTo")
    parser.add_argument("--preset", choices=["scale_width", "scale_refine"], default="scale_refine")
    parser.add_argument("--max-variants", type=int, help="Run only the first N variants for smoke tests.")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    models = _parse_models(args.models)
    base = yaml.safe_load(args.config.read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    variants = _variants(args.preset)
    if args.max_variants is not None:
        if args.max_variants <= 0:
            raise ValueError("--max-variants must be positive")
        variants = variants[: int(args.max_variants)]
    metadata = {
        "config": str(args.config),
        "output_root": str(args.output_root),
        "profile": "compact",
        "preset": args.preset,
        "models": models,
        "variants": [variant["name"] for variant in variants],
        "max_variants": args.max_variants,
        "scenario": "monokan_residual_parameter_sweep",
        "split": "validation",
        "uses_final_test": False,
    }
    (args.output_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for variant in variants:
        name = variant["name"]
        config = _variant_config(base, variant)
        config_path = args.output_root / f"{name}.yaml"
        variant_output = args.output_root / name
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        command = [
            sys.executable,
            "scripts/run_idp_benchmark.py",
            "--mode",
            "structure-ls",
            "--profile",
            "compact",
            "--config",
            str(config_path),
            "--models",
            ",".join(models),
            "--output-dir",
            str(variant_output),
        ]
        if args.progress:
            command.append("--progress")
        print(f"=== {name} ===", flush=True)
        subprocess.run(command, check=True)
        rows.extend(_read_policy_rows(variant_output / "summary.csv", name, models))

    aggregate = pd.DataFrame(rows)
    aggregate_path = args.output_root / "aggregate_summary.csv"
    aggregate.to_csv(aggregate_path, index=False)
    print(aggregate.sort_values(["variant", "reward_per_period"], ascending=[True, False]).to_string(index=False))
    print(f"saved_aggregate={aggregate_path}")


if __name__ == "__main__":
    main()
