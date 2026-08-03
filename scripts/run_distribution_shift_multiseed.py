"""Train residual policies across seeds and evaluate distribution-shift robustness."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import yaml

from inventory_control.policies.neural.registry import RESIDUAL_COMPARISON_POLICY_NAMES

DEFAULT_MODELS = ",".join(RESIDUAL_COMPARISON_POLICY_NAMES)
DEFAULT_SHIFTS = "in_distribution,low_cv,high_cv,low_mean_demand,high_mean_demand,high_lost_sale_penalty,high_holding_cost,mixed_stress"


def _write_seed_config(base_config: dict[str, Any], seed: int, directory: Path, vary_product_seed: bool) -> Path:
    config = copy.deepcopy(base_config)
    config["idp_benchmark"]["seed"] = int(seed)
    if vary_product_seed:
        config["product_generation"]["seed"] = int(seed)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_seed_{seed}.yaml",
        prefix="distribution_shift_multiseed_",
        dir=directory,
        delete=False,
        encoding="utf-8",
    ) as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
        return Path(handle.name)


def _run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def _read_seed_shift_summary(path: Path, seed: int) -> pd.DataFrame:
    summary = pd.read_csv(path)
    summary.insert(0, "seed", int(seed))
    return summary


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, samples: int = 1000) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    means = np.empty(samples, dtype=float)
    for index in range(samples):
        means[index] = float(rng.choice(values, size=values.size, replace=True).mean())
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired_reward_differences(
    operational: pd.DataFrame,
    learned_models: set[str],
    baseline: str = "PaperFCNNResidualOrderUpTo",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if operational.empty or baseline not in learned_models:
        return pd.DataFrame(), pd.DataFrame()
    required_columns = {"seed", "shift", "product_id", "policy_name", "mean_reward_per_period"}
    missing_columns = required_columns.difference(operational.columns)
    if missing_columns:
        raise ValueError(f"operational metrics are missing paired-comparison columns: {sorted(missing_columns)}")
    filtered = operational.loc[operational["policy_name"].isin(learned_models)].copy()
    index_columns = ["seed", "shift", "product_id"]
    if "evaluation_seed" in filtered.columns:
        index_columns.append("evaluation_seed")
    key_columns = [*index_columns, "policy_name"]
    duplicates = filtered.loc[filtered.duplicated(key_columns, keep=False), key_columns]
    if not duplicates.empty:
        raise ValueError(f"duplicate product-policy rows prevent paired comparison: {duplicates.head().to_dict('records')}")
    pivot = filtered.pivot(
        index=index_columns,
        columns="policy_name",
        values="mean_reward_per_period",
    )
    if baseline not in pivot.columns:
        raise ValueError(f"baseline policy {baseline!r} is missing from paired operational metrics")
    missing_models = sorted(learned_models.difference(set(pivot.columns)))
    if missing_models:
        raise ValueError(f"paired operational metrics are missing models: {missing_models}")
    rows = []
    for policy_name in sorted(learned_models):
        if policy_name == baseline or policy_name not in pivot.columns:
            continue
        paired = pivot[[baseline, policy_name]].dropna()
        if len(paired) != len(pivot):
            raise ValueError(f"missing baseline/comparator pairs for {policy_name!r}")
        diff = paired[policy_name] - paired[baseline]
        for index, value in diff.items():
            index_values = index if isinstance(index, tuple) else (index,)
            row = {
                column: int(index_values[position]) if column in {"seed", "evaluation_seed"} else str(index_values[position])
                for position, column in enumerate(index_columns)
            }
            row.update(
                {
                    "policy_name": policy_name,
                    "baseline_policy_name": baseline,
                    "reward_diff_vs_baseline": float(value),
                }
            )
            rows.append(row)
    differences = pd.DataFrame(rows)
    if differences.empty:
        return differences, pd.DataFrame()
    rng = np.random.default_rng(0)
    summary_rows = []
    for (shift, policy_name), group in differences.groupby(["shift", "policy_name"], sort=False):
        values = group["reward_diff_vs_baseline"].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_ci(values, rng)
        summary_rows.append(
            {
                "shift": shift,
                "policy_name": policy_name,
                "baseline_policy_name": baseline,
                "paired_products": int(values.size),
                "mean_reward_diff_vs_baseline": float(values.mean()),
                "median_reward_diff_vs_baseline": float(np.median(values)),
                "worst_decile_reward_diff_vs_baseline": float(np.quantile(values, 0.1)),
                "win_rate_vs_baseline": float(np.mean(values > 0.0)),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
            }
        )
    return differences, pd.DataFrame(summary_rows)


def _aggregate_shift_seed(
    summary: pd.DataFrame,
    learned_models: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    learned = summary.loc[summary["policy_name"].isin(learned_models)].copy()
    robustness = (
        learned.groupby("policy_name", as_index=False)
        .agg(
            seed_shift_cases=("shift", "count"),
            seeds=("seed", "nunique"),
            shifts=("shift", "nunique"),
            mean_reward_per_period=("mean_reward_per_period", "mean"),
            std_reward_per_period=("mean_reward_per_period", "std"),
            worst_reward_per_period=("mean_reward_per_period", "min"),
            mean_regret_vs_non_omniscient=("regret_vs_non_omniscient", "mean"),
            worst_regret_vs_non_omniscient=("regret_vs_non_omniscient", "max"),
            mean_regret_vs_omniscient=("regret_vs_omniscient", "mean"),
            mean_service_level=("service_level", "mean"),
            mean_lost_sales_rate=("lost_sales_rate", "mean"),
            mean_order_up_to_mae=("order_up_to_mae", "mean"),
            mean_cap_hit_rate=("cap_hit_rate", "mean"),
            mean_raw_violation_rate=("raw_violation_rate", "mean"),
        )
        .sort_values(["mean_regret_vs_non_omniscient", "mean_reward_per_period"], ascending=[True, False])
    )
    per_shift = (
        learned.groupby(["shift", "policy_name"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_reward_per_period=("mean_reward_per_period", "mean"),
            std_reward_per_period=("mean_reward_per_period", "std"),
            mean_regret_vs_non_omniscient=("regret_vs_non_omniscient", "mean"),
            worst_regret_vs_non_omniscient=("regret_vs_non_omniscient", "max"),
            mean_service_level=("service_level", "mean"),
            mean_lost_sales_rate=("lost_sales_rate", "mean"),
            mean_order_up_to_mae=("order_up_to_mae", "mean"),
            mean_cap_hit_rate=("cap_hit_rate", "mean"),
            mean_raw_violation_rate=("raw_violation_rate", "mean"),
        )
        .sort_values(["shift", "mean_regret_vs_non_omniscient"], ascending=[True, True])
    )
    winners = (
        learned.sort_values(["seed", "shift", "mean_reward_per_period"], ascending=[True, True, False])
        .groupby(["seed", "shift"], as_index=False)
        .first()[["seed", "shift", "policy_name", "mean_reward_per_period", "regret_vs_non_omniscient"]]
    )
    winner_counts = (
        winners.groupby("policy_name", as_index=False)
        .agg(wins=("shift", "count"))
        .sort_values("wins", ascending=False)
    )
    return robustness, per_shift, winner_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paper_lost_sales_fcnn_alignment_medium.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/monokan_distribution_shift_multiseed"))
    parser.add_argument("--seeds", default="2025,2026,2027,2028,2029,2030,2031,2032,2033,2034")
    parser.add_argument("--profile", choices=["compact", "full", "paper_replication"], default="full")
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--shifts", default=DEFAULT_SHIFTS)
    parser.add_argument("--eval-products", type=int, default=500)
    parser.add_argument("--diagnostic-products", type=int, default=200)
    parser.add_argument("--inventory-states", type=int, default=41)
    parser.add_argument("--oracle-mc-samples", type=int, default=200000)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--train-full-diagnostics",
        action="store_true",
        help="Keep training benchmark diagnostics. Default skips them to reduce runtime and disk use.",
    )
    parser.add_argument(
        "--vary-product-seed",
        action="store_true",
        help="Also set product_generation.seed for each seed to test product-split robustness.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse completed seed outputs if the expected summary files already exist.",
    )
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    models = [name.strip() for name in args.models.split(",") if name.strip()]
    if not models:
        raise ValueError("At least one model is required")
    model_set = set(models)

    base_config = yaml.safe_load(args.config.read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_summaries: list[pd.DataFrame] = []
    all_operational: list[pd.DataFrame] = []
    shift_names = [name.strip() for name in args.shifts.split(",") if name.strip()]
    for seed in seeds:
        print(f"=== seed {seed} ===", flush=True)
        seed_config = _write_seed_config(base_config, seed, args.output_root, args.vary_product_seed)
        train_dir = args.output_root / f"seed_{seed}" / "trained_models"
        shift_dir = args.output_root / f"seed_{seed}" / "distribution_shift"
        expected_shift_summary = shift_dir / "summary_all_shifts.csv"
        try:
            if args.skip_existing and expected_shift_summary.exists():
                print(f"reusing_existing_seed={seed}", flush=True)
            else:
                train_command = [
                    sys.executable,
                    "scripts/run_idp_benchmark.py",
                    "--mode",
                    "structure-ls",
                    "--profile",
                    args.profile,
                    "--config",
                    str(seed_config),
                    "--models",
                    args.models,
                    "--output-dir",
                    str(train_dir),
                ]
                if args.progress:
                    train_command.append("--progress")
                if not args.train_full_diagnostics:
                    train_command.append("--fast-diagnostics")
                _run(train_command)

                shift_command = [
                    sys.executable,
                    "scripts/run_distribution_shift_validation.py",
                    "--config",
                    str(seed_config),
                    "--source-benchmark-dir",
                    str(train_dir),
                    "--output-dir",
                    str(shift_dir),
                    "--profile",
                    args.profile,
                    "--eval-products",
                    str(args.eval_products),
                    "--diagnostic-products",
                    str(args.diagnostic_products),
                    "--inventory-states",
                    str(args.inventory_states),
                    "--oracle-mc-samples",
                    str(args.oracle_mc_samples),
                    "--models",
                    args.models,
                    "--shifts",
                    args.shifts,
                ]
                _run(shift_command)
            all_summaries.append(_read_seed_shift_summary(expected_shift_summary, seed))
            for shift_name in shift_names:
                operational_path = shift_dir / shift_name / "operational_metrics.csv"
                if operational_path.exists():
                    operational = pd.read_csv(operational_path)
                    operational.insert(0, "shift", shift_name)
                    operational.insert(0, "seed", int(seed))
                    all_operational.append(operational)
        finally:
            archived_config = args.output_root / f"seed_{seed}" / "effective_config.yaml"
            archived_config.parent.mkdir(parents=True, exist_ok=True)
            archived_config.write_text(seed_config.read_text(encoding="utf-8"), encoding="utf-8")
            seed_config.unlink(missing_ok=True)

    combined = pd.concat(all_summaries, ignore_index=True)
    combined_path = args.output_root / "summary_all_seeds_shifts.csv"
    combined.to_csv(combined_path, index=False)
    robustness, per_shift, winner_counts = _aggregate_shift_seed(combined, model_set)
    robustness_path = args.output_root / "robustness_summary_all_seeds.csv"
    per_shift_path = args.output_root / "per_shift_summary_all_seeds.csv"
    winners_path = args.output_root / "winner_counts.csv"
    robustness.to_csv(robustness_path, index=False)
    per_shift.to_csv(per_shift_path, index=False)
    winner_counts.to_csv(winners_path, index=False)
    if all_operational:
        operational_combined = pd.concat(all_operational, ignore_index=True)
        paired_differences, paired_summary = _paired_reward_differences(operational_combined, model_set)
        paired_differences_path = args.output_root / "paired_product_reward_differences.csv"
        paired_summary_path = args.output_root / "paired_product_reward_summary.csv"
        paired_differences.to_csv(paired_differences_path, index=False)
        paired_summary.to_csv(paired_summary_path, index=False)
    print(robustness.to_string(index=False))
    print(f"saved_combined={combined_path}")
    print(f"saved_robustness={robustness_path}")
    print(f"saved_per_shift={per_shift_path}")
    print(f"saved_winner_counts={winners_path}")
    if all_operational:
        print(f"saved_paired_differences={paired_differences_path}")
        print(f"saved_paired_summary={paired_summary_path}")


if __name__ == "__main__":
    main()
