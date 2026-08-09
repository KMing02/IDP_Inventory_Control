"""Run validation-only equal-capacity and data-efficiency follow-up analyses.

This is an orchestration/reporting layer.  It deliberately reuses the model
factories, trainer, action cap, prediction diagnostics, and operational
evaluation used by the primary tuning workflow.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Sequence

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from inventory_control.idp_benchmark import (  # noqa: E402
    _oracle_levels,
    _prediction_summary,
    _profile_settings,
)
from inventory_control.paper_configs import load_paper_lost_sales_config  # noqa: E402
from inventory_control.policies.neural.fcnn import compute_paper_fcnn_feature_stats  # noqa: E402
from inventory_control.products import PaperLostSalesProduct, PaperLostSalesProductGenerator  # noqa: E402
from scripts.tune_primary_neural_policies import (  # noqa: E402
    Candidate,
    PRIMARY_POLICIES,
    VALIDATION_GRID_SEED,
    _atomic_write_csv,
    _atomic_write_json,
    _completion_record,
    _environment_record,
    _fingerprint,
    _repository_relative_path,
    _resolved_fit_config,
    _run_fit,
    _safe_generate_products,
    _write_or_validate_json,
    data_fingerprints,
    validate_split_and_output,
)


TUNING_ROOT = Path("outputs/idp_tuning/primary_neural_balanced8_validation")
EQUAL_CANDIDATE = Candidate("EC", 2, 32, 0.001, 15, (1, 1, 1), 0.1)
SELECTED_CANDIDATES = {
    "PaperFCNN": Candidate("C8", 3, 64, 0.001, 15, (1, 1, 1), 0.1),
    "PenaltyMLP": Candidate("C8", 3, 64, 0.001, 15, (1, 1, 1), 0.1),
    "PositiveWeightMLP": Candidate("C6", 3, 32, 0.001, 30, (2, 2, 1), 1.0),
    "ConstrainedMonoMLP": Candidate("C2", 2, 32, 0.001, 15, (1, 1, 1), 10.0),
}
EXTREME_ACTION_FACTOR = 8.0
VIOLATION_TOLERANCE = 1e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validation_trace_fingerprint(products: Sequence[PaperLostSalesProduct]) -> str:
    return _fingerprint(
        [
            {
                "product_id": product.product_id,
                "historical_demand": product.historical_demand,
                "future_demand": product.future_demand,
            }
            for product in products
        ]
    )


def optimizer_update_counts(training_log: pd.DataFrame) -> dict[str, int]:
    """Count actual optimizer steps; DirectBackprop performs one step per row."""
    phase = pd.to_numeric(training_log.get("phase", pd.Series(dtype=float)), errors="coerce")
    imitation = int((phase == 0).sum())
    reward = int((phase == 1).sum())
    return {
        "imitation_optimizer_updates": imitation,
        "reward_optimizer_updates": reward,
        "total_optimizer_updates": imitation + reward,
    }


def output_diagnostics(predictions: pd.DataFrame) -> dict[str, float | int]:
    raw = predictions["raw_order"].to_numpy(dtype=float)
    deployed = predictions["deployed_order"].to_numpy(dtype=float)
    mean = predictions["historical_mean_demand"].to_numpy(dtype=float)
    nonfinite = ~(np.isfinite(raw) & np.isfinite(deployed))
    extreme = nonfinite | (raw < -1e-6) | (raw > EXTREME_ACTION_FACTOR * mean)
    return {
        "non_finite_output_count": int(nonfinite.sum()),
        "non_finite_output_rate": float(nonfinite.mean()),
        "extreme_raw_action_count": int(extreme.sum()),
        "extreme_raw_action_rate": float(extreme.mean()),
    }


def _analysis_record_dir(output_dir: Path, policy: str, key: str, seed: int) -> Path:
    return output_dir / "records" / policy / key / f"seed_{seed}"


def _source_run_dir(root: Path, policy: str, candidate_id: str, seed: int) -> Path:
    return root / "runs" / policy / candidate_id / f"seed_{seed}"


def _compatible_source(
    source_dir: Path,
    base_config: dict[str, Any],
    profile: str,
    policy: str,
    candidate: Candidate,
    seed: int,
    shared_data_fingerprint: str,
) -> bool:
    record = _completion_record(source_dir)
    resolved_path = source_dir / "resolved_config.json"
    if record is None or not resolved_path.exists():
        return False
    source = json.loads(resolved_path.read_text(encoding="utf-8"))
    _, expected = _resolved_fit_config(base_config, profile, policy, candidate, seed)
    ignored = {"candidate_id", "resolved_config_fingerprint", "shared_data_fingerprint"}
    for key, value in expected.items():
        if key not in ignored and source.get(key) != value:
            return False
    return record.get("shared_data_fingerprint") == shared_data_fingerprint


def _record_source(
    *,
    output_dir: Path,
    output_relative: Path,
    policy: str,
    analysis_key: str,
    seed: int,
    source_dir: Path,
    reused: bool,
    training_products: int,
) -> dict[str, Any]:
    source_completion = _completion_record(source_dir)
    if source_completion is None:
        raise RuntimeError(f"Source fit is incomplete: {source_dir}")
    training_log = pd.read_csv(source_dir / "training_log.csv")
    predictions = pd.read_csv(source_dir / "predictions.csv")
    source_relative = source_dir.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    record = {
        **source_completion,
        "analysis_key": analysis_key,
        "training_products": int(training_products),
        "reused_checkpoint": bool(reused),
        "source_run_dir": source_relative,
        "checkpoint_sha256": _sha256(source_dir / "model.pt"),
        **optimizer_update_counts(training_log),
        **output_diagnostics(predictions),
        "failed_fit": False,
    }
    if policy == "PenaltyMLP":
        record["penalty_active"] = bool(
            float(record.get("penalty_loss_max", 0.0)) > 0.0
            or int(record.get("violating_pair_count", 0)) > 0
        )
    record_dir = _analysis_record_dir(output_dir, policy, analysis_key, seed)
    record_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(record_dir / "record.json", record)
    _atomic_write_json(record_dir / "completion.json", record)
    return record


def _valid_analysis_record(record_dir: Path) -> dict[str, Any] | None:
    marker = record_dir / "completion.json"
    record_path = record_dir / "record.json"
    if not marker.exists() or not record_path.exists():
        return None
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
        source = REPOSITORY_ROOT / record["source_run_dir"]
    except (KeyError, OSError, json.JSONDecodeError):
        return None
    if _completion_record(source) is None or not (source / "model.pt").exists():
        return None
    if _sha256(source / "model.pt") != record.get("checkpoint_sha256"):
        return None
    return record


def _prediction_violations(predictions: pd.DataFrame, column: str) -> tuple[int, int]:
    violations = 0
    pairs = 0
    for _, rows in predictions.groupby(["training_seed", "product_id"], sort=False):
        values = rows.sort_values("state_index")[column].to_numpy(dtype=float)
        increases = np.diff(values)
        violations += int(np.sum(increases > VIOLATION_TOLERANCE))
        pairs += len(increases)
    return violations, pairs


def _aggregate_results(
    records: pd.DataFrame,
    output_dir: Path,
    expected_seeds: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if records.empty:
        return pd.DataFrame()
    group_columns = ["policy_name", "analysis_key", "training_products"]
    for keys, group in records.groupby(group_columns, sort=False):
        policy, analysis_key, training_products = keys
        prediction_frames = []
        for record in group.itertuples(index=False):
            source = REPOSITORY_ROOT / str(record.source_run_dir)
            frame = pd.read_csv(source / "predictions.csv")
            frame["training_seed"] = int(record.training_seed)
            prediction_frames.append(frame)
        predictions = pd.concat(prediction_frames, ignore_index=True)
        finite_errors = predictions["deployed_error"].to_numpy(dtype=float)
        finite_errors = finite_errors[np.isfinite(finite_errors)]
        zero = predictions.loc[predictions["state_index"] == 0]
        zero_errors = np.abs(
            zero["deployed_order_up_to_level"].to_numpy(dtype=float)
            - zero["deployed_omniscient_order_up_to_level"].to_numpy(dtype=float)
        )
        raw_v, raw_pairs = _prediction_violations(predictions, "raw_order")
        dep_v, dep_pairs = _prediction_violations(predictions, "deployed_order")
        failed_count = len(set(expected_seeds) - set(group["training_seed"].astype(int)))
        rows.append(
            {
                "policy_name": policy,
                "analysis_key": analysis_key,
                "training_products": int(training_products),
                "completed_seed_count": int(len(group)),
                "failed_fit_count": int(failed_count),
                "failed_fit_rate": failed_count / max(len(expected_seeds), 1),
                "mean_reward_per_period": float(group["mean_reward_per_period"].mean()),
                "reward_per_period_std": float(group["mean_reward_per_period"].std(ddof=1)),
                "service_level": float(group["service_level"].mean()),
                "lost_sales_rate": float(group["lost_sales_rate"].mean()),
                "mean_inventory": float(group["mean_inventory"].mean()),
                "mean_order_quantity": float(group["mean_order_quantity"].mean()),
                "pooled_deployed_rmse": float(np.sqrt(np.mean(np.square(finite_errors)))),
                "pooled_raw_violation_rate": raw_v / max(raw_pairs, 1),
                "pooled_deployed_violation_rate": dep_v / max(dep_pairs, 1),
                "zero_inventory_median_absolute_error": float(np.median(zero_errors)),
                "zero_inventory_p90_absolute_error": float(np.quantile(zero_errors, 0.9)),
                "zero_inventory_max_absolute_error": float(np.max(zero_errors)),
                "parameter_count": int(group["parameter_count"].iloc[0]),
                "non_finite_output_rate": float(group["non_finite_output_count"].sum())
                / max(sum(len(pd.read_csv(REPOSITORY_ROOT / p / "predictions.csv")) for p in group["source_run_dir"]), 1),
                "extreme_raw_action_rate": float(group["extreme_raw_action_count"].sum())
                / max(sum(len(pd.read_csv(REPOSITORY_ROOT / p / "predictions.csv")) for p in group["source_run_dir"]), 1),
                "mean_imitation_optimizer_updates": float(group["imitation_optimizer_updates"].mean()),
                "mean_reward_optimizer_updates": float(group["reward_optimizer_updates"].mean()),
                "mean_total_optimizer_updates": float(group["total_optimizer_updates"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _write_summaries(output_dir: Path, expected_seeds: Sequence[int]) -> None:
    completed = []
    for marker in sorted((output_dir / "records").glob("*/*/seed_*/completion.json")):
        record = _valid_analysis_record(marker.parent)
        if record is not None:
            completed.append(record)
    seed_results = pd.DataFrame(completed)
    if not seed_results.empty:
        seed_results = seed_results.sort_values(["policy_name", "analysis_key", "training_seed"])
        rewards = seed_results.pivot_table(
            index=["analysis_key", "training_products", "training_seed"],
            columns="policy_name",
            values="mean_reward_per_period",
        )
        if "PaperFCNN" in rewards:
            deltas = rewards.subtract(rewards["PaperFCNN"], axis=0).stack().rename("reward_difference_vs_paper_fcnn")
            delta_frame = deltas.reset_index().rename(columns={"level_3": "policy_name"})
            seed_results = seed_results.merge(
                delta_frame,
                on=["analysis_key", "training_products", "training_seed", "policy_name"],
                how="left",
            )
    _atomic_write_csv(output_dir / "seed_results.csv", seed_results)
    reuse_columns = [
        "policy_name",
        "analysis_key",
        "training_seed",
        "reused_checkpoint",
        "source_run_dir",
        "checkpoint_sha256",
    ]
    _atomic_write_csv(
        output_dir / "reuse_manifest.csv",
        seed_results[[column for column in reuse_columns if column in seed_results.columns]],
    )
    _atomic_write_csv(output_dir / "aggregate_results.csv", _aggregate_results(seed_results, output_dir, expected_seeds))


def _build_plan(
    analysis: str,
    training_counts: Sequence[int],
    seeds: Sequence[int],
) -> list[tuple[str, int, Candidate, str, int]]:
    plan = []
    if analysis in {"equal-capacity", "matched-width-extension"}:
        policies = (
            PRIMARY_POLICIES
            if analysis == "equal-capacity"
            else ("PaperFCNN", "ConstrainedMonoMLP")
        )
        key = "equal_capacity" if analysis == "equal-capacity" else "matched_width_depth"
        for policy in policies:
            for seed in seeds:
                plan.append((policy, 128, EQUAL_CANDIDATE, key, int(seed)))
    else:
        for count in training_counts:
            for policy in PRIMARY_POLICIES:
                candidate = SELECTED_CANDIDATES[policy]
                for seed in seeds:
                    plan.append(
                        (policy, int(count), candidate, f"N{count}_{candidate.candidate_id}", int(seed))
                    )
    return plan


def _source_candidate_id(analysis: str, policy: str, count: int) -> str | None:
    if analysis == "equal-capacity":
        return "C2"
    if count == 128:
        return SELECTED_CANDIDATES[policy].candidate_id
    return None


def _analysis_record_source(root: Path, policy: str, key: str, seed: int) -> Path | None:
    marker = root / "records" / policy / key / f"seed_{seed}" / "completion.json"
    if not marker.exists():
        return None
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
        source = REPOSITORY_ROOT / record["source_run_dir"]
    except (KeyError, OSError, json.JSONDecodeError):
        return None
    return source


def _write_paired_statistics(output_dir: Path) -> None:
    from scipy import stats

    seed_results = pd.read_csv(output_dir / "seed_results.csv")
    rewards = seed_results.pivot(
        index="training_seed", columns="policy_name", values="mean_reward_per_period"
    ).sort_index()
    differences = rewards["ConstrainedMonoMLP"] - rewards["PaperFCNN"]
    count = len(differences)
    mean = float(differences.mean())
    sample_std = float(differences.std(ddof=1))
    half_width = float(stats.t.ppf(0.975, count - 1) * sample_std / np.sqrt(count))
    paired_t = stats.ttest_rel(rewards["ConstrainedMonoMLP"], rewards["PaperFCNN"])
    wilcoxon = stats.wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
    _atomic_write_json(
        output_dir / "paired_statistics.json",
        {
            "seed_count": count,
            "mean_paired_difference_cmnn_minus_paper_fcnn": mean,
            "sample_std_paired_difference": sample_std,
            "confidence_interval_95": [mean - half_width, mean + half_width],
            "paired_t_statistic": float(paired_t.statistic),
            "paired_t_pvalue": float(paired_t.pvalue),
            "wilcoxon_statistic": float(wilcoxon.statistic),
            "wilcoxon_pvalue": float(wilcoxon.pvalue),
            "cmnn_seed_wins": int((differences > 0).sum()),
            "paper_fcnn_seed_wins": int((differences < 0).sum()),
            "ties": int((differences == 0).sum()),
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        choices=["equal-capacity", "data-efficiency", "matched-width-extension"],
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", default="full")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--training-products", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--hidden-units", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--reward-epochs", type=int, default=15)
    parser.add_argument("--penalty-lambda", type=float, default=0.1)
    parser.add_argument("--cmnn-allocation", nargs=3, type=int, default=[1, 1, 1])
    parser.add_argument("--selected-configs", type=Path)
    parser.add_argument("--reuse-root", type=Path, default=TUNING_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_split_and_output(args.split, args.output_dir)
    config_path, config_relative = _repository_relative_path(args.config, "--config")
    output_dir, output_relative = _repository_relative_path(args.output_dir, "--output-dir")
    reuse_root, _ = _repository_relative_path(args.reuse_root, "--reuse-root")
    if args.analysis in {"equal-capacity", "matched-width-extension"}:
        exact = (args.hidden_layers, args.hidden_units, args.learning_rate, args.reward_epochs, args.penalty_lambda, tuple(args.cmnn_allocation))
        required_seeds = (
            (0, 1, 2, 3, 4)
            if args.analysis == "equal-capacity"
            else tuple(range(10))
        )
        if exact != (2, 32, 0.001, 15, 0.1, (1, 1, 1)) or tuple(args.seeds) != required_seeds:
            raise ValueError("Matched-width/depth settings must match the preregistered specification")
        training_counts = [128]
    else:
        if tuple(args.training_products) != (16, 32, 64, 128) or tuple(args.seeds) != (0, 1, 2):
            raise ValueError("Data-efficiency counts/seeds must match the preregistered specification")
        if args.selected_configs is None:
            raise ValueError("Data-efficiency analysis requires --selected-configs")
        selected_path, _ = _repository_relative_path(args.selected_configs, "--selected-configs")
        selected = pd.read_csv(selected_path)
        expected_selection = {
            (policy, candidate.candidate_id) for policy, candidate in SELECTED_CANDIDATES.items()
        }
        if set(zip(selected["policy_name"], selected["candidate_id"])) != expected_selection:
            raise RuntimeError("selected_configs.csv does not match the preregistered configurations")
        training_counts = list(args.training_products)

    base_config = load_paper_lost_sales_config(config_path)
    base_settings = _profile_settings(base_config, args.profile)
    if float(base_settings["max_order_factor"]) != 4.0 or int(base_settings["validation_products"]) != 32:
        raise ValueError("Follow-ups require F=4 and exactly 32 validation products")
    generator = PaperLostSalesProductGenerator(base_config)
    train_pool = _safe_generate_products(generator, 128, "train")
    validation_products = _safe_generate_products(generator, 32, "validation")
    oracle_levels = _oracle_levels(validation_products, int(base_settings["oracle_mc_samples"]), 10_000)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_or_validate_json(output_dir / "environment.json", _environment_record(config_relative, base_config))

    fingerprint_map: dict[str, Any] = {}
    for count in training_counts:
        local_config = copy.deepcopy(base_config)
        local_config["idp_benchmark"][args.profile]["training_products"] = int(count)
        local_settings = _profile_settings(local_config, args.profile)
        values = data_fingerprints(local_config, local_settings, train_pool[:count], validation_products)
        shared_data_fingerprint = _fingerprint(values)
        values["validation_demand_traces_fingerprint"] = _validation_trace_fingerprint(validation_products)
        values["shared_data_fingerprint"] = shared_data_fingerprint
        fingerprint_map[str(count)] = values
    _write_or_validate_json(output_dir / "data_fingerprints.json", fingerprint_map)

    plan = _build_plan(args.analysis, training_counts, args.seeds)
    manifest_rows = plan
    manifest = pd.DataFrame(
        [
            {
                "analysis": args.analysis,
                "policy_name": policy,
                "training_products": count,
                "analysis_key": key,
                "training_seed": seed,
                "hidden_layers": candidate.hidden_layers,
                "hidden_units": candidate.hidden_units,
                "learning_rate": candidate.learning_rate,
                "reward_epochs": candidate.reward_epochs,
                "penalty_lambda": candidate.penalty_lambda if policy == "PenaltyMLP" else np.nan,
                "activation_weights": json.dumps(candidate.cmnn_activation_weights) if policy == "ConstrainedMonoMLP" else "",
            }
            for policy, count, candidate, key, seed in manifest_rows
        ]
    )
    _atomic_write_csv(output_dir / "manifest.csv", manifest)
    if args.dry_run:
        print(manifest.to_string(index=False))
        return 0

    completed = 0
    started = time.perf_counter()
    failures = []
    for policy, count, candidate, key, seed in manifest_rows:
        record_dir = _analysis_record_dir(output_dir, policy, key, seed)
        existing = _valid_analysis_record(record_dir)
        if existing is not None and args.resume:
            completed += 1
            continue
        if existing is not None and not args.resume:
            raise RuntimeError("Completed analysis records exist; pass --resume")
        local_config = copy.deepcopy(base_config)
        local_config["idp_benchmark"][args.profile]["training_products"] = int(count)
        local_train = train_pool[:count]
        local_settings = _profile_settings(local_config, args.profile)
        shared = str(fingerprint_map[str(count)]["shared_data_fingerprint"])
        if args.analysis == "matched-width-extension":
            source = _analysis_record_source(reuse_root, policy, "equal_capacity", seed)
            source_id = None
        else:
            source_id = _source_candidate_id(args.analysis, policy, count)
            source = _source_run_dir(reuse_root, policy, source_id, seed) if source_id else None
        reused = bool(source and _compatible_source(source, local_config, args.profile, policy, candidate, seed, shared))
        try:
            if not reused:
                source = _source_run_dir(output_dir, policy, key, seed)
                _run_fit(
                    base_config=local_config,
                    profile=args.profile,
                    stage=f"validation_followup_{args.analysis}",
                    policy=policy,
                    candidate=Candidate(key, candidate.hidden_layers, candidate.hidden_units, candidate.learning_rate, candidate.reward_epochs, candidate.cmnn_activation_weights, candidate.penalty_lambda),
                    training_seed=seed,
                    output_dir=output_dir,
                    output_relative=output_relative,
                    train_products=local_train,
                    validation_products=validation_products,
                    feature_stats=compute_paper_fcnn_feature_stats(local_train),
                    oracle_levels=oracle_levels,
                    shared_data_fingerprint=shared,
                )
            record = _record_source(
                output_dir=output_dir,
                output_relative=output_relative,
                policy=policy,
                analysis_key=key,
                seed=seed,
                source_dir=source,
                reused=reused,
                training_products=count,
            )
            completed += 1
            _write_summaries(output_dir, args.seeds)
            print(f"completed={completed}/{len(manifest_rows)} policy={policy} key={key} seed={seed} reused={str(reused).lower()} reward={record['mean_reward_per_period']:.6f}")
        except Exception as exc:  # continue so failed-fit rate is observable
            failures.append({"policy_name": policy, "analysis_key": key, "training_seed": seed, "error": repr(exc)})
            record_dir.mkdir(parents=True, exist_ok=True)
            (record_dir / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
            _atomic_write_csv(output_dir / "failures.csv", pd.DataFrame(failures))
            print(f"failed policy={policy} key={key} seed={seed} error={type(exc).__name__}")
    _write_summaries(output_dir, args.seeds)
    if args.analysis == "matched-width-extension" and completed == len(manifest_rows):
        _write_paired_statistics(output_dir)
    _atomic_write_csv(output_dir / "failures.csv", pd.DataFrame(failures, columns=["policy_name", "analysis_key", "training_seed", "error"]))
    _atomic_write_json(output_dir / "analysis_completion.json", {"analysis": args.analysis, "completed_fits": completed, "expected_fits": len(manifest_rows), "failed_fits": len(failures), "wall_seconds": time.perf_counter() - started, "protected_test_accessed": False})
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
