"""Run the post-protected validation-only primary convergence audit.

This runner reproduces the primary imitation stage and its no-reset Adam
transition, then observes fixed reward-update checkpoints without validation
selection.  It never generates or evaluates the protected split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from inventory_control.idp_benchmark import (  # noqa: E402
    _model_config,
    _operational_table,
    _oracle_levels,
    _parameter_count,
    _prediction_summary,
    _prediction_table,
    _profile_settings,
)
from inventory_control.paper_configs import load_paper_lost_sales_config  # noqa: E402
from inventory_control.policies.neural.fcnn import compute_paper_fcnn_feature_stats  # noqa: E402
from inventory_control.products import PaperLostSalesProduct, PaperLostSalesProductGenerator  # noqa: E402
from inventory_control.training.direct_backprop import train_paper_lost_sales_fcnn  # noqa: E402
from scripts.run_no_imitation_ablation import _operational_metrics  # noqa: E402
from scripts.tune_primary_neural_policies import (  # noqa: E402
    Candidate,
    _atomic_torch_save,
    _atomic_write_csv,
    _atomic_write_json,
    _environment_record,
    _fixed_validation_grid_seed,
    _make_model,
    _repository_relative_path,
    _resolved_fit_config,
    _safe_generate_products,
    _write_or_validate_json,
    data_fingerprints,
    validate_split_and_output,
)


POLICIES = ("PaperFCNN", "ConstrainedMonoMLP")
SEEDS = tuple(range(10))
CHECKPOINT_UPDATES = (0, 15, 30, 60, 120, 240)
CANDIDATE = Candidate("C2", 2, 32, 0.001, 15, (1, 1, 1), 10.0)
DEFAULT_OUTPUT = Path(
    "outputs/idp_followup/primary_imitation_assisted_convergence_validation"
)
PROVENANCE_ROOT = Path("outputs/idp_followup/primary_checkpoint_provenance_audit")


def _state_max_difference(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    if left.keys() != right.keys():
        return float("inf")
    return max(
        float(torch.max(torch.abs(left[key].detach().cpu() - right[key].detach().cpu())).item())
        for key in left
    )


def _fit_dir(output_dir: Path, policy: str, seed: int) -> Path:
    return output_dir / "runs" / policy / f"seed_{seed}"


def _complete(fit_dir: Path) -> bool:
    required = [
        fit_dir / "completion.json",
        fit_dir / "training_log.csv",
        fit_dir / "checkpoint_metrics.csv",
    ] + [fit_dir / "checkpoints" / f"model_update_{update}.pt" for update in CHECKPOINT_UPDATES]
    return all(path.exists() for path in required)


def _checkpoint_metrics(
    *,
    model: torch.nn.Module,
    policy: str,
    seed: int,
    update: int,
    validation_products: list[PaperLostSalesProduct],
    feature_stats: Any,
    settings: dict[str, Any],
    oracle_levels: dict[str, float],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    was_training = model.training
    operational = _operational_metrics(
        validation_products, policy, model, feature_stats, settings
    )
    predictions = _prediction_table(
        validation_products,
        {policy: model},
        feature_stats,
        settings,
        oracle_levels,
    )
    policy_predictions = predictions.loc[predictions["policy_name"] == policy].reset_index(
        drop=True
    )
    diagnostics = _prediction_summary(policy_predictions).iloc[0].to_dict()
    if was_training:
        model.train()
    cap_hits = int(operational["cap_hit_count"].sum())
    scored_actions = int(operational["scored_action_count"].sum())
    row = {
        "policy_name": policy,
        "training_seed": int(seed),
        "reward_update": int(update),
        "mean_reward_per_period": float(operational["mean_reward_per_period"].mean()),
        "service_level": float(operational["service_level"].mean()),
        "lost_sales_rate": float(operational["lost_sales_rate"].mean()),
        "mean_inventory": float(operational["mean_inventory"].mean()),
        "mean_order_quantity": float(operational["mean_order_quantity"].mean()),
        "mean_holding_cost": float(operational["mean_holding_cost"].mean()),
        "cap_hit_rate": cap_hits / max(scored_actions, 1),
        "raw_violation_rate": float(diagnostics["raw_violation_rate"]),
        "deployed_violation_rate": float(diagnostics["deployed_violation_rate"]),
        "raw_max_violation": float(diagnostics["raw_max_violation"]),
        "deployed_max_violation": float(diagnostics["deployed_max_violation"]),
        "raw_rmse": float(diagnostics["raw_rmse"]),
        "deployed_rmse": float(diagnostics["deployed_rmse"]),
        "active_slope_deviation_from_minus_one": float(
            diagnostics["active_slope_deviation_from_minus_one"]
        ),
    }
    operational = operational.copy()
    operational.insert(0, "reward_update", int(update))
    operational.insert(0, "training_seed", int(seed))
    policy_predictions = policy_predictions.copy()
    policy_predictions.insert(0, "reward_update", int(update))
    policy_predictions.insert(0, "training_seed", int(seed))
    return row, operational, policy_predictions


def run_fit(
    *,
    base_config: dict[str, Any],
    profile: str,
    policy: str,
    seed: int,
    output_dir: Path,
    train_products: list[PaperLostSalesProduct],
    validation_products: list[PaperLostSalesProduct],
    feature_stats: Any,
    oracle_levels: dict[str, float],
    shared_data_fingerprint: str,
    provenance_root: Path,
) -> dict[str, Any]:
    fit_dir = _fit_dir(output_dir, policy, seed)
    fit_dir.mkdir(parents=True, exist_ok=True)
    (fit_dir / "completion.json").unlink(missing_ok=True)
    config, resolved = _resolved_fit_config(base_config, profile, policy, CANDIDATE, seed)
    settings = _profile_settings(config, profile)
    resolved.update(
        {
            "audit_name": "primary_imitation_assisted_convergence_validation",
            "reward_epochs": 240,
            "reward_checkpoint_updates": list(CHECKPOINT_UPDATES),
            "reward_validation_checkpoint_selection": False,
            "reward_early_stopping": False,
            "shared_data_fingerprint": shared_data_fingerprint,
            "protected_test_accessed": False,
        }
    )
    _atomic_write_json(fit_dir / "resolved_config.json", resolved)
    model = _make_model(policy, _model_config(config, settings), seed)
    provenance_dir = provenance_root / "runs" / policy / f"seed_{seed}"
    provenance_record = json.loads(
        (provenance_dir / "completion.json").read_text(encoding="utf-8")
    )
    provenance_best_imitation = torch.load(
        provenance_dir / "best_imitation_checkpoint.pt", map_location="cpu", weights_only=True
    )
    provenance_update_15 = torch.load(
        provenance_dir / "actual_reward_update_15.pt", map_location="cpu", weights_only=True
    )
    checkpoint_rows: list[dict[str, Any]] = []
    operational_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    compatibility: dict[str, float] = {}

    def observe(update: int, current_model: torch.nn.Module) -> None:
        if update not in CHECKPOINT_UPDATES:
            return
        state = {
            key: value.detach().cpu().clone()
            for key, value in current_model.state_dict().items()
        }
        if update == 0:
            compatibility["update_0_max_abs_diff_vs_provenance_best_imitation"] = (
                _state_max_difference(state, provenance_best_imitation)
            )
            if compatibility["update_0_max_abs_diff_vs_provenance_best_imitation"] != 0.0:
                raise RuntimeError(
                    f"{policy} seed {seed} did not reproduce the best-imitation start state"
                )
        if update == 15:
            compatibility["update_15_max_abs_diff_vs_provenance_actual_update_15"] = (
                _state_max_difference(state, provenance_update_15)
            )
            if compatibility["update_15_max_abs_diff_vs_provenance_actual_update_15"] != 0.0:
                raise RuntimeError(
                    f"{policy} seed {seed} did not reproduce the audited fixed update-15 state"
                )
        row, operational, predictions = _checkpoint_metrics(
            model=current_model,
            policy=policy,
            seed=seed,
            update=update,
            validation_products=validation_products,
            feature_stats=feature_stats,
            settings=settings,
            oracle_levels=oracle_levels,
        )
        checkpoint_rows.append(row)
        operational_frames.append(operational)
        prediction_frames.append(predictions)
        checkpoint_dir = fit_dir / "checkpoints"
        _atomic_torch_save(checkpoint_dir / f"model_update_{update}.pt", state)
        _atomic_write_json(checkpoint_dir / f"metrics_update_{update}.json", row)
        _atomic_write_csv(
            checkpoint_dir / f"operational_update_{update}.csv", operational
        )
        _atomic_write_csv(
            checkpoint_dir / f"predictions_update_{update}.csv", predictions
        )

    started = time.perf_counter()
    with _fixed_validation_grid_seed(validation_products):
        result = train_paper_lost_sales_fcnn(
            train_products,
            model,
            horizon=int(settings["training_horizon"]),
            batch_size=min(int(settings["batch_size"]), len(train_products)),
            epochs=240,
            learning_rate=float(config["learning_rate"]),
            seed=int(settings["seed"]),
            imitation_epochs=int(settings["imitation_epochs"]),
            validation_products=validation_products,
            validation_interval=0,
            validation_callback=None,
            feature_stats=feature_stats,
            max_order_factor=float(settings["max_order_factor"]),
            scale_action_by_history_mean=bool(settings["scale_action_by_history_mean"]),
            normalize_reward_loss=bool(settings["normalize_reward_loss"]),
            monotonicity_penalty_weight=0.0,
            monotonicity_samples_per_product=int(settings["monotonicity_samples_per_product"]),
            clip_imitation_predictions=True,
            imitation_early_stopping=bool(settings["imitation_early_stopping"]),
            imitation_max_epochs=int(settings["imitation_max_epochs"]),
            imitation_patience=int(settings["imitation_patience"]),
            imitation_min_delta=float(settings["imitation_min_delta"]),
            track_imitation_validation=bool(settings.get("track_imitation_validation", False)),
            reset_optimizer_after_imitation=bool(settings["reset_optimizer_after_imitation"]),
            adam_kwargs=dict(config.get("adam", {})),
            reward_checkpoint_callback=observe,
        )
    elapsed = time.perf_counter() - started
    checkpoints = pd.DataFrame(checkpoint_rows).sort_values("reward_update")
    if checkpoints["reward_update"].astype(int).tolist() != list(CHECKPOINT_UPDATES):
        raise RuntimeError(f"Missing fixed checkpoint for {policy} seed {seed}")
    if result.best_state_dict is not None or result.validation_rewards is not None:
        raise RuntimeError("Reward validation selection was unexpectedly enabled")
    if result.optimizer_reset_after_imitation:
        raise RuntimeError("Adam was unexpectedly reset after imitation")
    if int(result.imitation_epochs_completed) != int(
        provenance_record["imitation_epochs_completed"]
    ) or int(result.best_imitation_epoch) != int(provenance_record["best_imitation_epoch"]):
        raise RuntimeError(f"Imitation provenance mismatch for {policy} seed {seed}")
    training_log = pd.DataFrame(result.training_log or [])
    reward_rows = training_log.loc[training_log["phase"] == 1]
    if len(reward_rows) != 240:
        raise RuntimeError("Run did not complete exactly 240 reward optimizer updates")
    _atomic_write_csv(fit_dir / "training_log.csv", training_log)
    _atomic_write_csv(fit_dir / "checkpoint_metrics.csv", checkpoints)
    _atomic_write_csv(
        fit_dir / "checkpoint_operational_metrics.csv",
        pd.concat(operational_frames, ignore_index=True),
    )
    _atomic_write_csv(
        fit_dir / "checkpoint_predictions.csv",
        pd.concat(prediction_frames, ignore_index=True),
    )
    completion = {
        "policy_name": policy,
        "training_seed": int(seed),
        "parameter_count": _parameter_count(model),
        "imitation_epochs_completed": int(result.imitation_epochs_completed),
        "best_imitation_epoch": int(result.best_imitation_epoch),
        "best_imitation_validation_mae": result.best_imitation_validation_mae,
        "best_imitation_validation_rmse": result.best_imitation_validation_rmse,
        "imitation_checkpoint_metric": result.imitation_selection_metric,
        "best_imitation_checkpoint_restored": True,
        "optimizer_reset_after_imitation": False,
        "reward_optimizer_updates": 240,
        "reward_validation_checkpoint_selection": False,
        "reward_early_stopping": False,
        "checkpoint_updates": list(CHECKPOINT_UPDATES),
        "shared_data_fingerprint": shared_data_fingerprint,
        "training_seconds": float(result.elapsed_seconds),
        "fit_wall_seconds": float(elapsed),
        "protected_test_accessed": False,
        **compatibility,
    }
    _atomic_write_json(fit_dir / "completion.json", completion)
    return completion


def _analytical_references(
    *,
    validation_products: list[PaperLostSalesProduct],
    feature_stats: Any,
    settings: dict[str, Any],
    oracle_levels: dict[str, float],
) -> pd.DataFrame:
    table = _operational_table(
        validation_products,
        {},
        feature_stats,
        settings,
        oracle_levels,
        "validation",
    )
    return (
        table.loc[table["policy_name"].isin(["NonOmniscientOptimal", "OmniscientOptimal"])]
        .groupby("policy_name", as_index=False)
        .agg(
            mean_reward_per_period=("mean_reward_per_period", "mean"),
            service_level=("service_level", "mean"),
            lost_sales_rate=("lost_sales_rate", "mean"),
            mean_inventory=("mean_inventory", "mean"),
            mean_order_quantity=("mean_order_quantity", "mean"),
            mean_holding_cost=("mean_holding_cost", "mean"),
        )
    )


def aggregate(output_dir: Path, references: pd.DataFrame) -> None:
    frames = [
        pd.read_csv(path)
        for path in sorted((output_dir / "runs").glob("*/seed_*/checkpoint_metrics.csv"))
    ]
    seed_results = pd.concat(frames, ignore_index=True).sort_values(
        ["policy_name", "training_seed", "reward_update"]
    )
    _atomic_write_csv(output_dir / "checkpoint_seed_results.csv", seed_results)
    metric_columns = [
        column
        for column in seed_results.columns
        if column not in {"policy_name", "training_seed", "reward_update"}
    ]
    aggregate_rows = []
    for (policy, update), group in seed_results.groupby(
        ["policy_name", "reward_update"], sort=False
    ):
        row = {
            "policy_name": policy,
            "reward_update": int(update),
            "seed_count": len(group),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sample_sd"] = float(group[metric].std(ddof=1))
        aggregate_rows.append(row)
    aggregate_frame = pd.DataFrame(aggregate_rows).sort_values(
        ["policy_name", "reward_update"]
    )
    _atomic_write_csv(output_dir / "checkpoint_aggregate_results.csv", aggregate_frame)

    rewards = seed_results.pivot(
        index=["reward_update", "training_seed"],
        columns="policy_name",
        values="mean_reward_per_period",
    ).reset_index()
    rewards["cmnn_minus_fcnn"] = rewards["ConstrainedMonoMLP"] - rewards["PaperFCNN"]
    _atomic_write_csv(output_dir / "paired_checkpoint_seed_differences.csv", rewards)
    paired_rows = []
    for update, group in rewards.groupby("reward_update", sort=True):
        differences = group["cmnn_minus_fcnn"]
        paired_rows.append(
            {
                "reward_update": int(update),
                "seed_count": len(group),
                "paired_mean_cmnn_minus_fcnn": float(differences.mean()),
                "paired_difference_sample_sd": float(differences.std(ddof=1)),
                "cmnn_wins": int((differences > 0.0).sum()),
                "fcnn_wins": int((differences < 0.0).sum()),
                "ties": int((differences == 0.0).sum()),
            }
        )
    _atomic_write_csv(
        output_dir / "paired_checkpoint_aggregate.csv", pd.DataFrame(paired_rows)
    )
    _atomic_write_csv(output_dir / "analytical_references.csv", references)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    colors = {"PaperFCNN": "#2F6FDB", "ConstrainedMonoMLP": "#C83E73"}
    labels = {
        "PaperFCNN": "Imitation-assisted FCNN",
        "ConstrainedMonoMLP": "Imitation-assisted direct CMNN",
    }
    for policy in POLICIES:
        rows = aggregate_frame.loc[aggregate_frame["policy_name"] == policy].sort_values(
            "reward_update"
        )
        x = rows["reward_update"].to_numpy(dtype=float)
        mean = rows["mean_reward_per_period_mean"].to_numpy(dtype=float)
        sd = rows["mean_reward_per_period_sample_sd"].to_numpy(dtype=float)
        ax.plot(x, mean, marker="o", linewidth=2.2, color=colors[policy], label=labels[policy])
        ax.fill_between(x, mean - sd, mean + sd, color=colors[policy], alpha=0.15)
    reference_labels = {
        "NonOmniscientOptimal": "Historical Gamma",
        "OmniscientOptimal": "Omniscient Gamma",
    }
    reference_colors = {
        "NonOmniscientOptimal": "#5B8C3A",
        "OmniscientOptimal": "#333333",
    }
    for row in references.itertuples(index=False):
        ax.axhline(
            row.mean_reward_per_period,
            linestyle="--",
            linewidth=1.6,
            color=reference_colors[row.policy_name],
            label=reference_labels[row.policy_name],
        )
    ax.axvline(15, linestyle=":", linewidth=1.4, color="#666666")
    ax.text(
        18,
        0.88,
        "original fixed-budget index",
        transform=ax.get_xaxis_transform(),
        va="top",
        color="#555555",
        fontsize=8,
    )
    ax.set_title("Imitation-assisted primary training dynamics")
    ax.set_xlabel("Reward optimizer updates")
    ax.set_ylabel("Validation reward per scored period")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "imitation_assisted_primary_training_dynamics.png", dpi=180)
    fig.savefig(output_dir / "imitation_assisted_primary_training_dynamics.pdf")
    plt.close(fig)

    paired = pd.DataFrame(paired_rows)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.axhline(0.0, color="#555555", linewidth=1.0)
    ax.errorbar(
        paired["reward_update"],
        paired["paired_mean_cmnn_minus_fcnn"],
        yerr=paired["paired_difference_sample_sd"],
        marker="o",
        capsize=3,
        color="#7A4EAB",
    )
    ax.set_title("Paired CMNN − FCNN validation reward")
    ax.set_xlabel("Reward optimizer updates")
    ax.set_ylabel("Mean paired difference ± sample SD")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "paired_reward_difference_by_update.png", dpi=180)
    fig.savefig(output_dir / "paired_reward_difference_by_update.pdf")
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", default="full")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provenance-root", type=Path, default=PROVENANCE_ROOT)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if tuple(args.seeds) != SEEDS:
        raise ValueError("The clean primary convergence audit requires seeds 0-9")
    validate_split_and_output(args.split, args.output_dir)
    config_path, config_relative = _repository_relative_path(args.config, "--config")
    output_dir, _ = _repository_relative_path(args.output_dir, "--output-dir")
    provenance_root, _ = _repository_relative_path(args.provenance_root, "--provenance-root")
    audit_completion = json.loads(
        (provenance_root / "audit_completion.json").read_text(encoding="utf-8")
    )
    if not audit_completion.get("all_archived_checkpoint_hashes_and_parameters_verified"):
        raise RuntimeError("Checkpoint provenance audit is incomplete")
    base_config = load_paper_lost_sales_config(config_path)
    settings = _profile_settings(base_config, args.profile)
    generator = PaperLostSalesProductGenerator(base_config)
    train_products = _safe_generate_products(
        generator, int(settings["training_products"]), "train"
    )
    validation_products = _safe_generate_products(
        generator, int(settings["validation_products"]), "validation"
    )
    feature_stats = compute_paper_fcnn_feature_stats(train_products)
    fingerprints = data_fingerprints(
        base_config, settings, train_products, validation_products
    )
    provenance_fingerprints = json.loads(
        (provenance_root / "data_fingerprints.json").read_text(encoding="utf-8")
    )
    if fingerprints != provenance_fingerprints:
        raise RuntimeError("Data/evaluation fingerprints differ from the provenance audit")
    shared_data_fingerprint = json.dumps(fingerprints, sort_keys=True)
    import hashlib

    shared_data_fingerprint = hashlib.sha256(shared_data_fingerprint.encode("utf-8")).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_or_validate_json(output_dir / "data_fingerprints.json", fingerprints)
    _write_or_validate_json(
        output_dir / "convergence_metadata.json",
        {
            "analysis": "post_protected_validation_only_primary_convergence",
            "split": "validation",
            "policies": list(POLICIES),
            "training_seeds": list(SEEDS),
            "reward_checkpoint_updates": list(CHECKPOINT_UPDATES),
            "reward_validation_checkpoint_selection": False,
            "reward_early_stopping": False,
            "reset_optimizer_after_imitation": False,
            "protected_test_accessed": False,
            "protected_test_executions": 0,
        },
    )
    _write_or_validate_json(
        output_dir / "environment.json",
        _environment_record(config_relative, base_config),
    )
    oracle_levels = _oracle_levels(
        validation_products, int(settings["oracle_mc_samples"]), 0
    )
    references = _analytical_references(
        validation_products=validation_products,
        feature_stats=feature_stats,
        settings=settings,
        oracle_levels=oracle_levels,
    )
    total = len(POLICIES) * len(SEEDS)
    completed = 0
    run_started = time.perf_counter()
    for policy in POLICIES:
        for seed in SEEDS:
            fit_dir = _fit_dir(output_dir, policy, seed)
            if args.resume and _complete(fit_dir):
                completed += 1
                continue
            completion = run_fit(
                base_config=base_config,
                profile=args.profile,
                policy=policy,
                seed=seed,
                output_dir=output_dir,
                train_products=train_products,
                validation_products=validation_products,
                feature_stats=feature_stats,
                oracle_levels=oracle_levels,
                shared_data_fingerprint=shared_data_fingerprint,
                provenance_root=provenance_root,
            )
            completed += 1
            latest = json.loads(
                (fit_dir / "checkpoints" / "metrics_update_240.json").read_text(
                    encoding="utf-8"
                )
            )
            print(
                f"{completed}/{total} policy={policy} seed={seed} "
                f"elapsed={completion['fit_wall_seconds']:.1f}s "
                f"reward240={latest['mean_reward_per_period']:.3f}",
                flush=True,
            )
    aggregate(output_dir, references)
    completed_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / "runs").glob("*/seed_*/completion.json"))
    ]
    _atomic_write_json(
        output_dir / "completion.json",
        {
            "completed_fits": completed,
            "expected_fits": total,
            "wall_seconds_this_invocation": time.perf_counter() - run_started,
            "sum_fit_wall_seconds": float(
                sum(float(record["fit_wall_seconds"]) for record in completed_records)
            ),
            "protected_test_accessed": False,
            "protected_test_executions": 0,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
