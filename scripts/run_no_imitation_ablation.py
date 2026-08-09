"""Run the fixed validation-only, reward-only neural-policy ablation.

This orchestration layer deliberately leaves model and DirectBackprop
mathematics unchanged.  It disables imitation completely and verifies that
every optimizer update belongs to the reward-training phase.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from inventory_control.idp_benchmark import (  # noqa: E402
    _model_config,
    _model_orders,
    _oracle_levels,
    _parameter_count,
    _prediction_summary,
    _prediction_table,
    _profile_settings,
    _train_model,
)
from inventory_control.paper_configs import load_paper_lost_sales_config  # noqa: E402
from inventory_control.paper_evaluation import (  # noqa: E402
    PAPER_VALIDATION_EVALUATION_SEED_START,
)
from inventory_control.paper_lost_sales import evaluate_paper_lost_sales_policy  # noqa: E402
from inventory_control.policies.neural.fcnn import (  # noqa: E402
    PaperFCNNFeatureStats,
    compute_paper_fcnn_feature_stats,
)
from inventory_control.products import (  # noqa: E402
    PaperLostSalesProduct,
    PaperLostSalesProductGenerator,
)
from scripts.run_primary_validation_followups import (  # noqa: E402
    output_diagnostics,
)
from scripts.tune_primary_neural_policies import (  # noqa: E402
    Candidate,
    PRIMARY_POLICIES,
    _atomic_torch_save,
    _atomic_write_csv,
    _atomic_write_json,
    _environment_record,
    _fingerprint,
    _make_model,
    _penalty_pair_recorder,
    _penalty_summary,
    _repository_relative_path,
    _safe_generate_products,
    _write_or_validate_json,
    data_fingerprints,
    validate_split_and_output,
)


FIXED_CANDIDATE = Candidate("NI", 2, 32, 0.001, 15, (1, 1, 1), 0.1)
FIXED_SEEDS = tuple(range(10))
VIOLATION_TOLERANCE = 1e-6
DEFAULT_OUTPUT_DIR = Path("outputs/idp_followup/no_imitation_matched_width_validation")
IMITATION_REFERENCE_ROOT = Path("outputs/idp_followup/matched_width_depth_10seed_validation")
EQUAL_CAPACITY_REFERENCE_ROOT = Path("outputs/idp_followup/equal_capacity_validation")


def resolve_no_imitation_config(
    base_config: dict[str, Any], profile: str, policy: str, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve the frozen ablation settings and explicitly disable imitation."""
    if policy not in PRIMARY_POLICIES:
        raise ValueError(f"Unsupported policy: {policy}")
    config = copy.deepcopy(base_config)
    config["learning_rate"] = FIXED_CANDIDATE.learning_rate
    config["fcnn"]["hidden_layers"] = FIXED_CANDIDATE.hidden_layers
    config["fcnn"]["hidden_units"] = FIXED_CANDIDATE.hidden_units
    config["constrained_monotone_mlp"]["activation_weights"] = list(
        FIXED_CANDIDATE.cmnn_activation_weights
    )
    benchmark = config["idp_benchmark"]
    benchmark["seed"] = int(seed)
    benchmark["hidden_layers"] = FIXED_CANDIDATE.hidden_layers
    benchmark["hidden_units"] = FIXED_CANDIDATE.hidden_units
    benchmark["imitation_early_stopping"] = False
    benchmark["imitation_max_epochs"] = None
    benchmark["track_imitation_validation"] = False
    benchmark["reset_optimizer_after_imitation"] = False
    benchmark[profile]["training_products"] = 128
    benchmark[profile]["imitation_epochs"] = 0
    benchmark[profile]["reward_epochs"] = FIXED_CANDIDATE.reward_epochs
    settings = _profile_settings(config, profile)
    resolved = {
        "policy_name": policy,
        "training_seed": int(seed),
        "hidden_layers": FIXED_CANDIDATE.hidden_layers,
        "hidden_units": FIXED_CANDIDATE.hidden_units,
        "learning_rate": FIXED_CANDIDATE.learning_rate,
        "reward_epochs": FIXED_CANDIDATE.reward_epochs,
        "training_products": int(settings["training_products"]),
        "validation_products": int(settings["validation_products"]),
        "batch_size": int(settings["batch_size"]),
        "training_horizon": int(settings["training_horizon"]),
        "evaluation_horizon": int(settings["evaluation_horizon"]),
        "burn_in": int(settings["burn_in"]),
        "deployment_action_cap_factor": float(settings["max_order_factor"]),
        "normalize_reward_loss": bool(settings["normalize_reward_loss"]),
        "imitation_epochs": int(settings["imitation_epochs"]),
        "imitation_early_stopping": bool(settings["imitation_early_stopping"]),
        "imitation_max_epochs": settings.get("imitation_max_epochs"),
        "track_imitation_validation": bool(settings.get("track_imitation_validation", False)),
        "reset_optimizer_after_imitation": bool(settings["reset_optimizer_after_imitation"]),
        "penalty_lambda": FIXED_CANDIDATE.penalty_lambda if policy == "PenaltyMLP" else None,
        "activation_weights": (
            list(FIXED_CANDIDATE.cmnn_activation_weights)
            if policy == "ConstrainedMonoMLP"
            else None
        ),
        "training_semantics": "reward_only_direct_backprop_from_random_initialization",
    }
    return config, resolved


class _CapTrackingPolicy:
    """Use the standard neural deployment action while recording scored cap hits."""

    def __init__(
        self,
        name: str,
        model: torch.nn.Module,
        product: PaperLostSalesProduct,
        feature_stats: PaperFCNNFeatureStats,
        max_order_factor: float,
        scale_action_by_history_mean: bool,
        burn_in: int,
    ) -> None:
        self.name = name
        self.model = model
        self.product = product
        self.feature_stats = feature_stats
        self.max_order_factor = max_order_factor
        self.scale_action_by_history_mean = scale_action_by_history_mean
        self.burn_in = burn_in
        self.scored_actions = 0
        self.cap_hits = 0
        self.total_clipping = 0.0
        self.maximum_clipping = 0.0

    def act(self, inventory: float, t: int, rng=None) -> float:
        del rng
        raw, deployed = _model_orders(
            self.model,
            self.product,
            self.feature_stats,
            np.asarray([inventory], dtype=float),
            self.max_order_factor,
            self.scale_action_by_history_mean,
        )
        raw_value = float(raw[0])
        deployed_value = float(deployed[0])
        if t >= self.burn_in:
            clipping = max(raw_value - deployed_value, 0.0)
            self.scored_actions += 1
            self.cap_hits += int(clipping > VIOLATION_TOLERANCE)
            self.total_clipping += clipping
            self.maximum_clipping = max(self.maximum_clipping, clipping)
        return deployed_value


def _operational_metrics(
    products: Sequence[PaperLostSalesProduct],
    policy_name: str,
    model: torch.nn.Module,
    feature_stats: PaperFCNNFeatureStats,
    settings: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, product in enumerate(products):
        policy = _CapTrackingPolicy(
            policy_name,
            model,
            product,
            feature_stats,
            float(settings["max_order_factor"]),
            bool(settings["scale_action_by_history_mean"]),
            int(settings["burn_in"]),
        )
        metrics = evaluate_paper_lost_sales_policy(
            product,
            policy,
            horizon=int(settings["evaluation_horizon"]),
            n_episodes=1,
            seed=PAPER_VALIDATION_EVALUATION_SEED_START + index,
            burn_in=int(settings["burn_in"]),
        )
        rows.append(
            {
                "product_id": product.product_id,
                "policy_name": policy_name,
                "evaluation_split": "validation",
                "evaluation_seed": PAPER_VALIDATION_EVALUATION_SEED_START + index,
                **metrics,
                "scored_action_count": policy.scored_actions,
                "cap_hit_count": policy.cap_hits,
                "cap_hit_rate": policy.cap_hits / max(policy.scored_actions, 1),
                "mean_clipping_amount": policy.total_clipping / max(policy.scored_actions, 1),
                "maximum_clipping_amount": policy.maximum_clipping,
            }
        )
    return pd.DataFrame(rows)


def _fit_dir(output_dir: Path, policy: str, seed: int) -> Path:
    return output_dir / "runs" / policy / f"seed_{seed}"


def _load_completed(fit_dir: Path, expected_fingerprint: str) -> dict[str, Any] | None:
    required = (
        "completion.json",
        "resolved_config.json",
        "training_log.csv",
        "operational_metrics.csv",
        "predictions.csv",
        "prediction_diagnostics.csv",
        "model.pt",
    )
    if any(not (fit_dir / name).exists() for name in required):
        return None
    record = json.loads((fit_dir / "completion.json").read_text(encoding="utf-8"))
    if record.get("resolved_config_fingerprint") != expected_fingerprint:
        raise RuntimeError(f"Completed fit has incompatible settings: {fit_dir}")
    return record


def _run_fit(
    *,
    base_config: dict[str, Any],
    profile: str,
    policy: str,
    seed: int,
    output_dir: Path,
    train_products: list[PaperLostSalesProduct],
    validation_products: list[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
    oracle_levels: dict[str, float],
    shared_data_fingerprint: str,
) -> dict[str, Any]:
    config, resolved = resolve_no_imitation_config(base_config, profile, policy, seed)
    settings = _profile_settings(config, profile)
    resolved["shared_data_fingerprint"] = shared_data_fingerprint
    resolved["resolved_config_fingerprint"] = _fingerprint(resolved)
    fit_dir = _fit_dir(output_dir, policy, seed)
    fit_dir.mkdir(parents=True, exist_ok=True)
    (fit_dir / "completion.json").unlink(missing_ok=True)
    _atomic_write_json(fit_dir / "resolved_config.json", resolved)

    model = _make_model(policy, _model_config(config, settings), seed)
    completion_path = output_dir / "analysis_completion.json"
    prior_completion = (
        json.loads(completion_path.read_text(encoding="utf-8"))
        if completion_path.exists()
        else None
    )
    started = time.perf_counter()
    with _penalty_pair_recorder(policy == "PenaltyMLP") as pair_records:
        result = _train_model(
            config,
            settings,
            train_products,
            validation_products,
            feature_stats,
            model,
            policy,
            "validation",
            penalty_weight=(FIXED_CANDIDATE.penalty_lambda if policy == "PenaltyMLP" else 0.0),
            clip_imitation_predictions=True,
        )
    wall_seconds = time.perf_counter() - started
    training_log = pd.DataFrame(result.training_log or [])
    if result.imitation_epochs_completed != 0 or result.best_imitation_state_dict is not None:
        raise RuntimeError("No-imitation fit unexpectedly produced imitation training state")
    if len(training_log) != FIXED_CANDIDATE.reward_epochs or not (training_log["phase"] == 1).all():
        raise RuntimeError("No-imitation fit did not contain exactly 15 reward-only updates")

    predictions = _prediction_table(
        validation_products, {policy: model}, feature_stats, settings, oracle_levels
    )
    predictions = predictions.loc[predictions["policy_name"] == policy].reset_index(drop=True)
    prediction_summary = _prediction_summary(predictions)
    operational = _operational_metrics(
        validation_products, policy, model, feature_stats, settings
    )
    diagnostics = prediction_summary.iloc[0].to_dict()
    cap_hits = int(operational["cap_hit_count"].sum())
    scored_actions = int(operational["scored_action_count"].sum())
    aggregate_columns = (
        "mean_total_reward",
        "mean_reward_per_period",
        "service_level",
        "lost_sales_rate",
        "mean_order_quantity",
        "mean_inventory",
        "mean_holding_cost",
    )
    aggregate = {column: float(operational[column].mean()) for column in aggregate_columns}
    aggregate.update(
        {
            "cap_hit_count": cap_hits,
            "scored_action_count": scored_actions,
            "cap_hit_rate": cap_hits / max(scored_actions, 1),
            "mean_clipping_amount": float(
                (operational["mean_clipping_amount"] * operational["scored_action_count"]).sum()
                / max(scored_actions, 1)
            ),
            "maximum_clipping_amount": float(operational["maximum_clipping_amount"].max()),
        }
    )
    penalty = _penalty_summary(
        policy,
        FIXED_CANDIDATE.penalty_lambda if policy == "PenaltyMLP" else None,
        training_log,
        pair_records,
    )
    output = output_diagnostics(predictions)
    record = {
        "policy_name": policy,
        "training_seed": int(seed),
        "resolved_config_fingerprint": resolved["resolved_config_fingerprint"],
        "shared_data_fingerprint": shared_data_fingerprint,
        "parameter_count": _parameter_count(model),
        "training_seconds": float(result.elapsed_seconds),
        "fit_wall_seconds": float(wall_seconds),
        "imitation_optimizer_updates": 0,
        "reward_optimizer_updates": int(len(training_log)),
        "failed_fit": False,
        **aggregate,
        **diagnostics,
        **output,
        **penalty,
    }
    if policy == "PenaltyMLP":
        pairs = pd.DataFrame(pair_records)
        pairs.insert(0, "reward_epoch", np.arange(1, len(pairs) + 1))
        _atomic_write_csv(fit_dir / "penalty_pair_diagnostics.csv", pairs)
        record["penalty_active"] = bool(
            float(record.get("penalty_loss_max", 0.0)) > 0.0
            or int(record.get("violating_pair_count", 0)) > 0
        )
    _atomic_write_csv(fit_dir / "training_log.csv", training_log)
    _atomic_write_csv(fit_dir / "operational_metrics.csv", operational)
    _atomic_write_csv(fit_dir / "predictions.csv", predictions)
    _atomic_write_csv(fit_dir / "prediction_diagnostics.csv", prediction_summary)
    _atomic_torch_save(fit_dir / "model.pt", model.state_dict())
    _atomic_write_json(fit_dir / "fit_result.json", record)
    _atomic_write_json(fit_dir / "completion.json", record)
    return record


def _scan_results(output_dir: Path) -> pd.DataFrame:
    records = []
    for marker in sorted((output_dir / "runs").glob("*/seed_*/completion.json")):
        records.append(json.loads(marker.read_text(encoding="utf-8")))
    return pd.DataFrame(records)


def _aggregate(seed_results: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "mean_reward_per_period",
        "mean_inventory",
        "service_level",
        "lost_sales_rate",
        "mean_order_quantity",
        "mean_holding_cost",
        "deployed_rmse",
        "raw_violation_rate",
        "deployed_violation_rate",
        "cap_hit_rate",
        "non_finite_output_rate",
    )
    rows = []
    for policy, group in seed_results.groupby("policy_name", sort=False):
        row: dict[str, Any] = {
            "policy_name": policy,
            "completed_seed_count": len(group),
            "failed_fit_count": len(FIXED_SEEDS) - len(group),
            "parameter_count": int(group["parameter_count"].iloc[0]),
            "reward_sample_sd": float(group["mean_reward_per_period"].std(ddof=1)),
        }
        for metric in metrics:
            row[metric] = float(group[metric].mean())
        row["raw_max_violation"] = float(group["raw_max_violation"].max())
        row["deployed_max_violation"] = float(group["deployed_max_violation"].max())
        rows.append(row)
    return pd.DataFrame(rows)


def _reference_results(root: Path) -> pd.DataFrame:
    path = root / "seed_results.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    return frame


def _write_comparison(output_dir: Path, no_imitation: pd.DataFrame) -> None:
    matched = _reference_results(REPOSITORY_ROOT / IMITATION_REFERENCE_ROOT)
    equal = _reference_results(REPOSITORY_ROOT / EQUAL_CAPACITY_REFERENCE_ROOT)
    references = []
    if not matched.empty:
        references.append(matched.loc[matched["policy_name"].isin(["PaperFCNN", "ConstrainedMonoMLP"])])
    if not equal.empty:
        references.append(equal.loc[equal["policy_name"].isin(["PenaltyMLP", "PositiveWeightMLP"])])
    if not references:
        return
    assisted = pd.concat(references, ignore_index=True)
    common = [
        column
        for column in (
            "policy_name",
            "training_seed",
            "mean_reward_per_period",
            "mean_inventory",
            "service_level",
            "lost_sales_rate",
            "mean_order_quantity",
            "deployed_rmse",
            "raw_violation_rate",
            "deployed_violation_rate",
            "raw_max_violation",
            "deployed_max_violation",
        )
        if column in assisted.columns and column in no_imitation.columns
    ]
    assisted = assisted[common].copy()
    assisted["training_regime"] = "imitation_assisted"
    reward_only = no_imitation[common].copy()
    reward_only["training_regime"] = "reward_only"
    seed_comparison = pd.concat([assisted, reward_only], ignore_index=True)
    _atomic_write_csv(output_dir / "imitation_comparison_seed_results.csv", seed_comparison)
    rows = []
    for (policy, regime), group in seed_comparison.groupby(
        ["policy_name", "training_regime"], sort=False
    ):
        rows.append(
            {
                "policy_name": policy,
                "training_regime": regime,
                "seed_count": len(group),
                "mean_reward_per_period": float(group["mean_reward_per_period"].mean()),
                "reward_sample_sd": float(group["mean_reward_per_period"].std(ddof=1)),
                "mean_inventory": float(group["mean_inventory"].mean()),
                "service_level": float(group["service_level"].mean()),
                "lost_sales_rate": float(group["lost_sales_rate"].mean()),
                "mean_order_quantity": float(group["mean_order_quantity"].mean()),
                "deployed_rmse": float(group["deployed_rmse"].mean()),
                "raw_violation_rate": float(group["raw_violation_rate"].mean()),
                "deployed_violation_rate": float(group["deployed_violation_rate"].mean()),
                "raw_max_violation": float(group["raw_max_violation"].max()),
                "deployed_max_violation": float(group["deployed_max_violation"].max()),
            }
        )
    _atomic_write_csv(output_dir / "imitation_comparison_aggregate.csv", pd.DataFrame(rows))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", default="full")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--policies", nargs="+", default=list(PRIMARY_POLICIES))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_split_and_output(args.split, args.output_dir)
    if tuple(args.seeds) != FIXED_SEEDS:
        raise ValueError("No-imitation ablation requires exactly seeds 0 through 9")
    if tuple(args.policies) != PRIMARY_POLICIES:
        raise ValueError("No-imitation ablation requires all four compatible primary policies")
    config_path, config_relative = _repository_relative_path(args.config, "--config")
    output_dir, _ = _repository_relative_path(args.output_dir, "--output-dir")
    base_config = load_paper_lost_sales_config(config_path)
    settings = _profile_settings(base_config, args.profile)
    if (
        int(settings["training_products"]) != 128
        or int(settings["validation_products"]) != 32
        or float(settings["max_order_factor"]) != 4.0
    ):
        raise ValueError("Ablation requires 128 train products, 32 validation products, and F=4")

    generator = PaperLostSalesProductGenerator(base_config)
    train_products = _safe_generate_products(generator, 128, "train")
    validation_products = _safe_generate_products(generator, 32, "validation")
    feature_stats = compute_paper_fcnn_feature_stats(train_products)
    oracle_levels = _oracle_levels(validation_products, int(settings["oracle_mc_samples"]), 10_000)
    fingerprints = data_fingerprints(base_config, settings, train_products, validation_products)
    shared_data_fingerprint = _fingerprint(fingerprints)
    fingerprints["shared_data_fingerprint"] = shared_data_fingerprint

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_or_validate_json(output_dir / "environment.json", _environment_record(config_relative, base_config))
    _write_or_validate_json(output_dir / "data_fingerprints.json", fingerprints)
    manifest = pd.DataFrame(
        [
            {
                "policy_name": policy,
                "training_seed": seed,
                "hidden_layers": 2,
                "hidden_units": 32,
                "learning_rate": 0.001,
                "reward_epochs": 15,
                "imitation_epochs": 0,
                "penalty_lambda": 0.1 if policy == "PenaltyMLP" else np.nan,
                "activation_weights": "[1, 1, 1]" if policy == "ConstrainedMonoMLP" else "",
            }
            for policy in args.policies
            for seed in args.seeds
        ]
    )
    _atomic_write_csv(output_dir / "manifest.csv", manifest)
    if args.dry_run:
        print(manifest.to_string(index=False))
        return 0

    started = time.perf_counter()
    completed = 0
    failures: list[dict[str, Any]] = []
    total = len(manifest)
    for policy in args.policies:
        for seed in args.seeds:
            _, resolved = resolve_no_imitation_config(base_config, args.profile, policy, seed)
            resolved["shared_data_fingerprint"] = shared_data_fingerprint
            expected = _fingerprint(resolved)
            fit_dir = _fit_dir(output_dir, policy, seed)
            existing = _load_completed(fit_dir, expected) if fit_dir.exists() else None
            if existing is not None and args.resume:
                completed += 1
                print(
                    f"completed={completed}/{total} policy={policy} seed={seed} "
                    f"reused=true reward={existing['mean_reward_per_period']:.6f}"
                )
                continue
            if existing is not None:
                raise RuntimeError("Completed fits exist; pass --resume")
            try:
                record = _run_fit(
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
                )
                completed += 1
                current = _scan_results(output_dir)
                _atomic_write_csv(output_dir / "seed_results.csv", current)
                _atomic_write_csv(output_dir / "aggregate_results.csv", _aggregate(current))
                print(
                    f"completed={completed}/{total} policy={policy} seed={seed} "
                    f"reused=false reward={record['mean_reward_per_period']:.6f}"
                )
            except Exception as exc:
                failures.append({"policy_name": policy, "training_seed": seed, "error": repr(exc)})
                fit_dir.mkdir(parents=True, exist_ok=True)
                (fit_dir / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
                _atomic_write_csv(output_dir / "failures.csv", pd.DataFrame(failures))
                print(f"failed policy={policy} seed={seed} error={type(exc).__name__}")

    seed_results = _scan_results(output_dir)
    _atomic_write_csv(output_dir / "seed_results.csv", seed_results)
    _atomic_write_csv(output_dir / "aggregate_results.csv", _aggregate(seed_results))
    _write_comparison(output_dir, seed_results)
    _atomic_write_csv(
        output_dir / "failures.csv",
        pd.DataFrame(failures, columns=["policy_name", "training_seed", "error"]),
    )
    elapsed = time.perf_counter() - started
    if len(seed_results) == total and not failures and prior_completion is not None:
        elapsed = max(elapsed, float(prior_completion.get("wall_seconds", 0.0)))
    _atomic_write_json(
        completion_path,
        {
            "completed_fits": len(seed_results),
            "expected_fits": total,
            "failed_fits": len(failures),
            "wall_seconds": time.perf_counter() - started,
            "protected_test_accessed": False,
            "imitation_enabled": False,
        },
    )
    return 0 if not failures and len(seed_results) == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
