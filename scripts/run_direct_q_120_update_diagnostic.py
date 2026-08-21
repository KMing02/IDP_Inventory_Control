"""Run the validation-only 120-update direct-q PaperFCNN diagnostic.

The runner uses the production DirectBackprop trainer and observes fixed
optimizer-update checkpoints without selecting among them.  It generates only
train and validation products and compares against the already completed
TargetStockFCNN dynamics artifacts.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Iterator, Sequence
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from inventory_control.idp_benchmark import (  # noqa: E402
    _model_config,
    _oracle_levels,
    _parameter_count,
    _prediction_table,
    _profile_settings,
)
from inventory_control.paper_configs import load_paper_lost_sales_config  # noqa: E402
from inventory_control.paper_lost_sales import (  # noqa: E402
    non_omniscient_gamma_base_stock_policy,
)
from inventory_control.policies.neural.fcnn import (  # noqa: E402
    PaperFCNNFeatureStats,
    compute_paper_fcnn_feature_stats,
    make_paper_fcnn_model,
)
from inventory_control.products import (  # noqa: E402
    PaperLostSalesProduct,
    PaperLostSalesProductGenerator,
)
from inventory_control.training.direct_backprop import (  # noqa: E402
    train_paper_lost_sales_fcnn,
)
from scripts.run_no_imitation_ablation import (  # noqa: E402
    _operational_metrics,
    resolve_no_imitation_config,
)
from scripts.run_target_stock_training_dynamics import (  # noqa: E402
    EVALUATION_UPDATES,
)
from scripts.run_target_stock_validation import (  # noqa: E402
    _historical_gamma_rmse,
    _read_json,
    verify_direct_baseline_reuse,
)
from scripts.tune_primary_neural_policies import (  # noqa: E402
    _atomic_torch_save,
    _atomic_write_csv,
    _atomic_write_json,
    _environment_record,
    _fingerprint,
    _repository_relative_path,
    _safe_generate_products,
    data_fingerprints,
    validate_split_and_output,
)


SEEDS = (0, 1, 2)
COMPARISON_UPDATES = (15, 30, 60, 90, 120)
POLICY_NAME = "PaperFCNNDirectQ120"
DIRECT_BASELINE_ROOT = Path("outputs/idp_followup/no_imitation_matched_width_validation")
TARGET_DYNAMICS_ROOT = Path("outputs/idp_followup/target_stock_training_dynamics")
DEFAULT_OUTPUT_DIR = Path("outputs/idp_followup/direct_q_120_update_diagnostic")


def resolve_direct_q_120_config(
    base_config: dict[str, Any], profile: str, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve the frozen 120-update reward-only direct-q configuration."""
    config, direct = resolve_no_imitation_config(base_config, profile, "PaperFCNN", seed)
    config["idp_benchmark"][profile]["reward_epochs"] = 120
    settings = _profile_settings(config, profile)
    resolved = {
        **direct,
        "policy_name": POLICY_NAME,
        "action_parameterization": "direct_q",
        "history_encoder": "causal_cnn",
        "reward_epochs": 120,
        "reward_optimizer_updates": 120,
        "imitation_optimizer_updates": 0,
        "checkpoint_updates": list(EVALUATION_UPDATES),
        "checkpoint_selection": None,
        "early_stopping": False,
        "training_semantics": "reward_only_direct_backprop_from_standard_random_initialization",
    }
    assert int(settings["reward_epochs"]) == 120
    return config, resolved


@contextmanager
def optimizer_checkpoint_hook(evaluate_checkpoint: Any) -> Iterator[dict[str, int]]:
    """Observe Adam updates without changing gradients, parameters, or RNG."""
    original_step = torch.optim.Adam.step
    state = {"updates": 0}

    def observed_step(optimizer: torch.optim.Adam, closure=None):
        result = original_step(optimizer, closure=closure)
        state["updates"] += 1
        if state["updates"] in EVALUATION_UPDATES:
            evaluate_checkpoint(state["updates"])
        return result

    with patch.object(torch.optim.Adam, "step", observed_step):
        yield state


def _checkpoint_metrics(
    *,
    model: torch.nn.Module,
    seed: int,
    update: int,
    validation_products: list[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
    settings: dict[str, Any],
    oracle_levels: dict[str, float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    was_training = model.training
    operational = _operational_metrics(
        validation_products, POLICY_NAME, model, feature_stats, settings
    )
    predictions = _prediction_table(
        validation_products,
        {POLICY_NAME: model},
        feature_stats,
        settings,
        oracle_levels,
    )
    predictions = predictions.loc[predictions["policy_name"] == POLICY_NAME].copy()
    if was_training:
        model.train()
    cap_hits = int(operational["cap_hit_count"].sum())
    scored_actions = int(operational["scored_action_count"].sum())
    row = {
        "policy_name": POLICY_NAME,
        "training_seed": int(seed),
        "reward_update": int(update),
        "mean_reward_per_period": float(operational["mean_reward_per_period"].mean()),
        "service_level": float(operational["service_level"].mean()),
        "lost_sales_rate": float(operational["lost_sales_rate"].mean()),
        "mean_inventory": float(operational["mean_inventory"].mean()),
        "mean_order_quantity": float(operational["mean_order_quantity"].mean()),
        "mean_holding_cost": float(operational["mean_holding_cost"].mean()),
        "cap_hit_rate": cap_hits / max(scored_actions, 1),
        "deployed_historical_gamma_rmse": _historical_gamma_rmse(
            predictions,
            validation_products,
            float(settings["max_order_factor"]),
        ),
    }
    predictions.insert(0, "reward_update", int(update))
    predictions.insert(0, "training_seed", int(seed))
    return row, predictions


def _fit_dir(output_dir: Path, seed: int) -> Path:
    return output_dir / "runs" / POLICY_NAME / f"seed_{seed}"


def _completed(fit_dir: Path, fingerprint: str) -> dict[str, Any] | None:
    required = (
        "completion.json",
        "resolved_config.json",
        "training_log.csv",
        "checkpoint_metrics.csv",
        "checkpoint_predictions.csv",
        "model_update_120.pt",
    )
    if any(not (fit_dir / name).exists() for name in required):
        return None
    completion = _read_json(fit_dir / "completion.json")
    if completion.get("resolved_config_fingerprint") != fingerprint:
        raise RuntimeError(f"Incompatible completed direct-q fit: {fit_dir}")
    return completion


def run_fit(
    *,
    base_config: dict[str, Any],
    profile: str,
    seed: int,
    output_dir: Path,
    train_products: list[PaperLostSalesProduct],
    validation_products: list[PaperLostSalesProduct],
    feature_stats: PaperFCNNFeatureStats,
    oracle_levels: dict[str, float],
    shared_data_fingerprint: str,
) -> dict[str, Any]:
    config, resolved = resolve_direct_q_120_config(base_config, profile, seed)
    settings = _profile_settings(config, profile)
    resolved["shared_data_fingerprint"] = shared_data_fingerprint
    resolved["resolved_config_fingerprint"] = _fingerprint(resolved)
    fit_dir = _fit_dir(output_dir, seed)
    fit_dir.mkdir(parents=True, exist_ok=True)
    (fit_dir / "completion.json").unlink(missing_ok=True)
    _atomic_write_json(fit_dir / "resolved_config.json", resolved)

    torch.manual_seed(seed)
    model = make_paper_fcnn_model(_model_config(config, settings))
    checkpoint_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    def evaluate(update: int) -> None:
        row, predictions = _checkpoint_metrics(
            model=model,
            seed=seed,
            update=update,
            validation_products=validation_products,
            feature_stats=feature_stats,
            settings=settings,
            oracle_levels=oracle_levels,
        )
        checkpoint_rows.append(row)
        prediction_frames.append(predictions)

    evaluate(0)
    started = time.perf_counter()
    with optimizer_checkpoint_hook(evaluate) as observed:
        result = train_paper_lost_sales_fcnn(
            train_products,
            model,
            horizon=int(settings["training_horizon"]),
            batch_size=min(int(settings["batch_size"]), len(train_products)),
            epochs=120,
            learning_rate=float(config["learning_rate"]),
            seed=int(settings["seed"]),
            imitation_epochs=0,
            validation_products=None,
            validation_interval=0,
            validation_callback=None,
            feature_stats=feature_stats,
            max_order_factor=float(settings["max_order_factor"]),
            scale_action_by_history_mean=bool(settings["scale_action_by_history_mean"]),
            normalize_reward_loss=bool(settings["normalize_reward_loss"]),
            monotonicity_penalty_weight=0.0,
            monotonicity_samples_per_product=int(
                settings["monotonicity_samples_per_product"]
            ),
            include_terminal_salvage=True,
            clip_imitation_predictions=True,
            imitation_early_stopping=False,
            imitation_max_epochs=None,
            track_imitation_validation=False,
            reset_optimizer_after_imitation=False,
            adam_kwargs=dict(config.get("adam", {})),
        )
    wall_seconds = time.perf_counter() - started
    training_log = pd.DataFrame(result.training_log or [])
    checkpoints = pd.DataFrame(checkpoint_rows).sort_values("reward_update")
    if (
        observed["updates"] != 120
        or checkpoints["reward_update"].astype(int).tolist() != list(EVALUATION_UPDATES)
        or len(training_log) != 120
        or not (training_log["phase"] == 1).all()
        or result.imitation_epochs_completed != 0
        or result.best_state_dict is not None
    ):
        raise RuntimeError("Fit was not exactly 120 reward-only updates without selection")
    training_at_checkpoints = training_log.loc[
        training_log["epoch"].isin(EVALUATION_UPDATES),
        ["epoch", "training_loss", "training_mean_total_reward"],
    ].rename(columns={"epoch": "reward_update"})
    checkpoints = checkpoints.merge(training_at_checkpoints, on="reward_update", how="left")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    _atomic_write_csv(fit_dir / "training_log.csv", training_log)
    _atomic_write_csv(fit_dir / "checkpoint_metrics.csv", checkpoints)
    _atomic_write_csv(fit_dir / "checkpoint_predictions.csv", predictions)
    _atomic_torch_save(fit_dir / "model_update_120.pt", model.state_dict())
    completion = {
        "policy_name": POLICY_NAME,
        "training_seed": int(seed),
        "resolved_config_fingerprint": resolved["resolved_config_fingerprint"],
        "shared_data_fingerprint": shared_data_fingerprint,
        "parameter_count": _parameter_count(model),
        "reward_optimizer_updates": 120,
        "imitation_optimizer_updates": 0,
        "checkpoint_selection_used": False,
        "early_stopping_used": False,
        "checkpoint_count": len(checkpoints),
        "training_seconds": float(result.elapsed_seconds),
        "fit_wall_seconds": float(wall_seconds),
        "failed_fit": False,
    }
    _atomic_write_json(fit_dir / "completion.json", completion)
    return completion


def _target_historical_gamma_rmse(
    target_root: Path,
    validation_products: list[PaperLostSalesProduct],
    settings: dict[str, Any],
) -> pd.DataFrame:
    """Reconstruct exact deployed target-stock actions from saved target levels."""
    product_map = {product.product_id: product for product in validation_products}
    rows: list[dict[str, Any]] = []
    paths = sorted(target_root.glob("runs/*/seed_*/checkpoint_target_stocks.csv"))
    for path in paths:
        targets = pd.read_csv(path)
        for (policy, seed, update), group in targets.groupby(
            ["policy_name", "training_seed", "reward_update"], sort=False
        ):
            squared_errors: list[float] = []
            for record in group.itertuples(index=False):
                product = product_map[str(record.product_id)]
                mean = max(float(np.mean(product.historical_demand)), 1.0)
                inventories = np.linspace(
                    0.0,
                    float(settings["max_order_factor"]) * mean,
                    int(settings["inventory_states"]),
                )
                raw = np.maximum(float(record.target_stock_level) - inventories, 0.0)
                cap = np.maximum(float(settings["max_order_factor"]) * mean - inventories, 0.0)
                deployed = np.minimum(raw, cap)
                historical = float(non_omniscient_gamma_base_stock_policy(product).S)
                reference = np.minimum(np.maximum(historical - inventories, 0.0), cap)
                squared_errors.extend(np.square(deployed - reference).tolist())
            rows.append(
                {
                    "policy_name": policy,
                    "training_seed": int(seed),
                    "reward_update": int(update),
                    "deployed_historical_gamma_rmse": float(np.sqrt(np.mean(squared_errors))),
                }
            )
    return pd.DataFrame(rows)


def aggregate_results(
    output_dir: Path,
    target_root: Path,
    validation_products: list[PaperLostSalesProduct],
    settings: dict[str, Any],
) -> None:
    direct = pd.concat(
        [
            pd.read_csv(path)
            for path in sorted((output_dir / "runs").glob("*/seed_*/checkpoint_metrics.csv"))
        ],
        ignore_index=True,
    )
    _atomic_write_csv(output_dir / "direct_q_checkpoint_seed_metrics.csv", direct)
    metric_columns = [
        column
        for column in direct.columns
        if column not in {"policy_name", "training_seed", "reward_update"}
        and pd.api.types.is_numeric_dtype(direct[column])
    ]
    aggregate_rows: list[dict[str, Any]] = []
    for (policy, update), group in direct.groupby(["policy_name", "reward_update"]):
        row: dict[str, Any] = {
            "policy_name": policy,
            "reward_update": int(update),
            "seed_count": len(group),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sample_sd"] = float(group[metric].std(ddof=1))
        aggregate_rows.append(row)
    _atomic_write_csv(output_dir / "direct_q_checkpoint_aggregate.csv", pd.DataFrame(aggregate_rows))

    target = pd.read_csv(target_root / "extended_checkpoint_seed_metrics.csv")
    target_rmse = _target_historical_gamma_rmse(target_root, validation_products, settings)
    target = target.merge(
        target_rmse,
        on=["policy_name", "training_seed", "reward_update"],
        how="left",
        validate="one_to_one",
    )
    common_columns = [
        "policy_name",
        "training_seed",
        "reward_update",
        "mean_reward_per_period",
        "service_level",
        "lost_sales_rate",
        "mean_inventory",
        "mean_order_quantity",
        "mean_holding_cost",
        "cap_hit_rate",
        "deployed_historical_gamma_rmse",
    ]
    matched = pd.concat(
        [direct[common_columns], target[common_columns]], ignore_index=True
    )
    matched = matched.loc[matched["reward_update"].isin(COMPARISON_UPDATES)].copy()
    _atomic_write_csv(output_dir / "matched_checkpoint_seed_results.csv", matched)
    matched_rows: list[dict[str, Any]] = []
    for (policy, update), group in matched.groupby(["policy_name", "reward_update"], sort=False):
        row = {
            "policy_name": policy,
            "reward_update": int(update),
            "seed_count": len(group),
        }
        for metric in common_columns[3:]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sample_sd"] = float(group[metric].std(ddof=1))
        matched_rows.append(row)
    _atomic_write_csv(output_dir / "matched_checkpoint_aggregate.csv", pd.DataFrame(matched_rows))

    final = matched.loc[matched["reward_update"] == 120].pivot(
        index="training_seed", columns="policy_name", values="mean_reward_per_period"
    )
    paired_seed_rows: list[dict[str, Any]] = []
    paired_summary_rows: list[dict[str, Any]] = []
    for target_policy in ("TargetStockFCNNStandard", "TargetStockFCNNInformed"):
        differences = final[target_policy] - final[POLICY_NAME]
        for seed, value in differences.items():
            paired_seed_rows.append(
                {
                    "comparison": f"{target_policy} - {POLICY_NAME}",
                    "training_seed": int(seed),
                    "reward_difference": float(value),
                }
            )
        paired_summary_rows.append(
            {
                "comparison": f"{target_policy} - {POLICY_NAME}",
                "seed_count": len(differences),
                "paired_mean_reward_difference": float(differences.mean()),
                "paired_difference_sample_sd": float(differences.std(ddof=1)),
                "target_wins": int((differences > 0.0).sum()),
                "direct_q_wins": int((differences < 0.0).sum()),
                "ties": int((differences == 0.0).sum()),
            }
        )
    _atomic_write_csv(output_dir / "paired_update_120_seed_differences.csv", pd.DataFrame(paired_seed_rows))
    _atomic_write_csv(output_dir / "paired_update_120_summary.csv", pd.DataFrame(paired_summary_rows))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "run"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", default="full")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--direct-baseline-root", type=Path, default=DIRECT_BASELINE_ROOT)
    parser.add_argument("--target-dynamics-root", type=Path, default=TARGET_DYNAMICS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if tuple(args.seeds) != SEEDS:
        raise ValueError("Direct-q 120-update diagnostic requires exactly seeds 0 1 2")
    validate_split_and_output(args.split, args.output_dir)
    config_path, config_relative = _repository_relative_path(args.config, "--config")
    direct_root, direct_relative = _repository_relative_path(
        args.direct_baseline_root, "--direct-baseline-root"
    )
    target_root, target_relative = _repository_relative_path(
        args.target_dynamics_root, "--target-dynamics-root"
    )
    output_dir, _ = _repository_relative_path(args.output_dir, "--output-dir")
    if direct_relative != DIRECT_BASELINE_ROOT or target_relative != TARGET_DYNAMICS_ROOT:
        raise ValueError("Source roots must be the frozen validation-only outputs")

    base_config = load_paper_lost_sales_config(config_path)
    config, _ = resolve_direct_q_120_config(base_config, args.profile, 0)
    settings = _profile_settings(config, args.profile)
    generator = PaperLostSalesProductGenerator(config)
    train_products = _safe_generate_products(generator, 128, "train")
    validation_products = _safe_generate_products(generator, 32, "validation")
    feature_stats = compute_paper_fcnn_feature_stats(train_products)
    fingerprints = data_fingerprints(config, settings, train_products, validation_products)
    shared_data_fingerprint = _fingerprint(fingerprints)
    direct_audit = verify_direct_baseline_reuse(
        baseline_root=direct_root,
        base_config=base_config,
        profile=args.profile,
        train_products=train_products,
        validation_products=validation_products,
        feature_stats=feature_stats,
        fingerprints=fingerprints,
    )
    target_fingerprints = _read_json(target_root / "data_fingerprints.json")
    target_completion = _read_json(target_root / "extended_completion.json")
    if target_fingerprints.get("shared_data_fingerprint") != shared_data_fingerprint:
        raise RuntimeError("Target-stock dynamics data fingerprint differs")
    if (
        target_completion.get("completed_fits") != 6
        or target_completion.get("reward_optimizer_updates_per_fit") != 120
        or target_completion.get("imitation_optimizer_updates") != 0
        or target_completion.get("checkpoint_selection_used") is not False
        or target_completion.get("early_stopping_used") is not False
        or target_completion.get("protected_test_accessed") is not False
    ):
        raise RuntimeError("Target-stock dynamics completion metadata is incompatible")

    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_record = dict(fingerprints)
    fingerprint_record["shared_data_fingerprint"] = shared_data_fingerprint
    _atomic_write_json(output_dir / "environment.json", _environment_record(config_relative, config))
    _atomic_write_json(output_dir / "data_fingerprints.json", fingerprint_record)
    _atomic_write_json(output_dir / "source_compatibility_audit.json", {
        "direct_q_15_update_baseline": direct_audit,
        "target_stock_dynamics_completed_fits": 6,
        "target_stock_dynamics_shared_data_fingerprint": shared_data_fingerprint,
        "protected_test_accessed": False,
    })
    manifest = pd.DataFrame([
        {
            "policy_name": POLICY_NAME,
            "training_seed": seed,
            "reward_optimizer_updates": 120,
            "imitation_optimizer_updates": 0,
            "checkpoint_updates": ",".join(map(str, EVALUATION_UPDATES)),
            "checkpoint_selection": False,
            "early_stopping": False,
            "shared_data_fingerprint": shared_data_fingerprint,
        }
        for seed in SEEDS
    ])
    _atomic_write_csv(output_dir / "manifest.csv", manifest)
    if args.stage == "audit":
        _atomic_write_json(output_dir / "audit_completion.json", {
            "source_compatibility_verified": True,
            "optimizer_constructions": 0,
            "optimizer_updates": 0,
            "training_function_calls": 0,
            "protected_test_accessed": False,
        })
        print(f"audit=passed shared_data_fingerprint={shared_data_fingerprint} optimizer_updates=0")
        return 0

    oracle_levels = _oracle_levels(
        validation_products, int(settings["oracle_mc_samples"]), int(settings["seed"])
    )
    started = time.perf_counter()
    failures: list[dict[str, Any]] = []
    completed = 0
    for seed in SEEDS:
        config_seed, resolved = resolve_direct_q_120_config(base_config, args.profile, seed)
        del config_seed
        resolved["shared_data_fingerprint"] = shared_data_fingerprint
        expected = _fingerprint(resolved)
        fit_dir = _fit_dir(output_dir, seed)
        existing = _completed(fit_dir, expected) if fit_dir.exists() else None
        if existing is not None and args.resume:
            completed += 1
            print(f"completed={completed}/3 policy={POLICY_NAME} seed={seed} reused=true")
            continue
        if existing is not None:
            raise RuntimeError("Completed diagnostic fits exist; pass --resume")
        try:
            result = run_fit(
                base_config=base_config,
                profile=args.profile,
                seed=seed,
                output_dir=output_dir,
                train_products=train_products,
                validation_products=validation_products,
                feature_stats=feature_stats,
                oracle_levels=oracle_levels,
                shared_data_fingerprint=shared_data_fingerprint,
            )
            completed += 1
            print(
                f"completed={completed}/3 policy={POLICY_NAME} seed={seed} "
                f"elapsed={result['fit_wall_seconds']:.1f}s reused=false"
            )
        except Exception as exc:
            failures.append({"training_seed": seed, "error": repr(exc)})
            fit_dir.mkdir(parents=True, exist_ok=True)
            (fit_dir / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"failed policy={POLICY_NAME} seed={seed} error={type(exc).__name__}")
            break

    markers = list((output_dir / "runs").glob("*/seed_*/completion.json"))
    _atomic_write_csv(
        output_dir / "failures.csv",
        pd.DataFrame(failures, columns=["training_seed", "error"]),
    )
    if len(markers) == 3 and not failures:
        aggregate_results(output_dir, target_root, validation_products, settings)
    wall_seconds = time.perf_counter() - started
    _atomic_write_json(output_dir / "completion.json", {
        "completed_fits": len(markers),
        "expected_fits": 3,
        "failed_fits": len(failures),
        "reward_optimizer_updates_per_fit": 120,
        "imitation_optimizer_updates": 0,
        "checkpoint_selection_used": False,
        "early_stopping_used": False,
        "training_function_calls": len(markers),
        "protected_test_accessed": False,
        "wall_seconds": wall_seconds,
    })
    if failures or len(markers) != 3:
        raise RuntimeError("Direct-q diagnostic did not complete all three fits")
    print(f"completed=3/3 wall_seconds={wall_seconds:.1f} protected_test_accessed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
