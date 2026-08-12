"""Run the frozen FCNN-versus-CMNN final evaluation without any training.

The default invocation is a validation-only dry run. Access to the protected
test split fails closed unless both ``--authorize-protected-test`` and the
exact frozen protocol fingerprint are supplied. The runner loads existing
post-reward checkpoints and never constructs an optimizer or calls a training
function.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from inventory_control.idp_benchmark import (  # noqa: E402
    _model_config,
    _operational_table,
    _oracle_levels,
    _prediction_summary,
    _prediction_table,
    _profile_settings,
)
from inventory_control.paper_configs import load_paper_lost_sales_config  # noqa: E402
from inventory_control.paper_evaluation import (  # noqa: E402
    PAPER_TEST_EVALUATION_SEED_START,
    PAPER_VALIDATION_EVALUATION_SEED_START,
    require_final_test_confirmation,
)
from inventory_control.policies.neural.constrained_mlp import (  # noqa: E402
    make_constrained_monotone_mlp_model,
)
from inventory_control.policies.neural.fcnn import (  # noqa: E402
    compute_paper_fcnn_feature_stats,
    make_paper_fcnn_model,
)
from inventory_control.products import (  # noqa: E402
    PaperLostSalesProduct,
    PaperLostSalesProductGenerator,
)


FROZEN_POLICIES = ("PaperFCNN", "ConstrainedMonoMLP")
FROZEN_SEEDS = tuple(range(10))
FROZEN_RESULTS_ROOT = Path("outputs/idp_followup/matched_width_depth_10seed_validation")
DEFAULT_CONFIG = Path("configs/paper_lost_sales_medium_seed0_converged.yaml")
DEFAULT_PROTOCOL = Path("outputs/idp_followup/final_test_protocol.json")
DEFAULT_VALIDATION_OUTPUT = Path("outputs/idp_followup/final_protocol_validation_dry_run")
FROZEN_TEST_OUTPUT = Path("outputs/idp_followup/final_protected_test_evaluation")
VALIDATION_TOLERANCE = 1e-9
VIOLATION_TOLERANCE = 1e-6

OUTPUT_FILES = {
    "checkpoint_manifest": "checkpoint_manifest.csv",
    "seed_results": "seed_results.csv",
    "aggregate_results": "aggregate_results.csv",
    "paired_reward_statistics": "paired_reward_statistics.json",
    "operational_metrics": "operational_metrics.csv",
    "predictions": "predictions.csv",
    "reward_reconstruction": "reward_reconstruction.csv",
    "evaluation_metadata": "evaluation_metadata.json",
    "completion": "completion.json",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, default=_json_default, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, default=_json_default, indent=2, sort_keys=True))


def _atomic_write_csv(path: Path, value: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    value.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _repository_path(path: Path, label: str) -> tuple[Path, Path]:
    if path.is_absolute():
        raise ValueError(f"{label} must be repository-relative")
    resolved = (REPOSITORY_ROOT / path).resolve()
    try:
        relative = resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the repository") from exc
    return resolved, relative


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={REPOSITORY_ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _validate_evaluation_request(
    split: str,
    authorize_protected_test: bool,
    expected_protocol_fingerprint: str | None,
    output_relative: Path,
) -> None:
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    require_final_test_confirmation(split, authorize_protected_test)
    if split == "test":
        if not expected_protocol_fingerprint:
            raise ValueError("Protected test requires --expected-protocol-fingerprint")
        if output_relative.as_posix() != FROZEN_TEST_OUTPUT.as_posix():
            raise ValueError(
                f"Protected-test output is frozen to {FROZEN_TEST_OUTPUT.as_posix()}"
            )
    elif authorize_protected_test:
        raise ValueError("--authorize-protected-test is invalid for a validation dry run")


def _safe_generate_products(
    generator: PaperLostSalesProductGenerator,
    count: int,
    split: str,
    authorize_protected_test: bool,
) -> list[PaperLostSalesProduct]:
    if split == "test" and not authorize_protected_test:
        raise RuntimeError("Protected-test generation was not explicitly authorized")
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported product split: {split}")
    return generator.generate(count, split=split)


def _assert_frozen_resolved_config(record: dict[str, Any], policy: str, seed: int) -> None:
    expected = {
        "policy_name": policy,
        "training_seed": seed,
        "hidden_layers": 2,
        "hidden_units": 32,
        "learning_rate": 0.001,
        "reward_epochs": 15,
        "imitation_early_stopping": True,
        "imitation_max_epochs": 100,
        "imitation_patience": 10,
        "imitation_min_delta": 0.5,
        "reset_optimizer_after_imitation": False,
        "deployment_action_cap_factor": 4.0,
        "evaluation_horizon": 120,
        "burn_in": 20,
        "inventory_states": 50,
        "imitation_semantics": "deployed_prediction_vs_capped_target",
        "imitation_checkpoint_metric": "deployed_imitation_rmse",
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(
                f"Frozen configuration mismatch for {policy} seed {seed}: "
                f"{key}={record.get(key)!r}, expected {value!r}"
            )
    expected_allocation = [1, 1, 1] if policy == "ConstrainedMonoMLP" else None
    if record.get("activation_weights") != expected_allocation:
        raise RuntimeError(f"Frozen activation allocation mismatch for {policy} seed {seed}")


def _load_checkpoint_manifest(results_root: Path) -> pd.DataFrame:
    reuse = pd.read_csv(results_root / "reuse_manifest.csv")
    reuse = reuse.loc[reuse["policy_name"].isin(FROZEN_POLICIES)].copy()
    expected = {(policy, seed) for policy in FROZEN_POLICIES for seed in FROZEN_SEEDS}
    observed = {
        (str(row.policy_name), int(row.training_seed)) for row in reuse.itertuples(index=False)
    }
    if observed != expected or len(reuse) != len(expected):
        raise RuntimeError("Frozen checkpoint manifest is not exactly the 20 matched policy/seed fits")

    environment = json.loads((results_root / "environment.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for row in reuse.sort_values(["policy_name", "training_seed"]).itertuples(index=False):
        policy = str(row.policy_name)
        seed = int(row.training_seed)
        source_relative = Path(str(row.source_run_dir))
        source, _ = _repository_path(source_relative, "source_run_dir")
        checkpoint = source / "model.pt"
        resolved_path = source / "resolved_config.json"
        completion_path = source / "completion.json"
        training_log_path = source / "training_log.csv"
        if (
            not checkpoint.exists()
            or not resolved_path.exists()
            or not completion_path.exists()
            or not training_log_path.exists()
        ):
            raise RuntimeError(f"Incomplete frozen source run: {source_relative.as_posix()}")
        actual_hash = _sha256(checkpoint)
        if actual_hash != str(row.checkpoint_sha256):
            raise RuntimeError(f"Checkpoint fingerprint mismatch: {checkpoint}")
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        _assert_frozen_resolved_config(resolved, policy, seed)
        json.loads(completion_path.read_text(encoding="utf-8"))
        training_log = pd.read_csv(training_log_path)
        reward_optimizer_updates = int((training_log["phase"] == 1).sum())
        if reward_optimizer_updates != 15:
            raise RuntimeError(f"Frozen fit does not contain exactly 15 reward updates: {source}")
        rows.append(
            {
                "policy_name": policy,
                "training_seed": seed,
                "checkpoint_path": checkpoint.relative_to(REPOSITORY_ROOT).as_posix(),
                "checkpoint_sha256": actual_hash,
                "source_run_dir": source_relative.as_posix(),
                "resolved_config_path": resolved_path.relative_to(REPOSITORY_ROOT).as_posix(),
                "resolved_config_fingerprint": resolved.get("resolved_config_fingerprint"),
                "hidden_layers": 2,
                "hidden_units": 32,
                "learning_rate": 0.001,
                "reward_updates": 15,
                "activation_allocation": (
                    "[1, 1, 1]" if policy == "ConstrainedMonoMLP" else ""
                ),
                "checkpoint_training_git_commit": environment["git_commit"],
            }
        )
    return pd.DataFrame(rows)


def _protocol_payload(
    checkpoint_manifest: pd.DataFrame,
    config_relative: Path,
    checkpoint_training_commit: str,
) -> dict[str, Any]:
    return {
        "protocol_version": 1,
        "status": "frozen_pre_test",
        "scope": "primary confirmatory matched hidden-width/depth direct-policy comparison",
        "models": list(FROZEN_POLICIES),
        "training_seeds": list(FROZEN_SEEDS),
        "checkpoint_training_git_commit": checkpoint_training_commit,
        "protocol_preparation_git_commit": _git_commit(),
        "base_config": config_relative.as_posix(),
        "checkpoint_manifest": checkpoint_manifest.to_dict(orient="records"),
        "frozen_model_configuration": {
            "PaperFCNN": {
                "hidden_layers": 2,
                "hidden_units": 32,
                "learning_rate": 0.001,
                "reward_updates": 15,
                "training_pipeline": "existing imitation-assisted DirectBackprop",
            },
            "ConstrainedMonoMLP": {
                "hidden_layers": 2,
                "hidden_units": 32,
                "learning_rate": 0.001,
                "reward_updates": 15,
                "activation_allocation": [1, 1, 1],
                "training_pipeline": "existing imitation-assisted DirectBackprop",
            },
        },
        "frozen_evaluation": {
            "evaluation_only": True,
            "training_or_optimizer_calls_permitted": False,
            "product_count": 32,
            "evaluation_horizon": 120,
            "burn_in": 20,
            "scored_periods": 100,
            "terminal_inventory_value": "purchase_cost * final_inventory, added exactly once",
            "deployment_action_cap_factor": 4.0,
            "inventory_grid_states": 50,
            "inventory_grid_range": "0 to 4 * historical_mean_demand",
            "violation_tolerance": VIOLATION_TOLERANCE,
            "validation_evaluation_seeds": [40000, 40031],
            "protected_test_evaluation_seeds": [30000, 30031],
        },
        "primary_metrics": {
            "operational": [
                "mean_reward_per_period",
                "service_level",
                "lost_sales_rate",
                "mean_inventory",
                "mean_order_quantity",
                "mean_holding_cost",
                "deployed_rmse",
            ],
            "structural": [
                "raw_adjacent_grid_violation_rate",
                "deployed_adjacent_grid_violation_rate",
                "maximum_positive_adjacent_raw_action_increase",
                "active_region_order_up_to_slope_error",
            ],
        },
        "reward_statistics": [
            "policy mean and sample standard deviation",
            "paired ConstrainedMonoMLP minus PaperFCNN seed differences",
            "mean paired difference",
            "95 percent Student-t confidence interval",
            "two-sided paired t-test",
            "two-sided Wilcoxon signed-rank test with zero_method=wilcox",
            "seed win counts and ties",
        ],
        "interpretation_questions": [
            "evidence for a meaningful FCNN-versus-CMNN reward difference",
            "CMNN zero observed violations consistent with its architectural guarantee",
            "empirical FCNN violations without treating their absence as a guarantee",
            "CMNN order-up-to slope error despite monotonicity",
        ],
        "excluded_from_primary_test": [
            "no-imitation ablation",
            "sample-efficiency analysis",
            "base-stock-informed initialization ablation",
            "product selection based on model outputs",
            "validation-selected stress cases",
            "residual order-up-to family",
        ],
        "output_directory": FROZEN_TEST_OUTPUT.as_posix(),
        "output_files": OUTPUT_FILES,
        "decision_freeze": (
            "No model, checkpoint, metric, statistical test, or primary conclusion may change "
            "after protected-test results are observed."
        ),
        "protected_test_accessed_during_protocol_preparation": False,
    }


def _write_or_validate_protocol(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    fingerprint = _fingerprint(payload)
    document = {**payload, "protocol_fingerprint": fingerprint}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != document:
            raise RuntimeError("Existing final-test protocol differs from the frozen checkpoint/specification")
    else:
        _atomic_write_json(path, document)
    return document


def _product_records(products: Sequence[PaperLostSalesProduct]) -> list[dict[str, Any]]:
    return [asdict(product) for product in products]


def _validation_fingerprints(
    config: dict[str, Any],
    settings: dict[str, Any],
    train_products: Sequence[PaperLostSalesProduct],
    validation_products: Sequence[PaperLostSalesProduct],
) -> dict[str, Any]:
    trace_records = [
        {
            "split": split,
            "product_id": product.product_id,
            "historical_demand": product.historical_demand,
            "future_demand": product.future_demand,
        }
        for split, products in (("train", train_products), ("validation", validation_products))
        for product in products
    ]
    grid_records = []
    for product in validation_products:
        historical_mean = max(float(np.mean(product.historical_demand)), 1.0)
        maximum_inventory = float(settings["max_order_factor"]) * historical_mean
        grid_records.append(
            {
                "product_id": product.product_id,
                "imitation_inventory": [
                    fraction * maximum_inventory for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
                ],
                "prediction_inventory": np.linspace(
                    0.0, maximum_inventory, int(settings["inventory_states"])
                ).tolist(),
            }
        )
    evaluation_seeds = [
        PAPER_VALIDATION_EVALUATION_SEED_START + index
        for index in range(len(validation_products))
    ]
    records = {
        "product_generation_seed": int(config["product_generation"].get("seed", 0)),
        "validation_grid_seed": 2_000_003,
        "evaluation_seed_start": PAPER_VALIDATION_EVALUATION_SEED_START,
        "training_product_count": len(train_products),
        "validation_product_count": len(validation_products),
        "training_products_fingerprint": _fingerprint(_product_records(train_products)),
        "validation_products_fingerprint": _fingerprint(_product_records(validation_products)),
        "demand_traces_fingerprint": _fingerprint(trace_records),
        "validation_grid_fingerprint": _fingerprint(grid_records),
        "operational_evaluation_scenarios_fingerprint": _fingerprint(
            {
                "products": _product_records(validation_products),
                "evaluation_seeds": evaluation_seeds,
                "evaluation_horizon": int(settings["evaluation_horizon"]),
                "burn_in": int(settings["burn_in"]),
            }
        ),
    }
    shared_data_fingerprint = _fingerprint(records)
    records["validation_demand_traces_fingerprint"] = _fingerprint(
        [
            {
                "product_id": product.product_id,
                "historical_demand": product.historical_demand,
                "future_demand": product.future_demand,
            }
            for product in validation_products
        ]
    )
    records["shared_data_fingerprint"] = shared_data_fingerprint
    return records


def _make_frozen_model(policy: str, model_config: dict[str, Any]) -> torch.nn.Module:
    if policy == "PaperFCNN":
        return make_paper_fcnn_model(model_config)
    if policy == "ConstrainedMonoMLP":
        return make_constrained_monotone_mlp_model(model_config)
    raise ValueError(f"Unsupported frozen policy: {policy}")


def _reward_reconstruction(operational: pd.DataFrame, scored_periods: int) -> pd.DataFrame:
    frame = operational.copy()
    frame["reconstructed_reward_per_period"] = (
        frame["mean_sales_revenue"]
        - frame["mean_purchase_cost"]
        - frame["mean_lost_sales_penalty"]
        - frame["mean_holding_cost"]
        + frame["mean_terminal_inventory_value"] / max(scored_periods, 1)
    )
    frame["reconstruction_error"] = (
        frame["reconstructed_reward_per_period"] - frame["mean_reward_per_period"]
    )
    return frame[
        [
            "policy_name",
            "training_seed",
            "product_id",
            "mean_reward_per_period",
            "reconstructed_reward_per_period",
            "reconstruction_error",
        ]
    ]


def _paired_statistics(seed_results: pd.DataFrame) -> dict[str, Any]:
    rewards = seed_results.pivot(
        index="training_seed", columns="policy_name", values="mean_reward_per_period"
    ).sort_index()
    differences = rewards["ConstrainedMonoMLP"] - rewards["PaperFCNN"]
    count = len(differences)
    mean = float(differences.mean())
    sample_std = float(differences.std(ddof=1))
    half_width = float(stats.t.ppf(0.975, count - 1) * sample_std / np.sqrt(count))
    paired_t = stats.ttest_rel(rewards["ConstrainedMonoMLP"], rewards["PaperFCNN"])
    if np.allclose(differences.to_numpy(dtype=float), 0.0):
        wilcoxon_statistic = 0.0
        wilcoxon_pvalue = 1.0
    else:
        wilcoxon = stats.wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
        wilcoxon_statistic = float(wilcoxon.statistic)
        wilcoxon_pvalue = float(wilcoxon.pvalue)
    return {
        "seed_count": count,
        "PaperFCNN_mean_reward_per_period": float(rewards["PaperFCNN"].mean()),
        "PaperFCNN_reward_sample_std": float(rewards["PaperFCNN"].std(ddof=1)),
        "ConstrainedMonoMLP_mean_reward_per_period": float(
            rewards["ConstrainedMonoMLP"].mean()
        ),
        "ConstrainedMonoMLP_reward_sample_std": float(
            rewards["ConstrainedMonoMLP"].std(ddof=1)
        ),
        "paired_differences_cmnn_minus_paper_fcnn": differences.to_dict(),
        "mean_paired_difference_cmnn_minus_paper_fcnn": mean,
        "sample_std_paired_difference": sample_std,
        "confidence_interval_95": [mean - half_width, mean + half_width],
        "paired_t_statistic": float(paired_t.statistic),
        "paired_t_pvalue": float(paired_t.pvalue),
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_pvalue": wilcoxon_pvalue,
        "cmnn_seed_wins": int((differences > 0).sum()),
        "paper_fcnn_seed_wins": int((differences < 0).sum()),
        "ties": int((differences == 0).sum()),
    }


def _aggregate_seed_results(seed_results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_reward_per_period",
        "service_level",
        "lost_sales_rate",
        "mean_inventory",
        "mean_order_quantity",
        "mean_holding_cost",
        "deployed_rmse",
        "raw_adjacent_grid_violation_rate",
        "deployed_adjacent_grid_violation_rate",
        "maximum_positive_adjacent_raw_action_increase",
        "active_region_order_up_to_slope_error",
    ]
    rows = []
    for policy, group in seed_results.groupby("policy_name", sort=False):
        row: dict[str, Any] = {
            "policy_name": policy,
            "seed_count": len(group),
            "parameter_count": int(group["parameter_count"].iloc[0]),
        }
        for metric in metrics:
            row[metric] = float(group[metric].mean())
        row["reward_per_period_sample_sd"] = float(
            group["mean_reward_per_period"].std(ddof=1)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _validation_reference_values(source: Path, policy: str) -> dict[str, float]:
    operational = pd.read_csv(source / "operational_metrics.csv")
    operational = operational.loc[operational["policy_name"] == policy]
    prediction = pd.read_csv(source / "prediction_diagnostics.csv")
    prediction = prediction.loc[prediction["policy_name"] == policy].iloc[0]
    return {
        "mean_reward_per_period": float(operational["mean_reward_per_period"].mean()),
        "service_level": float(operational["service_level"].mean()),
        "lost_sales_rate": float(operational["lost_sales_rate"].mean()),
        "mean_inventory": float(operational["mean_inventory"].mean()),
        "mean_order_quantity": float(operational["mean_order_quantity"].mean()),
        "mean_holding_cost": float(operational["mean_holding_cost"].mean()),
        "deployed_rmse": float(prediction["deployed_rmse"]),
        "raw_adjacent_grid_violation_rate": float(prediction["raw_violation_rate"]),
        "deployed_adjacent_grid_violation_rate": float(prediction["deployed_violation_rate"]),
        "maximum_positive_adjacent_raw_action_increase": float(prediction["raw_max_violation"]),
        "active_region_order_up_to_slope_error": float(
            prediction["active_slope_deviation_from_minus_one"]
        ),
    }


def _validation_comparison(
    seed_results: pd.DataFrame, checkpoint_manifest: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    indexed = checkpoint_manifest.set_index(["policy_name", "training_seed"])
    for result in seed_results.itertuples(index=False):
        key = (str(result.policy_name), int(result.training_seed))
        source = REPOSITORY_ROOT / str(indexed.loc[key, "source_run_dir"])
        expected = _validation_reference_values(source, key[0])
        for metric, expected_value in expected.items():
            observed_value = float(getattr(result, metric))
            absolute_difference = abs(observed_value - expected_value)
            rows.append(
                {
                    "policy_name": key[0],
                    "training_seed": key[1],
                    "metric": metric,
                    "frozen_validation_value": expected_value,
                    "dry_run_value": observed_value,
                    "absolute_difference": absolute_difference,
                    "within_tolerance": absolute_difference <= VALIDATION_TOLERANCE,
                }
            )
    return pd.DataFrame(rows)


def _run_evaluation(
    *,
    split: str,
    config: dict[str, Any],
    settings: dict[str, Any],
    checkpoint_manifest: pd.DataFrame,
    results_root: Path,
    output_dir: Path,
    output_relative: Path,
    protocol: dict[str, Any],
    authorize_protected_test: bool,
) -> None:
    generator = PaperLostSalesProductGenerator(config)
    train_products = _safe_generate_products(
        generator, int(settings["training_products"]), "train", False
    )
    evaluation_count = (
        int(settings["validation_products"])
        if split == "validation"
        else int(config["test_products"])
    )
    evaluation_products = _safe_generate_products(
        generator, evaluation_count, split, authorize_protected_test
    )
    feature_stats = compute_paper_fcnn_feature_stats(train_products)
    oracle_levels = _oracle_levels(
        evaluation_products, int(settings["oracle_mc_samples"]), 10_000
    )

    if split == "validation":
        actual_fingerprints = _validation_fingerprints(
            config, settings, train_products, evaluation_products
        )
        frozen_fingerprints = json.loads(
            (results_root / "data_fingerprints.json").read_text(encoding="utf-8")
        )[str(len(train_products))]
        if actual_fingerprints != frozen_fingerprints:
            raise RuntimeError("Validation data/grid/scenario fingerprints changed")
        _atomic_write_json(output_dir / "data_fingerprints.json", actual_fingerprints)

    model_config = _model_config(config, settings)
    operational_frames = []
    prediction_frames = []
    seed_rows = []
    for manifest_row in checkpoint_manifest.itertuples(index=False):
        policy = str(manifest_row.policy_name)
        seed = int(manifest_row.training_seed)
        model = _make_frozen_model(policy, model_config)
        checkpoint = REPOSITORY_ROOT / str(manifest_row.checkpoint_path)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()

        predictions = _prediction_table(
            evaluation_products, {policy: model}, feature_stats, settings, oracle_levels
        )
        predictions = predictions.loc[predictions["policy_name"] == policy].copy()
        predictions.insert(0, "training_seed", seed)
        prediction_summary = _prediction_summary(predictions).iloc[0]

        operational = _operational_table(
            evaluation_products,
            {policy: model},
            feature_stats,
            settings,
            oracle_levels,
            split,
        )
        operational = operational.loc[operational["policy_name"] == policy].copy()
        operational.insert(0, "training_seed", seed)
        operational_frames.append(operational)
        prediction_frames.append(predictions)
        seed_rows.append(
            {
                "policy_name": policy,
                "training_seed": seed,
                "checkpoint_sha256": str(manifest_row.checkpoint_sha256),
                "mean_reward_per_period": float(operational["mean_reward_per_period"].mean()),
                "service_level": float(operational["service_level"].mean()),
                "lost_sales_rate": float(operational["lost_sales_rate"].mean()),
                "mean_inventory": float(operational["mean_inventory"].mean()),
                "mean_order_quantity": float(operational["mean_order_quantity"].mean()),
                "mean_holding_cost": float(operational["mean_holding_cost"].mean()),
                "deployed_rmse": float(prediction_summary["deployed_rmse"]),
                "raw_adjacent_grid_violation_rate": float(
                    prediction_summary["raw_violation_rate"]
                ),
                "deployed_adjacent_grid_violation_rate": float(
                    prediction_summary["deployed_violation_rate"]
                ),
                "maximum_positive_adjacent_raw_action_increase": float(
                    prediction_summary["raw_max_violation"]
                ),
                "active_region_order_up_to_slope_error": float(
                    prediction_summary["active_slope_deviation_from_minus_one"]
                ),
                "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            }
        )

    operational_all = pd.concat(operational_frames, ignore_index=True)
    predictions_all = pd.concat(prediction_frames, ignore_index=True)
    seed_results = pd.DataFrame(seed_rows).sort_values(["policy_name", "training_seed"])
    reconstruction = _reward_reconstruction(
        operational_all, int(settings["evaluation_horizon"]) - int(settings["burn_in"])
    )
    maximum_reconstruction_error = float(np.max(np.abs(reconstruction["reconstruction_error"])))
    if maximum_reconstruction_error > VALIDATION_TOLERANCE:
        raise RuntimeError(
            f"Reward reconstruction failed: maximum error {maximum_reconstruction_error}"
        )

    _atomic_write_csv(output_dir / OUTPUT_FILES["checkpoint_manifest"], checkpoint_manifest)
    _atomic_write_csv(output_dir / OUTPUT_FILES["seed_results"], seed_results)
    _atomic_write_csv(
        output_dir / OUTPUT_FILES["aggregate_results"], _aggregate_seed_results(seed_results)
    )
    _atomic_write_json(
        output_dir / OUTPUT_FILES["paired_reward_statistics"], _paired_statistics(seed_results)
    )
    _atomic_write_csv(output_dir / OUTPUT_FILES["operational_metrics"], operational_all)
    _atomic_write_csv(output_dir / OUTPUT_FILES["predictions"], predictions_all)
    _atomic_write_csv(output_dir / OUTPUT_FILES["reward_reconstruction"], reconstruction)

    comparison_maximum = None
    comparison_passed = None
    if split == "validation":
        comparison = _validation_comparison(seed_results, checkpoint_manifest)
        comparison_maximum = float(comparison["absolute_difference"].max())
        comparison_passed = bool(comparison["within_tolerance"].all())
        _atomic_write_csv(output_dir / "validation_dry_run_comparison.csv", comparison)
        if not comparison_passed:
            raise RuntimeError(
                f"Validation dry run did not reproduce frozen metrics; max diff={comparison_maximum}"
            )

    metadata = {
        "split": split,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "git_commit": _git_commit(),
        "checkpoint_count": len(checkpoint_manifest),
        "evaluation_product_count": len(evaluation_products),
        "evaluation_seed_start": (
            PAPER_VALIDATION_EVALUATION_SEED_START
            if split == "validation"
            else PAPER_TEST_EVALUATION_SEED_START
        ),
        "training_function_called": False,
        "optimizer_constructed": False,
        "optimizer_update_count": 0,
        "model_parameters_modified": False,
        "maximum_reward_reconstruction_error": maximum_reconstruction_error,
        "validation_comparison_maximum_absolute_difference": comparison_maximum,
        "validation_comparison_passed": comparison_passed,
        "protected_test_accessed": split == "test",
        "output_directory": output_relative.as_posix(),
    }
    _atomic_write_json(output_dir / OUTPUT_FILES["evaluation_metadata"], metadata)
    _atomic_write_json(output_dir / OUTPUT_FILES["completion"], metadata)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", default="full")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--frozen-results-root", type=Path, default=FROZEN_RESULTS_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--authorize-protected-test", action="store_true")
    parser.add_argument("--expected-protocol-fingerprint")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path, config_relative = _repository_path(args.config, "--config")
    results_root, _ = _repository_path(args.frozen_results_root, "--frozen-results-root")
    protocol_path, _ = _repository_path(args.protocol, "--protocol")
    requested_output = args.output_dir or (
        DEFAULT_VALIDATION_OUTPUT if args.split == "validation" else FROZEN_TEST_OUTPUT
    )
    output_dir, output_relative = _repository_path(requested_output, "--output-dir")
    _validate_evaluation_request(
        args.split,
        args.authorize_protected_test,
        args.expected_protocol_fingerprint,
        output_relative,
    )

    config = load_paper_lost_sales_config(config_path)
    settings = _profile_settings(config, args.profile)
    checkpoint_manifest = _load_checkpoint_manifest(results_root)
    checkpoint_training_commits = set(checkpoint_manifest["checkpoint_training_git_commit"])
    if len(checkpoint_training_commits) != 1:
        raise RuntimeError("Frozen checkpoints do not share one recorded training git commit")
    protocol_payload = _protocol_payload(
        checkpoint_manifest, config_relative, checkpoint_training_commits.pop()
    )
    protocol = _write_or_validate_protocol(protocol_path, protocol_payload)
    if args.split == "test" and (
        args.expected_protocol_fingerprint != protocol["protocol_fingerprint"]
    ):
        raise ValueError("Provided protected-test protocol fingerprint does not match")

    _run_evaluation(
        split=args.split,
        config=config,
        settings=settings,
        checkpoint_manifest=checkpoint_manifest,
        results_root=results_root,
        output_dir=output_dir,
        output_relative=output_relative,
        protocol=protocol,
        authorize_protected_test=args.authorize_protected_test,
    )
    print(
        f"completed split={args.split} checkpoints={len(checkpoint_manifest)} "
        f"output={output_relative.as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
