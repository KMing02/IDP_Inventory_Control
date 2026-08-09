"""Two-stage local tuning for the four primary neural inventory policies.

This module is intentionally an orchestration and reporting layer. It reuses
the production model factories, DirectBackprop trainer, benchmark prediction
summary, and operational evaluation without changing their mathematics.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from inventory_control.idp_benchmark import (
    _model_config,
    _operational_table,
    _oracle_levels,
    _parameter_count,
    _prediction_summary,
    _prediction_table,
    _profile_settings,
    _train_model,
)
from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.paper_evaluation import PAPER_VALIDATION_EVALUATION_SEED_START
from inventory_control.policies.neural.constrained_mlp import (
    make_constrained_monotone_mlp_model,
    make_positive_weight_mlp_model,
)
from inventory_control.policies.neural.fcnn import (
    compute_paper_fcnn_feature_stats,
    make_paper_fcnn_model,
)
from inventory_control.products import PaperLostSalesProduct, PaperLostSalesProductGenerator
from inventory_control.training import direct_backprop
from inventory_control.training.direct_backprop import (
    default_torch_device,
    paper_lost_sales_features,
)


PRIMARY_POLICIES = (
    "PaperFCNN",
    "PenaltyMLP",
    "PositiveWeightMLP",
    "ConstrainedMonoMLP",
)
CANDIDATE_IDS = tuple(f"C{index}" for index in range(1, 9))
STAGE1_SEEDS = (0,)
STAGE2_SEEDS = (1, 2)
VALIDATION_GRID_SEED = 2_000_003
IMITATION_INVENTORY_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
VIOLATION_TOLERANCE = 1e-6
REWARD_TIE_RELATIVE_TOLERANCE = 0.005
REWARD_TIE_EPSILON = 1e-12
DEFAULT_OUTPUT_DIR = Path("outputs/idp_tuning/primary_neural_balanced8_validation")
REQUIRED_COMPLETION_FILES = (
    "resolved_config.json",
    "training_log.csv",
    "best_imitation_checkpoint.json",
    "operational_metrics.csv",
    "operational_summary.csv",
    "predictions.csv",
    "prediction_diagnostics.csv",
    "fit_result.json",
    "model.pt",
    "best_imitation.pt",
)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    hidden_layers: int
    hidden_units: int
    learning_rate: float
    reward_epochs: int
    cmnn_activation_weights: tuple[int, int, int]
    penalty_lambda: float


CANDIDATES = (
    Candidate("C1", 2, 32, 0.0003, 30, (1, 1, 1), 0.1),
    Candidate("C2", 2, 32, 0.0010, 15, (1, 1, 1), 10.0),
    Candidate("C3", 2, 64, 0.0003, 15, (2, 2, 1), 1.0),
    Candidate("C4", 2, 64, 0.0010, 30, (1, 1, 2), 100.0),
    Candidate("C5", 3, 32, 0.0003, 15, (1, 1, 2), 100.0),
    Candidate("C6", 3, 32, 0.0010, 30, (2, 2, 1), 1.0),
    Candidate("C7", 3, 64, 0.0003, 30, (1, 1, 1), 10.0),
    Candidate("C8", 3, 64, 0.0010, 15, (1, 1, 1), 0.1),
)
CANDIDATE_BY_ID = {candidate.candidate_id: candidate for candidate in CANDIDATES}


def candidate_manifest() -> pd.DataFrame:
    """Return the complete 32-row policy/candidate tuning manifest."""
    rows: list[dict[str, Any]] = []
    for policy in PRIMARY_POLICIES:
        for candidate in CANDIDATES:
            rows.append(
                {
                    "policy_name": policy,
                    "candidate_id": candidate.candidate_id,
                    "hidden_layers": candidate.hidden_layers,
                    "hidden_units": candidate.hidden_units,
                    "learning_rate": candidate.learning_rate,
                    "reward_epochs": candidate.reward_epochs,
                    "activation_weights": (
                        json.dumps(candidate.cmnn_activation_weights)
                        if policy == "ConstrainedMonoMLP"
                        else ""
                    ),
                    "penalty_lambda": (
                        candidate.penalty_lambda if policy == "PenaltyMLP" else np.nan
                    ),
                    "stage1_seeds": "0",
                    "stage2_seeds_if_promoted": "1,2",
                    "stage1_fit_count": 1,
                    "possible_stage2_fit_count": 2,
                }
            )
    return pd.DataFrame(rows)


def expected_fit_counts() -> dict[str, int]:
    """Return the fixed successive-halving fit budget."""
    stage1 = len(PRIMARY_POLICIES) * len(CANDIDATES) * len(STAGE1_SEEDS)
    stage2 = len(PRIMARY_POLICIES) * 2 * len(STAGE2_SEEDS)
    return {"stage1": stage1, "stage2": stage2, "maximum_total": stage1 + stage2}


def validate_split_and_output(split: str, output_dir: Path) -> None:
    """Fail closed if the protected test split or a test-labelled output is requested."""
    if split != "validation":
        raise ValueError("Primary neural tuning is validation-only; split must be 'validation'")
    suspicious_parts = {
        "test",
        "tests",
        "protected_test",
        "final_test",
        "test_output",
        "test_outputs",
    }
    normalized_parts = {part.casefold() for part in output_dir.parts}
    if normalized_parts & suspicious_parts:
        raise ValueError("Refusing to write tuning artifacts to a test-labelled output directory")


def _repository_relative_path(path: Path, label: str) -> tuple[Path, Path]:
    if path.is_absolute():
        raise ValueError(f"{label} must be repository-relative, got absolute path: {path}")
    resolved = (REPOSITORY_ROOT / path).resolve()
    try:
        relative = resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the repository: {path}") from exc
    return resolved, relative


def _safe_generate_products(
    generator: PaperLostSalesProductGenerator,
    count: int,
    split: str,
) -> list[PaperLostSalesProduct]:
    if split not in {"train", "validation"}:
        raise RuntimeError(f"Protected product split requested by tuning orchestration: {split}")
    return generator.generate(count, split=split)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, default=_json_default, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _product_records(products: Sequence[PaperLostSalesProduct]) -> list[dict[str, Any]]:
    return [asdict(product) for product in products]


def data_fingerprints(
    config: dict[str, Any],
    settings: dict[str, Any],
    train_products: Sequence[PaperLostSalesProduct],
    validation_products: Sequence[PaperLostSalesProduct],
) -> dict[str, Any]:
    """Fingerprint all data and evaluation state that must not vary by training seed."""
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
                    fraction * maximum_inventory for fraction in IMITATION_INVENTORY_FRACTIONS
                ],
                "prediction_inventory": np.linspace(
                    0.0,
                    maximum_inventory,
                    int(settings["inventory_states"]),
                ).tolist(),
            }
        )
    evaluation_seeds = [
        PAPER_VALIDATION_EVALUATION_SEED_START + index
        for index in range(len(validation_products))
    ]
    return {
        "product_generation_seed": int(config["product_generation"].get("seed", 0)),
        "validation_grid_seed": VALIDATION_GRID_SEED,
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


def _environment_record(config_relative: Path, config: dict[str, Any]) -> dict[str, Any]:
    try:
        git_commit = subprocess.run(
            ["git", "-c", f"safe.directory={REPOSITORY_ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unavailable"
    return {
        "git_commit": git_commit,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "operating_system": platform.platform(),
        "device": str(default_torch_device()),
        "base_config": config_relative.as_posix(),
        "base_config_fingerprint": _fingerprint(config),
    }


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, default=_json_default, indent=2, sort_keys=True))


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_or_validate_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise RuntimeError(f"Existing reproducibility record does not match current run: {path}")
    else:
        _atomic_write_json(path, value)


def _resolved_fit_config(
    base_config: dict[str, Any],
    profile: str,
    policy: str,
    candidate: Candidate,
    training_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = copy.deepcopy(base_config)
    config["learning_rate"] = float(candidate.learning_rate)
    config["fcnn"]["hidden_layers"] = int(candidate.hidden_layers)
    config["fcnn"]["hidden_units"] = int(candidate.hidden_units)
    benchmark = config["idp_benchmark"]
    benchmark["seed"] = int(training_seed)
    benchmark["hidden_layers"] = int(candidate.hidden_layers)
    benchmark["hidden_units"] = int(candidate.hidden_units)
    benchmark["imitation_early_stopping"] = True
    benchmark["imitation_max_epochs"] = 100
    benchmark["imitation_patience"] = 10
    benchmark["imitation_min_delta"] = 0.5
    benchmark["reset_optimizer_after_imitation"] = False
    benchmark[profile]["reward_epochs"] = int(candidate.reward_epochs)
    config["constrained_monotone_mlp"]["activation_weights"] = list(
        candidate.cmnn_activation_weights
    )
    settings = _profile_settings(config, profile)
    resolved = {
        "policy_name": policy,
        "candidate_id": candidate.candidate_id,
        "training_seed": int(training_seed),
        "product_generation_seed": int(config["product_generation"].get("seed", 0)),
        "validation_grid_seed": VALIDATION_GRID_SEED,
        "evaluation_seed_start": PAPER_VALIDATION_EVALUATION_SEED_START,
        "hidden_layers": candidate.hidden_layers,
        "hidden_units": candidate.hidden_units,
        "learning_rate": candidate.learning_rate,
        "reward_epochs": candidate.reward_epochs,
        "activation_weights": (
            list(candidate.cmnn_activation_weights) if policy == "ConstrainedMonoMLP" else None
        ),
        "penalty_lambda": candidate.penalty_lambda if policy == "PenaltyMLP" else None,
        "imitation_early_stopping": True,
        "imitation_max_epochs": 100,
        "imitation_patience": 10,
        "imitation_min_delta": 0.5,
        "reset_optimizer_after_imitation": False,
        "deployment_action_cap_factor": float(settings["max_order_factor"]),
        "batch_size": int(settings["batch_size"]),
        "training_horizon": int(settings["training_horizon"]),
        "evaluation_horizon": int(settings["evaluation_horizon"]),
        "burn_in": int(settings["burn_in"]),
        "inventory_states": int(settings["inventory_states"]),
        "normalize_reward_loss": bool(settings["normalize_reward_loss"]),
        "imitation_semantics": "deployed_prediction_vs_capped_target",
        "imitation_checkpoint_metric": "deployed_imitation_rmse",
    }
    return config, resolved


def _make_model(policy: str, model_config: dict[str, Any], training_seed: int) -> torch.nn.Module:
    torch.manual_seed(int(training_seed))
    if policy in {"PaperFCNN", "PenaltyMLP"}:
        return make_paper_fcnn_model(model_config)
    if policy == "PositiveWeightMLP":
        return make_positive_weight_mlp_model(model_config)
    if policy == "ConstrainedMonoMLP":
        return make_constrained_monotone_mlp_model(model_config)
    raise ValueError(f"Unsupported primary policy: {policy}")


@contextmanager
def _fixed_validation_grid_seed(
    validation_products: list[PaperLostSalesProduct],
) -> Iterator[None]:
    """Decouple the imitation-validation batch seed from training randomness."""
    original = direct_backprop.make_paper_lost_sales_training_batch

    def fixed_validation_batch(
        products: list[PaperLostSalesProduct],
        horizon: int,
        generator: torch.Generator,
        feature_stats: Any = None,
    ) -> dict[str, torch.Tensor]:
        if products is validation_products:
            generator = torch.Generator().manual_seed(VALIDATION_GRID_SEED)
        return original(
            products,
            horizon,
            generator,
            feature_stats=feature_stats,
        )

    direct_backprop.make_paper_lost_sales_training_batch = fixed_validation_batch
    try:
        yield
    finally:
        direct_backprop.make_paper_lost_sales_training_batch = original


@contextmanager
def _penalty_pair_recorder(enabled: bool) -> Iterator[list[dict[str, int | float]]]:
    """Replay penalty RNG draws for diagnostics without changing training RNG or loss."""
    records: list[dict[str, int | float]] = []
    if not enabled:
        yield records
        return

    original = direct_backprop.paper_inventory_monotonicity_penalty

    def instrumented(
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
        generator: torch.Generator,
        samples_per_product: int = 1,
        max_inventory_factor: float = 4.0,
        scale_action_by_history_mean: bool = False,
    ) -> torch.Tensor:
        generator_state = generator.get_state()
        penalty = original(
            model,
            batch,
            generator,
            samples_per_product=samples_per_product,
            max_inventory_factor=max_inventory_factor,
            scale_action_by_history_mean=scale_action_by_history_mean,
        )
        replay_generator = torch.Generator()
        replay_generator.set_state(generator_state)
        batch_size = int(batch["initial_inventory"].shape[0])
        maximum_inventory = float(max_inventory_factor) * torch.clamp(
            batch["historical_mean_demand"], min=1.0
        )
        violating_pairs = 0
        with torch.no_grad():
            for _ in range(int(samples_per_product)):
                first = torch.rand(batch_size, generator=replay_generator).to(
                    batch["initial_inventory"].device
                )
                second = torch.rand(batch_size, generator=replay_generator).to(
                    batch["initial_inventory"].device
                )
                inventory_low = torch.minimum(first, second) * maximum_inventory
                inventory_high = torch.maximum(first, second) * maximum_inventory
                low_order = model(paper_lost_sales_features(batch, inventory_low))
                high_order = model(paper_lost_sales_features(batch, inventory_high))
                if scale_action_by_history_mean:
                    action_scale = torch.clamp(batch["historical_mean_demand"], min=1.0)
                    low_order = low_order * action_scale
                    high_order = high_order * action_scale
                violating_pairs += int(
                    torch.sum((high_order - low_order) > VIOLATION_TOLERANCE).item()
                )
        sampled_pairs = batch_size * int(samples_per_product)
        records.append(
            {
                "sampled_pair_count": sampled_pairs,
                "violating_pair_count": violating_pairs,
                "violating_pair_proportion": violating_pairs / max(sampled_pairs, 1),
                "violation_tolerance": VIOLATION_TOLERANCE,
            }
        )
        return penalty

    direct_backprop.paper_inventory_monotonicity_penalty = instrumented
    try:
        yield records
    finally:
        direct_backprop.paper_inventory_monotonicity_penalty = original


def _penalty_summary(
    policy: str,
    penalty_lambda: float | None,
    training_log: pd.DataFrame,
    pair_records: list[dict[str, int | float]],
) -> dict[str, Any]:
    if policy != "PenaltyMLP":
        return {
            "configured_lambda": np.nan,
            "penalty_loss_min": np.nan,
            "penalty_loss_mean": np.nan,
            "penalty_loss_max": np.nan,
            "sampled_monotonicity_pair_count": np.nan,
            "violating_pair_count": np.nan,
            "violating_pair_proportion": np.nan,
        }
    penalties = training_log["monotonicity_penalty"].to_numpy(dtype=float)
    sampled = int(sum(int(row["sampled_pair_count"]) for row in pair_records))
    violating = int(sum(int(row["violating_pair_count"]) for row in pair_records))
    return {
        "configured_lambda": float(penalty_lambda),
        "penalty_loss_min": float(np.min(penalties)),
        "penalty_loss_mean": float(np.mean(penalties)),
        "penalty_loss_max": float(np.max(penalties)),
        "sampled_monotonicity_pair_count": sampled,
        "violating_pair_count": violating,
        "violating_pair_proportion": violating / max(sampled, 1),
    }


def _fit_directory(output_dir: Path, policy: str, candidate_id: str, seed: int) -> Path:
    return output_dir / "runs" / policy / candidate_id / f"seed_{seed}"


def _completion_record(fit_dir: Path) -> dict[str, Any] | None:
    marker = fit_dir / "completion.json"
    if not marker.exists() or any(not (fit_dir / name).exists() for name in REQUIRED_COMPLETION_FILES):
        return None
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if record.get("policy_name") == "PenaltyMLP" and not (
        fit_dir / "penalty_pair_diagnostics.csv"
    ).exists():
        return None
    return record


def _expected_fit_config_fingerprint(
    base_config: dict[str, Any],
    profile: str,
    policy: str,
    candidate: Candidate,
    training_seed: int,
    shared_data_fingerprint: str,
) -> str:
    _, resolved = _resolved_fit_config(
        base_config,
        profile,
        policy,
        candidate,
        training_seed,
    )
    resolved["shared_data_fingerprint"] = shared_data_fingerprint
    return _fingerprint(resolved)


def _validate_completed_fit(
    record: dict[str, Any],
    *,
    base_config: dict[str, Any],
    profile: str,
    policy: str,
    candidate: Candidate,
    training_seed: int,
    shared_data_fingerprint: str,
) -> None:
    expected_fingerprint = _expected_fit_config_fingerprint(
        base_config,
        profile,
        policy,
        candidate,
        training_seed,
        shared_data_fingerprint,
    )
    identity_matches = (
        record.get("policy_name") == policy
        and record.get("candidate_id") == candidate.candidate_id
        and int(record.get("training_seed", -1)) == int(training_seed)
    )
    if (
        not identity_matches
        or record.get("resolved_config_fingerprint") != expected_fingerprint
        or record.get("shared_data_fingerprint") != shared_data_fingerprint
    ):
        raise RuntimeError(
            "Completed fit does not match the requested identity, configuration, or shared data; "
            "refusing to skip it under --resume"
        )


def _run_fit(
    *,
    base_config: dict[str, Any],
    profile: str,
    stage: str,
    policy: str,
    candidate: Candidate,
    training_seed: int,
    output_dir: Path,
    output_relative: Path,
    train_products: list[PaperLostSalesProduct],
    validation_products: list[PaperLostSalesProduct],
    feature_stats: Any,
    oracle_levels: dict[str, float],
    shared_data_fingerprint: str,
) -> dict[str, Any]:
    config, resolved = _resolved_fit_config(
        base_config,
        profile,
        policy,
        candidate,
        training_seed,
    )
    settings = _profile_settings(config, profile)
    model_config = _model_config(config, settings)
    model = _make_model(policy, model_config, training_seed)
    fit_dir = _fit_directory(output_dir, policy, candidate.candidate_id, training_seed)
    fit_dir.mkdir(parents=True, exist_ok=True)
    # Completion is committed last. Removing a stale marker makes an
    # interrupted rerun unambiguously incomplete.
    (fit_dir / "completion.json").unlink(missing_ok=True)
    resolved["shared_data_fingerprint"] = shared_data_fingerprint
    resolved["resolved_config_fingerprint"] = _fingerprint(resolved)
    _atomic_write_json(fit_dir / "resolved_config.json", resolved)

    penalty_weight = candidate.penalty_lambda if policy == "PenaltyMLP" else 0.0
    started = time.perf_counter()
    with _fixed_validation_grid_seed(validation_products):
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
                penalty_weight=float(penalty_weight),
                clip_imitation_predictions=True,
            )
    if result.imitation_selection_metric != "deployed_imitation_rmse":
        raise RuntimeError("Fit did not use deployed imitation RMSE for checkpoint selection")
    if result.best_imitation_state_dict is None or result.best_imitation_epoch is None:
        raise RuntimeError("Fit did not produce a validation-best imitation checkpoint")
    elapsed = time.perf_counter() - started
    training_log = pd.DataFrame(result.training_log or [])
    if policy == "PenaltyMLP":
        pair_frame = pd.DataFrame(pair_records)
        pair_frame.insert(0, "epoch", np.arange(1, len(pair_frame) + 1))
        _atomic_write_csv(fit_dir / "penalty_pair_diagnostics.csv", pair_frame)

    predictions = _prediction_table(
        validation_products,
        {policy: model},
        feature_stats,
        settings,
        oracle_levels,
    )
    policy_predictions = predictions.loc[predictions["policy_name"] == policy].reset_index(drop=True)
    prediction_diagnostics = _prediction_summary(policy_predictions)
    operational = _operational_table(
        validation_products,
        {policy: model},
        feature_stats,
        settings,
        oracle_levels,
        "validation",
    )
    policy_operational = operational.loc[operational["policy_name"] == policy].reset_index(drop=True)
    operational_summary = (
        policy_operational.groupby("policy_name", as_index=False)
        .agg(
            mean_total_reward=("mean_total_reward", "mean"),
            std_total_reward=("mean_total_reward", "std"),
            mean_reward_per_period=("mean_reward_per_period", "mean"),
            service_level=("service_level", "mean"),
            lost_sales_rate=("lost_sales_rate", "mean"),
            mean_order_quantity=("mean_order_quantity", "mean"),
            mean_inventory=("mean_inventory", "mean"),
        )
    )
    diagnostics_row = prediction_diagnostics.iloc[0].to_dict()
    operational_row = operational_summary.iloc[0].to_dict()
    penalty_metrics = _penalty_summary(
        policy,
        candidate.penalty_lambda if policy == "PenaltyMLP" else None,
        training_log,
        pair_records,
    )
    fit_result = {
        "stage": stage,
        "policy_name": policy,
        "candidate_id": candidate.candidate_id,
        "training_seed": int(training_seed),
        "run_dir": (
            output_relative / "runs" / policy / candidate.candidate_id / f"seed_{training_seed}"
        ).as_posix(),
        "resolved_config_fingerprint": resolved["resolved_config_fingerprint"],
        "shared_data_fingerprint": shared_data_fingerprint,
        "parameter_count": _parameter_count(model),
        "training_seconds": float(result.elapsed_seconds),
        "fit_wall_seconds": float(elapsed),
        "imitation_epochs_run": int(result.imitation_epochs_completed),
        "selected_best_imitation_epoch": result.best_imitation_epoch,
        "best_validation_imitation_mae": result.best_imitation_validation_mae,
        "best_validation_imitation_rmse": result.best_imitation_validation_rmse,
        "imitation_checkpoint_metric": result.imitation_selection_metric,
        "best_imitation_checkpoint_restored_before_reward_training": True,
        **operational_row,
        **diagnostics_row,
        **penalty_metrics,
    }
    best_imitation_metadata = {
        "selected_epoch": result.best_imitation_epoch,
        "validation_mae": result.best_imitation_validation_mae,
        "validation_rmse": result.best_imitation_validation_rmse,
        "selection_metric": result.imitation_selection_metric,
        "epochs_run": result.imitation_epochs_completed,
        "early_stopped": result.imitation_early_stopped,
        "restored_before_reward_training": True,
        "checkpoint_saved": result.best_imitation_state_dict is not None,
    }

    _atomic_write_csv(fit_dir / "training_log.csv", training_log)
    _atomic_write_json(fit_dir / "best_imitation_checkpoint.json", best_imitation_metadata)
    _atomic_write_csv(fit_dir / "operational_metrics.csv", policy_operational)
    _atomic_write_csv(fit_dir / "operational_summary.csv", operational_summary)
    _atomic_write_csv(fit_dir / "predictions.csv", policy_predictions)
    _atomic_write_csv(fit_dir / "prediction_diagnostics.csv", prediction_diagnostics)
    _atomic_torch_save(fit_dir / "model.pt", model.state_dict())
    _atomic_torch_save(fit_dir / "best_imitation.pt", result.best_imitation_state_dict)
    _atomic_write_json(fit_dir / "fit_result.json", fit_result)
    _atomic_write_json(fit_dir / "completion.json", fit_result)
    return fit_result


def _scan_completed_results(output_dir: Path, stage: str | None = None) -> pd.DataFrame:
    rows = []
    runs_dir = output_dir / "runs"
    if not runs_dir.exists():
        return pd.DataFrame()
    for marker in sorted(runs_dir.glob("*/*/seed_*/completion.json")):
        fit_dir = marker.parent
        record = _completion_record(fit_dir)
        if record is not None and (stage is None or record.get("stage") == stage):
            rows.append(record)
    return pd.DataFrame(rows)


def _candidate_number(candidate_id: str) -> int:
    return int(candidate_id.removeprefix("C"))


def _is_hard_monotone(policy: str) -> bool:
    return policy in {"PositiveWeightMLP", "ConstrainedMonoMLP"}


def _tie_break_columns(policy: str, pooled: bool) -> list[str]:
    prefix = "pooled_" if pooled else ""
    columns = []
    if policy == "PenaltyMLP":
        columns.append(f"{prefix}raw_violation_rate")
    columns.extend(
        [
            f"{prefix}deployed_rmse",
            f"{prefix}deployed_zero_inventory_order_up_to_p90_absolute_error",
            f"{prefix}deployed_zero_inventory_order_up_to_max_absolute_error",
            "parameter_count",
            "candidate_number",
        ]
    )
    return columns


def _rank_with_reward_ties(
    frame: pd.DataFrame,
    policy: str,
    reward_column: str,
    pooled: bool,
) -> pd.DataFrame:
    """Rank reward bands, applying diagnostics only within each 0.5% band."""
    ranked_source = frame.copy()
    ranked_source["candidate_number"] = ranked_source["candidate_id"].map(_candidate_number)
    ranked_source["eligible"] = True
    if _is_hard_monotone(policy):
        raw_column = "pooled_raw_violation_rate" if pooled else "raw_violation_rate"
        deployed_column = (
            "pooled_deployed_violation_rate" if pooled else "deployed_violation_rate"
        )
        ranked_source["eligible"] = (
            (ranked_source[raw_column] == 0.0) & (ranked_source[deployed_column] == 0.0)
        )
    eligible = ranked_source.loc[ranked_source["eligible"]].copy()
    if eligible.empty:
        raise RuntimeError(f"No eligible candidates remain for {policy}")

    ordered_groups = []
    remaining = eligible.copy()
    while not remaining.empty:
        best_reward = float(remaining[reward_column].max())
        denominator = max(abs(best_reward), REWARD_TIE_EPSILON)
        relative_gap = (best_reward - remaining[reward_column]) / denominator
        tie_group = remaining.loc[relative_gap <= REWARD_TIE_RELATIVE_TOLERANCE].copy()
        tie_group = tie_group.sort_values(_tie_break_columns(policy, pooled), kind="stable")
        ordered_groups.append(tie_group)
        remaining = remaining.drop(index=tie_group.index)
    ranked = pd.concat(ordered_groups, ignore_index=True)
    ranked["selection_rank"] = np.arange(1, len(ranked) + 1)
    rejected = ranked_source.loc[~ranked_source["eligible"]].copy()
    if not rejected.empty:
        rejected["selection_rank"] = np.nan
        ranked = pd.concat([ranked, rejected], ignore_index=True, sort=False)
    return ranked


def _write_stage1_outputs(output_dir: Path, manifest: pd.DataFrame) -> bool:
    results = _scan_completed_results(output_dir, stage="screening")
    _atomic_write_csv(output_dir / "stage1_seed_results.csv", results)
    if results.empty:
        _atomic_write_csv(output_dir / "stage1_aggregate_results.csv", results)
        return False
    expected = {
        (row.policy_name, row.candidate_id, 0)
        for row in manifest.itertuples(index=False)
    }
    completed = {
        (str(row.policy_name), str(row.candidate_id), int(row.training_seed))
        for row in results.itertuples(index=False)
    }
    if completed != expected:
        (output_dir / "stage1_promoted_candidates.csv").unlink(missing_ok=True)
        partial = results.copy()
        hard_monotone = partial["policy_name"].isin(
            ["PositiveWeightMLP", "ConstrainedMonoMLP"]
        )
        zero_violations = (
            (partial["raw_violation_rate"] == 0.0)
            & (partial["deployed_violation_rate"] == 0.0)
        )
        partial["eligible"] = ~hard_monotone | zero_violations
        partial["selection_rank"] = np.nan
        partial["promoted"] = False
        _atomic_write_csv(output_dir / "stage1_aggregate_results.csv", partial)
        return False

    aggregate_rows = []
    promoted_rows = []
    for policy in PRIMARY_POLICIES:
        policy_rows = results.loc[results["policy_name"] == policy].copy()
        ranked = _rank_with_reward_ties(
            policy_rows,
            policy,
            reward_column="mean_reward_per_period",
            pooled=False,
        )
        ranked["promoted"] = ranked["selection_rank"].isin([1, 2])
        aggregate_rows.append(ranked)
        promoted_rows.append(ranked.loc[ranked["promoted"]].copy())
    aggregate = pd.concat(aggregate_rows, ignore_index=True)
    _atomic_write_csv(output_dir / "stage1_aggregate_results.csv", aggregate)
    promoted = pd.concat(promoted_rows, ignore_index=True)
    if len(promoted) != len(PRIMARY_POLICIES) * 2:
        raise RuntimeError("Stage 1 must promote exactly two candidates per policy")
    promoted_columns = [
        "policy_name",
        "candidate_id",
        "selection_rank",
        "mean_reward_per_period",
        "raw_violation_rate",
        "deployed_violation_rate",
        "deployed_rmse",
        "deployed_zero_inventory_order_up_to_p90_absolute_error",
        "deployed_zero_inventory_order_up_to_max_absolute_error",
        "parameter_count",
        "configured_lambda",
        "eligible",
    ]
    _atomic_write_csv(
        output_dir / "stage1_promoted_candidates.csv",
        promoted[[column for column in promoted_columns if column in promoted.columns]],
    )
    return True


def _pooled_prediction_diagnostics(predictions: pd.DataFrame) -> dict[str, float]:
    """Pool action and zero-inventory errors across all three training seeds."""
    deployed_errors = predictions["deployed_error"].to_numpy(dtype=float)
    raw_increases: list[float] = []
    deployed_increases: list[float] = []
    for _, rows in predictions.groupby(["training_seed", "product_id"], sort=False):
        ordered = rows.sort_values("state_index")
        raw_increases.extend(np.diff(ordered["raw_order"].to_numpy(dtype=float)).tolist())
        deployed_increases.extend(
            np.diff(ordered["deployed_order"].to_numpy(dtype=float)).tolist()
        )
    raw_increases_array = np.maximum(np.asarray(raw_increases, dtype=float), 0.0)
    deployed_increases_array = np.maximum(np.asarray(deployed_increases, dtype=float), 0.0)
    zero_rows = predictions.loc[predictions["state_index"] == 0]
    zero_absolute_errors = np.abs(
        zero_rows["deployed_order_up_to_level"].to_numpy(dtype=float)
        - zero_rows["deployed_omniscient_order_up_to_level"].to_numpy(dtype=float)
    )
    return {
        "pooled_deployed_rmse": float(np.sqrt(np.mean(deployed_errors**2))),
        "pooled_raw_violation_rate": float(
            np.mean(raw_increases_array > VIOLATION_TOLERANCE)
        ),
        "pooled_deployed_violation_rate": float(
            np.mean(deployed_increases_array > VIOLATION_TOLERANCE)
        ),
        "pooled_deployed_zero_inventory_order_up_to_p90_absolute_error": float(
            np.quantile(zero_absolute_errors, 0.9)
        ),
        "pooled_deployed_zero_inventory_order_up_to_max_absolute_error": float(
            np.max(zero_absolute_errors)
        ),
    }


def _write_final_outputs(output_dir: Path) -> bool:
    stage2 = _scan_completed_results(output_dir, stage="confirmation")
    _atomic_write_csv(output_dir / "stage2_seed_results.csv", stage2)
    promotion_path = output_dir / "stage1_promoted_candidates.csv"
    if not promotion_path.exists():
        return False
    promoted = pd.read_csv(promotion_path)
    stage1 = _scan_completed_results(output_dir, stage="screening")
    aggregate_rows = []
    for promoted_row in promoted.itertuples(index=False):
        policy = str(promoted_row.policy_name)
        candidate_id = str(promoted_row.candidate_id)
        seed_rows = pd.concat(
            [
                stage1.loc[
                    (stage1["policy_name"] == policy)
                    & (stage1["candidate_id"] == candidate_id)
                    & (stage1["training_seed"] == 0)
                ],
                stage2.loc[
                    (stage2["policy_name"] == policy)
                    & (stage2["candidate_id"] == candidate_id)
                    & (stage2["training_seed"].isin(STAGE2_SEEDS))
                ],
            ],
            ignore_index=True,
        )
        if set(seed_rows.get("training_seed", pd.Series(dtype=int)).astype(int)) != {0, 1, 2}:
            continue
        pooled_frame = pd.concat(
            [
                pd.read_csv(REPOSITORY_ROOT / Path(str(row.run_dir)) / "predictions.csv").assign(
                    training_seed=int(row.training_seed)
                )
                for row in seed_rows.itertuples(index=False)
            ],
            ignore_index=True,
        )
        pooled = _pooled_prediction_diagnostics(pooled_frame)
        rewards = seed_rows["mean_reward_per_period"].to_numpy(dtype=float)
        aggregate_rows.append(
            {
                "policy_name": policy,
                "candidate_id": candidate_id,
                "seed_count": 3,
                "mean_reward_per_period": float(np.mean(rewards)),
                "reward_per_period_std": float(np.std(rewards, ddof=1)),
                "parameter_count": int(seed_rows["parameter_count"].iloc[0]),
                "configured_lambda": seed_rows["configured_lambda"].iloc[0],
                "hidden_layers": CANDIDATE_BY_ID[candidate_id].hidden_layers,
                "hidden_units": CANDIDATE_BY_ID[candidate_id].hidden_units,
                "learning_rate": CANDIDATE_BY_ID[candidate_id].learning_rate,
                "reward_epochs": CANDIDATE_BY_ID[candidate_id].reward_epochs,
                "activation_weights": (
                    json.dumps(CANDIDATE_BY_ID[candidate_id].cmnn_activation_weights)
                    if policy == "ConstrainedMonoMLP"
                    else ""
                ),
                **pooled,
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    if aggregate.empty:
        (output_dir / "selected_configs.csv").unlink(missing_ok=True)
        _atomic_write_csv(output_dir / "final_aggregate_results.csv", aggregate)
        return False
    ranked_rows = []
    selected_rows = []
    for policy in PRIMARY_POLICIES:
        policy_rows = aggregate.loc[aggregate["policy_name"] == policy]
        if len(policy_rows) != 2:
            continue
        ranked = _rank_with_reward_ties(
            policy_rows,
            policy,
            reward_column="mean_reward_per_period",
            pooled=True,
        )
        ranked["selected"] = ranked["selection_rank"] == 1
        ranked_rows.append(ranked)
        selected_rows.append(ranked.loc[ranked["selected"]])
    final_aggregate = (
        pd.concat(ranked_rows, ignore_index=True) if ranked_rows else aggregate
    )
    _atomic_write_csv(output_dir / "final_aggregate_results.csv", final_aggregate)
    if len(selected_rows) != len(PRIMARY_POLICIES):
        (output_dir / "selected_configs.csv").unlink(missing_ok=True)
        return False
    selected = pd.concat(selected_rows, ignore_index=True)
    _atomic_write_csv(output_dir / "selected_configs.csv", selected)
    return True


def _prepare_common_state(
    config_path: Path,
    config_relative: Path,
    profile: str,
    output_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[PaperLostSalesProduct],
    list[PaperLostSalesProduct],
    Any,
    dict[str, float],
    dict[str, Any],
]:
    base_config = load_paper_lost_sales_config(config_path)
    settings = _profile_settings(base_config, profile)
    if float(settings["max_order_factor"]) != 4.0:
        raise ValueError("Primary tuning requires the shared deployment action cap F=4")
    if int(settings["training_products"]) != 128 or int(settings["validation_products"]) != 32:
        raise ValueError("Primary tuning requires exactly 128 training and 32 validation products")
    generator = PaperLostSalesProductGenerator(base_config)
    train_products = _safe_generate_products(
        generator,
        int(settings["training_products"]),
        "train",
    )
    validation_products = _safe_generate_products(
        generator,
        int(settings["validation_products"]),
        "validation",
    )
    feature_stats = compute_paper_fcnn_feature_stats(train_products)
    oracle_levels = _oracle_levels(
        validation_products,
        int(settings["oracle_mc_samples"]),
        10_000,
    )
    fingerprints = data_fingerprints(
        base_config,
        settings,
        train_products,
        validation_products,
    )
    _write_or_validate_json(output_dir / "data_fingerprints.json", fingerprints)
    _write_or_validate_json(
        output_dir / "environment.json",
        _environment_record(config_relative, base_config),
    )
    return (
        base_config,
        settings,
        train_products,
        validation_products,
        feature_stats,
        oracle_levels,
        fingerprints,
    )


def _stage_plan(
    stage: str,
    manifest: pd.DataFrame,
    output_dir: Path,
    policies: Sequence[str],
    candidate_ids: Sequence[str],
    seeds: Sequence[int],
) -> list[tuple[str, Candidate, int]]:
    if stage == "screening":
        if not set(seeds).issubset(STAGE1_SEEDS):
            raise ValueError("Screening supports seed 0 only")
        allowed = manifest
    else:
        if not set(seeds).issubset(STAGE2_SEEDS):
            raise ValueError("Confirmation supports additional seeds 1 and 2 only")
        promotion_path = output_dir / "stage1_promoted_candidates.csv"
        if not promotion_path.exists():
            raise RuntimeError("Stage 2 requires completed Stage-1 promotion decisions")
        stage1_results = _scan_completed_results(output_dir, stage="screening")
        expected_stage1 = {
            (row.policy_name, row.candidate_id, 0)
            for row in manifest.itertuples(index=False)
        }
        completed_stage1 = {
            (str(row.policy_name), str(row.candidate_id), int(row.training_seed))
            for row in stage1_results.itertuples(index=False)
        }
        if completed_stage1 != expected_stage1:
            raise RuntimeError("Stage 2 requires all 32 Stage-1 fits to remain fully complete")
        allowed = pd.read_csv(promotion_path)
    allowed_pairs = {
        (str(row.policy_name), str(row.candidate_id))
        for row in allowed.itertuples(index=False)
    }
    return [
        (policy, CANDIDATE_BY_ID[candidate_id], int(seed))
        for policy in PRIMARY_POLICIES
        if policy in policies
        for candidate_id in CANDIDATE_IDS
        if candidate_id in candidate_ids and (policy, candidate_id) in allowed_pairs
        for seed in seeds
    ]


def _print_manifest_summary(manifest: pd.DataFrame) -> None:
    display_columns = [
        "policy_name",
        "candidate_id",
        "hidden_layers",
        "hidden_units",
        "learning_rate",
        "reward_epochs",
        "activation_weights",
        "penalty_lambda",
    ]
    print(manifest[display_columns].to_string(index=False))
    penalty = manifest.loc[
        manifest["policy_name"] == "PenaltyMLP",
        ["candidate_id", "penalty_lambda"],
    ]
    counts = expected_fit_counts()
    print("\nPenaltyMLP lambda assignment")
    print(penalty.to_string(index=False))
    print(
        f"\nfit_budget stage1={counts['stage1']} stage2={counts['stage2']} "
        f"maximum_total={counts['maximum_total']}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["screening", "confirmation"], required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paper_lost_sales_medium_seed0_converged.yaml"),
    )
    parser.add_argument("--profile", choices=["compact", "full", "paper_replication"], default="full")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--policies", nargs="+", choices=PRIMARY_POLICIES, default=list(PRIMARY_POLICIES))
    parser.add_argument("--candidate-ids", nargs="+", choices=CANDIDATE_IDS, default=list(CANDIDATE_IDS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", "--manifest-only", dest="dry_run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_split_and_output(args.split, args.output_dir)
    config_path, config_relative = _repository_relative_path(args.config, "--config")
    output_dir, output_relative = _repository_relative_path(args.output_dir, "--output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = candidate_manifest()
    _atomic_write_csv(output_dir / "candidate_manifest.csv", manifest)
    (
        base_config,
        _,
        train_products,
        validation_products,
        feature_stats,
        oracle_levels,
        fingerprints,
    ) = _prepare_common_state(
        config_path,
        config_relative,
        args.profile,
        output_dir,
    )
    shared_data_fingerprint = _fingerprint(fingerprints)
    if args.dry_run:
        _print_manifest_summary(manifest)
        print("\ndata_fingerprints")
        print(json.dumps(fingerprints, indent=2, sort_keys=True))
        return 0

    default_seeds = STAGE1_SEEDS if args.stage == "screening" else STAGE2_SEEDS
    seeds = tuple(args.seeds) if args.seeds is not None else default_seeds
    plan = _stage_plan(
        args.stage,
        manifest,
        output_dir,
        args.policies,
        args.candidate_ids,
        seeds,
    )
    if not plan:
        raise RuntimeError("No fits match the requested stage/policy/candidate/seed filters")
    if not args.resume:
        existing = [
            _fit_directory(output_dir, policy, candidate.candidate_id, seed)
            for policy, candidate, seed in plan
            if _completion_record(
                _fit_directory(output_dir, policy, candidate.candidate_id, seed)
            )
            is not None
        ]
        if existing:
            raise RuntimeError("Completed fits already exist; pass --resume to skip them safely")

    completed_before = 0
    total = len(plan)
    for policy, candidate, seed in plan:
        fit_dir = _fit_directory(output_dir, policy, candidate.candidate_id, seed)
        completed = _completion_record(fit_dir)
        if completed is not None and args.resume:
            _validate_completed_fit(
                completed,
                base_config=base_config,
                profile=args.profile,
                policy=policy,
                candidate=candidate,
                training_seed=seed,
                shared_data_fingerprint=shared_data_fingerprint,
            )
            completed_before += 1
            continue
        result = _run_fit(
            base_config=base_config,
            profile=args.profile,
            stage=args.stage,
            policy=policy,
            candidate=candidate,
            training_seed=seed,
            output_dir=output_dir,
            output_relative=output_relative,
            train_products=train_products,
            validation_products=validation_products,
            feature_stats=feature_stats,
            oracle_levels=oracle_levels,
            shared_data_fingerprint=shared_data_fingerprint,
        )
        completed_before += 1
        if args.stage == "screening":
            _write_stage1_outputs(output_dir, manifest)
        else:
            _write_final_outputs(output_dir)
        print(
            f"completed={completed_before}/{total} policy={policy} "
            f"candidate={candidate.candidate_id} seed={seed} "
            f"elapsed={result['fit_wall_seconds']:.1f}s "
            f"validation_reward={result['mean_reward_per_period']:.6f}"
        )

    if args.stage == "screening":
        promoted = _write_stage1_outputs(output_dir, manifest)
        if promoted:
            print("stage1_complete promotions_saved=true stage2_started=false")
    else:
        selected = _write_final_outputs(output_dir)
        if selected:
            print("stage2_complete final_selections_saved=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
