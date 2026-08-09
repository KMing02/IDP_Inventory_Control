"""Inventory-range OOD structural diagnostic for selected validation checkpoints.

No model is trained and no operational reward evaluation is called here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
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
    _profile_settings,
)
from inventory_control.paper_configs import load_paper_lost_sales_config  # noqa: E402
from inventory_control.policies.neural.fcnn import compute_paper_fcnn_feature_stats  # noqa: E402
from inventory_control.products import PaperLostSalesProductGenerator  # noqa: E402
from scripts.run_primary_validation_followups import SELECTED_CANDIDATES  # noqa: E402
from scripts.tune_primary_neural_policies import (  # noqa: E402
    PRIMARY_POLICIES,
    _atomic_write_csv,
    _atomic_write_json,
    _completion_record,
    _environment_record,
    _fingerprint,
    _make_model,
    _repository_relative_path,
    _resolved_fit_config,
    _safe_generate_products,
    _write_or_validate_json,
    data_fingerprints,
    validate_split_and_output,
)


VIOLATION_TOLERANCE = 1e-6
MINIMUM_PAIR_SEPARATION_FACTOR = 0.01


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stratified_inventory_grid(
    historical_mean: float,
    inventory_factor: float,
    point_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return an exact-size stratified-random grid including 0, 4d, and 8d."""
    if point_count < 3 or inventory_factor != 8.0:
        raise ValueError("The preregistered grid requires factor 8 and at least three points")
    interior_count = point_count - 3
    edges = np.linspace(0.0, inventory_factor, interior_count + 1)
    interior = edges[:-1] + rng.random(interior_count) * np.diff(edges)
    normalized = np.concatenate([interior, np.array([0.0, 4.0, 8.0])])
    return np.sort(normalized * historical_mean)


def sample_ordered_pairs(
    rng: np.random.Generator,
    lower: float,
    upper: float,
    pair_count: int,
    minimum_separation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample deterministic ordered pairs with a required positive separation."""
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    collected = 0
    while collected < pair_count:
        draws = rng.uniform(lower, upper, size=(max(2 * (pair_count - collected), 64), 2))
        low = np.minimum(draws[:, 0], draws[:, 1])
        high = np.maximum(draws[:, 0], draws[:, 1])
        keep = high - low >= minimum_separation
        low = low[keep]
        high = high[keep]
        take = min(len(low), pair_count - collected)
        lows.append(low[:take])
        highs.append(high[:take])
        collected += take
    return np.concatenate(lows), np.concatenate(highs)


def positive_violation_metrics(low_order: np.ndarray, high_order: np.ndarray) -> dict[str, Any]:
    differences = np.asarray(high_order, dtype=float) - np.asarray(low_order, dtype=float)
    finite = np.isfinite(differences)
    positive = np.maximum(differences[finite], 0.0)
    strict = positive[positive > VIOLATION_TOLERANCE]
    return {
        "ordered_pair_count": int(len(differences)),
        "finite_ordered_pair_count": int(finite.sum()),
        "violating_pair_count": int(len(strict)),
        "raw_violation_rate": float(len(strict) / max(int(finite.sum()), 1)),
        "maximum_positive_violation": float(strict.max()) if len(strict) else 0.0,
        "p95_strictly_positive_violation": float(np.quantile(strict, 0.95)) if len(strict) else 0.0,
        "non_finite_pair_difference_count": int((~finite).sum()),
        "_strict_positive_values": strict,
    }


def _grid_metrics(
    inventory: np.ndarray,
    raw: np.ndarray,
    deployed: np.ndarray,
    oracle_level: float,
    historical_mean: float,
) -> dict[str, Any]:
    raw = np.asarray(raw, dtype=float)
    deployed = np.asarray(deployed, dtype=float)
    finite = np.isfinite(raw) & np.isfinite(deployed)
    clipping = raw - deployed
    hit = finite & (clipping > VIOLATION_TOLERANCE)
    finite_clipping = clipping[finite]
    dx = np.diff(inventory)
    raw_slopes = np.diff(raw) / np.maximum(dx, 1e-12)
    deployed_slopes = np.diff(deployed) / np.maximum(dx, 1e-12)
    finite_slopes = np.isfinite(raw_slopes) & np.isfinite(deployed_slopes)
    oracle_left = np.maximum(float(oracle_level) - inventory[:-1], 0.0)
    active = finite_slopes & (oracle_left > VIOLATION_TOLERANCE)
    inactive = finite_slopes & ~active
    raw_active = np.abs(raw_slopes[active] + 1.0)
    raw_inactive = np.abs(raw_slopes[inactive])
    dep_active = np.abs(deployed_slopes[active] + 1.0)
    dep_inactive = np.abs(deployed_slopes[inactive])
    finite_raw = raw[np.isfinite(raw)]
    raw_differences = np.diff(raw)
    finite_raw_differences = raw_differences[np.isfinite(raw_differences)]
    dense_positive = np.maximum(finite_raw_differences, 0.0)
    dense_strict = dense_positive[dense_positive > VIOLATION_TOLERANCE]
    return {
        "grid_point_count": int(len(inventory)),
        "non_finite_action_count": int((~finite).sum()),
        "dense_grid_finite_interval_count": int(len(finite_raw_differences)),
        "dense_grid_violating_interval_count": int(len(dense_strict)),
        "dense_grid_raw_violation_rate": float(
            len(dense_strict) / max(len(finite_raw_differences), 1)
        ),
        "dense_grid_maximum_positive_violation": (
            float(dense_strict.max()) if len(dense_strict) else 0.0
        ),
        "dense_grid_p95_strictly_positive_violation": (
            float(np.quantile(dense_strict, 0.95)) if len(dense_strict) else 0.0
        ),
        "raw_active_slope_deviation_sum": float(raw_active.sum()),
        "raw_active_slope_interval_count": int(len(raw_active)),
        "raw_inactive_slope_deviation_sum": float(raw_inactive.sum()),
        "raw_inactive_slope_interval_count": int(len(raw_inactive)),
        "deployed_active_slope_deviation_sum": float(dep_active.sum()),
        "deployed_active_slope_interval_count": int(len(dep_active)),
        "deployed_inactive_slope_deviation_sum": float(dep_inactive.sum()),
        "deployed_inactive_slope_interval_count": int(len(dep_inactive)),
        "maximum_raw_order": float(finite_raw.max()) if len(finite_raw) else np.nan,
        "maximum_raw_order_divided_by_historical_mean": (
            float(finite_raw.max() / historical_mean) if len(finite_raw) else np.nan
        ),
        "cap_hit_count": int(hit.sum()),
        "finite_grid_point_count": int(finite.sum()),
        "cap_hit_rate": float(hit.sum() / max(int(finite.sum()), 1)),
        "clipping_sum": float(finite_clipping.sum()) if len(finite_clipping) else 0.0,
        "clipping_hit_sum": float(clipping[hit].sum()) if np.any(hit) else 0.0,
        "mean_clipping_amount": float(finite_clipping.mean()) if len(finite_clipping) else 0.0,
        "mean_clipping_conditional_on_hit": float(clipping[hit].mean()) if np.any(hit) else 0.0,
        "maximum_clipping_amount": float(finite_clipping.max()) if len(finite_clipping) else 0.0,
    }


def _summarize_checkpoint(product_rows: pd.DataFrame, strict_values: dict[str, list[np.ndarray]]) -> pd.DataFrame:
    rows = []
    for region, group in product_rows.groupby("region", sort=False):
        strict_parts = strict_values[region]
        strict = np.concatenate(strict_parts) if strict_parts else np.array([], dtype=float)
        finite_pairs = int(group["finite_ordered_pair_count"].sum())
        violating = int(group["violating_pair_count"].sum())
        dense_intervals = int(group["dense_grid_finite_interval_count"].sum())
        dense_violating = int(group["dense_grid_violating_interval_count"].sum())
        active_count = int(group["raw_active_slope_interval_count"].sum())
        inactive_count = int(group["raw_inactive_slope_interval_count"].sum())
        dep_active_count = int(group["deployed_active_slope_interval_count"].sum())
        dep_inactive_count = int(group["deployed_inactive_slope_interval_count"].sum())
        hit_count = int(group["cap_hit_count"].sum())
        finite_points = int(group["finite_grid_point_count"].sum())
        rows.append(
            {
                "region": region,
                "raw_violation_rate": violating / max(finite_pairs, 1),
                "violating_pair_count": violating,
                "ordered_pair_count": int(group["ordered_pair_count"].sum()),
                "maximum_positive_violation": float(strict.max()) if len(strict) else 0.0,
                "p95_strictly_positive_violation": float(np.quantile(strict, 0.95)) if len(strict) else 0.0,
                "dense_grid_raw_violation_rate": dense_violating / max(dense_intervals, 1),
                "dense_grid_violating_interval_count": dense_violating,
                "dense_grid_finite_interval_count": dense_intervals,
                "dense_grid_maximum_positive_violation": float(
                    group["dense_grid_maximum_positive_violation"].max()
                ),
                "dense_grid_p95_strictly_positive_violation": float(
                    group["dense_grid_p95_strictly_positive_violation"].max()
                ),
                "raw_active_slope_deviation_from_minus_one": float(group["raw_active_slope_deviation_sum"].sum()) / max(active_count, 1),
                "raw_inactive_slope_deviation_from_zero": float(group["raw_inactive_slope_deviation_sum"].sum()) / max(inactive_count, 1),
                "deployed_active_slope_deviation_from_minus_one": float(group["deployed_active_slope_deviation_sum"].sum()) / max(dep_active_count, 1),
                "deployed_inactive_slope_deviation_from_zero": float(group["deployed_inactive_slope_deviation_sum"].sum()) / max(dep_inactive_count, 1),
                "maximum_raw_order": float(group["maximum_raw_order"].max()),
                "maximum_raw_order_divided_by_historical_mean": float(group["maximum_raw_order_divided_by_historical_mean"].max()),
                "cap_hit_rate": hit_count / max(finite_points, 1),
                "mean_clipping_amount": float(group["clipping_sum"].sum()) / max(finite_points, 1),
                "mean_clipping_conditional_on_hit": float(group["clipping_hit_sum"].sum()) / max(hit_count, 1),
                "maximum_clipping_amount": float(group["maximum_clipping_amount"].max()),
                "non_finite_action_count": int(group["non_finite_action_count"].sum()),
                "non_finite_pair_difference_count": int(group["non_finite_pair_difference_count"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _checkpoint_dir(output_dir: Path, policy: str, candidate_id: str, seed: int) -> Path:
    return output_dir / "checkpoints" / policy / candidate_id / f"seed_{seed}"


def aggregate_seed_region_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate seeds while retaining genuine maxima and total pair counts."""
    rows = []
    keys = ["policy_name", "candidate_id", "region"]
    maximum_columns = {
        "maximum_positive_violation",
        "maximum_raw_order",
        "maximum_raw_order_divided_by_historical_mean",
        "maximum_clipping_amount",
        "dense_grid_maximum_positive_violation",
        "dense_grid_p95_strictly_positive_violation",
    }
    sum_columns = {
        "violating_pair_count",
        "ordered_pair_count",
        "non_finite_action_count",
        "non_finite_pair_difference_count",
        "dense_grid_violating_interval_count",
        "dense_grid_finite_interval_count",
    }
    for key, group in seed_summary.groupby(keys, sort=False):
        row: dict[str, Any] = dict(zip(keys, key))
        for column in seed_summary.columns:
            if column in keys or column == "training_seed":
                continue
            values = group[column].to_numpy(dtype=float)
            if column in maximum_columns:
                row[column] = float(np.max(values))
            elif column in sum_columns:
                row[column] = int(np.sum(values))
            else:
                row[column] = float(np.mean(values))
        row["raw_violation_rate"] = row["violating_pair_count"] / max(
            row["ordered_pair_count"] - row["non_finite_pair_difference_count"], 1
        )
        if "dense_grid_violating_interval_count" in row:
            row["dense_grid_raw_violation_rate"] = row[
                "dense_grid_violating_interval_count"
            ] / max(row["dense_grid_finite_interval_count"], 1)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", default="full")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--selected-configs", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--inventory-factor", type=float, required=True)
    parser.add_argument("--random-grid-points", type=int, required=True)
    parser.add_argument("--pairs-per-region-per-product", type=int, required=True)
    parser.add_argument("--diagnostic-seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_split_and_output(args.split, args.output_dir)
    if tuple(args.seeds) != (0, 1, 2) or args.inventory_factor != 8.0 or args.random_grid_points != 4096 or args.pairs_per_region_per_product != 2048 or args.diagnostic_seed != 3_000_003:
        raise ValueError("Structural diagnostic settings must match the preregistered specification")
    config_path, config_relative = _repository_relative_path(args.config, "--config")
    selected_path, _ = _repository_relative_path(args.selected_configs, "--selected-configs")
    checkpoint_root, _ = _repository_relative_path(args.checkpoint_root, "--checkpoint-root")
    output_dir, _ = _repository_relative_path(args.output_dir, "--output-dir")
    base_config = load_paper_lost_sales_config(config_path)
    settings = _profile_settings(base_config, args.profile)
    if float(settings["max_order_factor"]) != 4.0 or int(settings["validation_products"]) != 32:
        raise ValueError("Structural diagnostic requires F=4 and 32 validation products")
    selected = pd.read_csv(selected_path)
    expected_selection = {(policy, candidate.candidate_id) for policy, candidate in SELECTED_CANDIDATES.items()}
    actual_selection = set(zip(selected["policy_name"], selected["candidate_id"]))
    if actual_selection != expected_selection:
        raise RuntimeError("selected_configs.csv does not contain the preregistered selected policies")

    generator = PaperLostSalesProductGenerator(base_config)
    train_products = _safe_generate_products(generator, 128, "train")
    validation_products = _safe_generate_products(generator, 32, "validation")
    feature_stats = compute_paper_fcnn_feature_stats(train_products)
    oracle_levels = _oracle_levels(validation_products, int(settings["oracle_mc_samples"]), 10_000)
    fingerprints = data_fingerprints(base_config, settings, train_products, validation_products)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_or_validate_json(output_dir / "environment.json", _environment_record(config_relative, base_config))
    _write_or_validate_json(output_dir / "data_fingerprints.json", fingerprints)

    manifest_rows = []
    for policy in PRIMARY_POLICIES:
        candidate = SELECTED_CANDIDATES[policy]
        for seed in args.seeds:
            source = checkpoint_root / policy / candidate.candidate_id / f"seed_{seed}"
            completion = _completion_record(source)
            if completion is None:
                raise RuntimeError(f"Selected source checkpoint is incomplete: {source}")
            manifest_rows.append(
                {
                    "policy_name": policy,
                    "candidate_id": candidate.candidate_id,
                    "training_seed": seed,
                    "source_run_dir": source.resolve().relative_to(REPOSITORY_ROOT).as_posix(),
                    "checkpoint_sha256": _sha256(source / "model.pt"),
                }
            )
    manifest = pd.DataFrame(manifest_rows)
    _atomic_write_csv(output_dir / "checkpoint_manifest.csv", manifest)
    if args.dry_run:
        print(manifest.to_string(index=False))
        return 0

    started = time.perf_counter()
    completed = 0
    for manifest_row in manifest.itertuples(index=False):
        policy = str(manifest_row.policy_name)
        candidate = SELECTED_CANDIDATES[policy]
        seed = int(manifest_row.training_seed)
        source = REPOSITORY_ROOT / str(manifest_row.source_run_dir)
        destination = _checkpoint_dir(output_dir, policy, candidate.candidate_id, seed)
        marker = destination / "completion.json"
        if marker.exists() and args.resume:
            prior = json.loads(marker.read_text(encoding="utf-8"))
            if prior.get("checkpoint_sha256") != manifest_row.checkpoint_sha256:
                raise RuntimeError(f"Checkpoint changed since completed diagnostic: {source}")
            if prior.get("diagnostic_schema_version") == 2:
                completed += 1
                continue
        if marker.exists() and not args.resume:
            raise RuntimeError("Completed structural diagnostics exist; pass --resume")
        destination.mkdir(parents=True, exist_ok=True)
        marker.unlink(missing_ok=True)
        resolved_config, _ = _resolved_fit_config(base_config, args.profile, policy, candidate, seed)
        local_settings = _profile_settings(resolved_config, args.profile)
        model = _make_model(policy, _model_config(resolved_config, local_settings), seed)
        model.load_state_dict(torch.load(source / "model.pt", map_location="cpu", weights_only=True))

        product_rows = []
        strict_values = {"in_range": [], "extended": [], "full_range": []}
        for product_index, product in enumerate(validation_products):
            historical_mean = max(float(np.mean(product.historical_demand)), 1.0)
            rng = np.random.default_rng(args.diagnostic_seed + product_index)
            dense = stratified_inventory_grid(historical_mean, args.inventory_factor, args.random_grid_points, rng)
            regions = {
                "in_range": np.linspace(0.0, 4.0 * historical_mean, 50),
                "extended": dense[dense > 4.0 * historical_mean],
                "full_range": dense,
            }
            pair_bounds = {
                "in_range": (0.0, 4.0 * historical_mean),
                "extended": (4.0 * historical_mean, 8.0 * historical_mean),
                "full_range": (0.0, 8.0 * historical_mean),
            }
            for region, inventory in regions.items():
                raw, deployed = _model_orders(model, product, feature_stats, inventory, 4.0, True)
                grid_metrics = _grid_metrics(inventory, raw, deployed, oracle_levels[product.product_id], historical_mean)
                low, high = sample_ordered_pairs(
                    rng,
                    *pair_bounds[region],
                    args.pairs_per_region_per_product,
                    MINIMUM_PAIR_SEPARATION_FACTOR * historical_mean,
                )
                pair_inventory = np.concatenate([low, high])
                pair_raw, _ = _model_orders(model, product, feature_stats, pair_inventory, 4.0, True)
                violation = positive_violation_metrics(pair_raw[: len(low)], pair_raw[len(low) :])
                strict_values[region].append(violation.pop("_strict_positive_values"))
                product_rows.append(
                    {
                        "policy_name": policy,
                        "candidate_id": candidate.candidate_id,
                        "training_seed": seed,
                        "product_id": product.product_id,
                        "region": region,
                        **grid_metrics,
                        **violation,
                    }
                )
        product_frame = pd.DataFrame(product_rows)
        summary = _summarize_checkpoint(product_frame, strict_values)
        summary.insert(0, "training_seed", seed)
        summary.insert(0, "candidate_id", candidate.candidate_id)
        summary.insert(0, "policy_name", policy)
        _atomic_write_csv(destination / "per_product_diagnostics.csv", product_frame)
        _atomic_write_csv(destination / "seed_region_summary.csv", summary)
        completion = {
            "policy_name": policy,
            "candidate_id": candidate.candidate_id,
            "training_seed": seed,
            "source_run_dir": manifest_row.source_run_dir,
            "checkpoint_sha256": manifest_row.checkpoint_sha256,
            "operational_reward_evaluation_called": False,
            "protected_test_accessed": False,
            "diagnostic_schema_version": 2,
        }
        _atomic_write_json(marker, completion)
        completed += 1
        print(f"completed={completed}/{len(manifest)} policy={policy} candidate={candidate.candidate_id} seed={seed}")

    seed_frames = [pd.read_csv(path) for path in sorted((output_dir / "checkpoints").glob("*/*/seed_*/seed_region_summary.csv"))]
    seed_summary = pd.concat(seed_frames, ignore_index=True)
    _atomic_write_csv(output_dir / "seed_region_summary.csv", seed_summary)
    aggregate = aggregate_seed_region_summary(seed_summary)
    _atomic_write_csv(output_dir / "aggregate_region_summary.csv", aggregate)
    _atomic_write_json(output_dir / "analysis_completion.json", {"completed_checkpoints": completed, "expected_checkpoints": len(manifest), "wall_seconds": time.perf_counter() - started, "operational_reward_evaluation_called": False, "protected_test_accessed": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
