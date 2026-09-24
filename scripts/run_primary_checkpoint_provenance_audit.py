"""Audit primary FCNN/CMNN checkpoint provenance on validation only."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inventory_control.idp_benchmark import (  # noqa: E402
    _model_config,
    _operational_table,
    _oracle_levels,
    _profile_settings,
)
from inventory_control.paper_configs import load_paper_lost_sales_config  # noqa: E402
from inventory_control.policies.neural.fcnn import compute_paper_fcnn_feature_stats  # noqa: E402
from inventory_control.products import PaperLostSalesProductGenerator  # noqa: E402
from inventory_control.training.direct_backprop import train_paper_lost_sales_fcnn  # noqa: E402
from scripts.tune_primary_neural_policies import (  # noqa: E402
    Candidate,
    _atomic_torch_save,
    _atomic_write_csv,
    _atomic_write_json,
    _fixed_validation_grid_seed,
    _make_model,
    _resolved_fit_config,
    _safe_generate_products,
    data_fingerprints,
)


OUTPUT = ROOT / "outputs/idp_followup/primary_checkpoint_provenance_audit"
PROTOCOL = ROOT / "outputs/idp_followup/final_test_protocol.json"
MATCHED_RECORDS = ROOT / "outputs/idp_followup/matched_width_depth_10seed_validation/records"
CONFIG = ROOT / "configs/paper_lost_sales_medium_seed0_converged.yaml"
CANDIDATE = Candidate("C2", 2, 32, 0.001, 15, (1, 1, 1), 10.0)
POLICIES = ("PaperFCNN", "ConstrainedMonoMLP")
SEEDS = tuple(range(10))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_max_difference(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    if left.keys() != right.keys():
        return float("inf")
    return max(
        float(torch.max(torch.abs(left[key].detach().cpu() - right[key].detach().cpu())).item())
        for key in left
    )


def _reward(
    model,
    policy,
    validation_products,
    feature_stats,
    settings,
    oracle_levels,
) -> float:
    table = _operational_table(
        validation_products,
        {policy: model},
        feature_stats,
        settings,
        oracle_levels,
        "validation",
    )
    return float(table.loc[table["policy_name"] == policy, "mean_reward_per_period"].mean())


def _load_completed_rows() -> list[dict]:
    rows = []
    for path in sorted((OUTPUT / "runs").glob("*/seed_*/completion.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _write_summary() -> None:
    rows = _load_completed_rows()
    if not rows:
        return
    frame = pd.DataFrame(rows).sort_values(["policy_name", "training_seed"])
    _atomic_write_csv(OUTPUT / "checkpoint_provenance.csv", frame)
    summary_rows = []
    for policy, group in frame.groupby("policy_name", sort=False):
        phase_counts = group["archived_selected_phase"].value_counts()
        reward_updates = (
            group.loc[group["archived_selected_phase"] == "reward", "archived_selected_reward_update"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )
        summary_rows.append(
            {
                "policy_name": policy,
                "seed_count": len(group),
                "archived_from_imitation_count": int(phase_counts.get("imitation", 0)),
                "archived_from_reward_count": int(phase_counts.get("reward", 0)),
                "selected_reward_update_distribution": json.dumps(
                    {str(int(key)): int(value) for key, value in reward_updates.items()},
                    sort_keys=True,
                ),
                "mean_archived_minus_actual_update_15_reward": float(
                    group["archived_minus_actual_update_15_reward"].mean()
                ),
                "all_protocol_hashes_match": bool(group["protocol_sha256_match"].all()),
                "all_record_hashes_match": bool(group["matched_record_sha256_match"].all()),
                "all_reproduced_selected_parameters_match": bool(
                    (group["reproduced_selected_state_max_abs_diff_vs_archived"] == 0.0).all()
                ),
            }
        )
    _atomic_write_csv(OUTPUT / "checkpoint_provenance_summary.csv", pd.DataFrame(summary_rows))


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    manifest = {
        (row["policy_name"], int(row["training_seed"])): row
        for row in protocol["checkpoint_manifest"]
        if row["policy_name"] in POLICIES
    }
    if set(manifest) != {(policy, seed) for policy in POLICIES for seed in SEEDS}:
        raise RuntimeError("Frozen protocol does not contain the expected 20 checkpoints")

    base = load_paper_lost_sales_config(CONFIG)
    base_settings = _profile_settings(base, "full")
    generator = PaperLostSalesProductGenerator(base)
    train_products = _safe_generate_products(generator, int(base_settings["training_products"]), "train")
    validation_products = _safe_generate_products(
        generator, int(base_settings["validation_products"]), "validation"
    )
    feature_stats = compute_paper_fcnn_feature_stats(train_products)
    fingerprints = data_fingerprints(base, base_settings, train_products, validation_products)
    _atomic_write_json(OUTPUT / "data_fingerprints.json", fingerprints)
    _atomic_write_json(
        OUTPUT / "audit_metadata.json",
        {
            "scope": "primary_checkpoint_provenance",
            "split": "validation",
            "policies": list(POLICIES),
            "training_seeds": list(SEEDS),
            "reward_updates_replayed": 15,
            "protected_test_accessed": False,
            "protected_test_executions": 0,
            "protocol_fingerprint_read_for_checkpoint_manifest_only": protocol.get(
                "protocol_fingerprint"
            ),
        },
    )

    total = len(POLICIES) * len(SEEDS)
    completed = len(_load_completed_rows())
    for policy in POLICIES:
        for seed in SEEDS:
            fit_dir = OUTPUT / "runs" / policy / f"seed_{seed}"
            marker = fit_dir / "completion.json"
            if marker.exists():
                continue
            fit_dir.mkdir(parents=True, exist_ok=True)
            config, resolved = _resolved_fit_config(base, "full", policy, CANDIDATE, seed)
            settings = _profile_settings(config, "full")
            model = _make_model(policy, _model_config(config, settings), seed)
            oracle_levels = _oracle_levels(validation_products, int(settings["oracle_mc_samples"]), seed)

            def validation_callback(epoch, current_model):
                del epoch
                return 100.0 * _reward(
                    current_model,
                    policy,
                    validation_products,
                    feature_stats,
                    settings,
                    oracle_levels,
                )

            started = time.perf_counter()
            with _fixed_validation_grid_seed(validation_products):
                result = train_paper_lost_sales_fcnn(
                    train_products,
                    model,
                    horizon=int(settings["training_horizon"]),
                    batch_size=min(int(settings["batch_size"]), len(train_products)),
                    epochs=15,
                    learning_rate=float(config["learning_rate"]),
                    seed=int(settings["seed"]),
                    imitation_epochs=int(settings["imitation_epochs"]),
                    validation_products=validation_products,
                    validation_interval=int(settings["validation_interval"]),
                    validation_callback=validation_callback,
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
                    track_imitation_validation=bool(
                        settings.get("track_imitation_validation", False)
                    ),
                    reset_optimizer_after_imitation=bool(
                        settings["reset_optimizer_after_imitation"]
                    ),
                    adam_kwargs=dict(config.get("adam", {})),
                )
            if result.best_imitation_state_dict is None or result.best_state_dict is None:
                raise RuntimeError(f"Missing reproduced checkpoint for {policy} seed {seed}")

            actual_update_15_state = copy.deepcopy(model.state_dict())
            actual_update_15_reward = _reward(
                model, policy, validation_products, feature_stats, settings, oracle_levels
            )
            model.load_state_dict(result.best_imitation_state_dict)
            best_imitation_reward = _reward(
                model, policy, validation_products, feature_stats, settings, oracle_levels
            )
            model.load_state_dict(result.best_state_dict)
            reproduced_selected_reward = _reward(
                model, policy, validation_products, feature_stats, settings, oracle_levels
            )

            protocol_row = manifest[(policy, seed)]
            checkpoint_path = ROOT / protocol_row["checkpoint_path"]
            archived_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            actual_hash = _sha256(checkpoint_path)
            matched_record_path = (
                MATCHED_RECORDS
                / policy
                / "matched_width_depth"
                / f"seed_{seed}"
                / "record.json"
            )
            matched_record = json.loads(matched_record_path.read_text(encoding="utf-8"))
            training_log = pd.DataFrame(result.training_log or [])
            validation_rows = training_log.loc[
                training_log["validation_mean_total_reward"].notna()
            ].copy()
            selected_row = validation_rows.loc[
                validation_rows["validation_mean_total_reward"].idxmax()
            ]
            selected_phase = "imitation" if int(selected_row["phase"]) == 0 else "reward"
            selected_reward_update = (
                None
                if selected_phase == "imitation"
                else int(selected_row["epoch"]) - int(result.imitation_epochs_completed)
            )
            row = {
                "policy_name": policy,
                "training_seed": seed,
                "imitation_epochs_completed": int(result.imitation_epochs_completed),
                "best_imitation_epoch": int(result.best_imitation_epoch),
                "best_imitation_validation_reward_per_period": best_imitation_reward,
                "actual_update_15_validation_reward_per_period": actual_update_15_reward,
                "archived_selected_validation_reward_per_period": reproduced_selected_reward,
                "archived_recorded_validation_reward_per_period": float(
                    matched_record["mean_reward_per_period"]
                ),
                "archived_selected_phase": selected_phase,
                "archived_selected_global_epoch": int(selected_row["epoch"]),
                "archived_selected_reward_update": selected_reward_update,
                "archived_minus_actual_update_15_reward": reproduced_selected_reward
                - actual_update_15_reward,
                "checkpoint_path": protocol_row["checkpoint_path"],
                "protocol_checkpoint_sha256": protocol_row["checkpoint_sha256"],
                "actual_checkpoint_sha256": actual_hash,
                "protocol_sha256_match": actual_hash == protocol_row["checkpoint_sha256"],
                "matched_record_checkpoint_sha256": matched_record["checkpoint_sha256"],
                "matched_record_sha256_match": actual_hash
                == matched_record["checkpoint_sha256"],
                "reproduced_selected_state_max_abs_diff_vs_archived": _state_max_difference(
                    result.best_state_dict, archived_state
                ),
                "actual_update_15_state_max_abs_diff_vs_archived": _state_max_difference(
                    actual_update_15_state, archived_state
                ),
                "reproduced_selected_reward_minus_archived_record": reproduced_selected_reward
                - float(matched_record["mean_reward_per_period"]),
                "optimizer_reset_after_imitation": bool(
                    result.optimizer_reset_after_imitation
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
            _atomic_write_json(fit_dir / "resolved_config.json", resolved)
            _atomic_write_csv(fit_dir / "training_log.csv", training_log)
            _atomic_torch_save(
                fit_dir / "best_imitation_checkpoint.pt", result.best_imitation_state_dict
            )
            _atomic_torch_save(
                fit_dir / "actual_reward_update_15.pt", actual_update_15_state
            )
            _atomic_torch_save(
                fit_dir / "reproduced_archived_selected_checkpoint.pt",
                result.best_state_dict,
            )
            _atomic_write_json(fit_dir / "completion.json", row)
            completed += 1
            _write_summary()
            print(
                f"{completed}/{total} {policy} seed={seed} "
                f"phase={selected_phase} reward_update={selected_reward_update} "
                f"archived={reproduced_selected_reward:.6f} "
                f"actual15={actual_update_15_reward:.6f}",
                flush=True,
            )

    _write_summary()
    frame = pd.read_csv(OUTPUT / "checkpoint_provenance.csv")
    complete = len(frame) == total
    all_verified = bool(
        frame["protocol_sha256_match"].all()
        and frame["matched_record_sha256_match"].all()
        and (frame["reproduced_selected_state_max_abs_diff_vs_archived"] == 0.0).all()
    )
    _atomic_write_json(
        OUTPUT / "audit_completion.json",
        {
            "completed_rows": len(frame),
            "expected_rows": total,
            "complete": complete,
            "all_archived_checkpoint_hashes_and_parameters_verified": all_verified,
            "protected_test_accessed": False,
            "protected_test_executions": 0,
        },
    )
    return 0 if complete and all_verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
