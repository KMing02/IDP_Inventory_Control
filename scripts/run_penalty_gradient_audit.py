"""Validation-only autograd and completed-run audit for PenaltyMLP."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from inventory_control.idp_benchmark import _model_config, _profile_settings  # noqa: E402
from inventory_control.paper_configs import load_paper_lost_sales_config  # noqa: E402
from inventory_control.policies.neural.fcnn import compute_paper_fcnn_feature_stats  # noqa: E402
from inventory_control.products import PaperLostSalesProductGenerator  # noqa: E402
from inventory_control.training.direct_backprop import (  # noqa: E402
    make_paper_lost_sales_training_batch,
    paper_inventory_monotonicity_penalty,
    paper_lost_sales_fcnn_rollout,
    paper_reward_normalization_scale,
    train_paper_lost_sales_fcnn,
)
from scripts.tune_primary_neural_policies import (  # noqa: E402
    CANDIDATE_BY_ID,
    _atomic_write_csv,
    _atomic_write_json,
    _completion_record,
    _fingerprint,
    _make_model,
    _repository_relative_path,
    _resolved_fit_config,
    _safe_generate_products,
    validate_split_and_output,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_collection_fingerprint(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(values):
        tensor = values[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _flatten_gradients(
    gradients: Sequence[torch.Tensor | None], parameters: Sequence[torch.nn.Parameter]
) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.detach().reshape(-1)
            for gradient, parameter in zip(gradients, parameters)
        ]
    )


def _state_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1).cpu() for parameter in model.parameters()])


def controlled_gradient_audit(
    config: dict[str, Any], lambdas: Sequence[float], controlled_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    candidate = CANDIDATE_BY_ID["C8"]
    local_config, _ = _resolved_fit_config(
        config, "full", "PenaltyMLP", candidate, controlled_seed
    )
    settings = _profile_settings(local_config, "full")
    generator = PaperLostSalesProductGenerator(local_config)
    products = _safe_generate_products(generator, 128, "train")
    feature_stats = compute_paper_fcnn_feature_stats(products)

    batch_generator = torch.Generator().manual_seed(controlled_seed)
    indices = torch.randint(
        0, len(products), (int(settings["batch_size"]),), generator=batch_generator
    ).tolist()
    batch = make_paper_lost_sales_training_batch(
        [products[index] for index in indices],
        int(settings["training_horizon"]),
        batch_generator,
        feature_stats=feature_stats,
    )
    batch_fingerprint = _tensor_collection_fingerprint(batch)

    pair_generator = torch.Generator().manual_seed(controlled_seed + 1_000_003)
    maximum_inventory = float(settings["max_order_factor"]) * torch.clamp(
        batch["historical_mean_demand"], min=1.0
    )
    pair_values: dict[str, torch.Tensor] = {}
    for sample in range(int(settings["monotonicity_samples_per_product"])):
        first = torch.rand(len(indices), generator=pair_generator)
        second = torch.rand(len(indices), generator=pair_generator)
        pair_values[f"low_{sample}"] = torch.minimum(first, second) * maximum_inventory
        pair_values[f"high_{sample}"] = torch.maximum(first, second) * maximum_inventory
    pair_fingerprint = _tensor_collection_fingerprint(pair_values)

    initial_model = _make_model(
        "PenaltyMLP", _model_config(local_config, settings), controlled_seed
    )
    initial_state = {
        key: value.detach().clone() for key, value in initial_model.state_dict().items()
    }
    initial_vector = _state_vector(initial_model)
    initial_fingerprint = _fingerprint(
        {key: value.detach().cpu().numpy().tolist() for key, value in initial_state.items()}
    )

    metric_rows = []
    layer_rows = []
    updated_vectors: dict[float, torch.Tensor] = {}
    penalty_values = []
    for penalty_lambda in lambdas:
        model = _make_model(
            "PenaltyMLP", _model_config(local_config, settings), controlled_seed
        )
        model.load_state_dict(initial_state)
        parameters = tuple(model.parameters())
        mean_reward = paper_lost_sales_fcnn_rollout(
            model,
            batch,
            max_order_factor=float(settings["max_order_factor"]),
            scale_action_by_history_mean=bool(settings["scale_action_by_history_mean"]),
            include_terminal_salvage=True,
        ).mean()
        task_loss = -mean_reward / paper_reward_normalization_scale(
            batch, int(settings["training_horizon"])
        )
        penalty_generator = torch.Generator().manual_seed(controlled_seed + 1_000_003)
        penalty = paper_inventory_monotonicity_penalty(
            model,
            batch,
            penalty_generator,
            samples_per_product=int(settings["monotonicity_samples_per_product"]),
            max_inventory_factor=float(settings["max_order_factor"]),
            scale_action_by_history_mean=bool(settings["scale_action_by_history_mean"]),
        )
        weighted_penalty = float(penalty_lambda) * penalty
        total_loss = task_loss + weighted_penalty
        task_gradients = torch.autograd.grad(
            task_loss, parameters, retain_graph=True, allow_unused=True
        )
        raw_penalty_gradients = torch.autograd.grad(
            penalty, parameters, retain_graph=True, allow_unused=True
        )
        task_vector = _flatten_gradients(task_gradients, parameters)
        raw_penalty_vector = _flatten_gradients(raw_penalty_gradients, parameters)
        weighted_penalty_vector = float(penalty_lambda) * raw_penalty_vector
        denominator = torch.linalg.vector_norm(task_vector) * torch.linalg.vector_norm(
            raw_penalty_vector
        )
        cosine = (
            float(torch.dot(task_vector, raw_penalty_vector) / denominator)
            if float(denominator) > 0.0
            else np.nan
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=float(candidate.learning_rate))
        optimizer.zero_grad()
        total_loss.backward()
        total_backward_gradients = {
            name: parameter.grad.detach().clone() if parameter.grad is not None else None
            for name, parameter in model.named_parameters()
        }
        optimizer.step()
        updated = _state_vector(model)
        updated_vectors[float(penalty_lambda)] = updated
        penalty_values.append(float(penalty.detach()))
        metric_rows.append(
            {
                "lambda": float(penalty_lambda),
                "task_reward_loss": float(task_loss.detach()),
                "raw_monotonicity_penalty": float(penalty.detach()),
                "weighted_penalty": float(weighted_penalty.detach()),
                "total_loss": float(total_loss.detach()),
                "task_gradient_norm": float(torch.linalg.vector_norm(task_vector)),
                "raw_penalty_gradient_norm": float(
                    torch.linalg.vector_norm(raw_penalty_vector)
                ),
                "weighted_penalty_gradient_norm": float(
                    torch.linalg.vector_norm(weighted_penalty_vector)
                ),
                "task_penalty_gradient_cosine_similarity": cosine,
                "parameter_update_norm": float(torch.linalg.vector_norm(updated - initial_vector)),
                "penalty_requires_grad": bool(penalty.requires_grad),
                "penalty_has_grad_fn": penalty.grad_fn is not None,
                "grad_mode_enabled": bool(torch.is_grad_enabled()),
                "layers_reached_by_raw_penalty_gradient": int(
                    sum(
                        gradient is not None and float(torch.linalg.vector_norm(gradient)) > 0.0
                        for gradient in raw_penalty_gradients
                    )
                ),
                "intended_parameter_tensor_count": len(parameters),
                "layers_reached_by_total_backward": int(
                    sum(
                        gradient is not None and float(torch.linalg.vector_norm(gradient)) > 0.0
                        for gradient in total_backward_gradients.values()
                    )
                ),
            }
        )
        for (name, _), task_gradient, raw_gradient in zip(
            model.named_parameters(), task_gradients, raw_penalty_gradients
        ):
            layer_rows.append(
                {
                    "lambda": float(penalty_lambda),
                    "parameter_name": name,
                    "task_gradient_norm": (
                        float(torch.linalg.vector_norm(task_gradient))
                        if task_gradient is not None
                        else 0.0
                    ),
                    "raw_penalty_gradient_norm": (
                        float(torch.linalg.vector_norm(raw_gradient))
                        if raw_gradient is not None
                        else 0.0
                    ),
                    "weighted_penalty_gradient_norm": (
                        float(penalty_lambda) * float(torch.linalg.vector_norm(raw_gradient))
                        if raw_gradient is not None
                        else 0.0
                    ),
                    "reached_by_raw_penalty_gradient": bool(
                        raw_gradient is not None
                        and float(torch.linalg.vector_norm(raw_gradient)) > 0.0
                    ),
                }
            )

    baseline = updated_vectors[0.0]
    metrics = pd.DataFrame(metric_rows)
    metrics["maximum_parameter_difference_vs_lambda_0"] = [
        float(torch.max(torch.abs(updated_vectors[float(value)] - baseline)))
        for value in metrics["lambda"]
    ]
    metrics["parameter_l2_difference_vs_lambda_0"] = [
        float(torch.linalg.vector_norm(updated_vectors[float(value)] - baseline))
        for value in metrics["lambda"]
    ]
    source = inspect.getsource(train_paper_lost_sales_fcnn)
    verification = {
        "controlled_training_seed": controlled_seed,
        "batch_fingerprint": batch_fingerprint,
        "sampled_pair_fingerprint": pair_fingerprint,
        "initial_parameter_fingerprint": initial_fingerprint,
        "raw_penalty_identical_across_lambdas": bool(
            np.allclose(penalty_values, penalty_values[0], rtol=0.0, atol=0.0)
        ),
        "penalty_tensor_not_detached": bool(metrics["penalty_requires_grad"].all()),
        "penalty_has_autograd_function": bool(metrics["penalty_has_grad_fn"].all()),
        "no_no_grad_blocks_controlled_penalty": bool(metrics["grad_mode_enabled"].all()),
        "raw_penalty_gradient_reaches_all_intended_parameter_tensors": bool(
            (metrics["layers_reached_by_raw_penalty_gradient"] == metrics["intended_parameter_tensor_count"]).all()
        ),
        "production_source_adds_logged_penalty_to_loss": (
            "loss = primary_loss + float(monotonicity_penalty_weight) * monotonicity_penalty"
            in source
        ),
        "production_source_backpropagates_total_loss": "loss.backward()" in source,
        "production_source_logs_same_penalty_tensor_value": (
            '"monotonicity_penalty": float(monotonicity_penalty.detach().item())'
            in source
        ),
    }
    return metrics, pd.DataFrame(layer_rows), verification


def completed_c8_comparison(checkpoint_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison_rows = []
    ratio_rows = []
    for seed in (0, 1, 2):
        paper_dir = checkpoint_root / "PaperFCNN" / "C8" / f"seed_{seed}"
        penalty_dir = checkpoint_root / "PenaltyMLP" / "C8" / f"seed_{seed}"
        if _completion_record(paper_dir) is None or _completion_record(penalty_dir) is None:
            raise RuntimeError(f"Incomplete C8 checkpoint pair for seed {seed}")
        paper_state = torch.load(paper_dir / "model.pt", map_location="cpu", weights_only=True)
        penalty_state = torch.load(penalty_dir / "model.pt", map_location="cpu", weights_only=True)
        differences = [
            (penalty_state[key].to(dtype=torch.float64) - paper_state[key].to(dtype=torch.float64)).reshape(-1)
            for key in paper_state
        ]
        difference = torch.cat(differences)
        paper_predictions = pd.read_csv(paper_dir / "predictions.csv")
        penalty_predictions = pd.read_csv(penalty_dir / "predictions.csv")
        if not paper_predictions["product_id"].str.startswith("validation_").all() or not penalty_predictions["product_id"].str.startswith("validation_").all():
            raise RuntimeError("Non-validation prediction found in C8 audit")
        raw_difference = penalty_predictions["raw_order"].to_numpy() - paper_predictions["raw_order"].to_numpy()
        deployed_difference = penalty_predictions["deployed_order"].to_numpy() - paper_predictions["deployed_order"].to_numpy()
        comparison_rows.append(
            {
                "training_seed": seed,
                "paper_checkpoint_sha256": _sha256(paper_dir / "model.pt"),
                "penalty_checkpoint_sha256": _sha256(penalty_dir / "model.pt"),
                "checkpoint_parameter_l2_distance": float(torch.linalg.vector_norm(difference)),
                "checkpoint_maximum_tensor_difference": float(torch.max(torch.abs(difference))),
                "raw_prediction_l2_distance": float(np.linalg.norm(raw_difference)),
                "raw_prediction_maximum_absolute_difference": float(np.max(np.abs(raw_difference))),
                "deployed_prediction_l2_distance": float(np.linalg.norm(deployed_difference)),
                "deployed_prediction_maximum_absolute_difference": float(np.max(np.abs(deployed_difference))),
                "deployed_prediction_rmse_difference": float(np.sqrt(np.mean(deployed_difference**2))),
            }
        )
        log = pd.read_csv(penalty_dir / "training_log.csv")
        weighted = np.abs(log["monotonicity_penalty_weight"] * log["monotonicity_penalty"])
        main = np.abs(log["primary_loss"].to_numpy(dtype=float))
        ratio = weighted / np.maximum(main, 1e-12)
        for phase, label in ((None, "all"), (0, "imitation"), (1, "reward")):
            selected = np.ones(len(log), dtype=bool) if phase is None else log["phase"].to_numpy() == phase
            ratio_rows.append(
                {
                    "training_seed": seed,
                    "phase": label,
                    "configured_lambda": float(log["monotonicity_penalty_weight"].iloc[0]),
                    "active_step_count": int(np.sum(weighted[selected] > 0.0)),
                    "step_count": int(selected.sum()),
                    "weighted_penalty_to_main_loss_ratio_mean": float(np.mean(ratio[selected])),
                    "weighted_penalty_to_main_loss_ratio_median": float(np.median(ratio[selected])),
                    "weighted_penalty_to_main_loss_ratio_max": float(np.max(ratio[selected])),
                    "weighted_penalty_magnitude_mean": float(np.mean(weighted[selected])),
                    "main_loss_magnitude_mean": float(np.mean(main[selected])),
                }
            )
    return pd.DataFrame(comparison_rows), pd.DataFrame(ratio_rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--controlled-seed", type=int, default=1)
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.1, 1.0, 10.0])
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_split_and_output(args.split, args.output_dir)
    if tuple(args.lambdas) != (0.0, 0.1, 1.0, 10.0) or args.controlled_seed != 1:
        raise ValueError("Penalty gradient audit settings must match the preregistered values")
    config_path, _ = _repository_relative_path(args.config, "--config")
    checkpoint_root, _ = _repository_relative_path(args.checkpoint_root, "--checkpoint-root")
    output_dir, _ = _repository_relative_path(args.output_dir, "--output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_paper_lost_sales_config(config_path)
    metrics, layers, verification = controlled_gradient_audit(
        config, args.lambdas, args.controlled_seed
    )
    checkpoints, ratios = completed_c8_comparison(checkpoint_root)
    _atomic_write_csv(output_dir / "controlled_step_metrics.csv", metrics)
    _atomic_write_csv(output_dir / "layer_gradient_reach.csv", layers)
    _atomic_write_json(output_dir / "autograd_verification.json", verification)
    _atomic_write_csv(output_dir / "c8_checkpoint_comparison.csv", checkpoints)
    _atomic_write_csv(output_dir / "c8_penalty_loss_scale.csv", ratios)
    _atomic_write_json(
        output_dir / "completion.json",
        {
            "controlled_seed": args.controlled_seed,
            "lambdas": list(args.lambdas),
            "completed_c8_seeds": [0, 1, 2],
            "protected_test_accessed": False,
            "training_logic_modified": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
