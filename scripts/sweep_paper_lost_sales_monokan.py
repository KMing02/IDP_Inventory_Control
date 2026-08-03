"""Legacy raw-action MonoKAN validation sweep.

Prefer `scripts/run_monokan_optimization_sweep.py` and the residual order-up-to
benchmark path for current MONOkan experiments.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.paper_evaluation import PAPER_VALIDATION_EVALUATION_SEED_START
from inventory_control.paper_lost_sales import evaluate_paper_lost_sales_policy, historical_demand_base_stock_policy
from inventory_control.policies.neural.fcnn import compute_paper_fcnn_feature_stats
from inventory_control.policies.neural.monokan import PaperMonoKANPolicyAdapter, make_paper_monokan_model
from inventory_control.products import PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import train_paper_lost_sales_fcnn


METRIC_COLUMNS = [
    "mean_total_reward",
    "service_level",
    "lost_sales_rate",
    "mean_order_quantity",
    "mean_inventory",
    "mean_holding_cost",
    "mean_purchase_cost",
    "mean_sales_revenue",
    "mean_lost_sales_penalty",
]


def _policy_metrics_frame(config: dict, products: list, policy_factory) -> pd.DataFrame:
    """Evaluate one policy factory and return product-level validation metrics."""
    rows = []
    for index, product in enumerate(products):
        metrics = evaluate_paper_lost_sales_policy(
            product,
            policy_factory(product),
            horizon=int(config["evaluation_horizon"]),
            n_episodes=1,
            seed=PAPER_VALIDATION_EVALUATION_SEED_START + index,
            burn_in=int(config["burn_in"]),
        )
        rows.append({"product_id": product.product_id, **metrics})
    return pd.DataFrame(rows)


def _mean_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return {column: float(frame[column].mean()) for column in METRIC_COLUMNS}


def _sweep_configs() -> list[dict[str, float | int | str]]:
    return [
        {
            "label": "1x8_grid3_lr1e-3",
            "hidden_layers": 1,
            "hidden_units": 8,
            "grid": 3,
            "max_order_factor": 4.0,
            "learning_rate": 1e-3,
            "imitation_epochs": 20,
        },
        {
            "label": "1x16_grid3_lr3e-4",
            "hidden_layers": 1,
            "hidden_units": 16,
            "grid": 3,
            "max_order_factor": 4.0,
            "learning_rate": 3e-4,
            "imitation_epochs": 20,
        },
        {
            "label": "2x16_grid3_lr3e-4",
            "hidden_layers": 2,
            "hidden_units": 16,
            "grid": 3,
            "max_order_factor": 4.0,
            "learning_rate": 3e-4,
            "imitation_epochs": 20,
        },
        {
            "label": "1x16_grid5_lr3e-4",
            "hidden_layers": 1,
            "hidden_units": 16,
            "grid": 5,
            "max_order_factor": 4.0,
            "learning_rate": 3e-4,
            "imitation_epochs": 20,
        },
    ]


def main() -> None:
    """Select on 20 validation products and full-evaluate only the winner."""
    config = load_paper_lost_sales_config()
    config_name = Path(os.environ.get("PAPER_LOST_SALES_CONFIG", "configs/paper_lost_sales.yaml")).stem
    output_suffix = "_debug" if "debug" in config_name else ""
    runtime = config.get("runtime", {})
    generator = PaperLostSalesProductGenerator(config)
    n_products = min(
        int(config["training_products"]),
        int(runtime.get("monokan_sweep_train_products", min(128, int(runtime.get("fcnn_train_products", 128))))),
    )
    full_validation_count = int(runtime.get("fcnn_validation_products", 100))
    selection_count = min(
        int(runtime.get("monokan_sweep_selection_products", 20)),
        full_validation_count,
    )
    train_products = generator.generate(n_products, split="train")
    full_validation_products = generator.generate(full_validation_count, split="validation")
    selection_products = full_validation_products[:selection_count]
    feature_stats = compute_paper_fcnn_feature_stats(train_products)
    base = config.get("monokan", {})
    train_horizon = min(int(config["training_horizon_T"]), int(runtime.get("monokan_sweep_horizon", 40)))
    reward_epochs = min(int(config["epochs"]), int(runtime.get("monokan_sweep_epochs", 8)))
    batch_size = min(int(config["batch_size"]), int(runtime.get("monokan_sweep_batch_size", 64)), n_products)
    validation_interval = int(config["fcnn"].get("validation_interval", 10))
    seed = int(config["product_generation"]["seed"])
    checkpoint_root = Path(f"outputs/models/monokan_sweep{output_suffix}")
    log_root = Path(f"outputs/results/monokan_sweep_logs{output_suffix}")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    historical_selection = _mean_metrics(
        _policy_metrics_frame(config, selection_products, historical_demand_base_stock_policy)
    )
    candidates: list[dict] = []
    for sweep in _sweep_configs():
        model_config = {
            **config,
            "monokan": {
                **base,
                "hidden_layers": sweep["hidden_layers"],
                "hidden_units": sweep["hidden_units"],
                "grid": sweep["grid"],
            },
        }
        torch.manual_seed(seed)
        model = make_paper_monokan_model(model_config)

        def validation_callback(epoch: int, current_model) -> float:
            del epoch
            frame = _policy_metrics_frame(
                config,
                selection_products,
                lambda product: PaperMonoKANPolicyAdapter(
                    current_model,
                    product,
                    feature_stats=feature_stats,
                    max_order_factor=float(sweep["max_order_factor"]),
                    scale_action_by_history_mean=bool(base.get("scale_action_by_history_mean", True)),
                ),
            )
            return float(frame["mean_total_reward"].mean())

        result = train_paper_lost_sales_fcnn(
            train_products,
            model,
            horizon=train_horizon,
            batch_size=batch_size,
            epochs=reward_epochs,
            learning_rate=float(sweep["learning_rate"]),
            seed=seed,
            imitation_epochs=int(sweep["imitation_epochs"]),
            validation_products=selection_products,
            validation_interval=validation_interval,
            validation_callback=validation_callback,
            feature_stats=feature_stats,
            max_order_factor=float(sweep["max_order_factor"]),
            scale_action_by_history_mean=bool(base.get("scale_action_by_history_mean", True)),
            clip_imitation_predictions=False,
        )

        candidate_dir = checkpoint_root / str(sweep["label"])
        candidate_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), candidate_dir / "final.pt")
        if result.best_imitation_state_dict is not None:
            torch.save(result.best_imitation_state_dict, candidate_dir / "best_imitation.pt")
        if result.best_state_dict is not None:
            torch.save(result.best_state_dict, candidate_dir / "best_validation.pt")
            model.load_state_dict(result.best_state_dict)

        selection_frame = _policy_metrics_frame(
            config,
            selection_products,
            lambda product: PaperMonoKANPolicyAdapter(
                model,
                product,
                feature_stats=feature_stats,
                max_order_factor=float(sweep["max_order_factor"]),
                scale_action_by_history_mean=bool(base.get("scale_action_by_history_mean", True)),
            ),
        )
        selection_metrics = _mean_metrics(selection_frame)
        log_path = log_root / f"{sweep['label']}.csv"
        pd.DataFrame(result.training_log or []).to_csv(log_path, index=False)
        best_validation_epoch = (
            result.validation_epochs[result.validation_rewards.index(result.best_validation_reward)]
            if result.validation_epochs and result.validation_rewards and result.best_validation_reward is not None
            else None
        )
        metadata = {
            **sweep,
            "seed": seed,
            "train_products": n_products,
            "selection_validation_products": selection_count,
            "full_validation_products": full_validation_count,
            "train_horizon": train_horizon,
            "reward_epochs": reward_epochs,
            "batch_size": batch_size,
            "validation_interval": validation_interval,
            "best_imitation_epoch": result.best_imitation_epoch,
            "best_imitation_loss": result.best_imitation_loss,
            "best_validation_epoch": best_validation_epoch,
            "best_validation_reward": result.best_validation_reward,
            "feature_demand_scale": feature_stats.demand_scale,
            "feature_price_scale": feature_stats.price_scale,
        }
        (candidate_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        row = {
            **sweep,
            "train_products": n_products,
            "selection_validation_products": selection_count,
            "train_horizon": train_horizon,
            "reward_epochs": reward_epochs,
            "batch_size": batch_size,
            "best_imitation_epoch": result.best_imitation_epoch,
            "best_imitation_loss": result.best_imitation_loss,
            "best_validation_epoch": best_validation_epoch,
            "best_validation_reward": result.best_validation_reward,
            "final_validation_reward": result.validation_rewards[-1] if result.validation_rewards else None,
            "final_raw_imitation_rmse": float((result.training_log or [{}])[-1].get("raw_imitation_rmse", float("nan"))),
            "final_cap_hit_rate": float((result.training_log or [{}])[-1].get("cap_hit_rate", float("nan"))),
            "selection_historical_mean_total_reward": historical_selection["mean_total_reward"],
            **{f"selection_{key}": value for key, value in selection_metrics.items()},
        }
        candidates.append({"row": row, "model_config": model_config, "checkpoint": candidate_dir / "best_validation.pt"})
        print(
            f"label={sweep['label']} selection_reward={selection_metrics['mean_total_reward']:.4f} "
            f"cap_hit={row['final_cap_hit_rate']:.4f}"
        )

    winner = max(candidates, key=lambda candidate: candidate["row"]["selection_mean_total_reward"])
    winner_model = make_paper_monokan_model(winner["model_config"])
    winner_model.load_state_dict(torch.load(winner["checkpoint"], map_location="cpu"))
    winner_sweep = winner["row"]
    winner_full_frame = _policy_metrics_frame(
        config,
        full_validation_products,
        lambda product: PaperMonoKANPolicyAdapter(
            winner_model,
            product,
            feature_stats=feature_stats,
            max_order_factor=float(winner_sweep["max_order_factor"]),
            scale_action_by_history_mean=bool(base.get("scale_action_by_history_mean", True)),
        ),
    )
    winner_full_frame.insert(1, "policy_name", "MonoKAN")
    historical_full_frame = _policy_metrics_frame(
        config,
        full_validation_products,
        historical_demand_base_stock_policy,
    )
    historical_full_frame.insert(1, "policy_name", "HistoricalDemandBaseStock")
    product_output = Path(
        f"outputs/results/paper_lost_sales_monokan_winner_product_validation{output_suffix}.csv"
    )
    pd.concat([winner_full_frame, historical_full_frame], ignore_index=True).to_csv(product_output, index=False)
    winner_full_metrics = _mean_metrics(winner_full_frame)
    historical_full_metrics = _mean_metrics(historical_full_frame)
    for candidate in candidates:
        candidate["row"]["selected_winner"] = candidate is winner
        if candidate is winner:
            candidate["row"].update({f"full_validation_{key}": value for key, value in winner_full_metrics.items()})
            candidate["row"]["full_validation_historical_mean_total_reward"] = historical_full_metrics[
                "mean_total_reward"
            ]

    results = pd.DataFrame([candidate["row"] for candidate in candidates]).sort_values(
        "selection_mean_total_reward",
        ascending=False,
    )
    output_path = Path(f"outputs/results/paper_lost_sales_monokan_sweep_validation{output_suffix}.csv")
    results.to_csv(output_path, index=False)
    print(results.to_string(index=False))
    print(f"selected_winner={winner_sweep['label']}")
    print(f"saved_sweep={output_path}")
    print(f"saved_product_validation={product_output}")
    print(f"saved_checkpoints={checkpoint_root}")


if __name__ == "__main__":
    main()
