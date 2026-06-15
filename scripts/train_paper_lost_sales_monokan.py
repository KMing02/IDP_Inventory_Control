"""Train adapted MonoKAN for Maggiar-style lost-sales replication."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from torch import nn

from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.paper_evaluation import PAPER_VALIDATION_EVALUATION_SEED_START
from inventory_control.paper_lost_sales import evaluate_paper_lost_sales_policy, historical_demand_base_stock_policy
from inventory_control.policies.neural.fcnn import compute_paper_fcnn_feature_stats
from inventory_control.policies.neural.monokan import PaperMonoKANPolicyAdapter, make_paper_monokan_model
from inventory_control.products import PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import default_torch_device, train_paper_lost_sales_fcnn


def main() -> None:
    """Train and save the paper lost-sales MonoKAN."""
    config = load_paper_lost_sales_config()
    generator = PaperLostSalesProductGenerator(config)
    runtime = config.get("runtime", {})
    monokan_config = config.get("monokan", {})
    n_products = min(
        int(config["training_products"]),
        int(runtime.get("monokan_train_products", runtime.get("fcnn_train_products", config["training_products"]))),
    )
    products = generator.generate(n_products, split="train")
    validation_products = generator.generate(
        min(int(runtime.get("fcnn_validation_products", 20)), int(runtime.get("fcnn_test_products", 20))),
        split="validation",
    )
    feature_stats = compute_paper_fcnn_feature_stats(products)
    max_order_factor = float(monokan_config.get("max_order_factor", config["fcnn"].get("max_order_factor", 2.0)))
    scale_action_by_history_mean = bool(
        monokan_config.get("scale_action_by_history_mean", config["fcnn"].get("scale_action_by_history_mean", False))
    )
    device = default_torch_device()
    torch.manual_seed(int(config["product_generation"]["seed"]))
    model = make_paper_monokan_model(config).to(device)

    baseline_rewards = [
        evaluate_paper_lost_sales_policy(
            product,
            historical_demand_base_stock_policy(product),
            horizon=int(config["evaluation_horizon"]),
            n_episodes=1,
            seed=PAPER_VALIDATION_EVALUATION_SEED_START + index,
            burn_in=int(config["burn_in"]),
        )["mean_total_reward"]
        for index, product in enumerate(validation_products)
    ]
    baseline_mean = sum(baseline_rewards) / len(baseline_rewards)

    def validation_callback(epoch: int, current_model: nn.Module) -> float:
        rewards = [
            evaluate_paper_lost_sales_policy(
                product,
                PaperMonoKANPolicyAdapter(
                    current_model,
                    product,
                    feature_stats=feature_stats,
                    max_order_factor=max_order_factor,
                    scale_action_by_history_mean=scale_action_by_history_mean,
                ),
                horizon=int(config["evaluation_horizon"]),
                n_episodes=1,
                seed=PAPER_VALIDATION_EVALUATION_SEED_START + index,
                burn_in=int(config["burn_in"]),
            )["mean_total_reward"]
            for index, product in enumerate(validation_products)
        ]
        return sum(rewards) / len(rewards)

    result = train_paper_lost_sales_fcnn(
        products,
        model,
        horizon=int(config["training_horizon_T"]),
        batch_size=min(int(config["batch_size"]), n_products),
        epochs=min(int(config["epochs"]), int(runtime.get("monokan_train_epochs", runtime.get("fcnn_train_epochs", 50)))),
        learning_rate=float(monokan_config.get("learning_rate", config["learning_rate"])),
        seed=int(config["product_generation"]["seed"]),
        imitation_epochs=int(monokan_config.get("imitation_epochs", config["fcnn"].get("imitation_epochs", 0))),
        validation_products=validation_products,
        validation_interval=int(monokan_config.get("validation_interval", config["fcnn"].get("validation_interval", 0))),
        validation_callback=validation_callback,
        feature_stats=feature_stats,
        max_order_factor=max_order_factor,
        scale_action_by_history_mean=scale_action_by_history_mean,
        device=device,
    )
    if result.best_state_dict is not None:
        model.load_state_dict(result.best_state_dict)

    output_path = Path("outputs/models/paper_lost_sales_monokan.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    metadata_path = Path("outputs/models/paper_lost_sales_monokan_metadata.json")
    metadata = {
        "history_length_H": int(config["history_length_H"]),
        "hidden_layers": int(monokan_config.get("hidden_layers", 1)),
        "hidden_units": int(monokan_config.get("hidden_units", 16)),
        "grid": int(monokan_config.get("grid", 5)),
        "grid_range": list(monokan_config.get("grid_range", [-2.0, 4.0])),
        "monotonic_inventory": "nonincreasing",
        "feature_demand_scale": feature_stats.demand_scale,
        "feature_price_scale": feature_stats.price_scale,
        "max_order_factor": max_order_factor,
        "scale_action_by_history_mean": scale_action_by_history_mean,
        "learning_rate": float(monokan_config.get("learning_rate", config["learning_rate"])),
        "imitation_epochs": int(monokan_config.get("imitation_epochs", config["fcnn"].get("imitation_epochs", 0))),
        "best_validation_mean_total_reward": result.best_validation_reward,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    log_rows = result.training_log or []
    for row in log_rows:
        row["validation_historical_mean_total_reward"] = baseline_mean
    log_path = Path("outputs/results/paper_lost_sales_monokan_training_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(log_path, index=False)

    print(f"initial_training_loss={result.training_losses[0]:.4f}")
    print(f"final_training_loss={result.training_losses[-1]:.4f}")
    print(f"device={device}")
    if result.best_validation_reward is not None:
        print(f"best_validation_mean_total_reward={result.best_validation_reward:.4f}")
    print(f"validation_historical_mean_total_reward={baseline_mean:.4f}")
    print(f"saved_model={output_path}")
    print(f"saved_model_metadata={metadata_path}")
    print(f"saved_training_log={log_path}")


if __name__ == "__main__":
    main()
