"""Train FCNN for Maggiar-style lost-sales replication."""

from __future__ import annotations

import os
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs/.cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.paper_evaluation import PAPER_VALIDATION_EVALUATION_SEED_START
from inventory_control.paper_lost_sales import evaluate_paper_lost_sales_policy, historical_demand_base_stock_policy
from torch import nn

from inventory_control.policies.neural.fcnn import compute_paper_fcnn_feature_stats, make_paper_fcnn_model
from inventory_control.policies.neural.fcnn import PaperFCNNPolicyAdapter
from inventory_control.products import PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import train_paper_lost_sales_fcnn


def main() -> None:
    """Train and save the paper lost-sales FCNN."""
    config = load_paper_lost_sales_config()
    generator = PaperLostSalesProductGenerator(config)
    runtime = config.get("runtime", {})
    n_products = min(int(config["training_products"]), int(config["runtime"]["fcnn_train_products"]))
    products = generator.generate(n_products, split="train")
    validation_products = generator.generate(
        min(int(runtime.get("fcnn_validation_products", 20)), int(runtime["fcnn_test_products"])),
        split="validation",
    )
    feature_stats = compute_paper_fcnn_feature_stats(products)
    max_order_factor = float(config["fcnn"].get("max_order_factor", 2.0))
    scale_action_by_history_mean = bool(config["fcnn"].get("scale_action_by_history_mean", False))
    torch.manual_seed(int(config["product_generation"]["seed"]))
    model = make_paper_fcnn_model(config)
    baseline_rewards = [
        evaluate_paper_lost_sales_policy(
            product,
            historical_demand_base_stock_policy(product),
            horizon=int(config["evaluation_horizon"]),
            n_episodes=1,
            seed=PAPER_VALIDATION_EVALUATION_SEED_START + idx,
            burn_in=int(config["burn_in"]),
        )["mean_total_reward"]
        for idx, product in enumerate(validation_products)
    ]
    baseline_mean = sum(baseline_rewards) / len(baseline_rewards)

    def validation_callback(epoch: int, current_model: nn.Module) -> float:
        rewards = [
            evaluate_paper_lost_sales_policy(
                product,
                PaperFCNNPolicyAdapter(
                    current_model,
                    product,
                    feature_stats=feature_stats,
                    max_order_factor=max_order_factor,
                    scale_action_by_history_mean=scale_action_by_history_mean,
                ),
                horizon=int(config["evaluation_horizon"]),
                n_episodes=1,
                seed=PAPER_VALIDATION_EVALUATION_SEED_START + idx,
                burn_in=int(config["burn_in"]),
            )["mean_total_reward"]
            for idx, product in enumerate(validation_products)
        ]
        return sum(rewards) / len(rewards)

    result = train_paper_lost_sales_fcnn(
        products,
        model,
        horizon=int(config["training_horizon_T"]),
        batch_size=min(int(config["batch_size"]), n_products),
        epochs=min(int(config["epochs"]), int(runtime.get("fcnn_train_epochs", 50))),
        learning_rate=float(config["learning_rate"]),
        seed=int(config["product_generation"]["seed"]),
        imitation_epochs=int(config["fcnn"].get("imitation_epochs", 0)),
        validation_products=validation_products,
        validation_interval=int(config["fcnn"].get("validation_interval", 0)),
        validation_callback=validation_callback,
        feature_stats=feature_stats,
        max_order_factor=max_order_factor,
        scale_action_by_history_mean=scale_action_by_history_mean,
    )
    if result.best_state_dict is not None:
        model.load_state_dict(result.best_state_dict)
    output_path = Path("outputs/models/paper_lost_sales_fcnn.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    metadata_path = Path("outputs/models/paper_lost_sales_fcnn_metadata.json")
    metadata = {
        "history_length_H": int(config["history_length_H"]),
        "history_encoder": str(config["fcnn"].get("history_encoder", "flat")),
        "hidden_layers": int(config["fcnn"]["hidden_layers"]),
        "hidden_units": int(config["fcnn"]["hidden_units"]),
        "cnn_channels": int(config["fcnn"].get("cnn_channels", 0)),
        "cnn_dilations": list(config["fcnn"].get("cnn_dilations", [])),
        "activation": str(config["fcnn"]["activation"]),
        "feature_demand_scale": feature_stats.demand_scale,
        "feature_price_scale": feature_stats.price_scale,
        "max_order_factor": max_order_factor,
        "scale_action_by_history_mean": scale_action_by_history_mean,
        "best_validation_mean_total_reward": result.best_validation_reward,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    log_rows = result.training_log or []
    for row in log_rows:
        row["validation_historical_mean_total_reward"] = baseline_mean
    log_path = Path("outputs/results/paper_lost_sales_fcnn_training_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    plot_path = Path("outputs/plots/paper_lost_sales_fcnn_training_curve.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    if log_rows:
        log = pd.DataFrame(log_rows)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(log["epoch"], log["training_mean_total_reward"], label="Training reward")
        ax.plot(log["epoch"], log["validation_mean_total_reward"], label="Validation FCNN reward")
        ax.plot(log["epoch"], log["validation_historical_mean_total_reward"], label="Validation historical base-stock")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean total reward")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    print(f"initial_training_loss={result.training_losses[0]:.4f}")
    print(f"final_training_loss={result.training_losses[-1]:.4f}")
    print(f"initial_mean_total_reward={result.mean_total_rewards[0]:.4f}")
    print(f"final_mean_total_reward={result.mean_total_rewards[-1]:.4f}")
    if result.best_validation_reward is not None:
        print(f"best_validation_mean_total_reward={result.best_validation_reward:.4f}")
    if result.validation_rewards:
        for epoch, reward in zip(result.validation_epochs or [], result.validation_rewards):
            print(
                f"validation_epoch={epoch} "
                f"fcnn_true_reward={reward:.4f} "
                f"historical_true_reward={baseline_mean:.4f}"
            )
    print(f"saved_model={output_path}")
    print(f"saved_model_metadata={metadata_path}")
    print(f"saved_training_log={log_path}")
    print(f"saved_training_curve={plot_path}")


if __name__ == "__main__":
    main()
