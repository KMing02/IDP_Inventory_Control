"""One-product MonoKAN calibration and overfit diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from inventory_control.paper_configs import load_paper_lost_sales_config
from inventory_control.paper_lost_sales import (
    evaluate_paper_lost_sales_policy,
    historical_demand_base_stock_policy,
)
from inventory_control.policies.neural.fcnn import compute_paper_fcnn_feature_stats
from inventory_control.policies.neural.monokan import PaperMonoKANPolicyAdapter, make_paper_monokan_model
from inventory_control.products import PaperLostSalesProductGenerator
from inventory_control.training.direct_backprop import train_paper_lost_sales_fcnn


def main() -> None:
    """Train MonoKAN on one product and compare it with its imitation teacher."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--imitation-epochs", type=int, default=50)
    parser.add_argument("--reward-epochs", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=100)
    args = parser.parse_args()

    config = load_paper_lost_sales_config()
    product = PaperLostSalesProductGenerator(config).generate(1, split="train")[0]
    feature_stats = compute_paper_fcnn_feature_stats([product])
    monokan_config = config["monokan"]
    max_order_factor = float(monokan_config.get("max_order_factor", 4.0))
    scale_action = bool(monokan_config.get("scale_action_by_history_mean", True))
    seed = int(config["product_generation"]["seed"])
    torch.manual_seed(seed)
    model = make_paper_monokan_model(config)

    result = train_paper_lost_sales_fcnn(
        [product],
        model,
        horizon=args.horizon,
        batch_size=1,
        epochs=args.reward_epochs,
        learning_rate=float(monokan_config.get("learning_rate", config["learning_rate"])),
        seed=seed,
        imitation_epochs=args.imitation_epochs,
        feature_stats=feature_stats,
        max_order_factor=max_order_factor,
        scale_action_by_history_mean=scale_action,
        normalize_reward_loss=True,
        clip_imitation_predictions=False,
    )
    neural_metrics = evaluate_paper_lost_sales_policy(
        product,
        PaperMonoKANPolicyAdapter(
            model,
            product,
            feature_stats=feature_stats,
            max_order_factor=max_order_factor,
            scale_action_by_history_mean=scale_action,
        ),
        horizon=int(config["evaluation_horizon"]),
        n_episodes=1,
        seed=55_000,
        burn_in=int(config["burn_in"]),
    )
    historical_metrics = evaluate_paper_lost_sales_policy(
        product,
        historical_demand_base_stock_policy(product),
        horizon=int(config["evaluation_horizon"]),
        n_episodes=1,
        seed=55_000,
        burn_in=int(config["burn_in"]),
    )

    log = pd.DataFrame(result.training_log or [])
    output = Path("outputs/results/paper_lost_sales_monokan_one_product_overfit.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(output, index=False)
    final = log.iloc[-1]
    print(f"initial_raw_imitation_rmse={log.iloc[0]['raw_imitation_rmse']:.4f}")
    print(f"final_raw_imitation_rmse={final['raw_imitation_rmse']:.4f}")
    print(f"final_cap_hit_rate={final['cap_hit_rate']:.4f}")
    print(f"monokan_mean_total_reward={neural_metrics['mean_total_reward']:.4f}")
    print(f"historical_mean_total_reward={historical_metrics['mean_total_reward']:.4f}")
    print(f"saved_log={output}")


if __name__ == "__main__":
    main()
