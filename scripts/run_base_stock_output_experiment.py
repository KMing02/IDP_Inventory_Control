"""Run the direct base-stock-output lost-sales experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inventory_control.base_stock_output_experiment import run_base_stock_output_experiment
from inventory_control.policies.neural.registry import (
    CAPACITY_DIRECT_TARGET_STOCK_POLICY_NAMES,
    DIRECT_TARGET_STOCK_POLICY_NAMES,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train policies that output base-stock levels S_theta(z), "
            "then deploy q=max(S_theta-inventory,0). This path uses no imitation pretraining."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/paper_lost_sales_debug.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/base_stock_output_smoke"))
    supported_models = DIRECT_TARGET_STOCK_POLICY_NAMES + CAPACITY_DIRECT_TARGET_STOCK_POLICY_NAMES
    parser.add_argument("--models", nargs="+", choices=supported_models, default=list(DIRECT_TARGET_STOCK_POLICY_NAMES))
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], help="Override the torch training device.")
    parser.add_argument("--seed", type=int, help="Override the training and evaluation run seed.")
    parser.add_argument(
        "--activation-checkpointing",
        choices=["on", "off"],
        default="on",
        help="Engineering-only memory tradeoff for rolling target evaluation.",
    )
    args = parser.parse_args()

    artifacts = run_base_stock_output_experiment(
        config_path=args.config,
        output_dir=args.output_dir,
        models=tuple(args.models),
        device=args.device,
        seed=args.seed,
        activation_checkpointing=args.activation_checkpointing == "on",
    )
    print(artifacts.summary.to_string(index=False))
    print(f"saved_artifacts={artifacts.output_dir}")


if __name__ == "__main__":
    main()
