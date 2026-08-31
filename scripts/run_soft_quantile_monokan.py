"""Train and evaluate the final soft-quantile MonoKAN target-stock policy."""

from __future__ import annotations

import argparse
from pathlib import Path

from inventory_control.base_stock_output_experiment import run_base_stock_output_experiment


DEFAULT_CONFIG = Path("configs/paper_lost_sales_soft_quantile_monokan.yaml")
REPORT_COLUMNS = (
    "policy_name",
    "mean_reward_per_period",
    "service_level",
    "parameter_count",
    "training_seconds",
    "selected_checkpoint_epoch",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["MonoKANSoftQuantileTargetStock"],
        help="Policies to train. Add PaperFCNNDirectTargetStock for the shipped FCNN anchor.",
    )
    parser.add_argument(
        "--no-activation-checkpointing",
        action="store_true",
        help="Trade memory for speed when the rollout graph fits without checkpointing.",
    )
    return parser.parse_args()


def main() -> None:
    """Run one seed of the final soft-quantile protocol."""
    args = parse_args()
    artifacts = run_base_stock_output_experiment(
        config_path=args.config,
        output_dir=args.output_dir,
        models=tuple(args.models),
        device=args.device,
        seed=args.seed,
        activation_checkpointing=not args.no_activation_checkpointing,
        shared_training_seed=True,
    )
    columns = [name for name in REPORT_COLUMNS if name in artifacts.summary.columns]
    print(artifacts.summary[columns].to_string(index=False))
    print(f"\nartifacts written to {artifacts.output_dir}")


if __name__ == "__main__":
    main()
