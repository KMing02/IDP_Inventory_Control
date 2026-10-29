"""Train and evaluate the soft-quantile target-stock policy (the alternative design) for one seed.

``--set`` changes single config entries, which is how the report's controls are run from the final config, e.g.
``--set monokan.level_network=kan`` (the same network without the certificate), ``--set monokan.window_mode=sliding``
(only the last 32 days) or ``--set monokan.roll_history=false`` (one fixed estimate from the first 32 days).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from inventory_control.base_stock_output_experiment import run_base_stock_output_experiment
from inventory_control.paper_configs import apply_config_overrides, load_paper_lost_sales_config


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
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one config entry, e.g. monokan.level_network=kan. Repeatable.",
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
    config_path = args.config
    if args.overrides:
        # Record the resolved config next to the results, so every run says exactly what it used.
        config = apply_config_overrides(load_paper_lost_sales_config(args.config), args.overrides)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        config_path = args.output_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    artifacts = run_base_stock_output_experiment(
        config_path=config_path,
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
