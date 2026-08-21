"""Run the supervised empirical-newsvendor target identification diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inventory_control.supervised_target_diagnostic import (
    SUPERVISED_TARGET_MODEL_NAMES,
    run_supervised_target_diagnostic,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic only: supervise direct target-stock networks on the observable empirical "
            "newsvendor target. This does not pretrain or evaluate a deployed policy."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/paper_lost_sales_base_stock_small.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/supervised_target_diagnostic/medium"))
    parser.add_argument("--models", nargs="+", choices=SUPERVISED_TARGET_MODEL_NAMES, default=list(SUPERVISED_TARGET_MODEL_NAMES))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--training-products", type=int)
    parser.add_argument("--validation-products", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()

    artifacts = run_supervised_target_diagnostic(
        config_path=args.config,
        output_dir=args.output_dir,
        models=tuple(args.models),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_interval=args.validation_interval,
        training_products=args.training_products,
        validation_products=args.validation_products,
        seed=args.seed,
        device=args.device,
    )
    validation = artifacts.summary[artifacts.summary["split"] == "validation"]
    print(validation.to_string(index=False))
    print(f"saved_artifacts={artifacts.output_dir}")


if __name__ == "__main__":
    main()
