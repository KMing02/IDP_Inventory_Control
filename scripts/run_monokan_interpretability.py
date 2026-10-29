"""Produce partial-dependence interpretability evidence from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from inventory_control.monokan_interpretability import run_monokan_interpretability


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default="MonoKANSoftQuantileTargetStock")
    parser.add_argument("--no-figure", action="store_true", help="Write tables only.")
    return parser.parse_args()


def main() -> None:
    """Write partial-dependence tables, a report, and (optionally) a figure."""
    args = parse_args()
    artifacts = run_monokan_interpretability(
        config_path=args.config,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        model_name=args.model,
        make_figure=not args.no_figure,
    )
    print((artifacts.output_dir / "INTERPRETABILITY.md").read_text(encoding="utf-8"))
    if artifacts.figure_path is not None:
        print(f"figure: {artifacts.figure_path}")


if __name__ == "__main__":
    main()
