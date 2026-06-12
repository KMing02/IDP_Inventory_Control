"""Run the unified IDP monotonic inventory-control benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inventory_control.idp_benchmark import run_idp_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["compact", "full"], default="compact")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--final-test", action="store_true", help="Required for held-out test evaluation.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    artifacts = run_idp_benchmark(
        config_path=args.config,
        profile=args.profile,
        split=args.split,
        final_test=args.final_test,
        output_dir=args.output_dir,
    )
    print(artifacts.summary.to_string(index=False))
    print(f"saved_artifacts={artifacts.output_dir}")


if __name__ == "__main__":
    main()
