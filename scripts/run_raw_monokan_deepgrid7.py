"""Run the paper-aligned raw-action DeepGrid7 MonoKAN experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inventory_control.raw_monokan_experiment import (
    run_full_shape_benchmark,
    run_raw_monokan_seed,
    validate_raw_protocol,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper-aligned 5k raw-action DeepGrid7 MonoKAN experiment."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Validate protocol and architecture metadata.")
    audit.set_defaults(handler=_audit)

    benchmark = commands.add_parser("benchmark", help="Run one full-shape training epoch only.")
    benchmark.add_argument("--output-dir", type=Path, required=True)
    benchmark.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    benchmark.add_argument("--checkpoint", choices=["on", "off"], default="on")
    benchmark.add_argument("--seed", type=int, default=2025)
    benchmark.set_defaults(handler=_benchmark)

    run_seed = commands.add_parser("run-seed", help="Run exactly one 1000-epoch formal seed.")
    run_seed.add_argument("--seed", type=int, choices=[2025, 2026, 2027], required=True)
    run_seed.add_argument("--output-dir", type=Path, required=True)
    run_seed.add_argument("--raw-fcnn-result-dir", type=Path)
    run_seed.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    run_seed.add_argument("--checkpoint", choices=["on", "off"], default="on")
    run_seed.set_defaults(handler=_run_seed)
    args = parser.parse_args()
    args.handler(args)


def _audit(_args: argparse.Namespace) -> None:
    print(json.dumps(validate_raw_protocol(), indent=2, sort_keys=True))


def _benchmark(args: argparse.Namespace) -> None:
    report = run_full_shape_benchmark(
        output_dir=args.output_dir,
        device=args.device,
        checkpoint_raw_action=args.checkpoint == "on",
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _run_seed(args: argparse.Namespace) -> None:
    paths = run_raw_monokan_seed(
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device,
        checkpoint_raw_action=args.checkpoint == "on",
        raw_fcnn_result_dir=args.raw_fcnn_result_dir,
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
