"""Run or benchmark the frozen rolling/static direct-target comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inventory_control.rolling_direct_target_matched import (
    run_full_shape_benchmark,
    run_matched_seed,
    run_small_development_benchmark,
    run_small_matched_development,
    validate_frozen_protocol,
    validate_small_development_protocol,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen 5k/1000-epoch matched rolling/static direct-target experiment."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Validate the three matched scientific configs.")
    audit.set_defaults(handler=_audit)

    benchmark = subparsers.add_parser("benchmark", help="Run 1-5 full-shape timing epochs only.")
    benchmark.add_argument("--output-dir", type=Path, required=True)
    benchmark.add_argument("--epochs", type=int, choices=range(1, 6), default=3)
    benchmark.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    benchmark.add_argument("--checkpoint", choices=["on", "off"], required=True)
    benchmark.add_argument("--seed", type=int, default=2025)
    benchmark.set_defaults(handler=_benchmark)

    run_seed = subparsers.add_parser("run-seed", help="Run exactly one formal matched training seed.")
    run_seed.add_argument("--seed", type=int, choices=[2025, 2026, 2027], required=True)
    run_seed.add_argument("--output-dir", type=Path, required=True)
    run_seed.add_argument("--raw-result-dir", type=Path)
    run_seed.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    run_seed.add_argument("--checkpoint", choices=["on", "off"], required=True)
    run_seed.set_defaults(handler=_run_seed)

    dev_audit = subparsers.add_parser(
        "audit-development",
        help="Validate the fixed 1k/200-epoch matched development protocol.",
    )
    dev_audit.set_defaults(handler=_audit_development)

    dev_benchmark = subparsers.add_parser(
        "benchmark-development",
        help="Run one full-shape epoch of the rolling DeepGrid7 MonoKAN development profile.",
    )
    dev_benchmark.add_argument("--output-dir", type=Path, required=True)
    dev_benchmark.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    dev_benchmark.add_argument("--seed", type=int, default=2025)
    dev_benchmark.set_defaults(handler=_benchmark_development)

    dev_run = subparsers.add_parser(
        "run-development",
        help="Run the fixed one-seed rolling/static direct-target development comparison.",
    )
    dev_run.add_argument("--output-dir", type=Path, required=True)
    dev_run.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    dev_run.add_argument("--seed", type=int, choices=[2025], default=2025)
    dev_run.set_defaults(handler=_run_development)

    args = parser.parse_args()
    args.handler(args)


def _audit(_args: argparse.Namespace) -> None:
    print(json.dumps(validate_frozen_protocol(), indent=2, sort_keys=True))


def _benchmark(args: argparse.Namespace) -> None:
    report = run_full_shape_benchmark(
        output_dir=args.output_dir,
        epochs=args.epochs,
        device=args.device,
        activation_checkpointing=args.checkpoint == "on",
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _run_seed(args: argparse.Namespace) -> None:
    paths = run_matched_seed(
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device,
        activation_checkpointing=args.checkpoint == "on",
        raw_result_dir=args.raw_result_dir,
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2, sort_keys=True))


def _audit_development(_args: argparse.Namespace) -> None:
    print(json.dumps(validate_small_development_protocol(), indent=2, sort_keys=True))


def _benchmark_development(args: argparse.Namespace) -> None:
    report = run_small_development_benchmark(
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _run_development(args: argparse.Namespace) -> None:
    paths = run_small_matched_development(
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
