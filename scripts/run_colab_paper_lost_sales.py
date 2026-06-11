"""Run Colab-scale paper lost-sales validation experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def _run(command: list[str], env: dict[str, str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    """Run validation-first FCNN/MonoKAN experiments with a chosen config."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_lost_sales_colab.yaml")
    parser.add_argument("--models", choices=["fcnn", "monokan", "both"], default="both")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-sweeps", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PAPER_LOST_SALES_CONFIG"] = str((repo_root / args.config).resolve())
    python = sys.executable

    if not args.skip_baselines:
        _run([python, "scripts/run_paper_lost_sales_baselines.py"], env)

    if args.models in {"fcnn", "both"}:
        _run([python, "scripts/train_paper_lost_sales_fcnn.py"], env)
        _run([python, "scripts/evaluate_paper_lost_sales_fcnn.py"], env)
        if not args.skip_sweeps:
            _run([python, "scripts/sweep_paper_lost_sales_fcnn.py"], env)

    if args.models in {"monokan", "both"}:
        if not args.skip_sweeps:
            _run([python, "scripts/sweep_paper_lost_sales_monokan.py"], env)
        _run([python, "scripts/train_paper_lost_sales_monokan.py"], env)
        _run([python, "scripts/evaluate_paper_lost_sales_monokan.py"], env)

    print("\nValidation run complete. Do not run the test split until final model selection is frozen.", flush=True)


if __name__ == "__main__":
    main()
