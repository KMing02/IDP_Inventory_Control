"""Run complete-target and MonoKAN spline diagnostics from saved checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inventory_control.direct_target_structural_audit import run_direct_target_structural_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--products-per-domain", type=int, default=100)
    args = parser.parse_args()
    artifacts = run_direct_target_structural_audit(
        args.config,
        args.checkpoint_root,
        args.output_dir,
        products_per_domain=args.products_per_domain,
    )
    print(artifacts.target_monotonicity.to_string(index=False))
    print(f"saved_artifacts={artifacts.output_dir}")


if __name__ == "__main__":
    main()
