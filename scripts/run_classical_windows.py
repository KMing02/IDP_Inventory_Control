"""Evaluate the classical percentile and Gamma-fit base-stock rules in all three information windows.

Usage:
    python scripts/run_classical_windows.py --out outputs/classical_windows.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inventory_control.classical_windows import CONFIG, evaluate_classical_windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out", type=Path, default=Path("outputs/classical_windows.json"))
    args = parser.parse_args()
    results = evaluate_classical_windows(args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    for name, row in results.items():
        print(f"{name:22s} reward {row['reward']:9.2f}   regret {row['regret_pct']:5.2f}%")


if __name__ == "__main__":
    main()
