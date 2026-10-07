"""Regret of saved target-stock runs, and paired comparisons between them.

A run root holds ``seed_*/product_metrics.csv`` as written by ``run_soft_quantile_monokan.py``. The policy name is
the model key, so it stays ``MonoKANSoftQuantileTargetStock`` when ``--set monokan.level_network=kan`` swaps the
level network.

Usage:
    python scripts/compare_target_stock_runs.py summary outputs/ts_monokan outputs/ts_kan
    python scripts/compare_target_stock_runs.py pair outputs/ts_monokan outputs/ts_kan
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inventory_control.analysis import compare_runs, summarize_run

POLICY = "MonoKANSoftQuantileTargetStock"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("summary", help="reward and regret of each run root")
    summary.add_argument("roots", nargs="+", type=Path)
    pair = commands.add_parser("pair", help="paired FIRST - SECOND reward difference over shared products")
    pair.add_argument("first", type=Path)
    pair.add_argument("second", type=Path)
    for command in (summary, pair):
        command.add_argument("--policy", default=POLICY)
    args = parser.parse_args()

    if args.command == "summary":
        for root in args.roots:
            result = summarize_run(root, args.policy)
            by_seed = ", ".join(f"{row['regret_pct']:.2f}" for row in result["per_seed"].values())
            print(f"{root}: regret {result['regret_pct']:.2f}% (by seed {by_seed}), reward {result['reward']:.2f}")
    else:
        result = compare_runs(args.first, args.second, args.policy)
        low, high = result["ci95"]
        print(f"{args.first} - {args.second}: {result['mean']:+.2f} per period, 95% CI [{low:+.2f}, {high:+.2f}], "
              f"paired t p = {result['p_paired_t']:.4f}, Wilcoxon p = {result['p_wilcoxon']:.4f}, "
              f"first ahead on {result['first_ahead_pct']:.0f}% of {result['products']} products")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
