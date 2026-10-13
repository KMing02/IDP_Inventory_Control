"""Direct-quantity comparison (main design): MonoKAN, the same KAN without the certificate, and the paper FCNN.

Usage:
    python scripts/run_direct_quantity_comparison.py train --models monokan,kan,fcnn \
        --seeds 2025,2026,2027 --epochs 2400 --out outputs/direct_quantity
    python scripts/run_direct_quantity_comparison.py summarize outputs/direct_quantity
    python scripts/run_direct_quantity_comparison.py interpret outputs/direct_quantity
    python scripts/run_direct_quantity_comparison.py curves outputs/direct_quantity

Every (model, seed) run writes its own result file and checkpoint, so ``train`` can run in parallel processes,
for example one per model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from inventory_control.direct_quantity_experiment import (
    MODELS,
    REPORTED_MODELS,
    implied_order_up_to,
    largest_violation_curves,
    summarize_runs,
    train_and_evaluate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train and evaluate (model, seed) runs")
    train.add_argument("--models", default=",".join(REPORTED_MODELS),
                       help=f"comma-separated subset of {','.join(MODELS)}; kan_open is the KAN initialised like MonoKAN")
    train.add_argument("--seeds", default="2025,2026,2027")
    train.add_argument("--epochs", type=int, default=2400)
    train.add_argument("--threads", type=int, default=2)
    train.add_argument("--out", type=Path, default=Path("outputs/direct_quantity"))
    for name, text in (
        ("summarize", "pool the run files into summary.json"),
        ("interpret", "implied order-up-to level and decision speed (writes interpretability.json)"),
        ("curves", "order-versus-stock curves where the KAN breaks the stock rule most (writes curves.json)"),
    ):
        command = commands.add_parser(name, help=text)
        command.add_argument("run_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        torch.set_num_threads(args.threads)
        for name in args.models.split(","):
            for seed in (int(value) for value in args.seeds.split(",")):
                row = train_and_evaluate(name, seed, args.epochs, args.out)
                print(json.dumps({key: value for key, value in row.items() if key != "per_product_reward"}), flush=True)
    elif args.command == "summarize":
        print(json.dumps(summarize_runs(args.run_dir), indent=1))
    elif args.command == "interpret":
        torch.set_num_threads(4)
        results = implied_order_up_to(args.run_dir)
        (args.run_dir / "interpretability.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        for name, rows in results.items():
            def mean(key: str) -> float:
                return float(np.mean([row[key] for row in rows]))

            print(f"{name:8s} params {rows[0]['params']:5d} | implied vs optimal level: corr {mean('raw_corr'):.3f}, "
                  f"per unit of demand {mean('scaled_corr'):.3f} | {mean('ms_one'):.3f} ms per decision, "
                  f"{mean('ms_10k'):.1f} ms per 10,000")
    else:
        curves = largest_violation_curves(args.run_dir)
        (args.run_dir / "curves.json").write_text(json.dumps(curves), encoding="utf-8")
        print(f"product {curves['product_index']}, seed {curves['seed']}: largest KAN rise "
              f"{curves['largest_rise_share'] * 100:.2f}% of a day's demand")


if __name__ == "__main__":
    main()
