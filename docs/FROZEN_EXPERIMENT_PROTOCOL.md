# Frozen Experiment Protocol

Status: superseded historical draft. The current experiment inventory and development evidence are indexed in `docs/EXPERIMENTS.md`. This file records the earlier residual order-up-to final-run proposal and should not be treated as the active final protocol unless the team explicitly returns to the residual framing.

Historical status: frozen for the earlier residual final evaluation run. Do not tune model classes,
training schedules, seeds, shift definitions, equivalence margin, or analysis
logic after viewing final outputs.

## Historical Residual Models

- `PaperFCNNResidualOrderUpTo`
- `KANResidualOrderUpTo`
- `ConstrainedMLPResidualOrderUpTo`
- `MonoKANResidualOrderUpTo`

## Configuration

- Development/compact evidence: `configs/paper_lost_sales_factorial_small.yaml`.
- Medium/paper-alignment evidence: `configs/paper_lost_sales_fcnn_alignment_medium.yaml`.
- Final evaluation configuration: `configs/paper_lost_sales_fcnn_alignment_medium.yaml`
  with the `full` profile unless the team explicitly approves a larger
  paper-replication profile before running.

## Action Decoder And Anchor

All primary models use residual order-up-to:

```text
S_theta(z) = S_hist(z) + alpha * r_theta(z)
q_theta(y,z) = max(S_theta(z) - y, 0)
```

Current inventory is excluded from the target-stock residual network. The residual anchor is the configured Gamma historical/non-omniscient base-stock anchor.

## Seeds

- Training/model seeds: `2025` through `2034`.
- Product-generation seed may vary with training seed when `--vary-product-seed` is enabled.
- Evaluation demand seeds are the deterministic per-product seeds recorded in `operational_metrics.csv`.

## Shift Definitions

Use the eight existing shifts:

- `in_distribution`
- `low_cv`
- `high_cv`
- `low_mean_demand`
- `high_mean_demand`
- `high_lost_sale_penalty`
- `high_holding_cost`
- `mixed_stress`

## Checkpoint Selection

Use the existing validation-based checkpoint selection in `run_idp_benchmark.py`. Do not select models on the protected final test split.

## Primary Metrics

- Mean reward per period.
- Product-level paired difference `MonoKAN - KAN`.
- Hierarchical paired bootstrap confidence interval with fixed shifts and resampled seeds/products.
- Practical equivalence with predeclared margin.

## Secondary Metrics

- Regret versus non-omniscient and omniscient baselines.
- Worst-decile paired reward delta.
- Win/loss/tie rates.
- Target-stock monotonicity violation rates.
- Spline interval and extrapolation certification checks.
- Cap-hit rate and raw/deployed action difference.
- Residual correction and target-stock distribution.

## Equivalence Margin

Frozen margin for the next final evaluation: +/- 5 reward per period, reported
also as a percentage of mean KAN reward.

## Protected Final Evaluation Rule

The protected final evaluation may be run once after this protocol is accepted.
After viewing protected final results, do not tune architectures, residual scale,
capacity guards, training schedules, seeds, shift definitions, model-selection
rules, or analysis logic. If a concrete correctness bug is discovered, document
the bug and rerun only after recording the invalidated run.

## Historical Residual Final Run Commands

Primary final four-model benchmark:

```bash
uv run python scripts/run_idp_benchmark.py \
  --mode structure-ls \
  --profile full \
  --config configs/paper_lost_sales_fcnn_alignment_medium.yaml \
  --models PaperFCNNResidualOrderUpTo,KANResidualOrderUpTo,ConstrainedMLPResidualOrderUpTo,MonoKANResidualOrderUpTo \
  --output-dir outputs/monokan_final_four_model \
  --progress
```

Final paired distribution-shift evaluation:

```bash
uv run python scripts/run_distribution_shift_multiseed.py \
  --config configs/paper_lost_sales_fcnn_alignment_medium.yaml \
  --output-root outputs/monokan_final_distribution_shift_10seed \
  --seeds 2025,2026,2027,2028,2029,2030,2031,2032,2033,2034 \
  --profile full \
  --models PaperFCNNResidualOrderUpTo,KANResidualOrderUpTo,ConstrainedMLPResidualOrderUpTo,MonoKANResidualOrderUpTo \
  --eval-products 500 \
  --diagnostic-products 200 \
  --inventory-states 41 \
  --vary-product-seed \
  --progress
```

Final evidence analysis:

```bash
uv run python scripts/analyze_monokan_evidence.py \
  --benchmark-dir outputs/monokan_final_four_model \
  --multiseed-root outputs/monokan_final_distribution_shift_10seed \
  --output-dir outputs/analysis_final
```
