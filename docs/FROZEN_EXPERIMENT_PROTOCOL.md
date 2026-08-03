# Proposed Frozen Experiment Protocol

Status: proposed for review, not automatically frozen.

## Models

- `PaperFCNNResidualOrderUpTo`
- `KANResidualOrderUpTo`
- `ConstrainedMLPResidualOrderUpTo`
- `MonoKANResidualOrderUpTo`

## Configuration

- Development/compact evidence: `configs/paper_lost_sales_factorial_small.yaml`.
- Medium/paper-alignment evidence: `configs/paper_lost_sales_fcnn_alignment_medium.yaml`.
- Final held-out evaluation should use the agreed paper-scale profile once choices are frozen.

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

Proposed development margin: +/- 5 reward per period, reported also as a percentage of mean KAN reward. This must be reviewed before any formal equivalence claim.

## Protected Final Evaluation Rule

Freeze model classes, configs, seeds, equivalence margin, metrics, and analysis script before running a final held-out/paper-scale evaluation. Do not tune after viewing protected final results.
