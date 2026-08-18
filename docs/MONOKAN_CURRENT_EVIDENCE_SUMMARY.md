# MonoKAN Current Evidence Summary

This note summarizes the current post-supervisor-feedback evidence for the
direct base-stock-output MonoKAN experiments. It is intended as a working
research summary, not a final paper-scale result.

## Current Framing

The current main policy follows the supervisor's base-stock-output suggestion:

```text
NN(z) -> S_theta
q_theta(y, z) = max(S_theta(z) - y, 0)
```

The neural network predicts a target stock/base-stock level from observable
product context `z`. The order quantity is then decoded by the order-up-to rule.
This is different from the earlier raw-action framing:

```text
NN(y, z) -> q
```

The raw-action framing caused KAN/MonoKAN to predict overly large orders, rely
heavily on the deployment cap, and accumulate excessive inventory. The direct
base-stock-output framing fixes the action-coordinate mismatch: current
inventory is handled by the shared order-up-to decoder rather than learned by
the network.

## Current Models

Primary direct base-stock-output policies:

- `PaperFCNNBaseStockOrderUpTo`
- `KANBaseStockOrderUpTo`
- `MonoKANBaseStockOrderUpTo`

Classical reference:

- `HistoricalDemandBaseStock`

Current KAN/MonoKAN output stabilization:

- final output parameterized inside `[0, max_order_factor]`;
- empirical newsvendor/base-stock output anchor;
- logit residual scale for KAN/MonoKAN;
- current best tested scale: `base_stock_logit_residual_scale = 0.03`.

## Feature Framing

Two history representations have been tested.

`raw_lags`:

```text
[d_1, d_2, ..., d_32, price, purchase_cost, holding_cost, lost_sales_penalty]
```

`summary`:

```text
[
  historical_mean,
  historical_std,
  historical_cv,
  recent_mean_8,
  recent_mean_16,
  trend_last8_minus_first8,
  p50,
  p75,
  p90,
  price,
  purchase_cost,
  holding_cost,
  lost_sales_penalty
]
```

The summary representation uses only observable demand history. It does not use
hidden Gamma parameters. The motivation is that FCNN has a causal CNN history
encoder, while the small KAN/MonoKAN models otherwise receive 32 noisy lag
features directly.

For MonoKAN with summary features, monotonic constraints are applied only to
summary variables with clear demand-level interpretation:

- increasing: `historical_mean`, `recent_mean_8`, `recent_mean_16`, `p50`,
  `p75`, `p90`, `price`, `lost_sales_penalty`;
- decreasing: `purchase_cost`, `holding_cost`;
- unconstrained: `historical_std`, `historical_cv`, `trend_last8_minus_first8`.

This avoids overclaiming monotonic directions for volatility or trend features.

## Action Scaling Evidence

Fine scale ablation on 3 seeds, raw-history direct base-stock-output:

| scale | KAN reward | KAN inventory | KAN cap-near | MonoKAN reward | MonoKAN inventory | MonoKAN cap-near |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 4496.97 | 132.24 | 1.67% | 4539.59 | 104.43 | 0.00% |
| 0.05 | 4459.88 | 147.99 | 2.92% | 4505.79 | 118.79 | 0.00% |
| 0.10 | 4368.69 | 174.94 | 5.83% | 4421.76 | 144.86 | 0.00% |

Interpretation: smaller logit residual scale improves reward and inventory
behavior for KAN/MonoKAN. This supports the supervisor's point that action
scaling can affect performance drastically.

## In-Distribution Small Validation

10-seed, direct base-stock-output, scale `0.03`.

### Raw History

| policy | mean reward | median reward | mean inventory | target factor | cap-near |
|---|---:|---:|---:|---:|---:|
| HistoricalDemandBaseStock | 4443.43 | 4497.96 | 71.14 | n/a | n/a |
| PaperFCNNBaseStockOrderUpTo | 4215.81 | 4281.46 | 70.58 | 1.64 | 0.00% |
| MonoKANBaseStockOrderUpTo | 4161.30 | 4166.00 | 89.36 | 1.63 | 0.00% |
| KANBaseStockOrderUpTo | 4126.69 | 4075.30 | 110.12 | 1.74 | 1.00% |

Key deltas:

- MonoKAN - KAN: `+34.61` reward/period, wins `7/10` seeds.
- MonoKAN - FCNN: `-54.51` reward/period, wins `2/10` seeds.

### Summary History

| policy | mean reward | median reward | mean inventory | target factor | cap-near |
|---|---:|---:|---:|---:|---:|
| HistoricalDemandBaseStock | 4443.43 | 4497.96 | 71.14 | n/a | n/a |
| PaperFCNNBaseStockOrderUpTo | 4253.31 | 4293.78 | 74.81 | 1.69 | 0.00% |
| MonoKANBaseStockOrderUpTo | 4193.91 | 4213.09 | 77.65 | 1.60 | 0.00% |
| KANBaseStockOrderUpTo | 4189.65 | 4170.06 | 83.79 | 1.63 | 0.00% |

Key deltas:

- MonoKAN - KAN: `+4.26` reward/period, wins `6/10` seeds.
- MonoKAN - FCNN: `-59.40` reward/period, wins `0/10` seeds.

Summary-history improvement over raw-history:

- FCNN: `+37.50`
- KAN: `+62.96`
- MonoKAN: `+32.60`

Interpretation: summary features improve all neural policies, especially KAN.
The reward gap between unconstrained KAN and MonoKAN narrows after fixing the
history representation, suggesting that part of KAN's earlier weakness came
from poor feature representation rather than lack of monotonicity alone.

## Capacity Sweep Evidence

3-seed capacity sweep, raw-history, scale `0.03`.

Increasing KAN/MonoKAN hidden units or grid size did not improve absolute
performance.

| variant | KAN reward | KAN inventory | MonoKAN reward | MonoKAN inventory |
|---|---:|---:|---:|---:|
| h4_g3 | 4496.97 | 132.24 | 4539.59 | 104.43 |
| h8_g3 | 4476.65 | 140.68 | 4520.98 | 111.92 |
| h8_g5 | 4439.24 | 151.67 | 4520.82 | 114.30 |
| h16_g3 | 4452.05 | 147.98 | 4498.03 | 118.37 |

Interpretation: the current weakness is not solved by simply making KAN/MonoKAN
larger. Feature representation and training/optimization are more plausible
bottlenecks.

## OOD / Generalization Evidence

5-seed x 8-shift OOD evaluation. Models are trained in-distribution and
evaluated under distribution shifts.

### Raw History OOD

| policy | mean reward | worst seed-shift reward | mean inventory | cap-near |
|---|---:|---:|---:|---:|
| HistoricalDemandBaseStock | 4486.59 | 1206.78 | 96.83 | n/a |
| PaperFCNNBaseStockOrderUpTo | 3968.56 | 1147.69 | 93.47 | 0.00% |
| MonoKANBaseStockOrderUpTo | 3843.96 | 1144.95 | 131.13 | 0.00% |
| KANBaseStockOrderUpTo | 3542.21 | 219.69 | 180.88 | 3.03% |

MonoKAN - KAN by shift:

| shift | mean delta | wins |
|---|---:|---:|
| mixed_stress | +1551.73 | 5/5 |
| high_mean_demand | +488.47 | 5/5 |
| high_holding_cost | +284.31 | 5/5 |
| in_distribution | +56.64 | 5/5 |
| high_lost_sale_penalty | +47.25 | 5/5 |
| low_cv | +9.81 | 2/5 |
| low_mean_demand | -6.98 | 0/5 |
| high_cv | -17.22 | 2/5 |

### Summary History OOD

| policy | mean reward | worst seed-shift reward | mean inventory | cap-near |
|---|---:|---:|---:|---:|
| HistoricalDemandBaseStock | 4486.59 | 1206.78 | 96.83 | n/a |
| PaperFCNNBaseStockOrderUpTo | 4018.61 | 1158.61 | 107.95 | 0.00% |
| MonoKANBaseStockOrderUpTo | 3930.97 | 1148.69 | 112.98 | 0.00% |
| KANBaseStockOrderUpTo | 3826.33 | 1149.96 | 132.99 | 0.31% |

MonoKAN - KAN by shift:

| shift | mean delta | wins |
|---|---:|---:|
| mixed_stress | +466.22 | 5/5 |
| high_mean_demand | +248.45 | 5/5 |
| high_holding_cost | +93.19 | 5/5 |
| in_distribution | +11.91 | 4/5 |
| low_cv | +10.91 | 3/5 |
| high_lost_sale_penalty | +8.27 | 3/5 |
| high_cv | +0.49 | 2/5 |
| low_mean_demand | -2.30 | 1/5 |

Interpretation: after summary-history feature improvement, MonoKAN still has a
clear advantage over KAN under high-demand, high-cost, and mixed-stress OOD
conditions, but the advantage is smaller than in raw-history mode because KAN is
less pathological. MonoKAN remains below FCNN on average.

## Claim Audit

| Claim | Status | Evidence | Caveat |
|---|---|---|---|
| Direct base-stock-output fixes the raw-action failure mode. | Supported | Cap-near rates for direct base-stock models are near zero after logit scaling; inventories are no longer cap-dominated. | Current evidence is small/medium validation, not final paper-scale. |
| Action scaling matters substantially. | Supported | Scale `0.03` improves KAN/MonoKAN reward and inventory over `0.05` and `0.10`. | Scale was selected from diagnostics; avoid further tuning without predeclared protocol. |
| Summary history is a useful observable feature representation. | Supported | 10-seed improvements: FCNN +37.50, KAN +62.96, MonoKAN +32.60. | Needs literature framing as demand-history feature engineering, not hidden parameter access. |
| MonoKAN consistently beats unconstrained KAN in reward. | Partially supported | Raw 10-seed: +34.61, wins 7/10. Summary 10-seed: +4.26, wins 6/10. OOD summary: +104.64 overall. | Summary in-distribution reward advantage is small. |
| MonoKAN reduces severe KAN failures under stress OOD. | Supported in small OOD | Summary OOD: mixed stress +466.22, high mean demand +248.45, high holding cost +93.19, each wins 5/5. | Needs larger paired protocol before final claim. |
| MonoKAN beats FCNN. | Unsupported | Summary 10-seed MonoKAN - FCNN: -59.40; summary OOD MonoKAN - FCNN: -87.63. | MonoKAN sometimes wins individual seeds/shifts, but not average. |
| MonoKAN beats historical base-stock. | Unsupported | HistoricalDemandBaseStock remains best in current small validations and OOD summary. | This may reflect strong heuristic and small training budget. |
| Larger KAN/MonoKAN capacity solves the gap. | Unsupported | h8/h16/grid5 variants did not improve reward. | Only small capacity sweep; not exhaustive architecture optimization. |
| MonoKAN's main contribution is average reward dominance. | Unsupported | Current reward evidence is weaker than FCNN/historical. | Better framing is structure/generalization/auditability. |
| MonoKAN offers structural auditability with modest reward cost. | Supported | MonoKAN enforces selected monotone comparative statics while staying close to KAN and below FCNN by a moderate gap. | Need final monotonicity/spline diagnostics under the chosen direct base-stock-output setup. |

## Current Best Research Story

The strongest defensible framing is:

> We follow recent inventory-policy learning work by letting the neural network
> output a base-stock/target-stock level rather than a purchase quantity. The
> order-up-to decoder handles inventory dependence exactly. With appropriate
> action scaling and observable demand-history summaries, KAN/MonoKAN no longer
> exhibit the previous clipping-dominated failure. MonoKAN does not outperform
> FCNN or historical base-stock in average reward, but it improves over
> unconstrained KAN under high-demand/high-cost/mixed-stress OOD conditions while
> preserving selected monotone comparative statics of the target-stock function.

Avoid claiming:

- MonoKAN is the best-performing policy.
- Monotonicity alone causes robustness.
- Summary-history features prove architecture superiority.
- Results are paper-scale.

## What To Do Next

Recommended next steps:

1. Literature/frame audit:
   - base-stock-output neural policies;
   - observable product features and demand-history summaries;
   - action scaling / bounded action parameterization;
   - monotone target-stock comparative statics.
2. Freeze one small protocol:
   - direct base-stock-output;
   - summary history;
   - scale `0.03`;
   - FCNN / KAN / MonoKAN / HistoricalDemandBaseStock;
   - fixed seeds and OOD shifts.
3. Add direct base-stock-output structural diagnostics:
   - complete target-stock monotonicity, not only spline-level checks;
   - KAN vs MonoKAN violation rates under raw and summary history.
4. Run sample-efficiency only after freezing the feature/scale protocol:
   - training products: 50 / 100 / 300;
   - same seeds and evaluation products;
   - focus on reward, inventory, OOD stress, and seed stability.

Do not keep sweeping scale/hidden/grid unless a protocol is predeclared. More
tuning would weaken the scientific story.
