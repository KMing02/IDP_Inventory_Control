# Final Experiment Index

Evidence relevant to the final IDP report. All rows use three seeds (2025, 2026, 2027) on identical products and demand paths unless stated otherwise. Historical exploratory branches live in Git history, not in the current tree.

## Metrics

Two metrics are reported throughout, because they disagree and the disagreement is informative.

- **Reward per period**: the mean over validation products. Product economics are heavy tailed, so this is dominated by a handful of large price-times-demand products.
- **Regret**: the mean per-product shortfall against the omniscient base-stock policy, in percent. Scale free, and therefore far more sensitive to typical-SKU behaviour.

The original headline metric hid a 21% typical-product failure behind a 3.6% mean gap, which is why both are now reported.

## Evidence Status

- **FINAL MAIN**: central evidence for the report's main comparison.
- **FINAL CONTROL**: isolates one factor in the main comparison.
- **FINAL ABLATION**: supports interpretation of the main result.
- **SUPERSEDED**: retained to explain how the final formulation was reached.

## Final Evidence

| Experiment | Report role | Reward | Regret | Config | Status |
|---|---|---:|---:|---|---|
| Soft-quantile MonoKAN, weighted objective, 5k products | Recommended final policy. | `4285.53 +/- 0.19` | `0.80%` | `configs/paper_lost_sales_soft_quantile_monokan.yaml` | FINAL MAIN |
| Soft-quantile MonoKAN, weighted objective, 1k products | Same policy at the smaller training set. | `4285.30 +/- 0.54` | `1.04%` | same, `training_products: 1000` | FINAL ABLATION |
| Soft-quantile MonoKAN, profit objective | Same policy maximising total profit. | `4286.46 +/- 0.23` | `1.77%` | `configs/paper_lost_sales_soft_quantile_monokan_profit.yaml` | FINAL MAIN |
| Soft-quantile MLP level network | Cost of the monotonicity certificate, decoder and parameters held fixed. | `4286.29 +/- 0.80` | `2.23%` | profit config, `level_network: mlp`, `level_hidden_units: 40` | FINAL CONTROL |
| Soft-quantile unconstrained KAN level network | Separates the constraint from the KAN representation. | `4274.10 +/- 2.31` | `2.41%` | profit config, `level_network: kan` | FINAL CONTROL |
| Unconstrained FCNN direct target stock | Shipped neural anchor. | `4174.41 +/- 4.36` | `9.98%` | `configs/paper_lost_sales_rolling_direct_target_dev_1k_normalized.yaml` | FINAL MAIN |
| DeepGrid7 MonoKAN direct target stock | Shipped monotone policy; the starting point. | `4111.59 +/- 3.31` | `21.40%` | same | FINAL MAIN |
| Unconstrained DeepGrid7 KAN direct target stock | Shows the shipped gap was representation, not monotonicity. | `4130.54 +/- 3.95` | `20.68%` | same, `KANDirectTargetStockDeepGrid7` | FINAL CONTROL |

## Superseded Evidence

These runs characterise the DeepGrid7 MonoKAN that the soft-quantile policy replaced. They are retained because they explain how the original result was reached and why particular explanations for it were ruled out. Numbers are reward per period on the 300-product protocol; regret was not computed for these runs.

| Experiment | Question it answered | Result | Config |
|---|---|---:|---|
| Paper raw-action FCNN anchor | Can the pipeline reproduce a strong paper-style FCNN? | `4580.54 +/- 4.09` | `configs/paper_lost_sales_raw_fcnn_replication_5k.yaml` |
| Compact vs DeepGrid7 MonoKAN | Does MonoKAN capacity explain the gap? Partly. | DeepGrid7 `4185.19`, compact `4126.50` | `configs/paper_lost_sales_direct_target_capacity_medium.yaml` |
| 1k vs 5k training products | Does more data explain the gap? No. | 5k DeepGrid7 `4192.54 +/- 1.10` | `configs/paper_lost_sales_direct_target_capacity_5k.yaml` |
| 500 vs 1,000 epochs | Does optimisation time explain the gap? No. | `4196.20` at 1,000 epochs | `configs/paper_lost_sales_direct_target_capacity_5k_1000e.yaml` |
| Temporal DeepGrid7 MonoKAN | Do causal temporal summaries help? Slightly, but they break the certificate. | `4135.64 +/- 9.08`; about `37.5%` raw-demand reversals | `configs/paper_lost_sales_rolling_direct_target_temporal_dev_1k_normalized.yaml` |
| Supervised target-fitting diagnostic | Is the limit representational or reward-driven? | DeepGrid7 fits the target map better than compact models | `configs/paper_lost_sales_base_stock_small.yaml` |

The capacity, data-scale, and epoch ablations each ruled out one explanation for the shipped model's weakness. The explanation they were missing was the spline grid; see `docs/DEBUGGING.md`.

The raw-action FCNN anchor is not comparable to the target-stock rows: it uses a different product draw and validation size, so its higher reward reflects a different product population, not a better policy.

## Classical Baselines

Evaluated on the same products and demand paths. Information access matters and must be matched when quoting these.

| Baseline | Information | Reward | Regret |
|---|---|---:|---:|
| Omniscient base stock | true Gamma parameters | `4297.45` | `0.00%` |
| Expanding-window empirical quantile | all demand observed so far | `4290.64` | `0.27%` |
| Running mean of rolling 32-window quantile | all demand observed so far | `4282.26` | `0.44%` |
| Gamma fit | initial 32 lags | `4274.91` | `1.35%` |
| Empirical quantile | initial 32 lags | `4264.52` | `1.76%` |
| Rolling 32-window quantile | last 32 observations | `4255.93` | `1.90%` |

The learned policy is competitive with, but does not beat, a well-specified classical estimator at matched information. Quoting only the fixed 32-lag baselines overstates the learned policy's standing, because the final decoder uses an expanding window.

## What Moved The Result

Cumulative, three seeds each. Each step keeps the end-to-end monotonicity certificate.

| Step | Reward | Regret |
|---|---:|---:|
| DeepGrid7 MonoKAN as shipped | `4111.59` | `21.40%` |
| Data-adapted spline knots | `4189.07` | `9.28%` |
| Sorted demand lags | `4192.86` | `8.71%` |
| 400 epochs, corrected monotone projection | `4210.70` | `7.06%` |
| Soft-quantile decoder | `4240.50` | `3.81%` |
| Level-dependent gain, 2,400 epochs | `4260.24` | `2.50%` |
| Expanding observation window | `4286.46` | `1.77%` |
| Per-product reward weighting | `4285.30` | `1.04%` |

## Levers That Did Not Work

Recorded so they are not retried.

| Lever | Outcome |
|---|---|
| Epochs 1,200 to 2,400 (final policy) | No change; reward is flat. |
| Training products 1,000 to 5,000 at a 600-epoch budget | Worse, but confounded by the short budget. Superseded: at the full budget it *helps*. |
| Imitation warm-start from the empirical base-stock policy | Severely harmful: regret `8.61%` at 200 epochs, `21.64%` at 50, against `1.04%` without. The imitation target is a static base stock while the policy deploys an expanding window, and its raw-unit MSE reintroduces large-product domination. |
| Training horizon 100 to 250 | No help: regret `2.06%`. |
| Larger KAN level network | Worse: supervised RMSE `0.026` (2x8) to `0.033` (3x16, grid 12). |
| Running mean on top of the expanding window | Worse by `6.8` reward; double smoothing. |
| log1p-transformed economics | Slightly worse: `corr 0.987` to `0.984`. |
| Blending the sample mean into the target | Null: `4258.32` vs `4258.28`. |
| Per-product weighting on the *flat* DeepGrid7 MonoKAN | Worse on both metrics; helps only once the level network is small and shared. |
| Per-product price normalisation | Rejected: breaks the certificate, since raising price moves the cost ratios in conflicting directions. |

## Structural Claims

The final policy is certified nondecreasing in every observed demand and in selling price and lost-sales penalty, nonincreasing in purchase and holding cost, and nonincreasing in current inventory through `q_t = max(S_t - I_t, 0)`.

Verified three ways:

- `scripts/run_direct_target_structural_audit.py` reports `0.0` violation rate in domain and under stress extrapolation, by finite differences and by autograd sign.
- A float64 perturbation sweep over all 32 historical lags, future observed demands, and all four economics features returns a worst wrong-direction change of exactly `0.000e+00`, on the deployed running-mean and expanding-window targets rather than only the static one.
- Scale equivariance in observed demand is exact to `0.00e+00`.

The certificate now holds for the right reason. The projection previously admitted small violations; see `docs/DEBUGGING.md`.

## Interpretability

`scripts/run_monokan_interpretability.py` produces partial-dependence curves for every driver and checks each against its certified direction while plotting it. On the final checkpoint all five drivers report `direction_holds = True`, and the learned order-up-to level correlates `0.982` with the analytic newsvendor fractile `(p + k - c) / (p + k - c + h)` — which is never supplied to the model as a feature, an initialisation, or a training target.

## Sample Efficiency

Three seeds per cell, MonoKAN against the parameter-matched MLP inside the identical decoder. Batch size is fixed at 100 for every training-set size, and every cell is trained to a fixed 2,400-epoch budget with best-checkpoint selection on a common validation set.

Both controls matter. Capping the batch at the training-set size makes small `n` silently mean full-batch gradients. Holding the *gradient-step* count constant instead of the epoch budget leaves the large-`n` cells undertrained, which reads as a spurious loss of sample efficiency: under that protocol `n = 1000` scored `4249.57` against `4285.30` when trained to convergence.

Reward per period:

| Level network | n=100 | n=300 | n=1000 |
|---|---:|---:|---:|
| MonoKAN | `4279.35 +/- 0.39` | `4282.60 +/- 0.24` | `4284.08 +/- 0.22` |
| MLP | `4285.05 +/- 0.45` | `4284.38 +/- 0.61` | `4278.53 +/- 0.29` |

Regret:

| Level network | n=100 | n=300 | n=1000 |
|---|---:|---:|---:|
| MonoKAN | `2.61 +/- 0.11` | `1.62 +/- 0.10` | `1.00 +/- 0.06` |
| MLP | `2.33 +/- 0.10` | `1.37 +/- 0.02` | `2.44 +/- 0.05` |

**The proposal's sample-efficiency claim is not supported, and the direction is reversed.** At 100 training products the monotone model is worse than the unconstrained MLP on both metrics. It needs more data, not less, and overtakes only at `n = 1000`.

What the constraint does buy is reliable scaling: MonoKAN improves monotonically with data on both metrics, while the MLP peaks around `n = 100` to `300` and then degrades. Selected checkpoints sit near the end of the budget for both, so this is not undertraining.

The two metrics disagree at `n = 300`, so any reported crossover point depends on which is quoted.

## Evaluation Precision

Per-product reward has a standard deviation of `8,537` against a mean of `4,284`, so absolute means are far less precise than the seed spread suggests. Two different error bars must not be confused:

- **seed SEM**: training variability on a fixed product set. This is what `+/-` denotes elsewhere in this file.
- **product SEM**: sampling variability of the product population. At 300 validation products the 95% CI on the population mean is roughly `+/- 966`.

Comparisons are unaffected, because every policy is evaluated on identical products and demand paths, so the product variance cancels: the paired SEM is `+/- 2.40` against differences of 1 to 10 reward per period.

Confirmation on 3,000 validation products, a strict superset of the 300 used for checkpoint selection (the generator draws sequentially from one seeded stream):

| Policy | Reward | seed SEM | product SEM | Regret |
|---|---:|---:|---:|---:|
| Omniscient base stock | `4693.34` | — | `+/- 185.90` | `0.00%` |
| Soft-quantile MonoKAN, 5k products | `4679.59` | `+/- 0.04` | `+/- 185.17` | `0.78%` |
| Soft-quantile MonoKAN, 1k products | `4678.87` | `+/- 0.44` | `+/- 185.16` | `1.28%` |
| Classical Gamma fit | `4671.74` | — | `+/- 185.19` | `1.46%` |
| Classical empirical quantile | `4651.87` | — | `+/- 184.24` | `2.34%` |

The absolute reward level differs from the 300-product tables because a different product draw has a different mean; that is the population uncertainty above, not a change in policy behaviour. The ordering is unchanged on ten times the products.

Report paired differences and regret as primary evidence, quote absolute rewards to few significant figures, and label which standard error is which.
