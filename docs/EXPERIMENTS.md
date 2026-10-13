# Final Experiment Index

Evidence relevant to the final IDP report. All rows use three seeds (2025, 2026, 2027) on identical products and demand paths unless stated otherwise. Historical exploratory branches live in Git history, not in the current tree; the configs and modules of the superseded experiments were removed after commit `226f001` and can be restored from it (`git show 226f001:<path>`). The commands that reproduce each table are collected under Reproduction at the end.

## Metrics

Two metrics are reported throughout, because they disagree and the disagreement is informative.

- **Reward per period**: the mean over validation products. Product economics are heavy tailed, so this is dominated by a handful of large price-times-demand products.
- **Regret**: the mean per-product shortfall against the omniscient base-stock policy, in percent. Scale free, and therefore far more sensitive to typical-SKU behaviour. Products on which the omniscient policy itself earns nothing or loses money (21 of the 300 validation products) have no meaningful relative shortfall and are excluded; this reproduces the classical baselines' `1.35%` and `1.76%` exactly.

The original headline metric hid a 21% typical-product failure behind a 3.6% mean gap, which is why both are now reported.

## Evidence Status

- **FINAL MAIN**: central evidence for the report's main comparison.
- **FINAL CONTROL**: isolates one factor in the main comparison.
- **FINAL ABLATION**: supports interpretation of the main result.
- **SUPERSEDED**: retained to explain how the final formulation was reached.

## Direct-Quantity Evidence (Main Design)

Following the supervisor's feedback this is the main formulation. Under the order-up-to decoder,
`q = max(S - I, 0)` supplies monotonicity in inventory for every model, so the certificate only binds on the four
economic inputs. Here the network emits the order with inventory as input 0, and all 37 input directions are
certified. Three seeds, 2,400 epochs, 1,000 training products, profit objective, and the same 300 validation
products as the target-stock controls; reproduced by
`python scripts/run_direct_quantity_comparison.py train` followed by `summarize`.

| Policy | Reward | Regret | Products violating | Order drop per unit of stock | Params | Status |
|---|---:|---:|---:|---:|---:|---|
| Raw-action MonoKAN | `4206.43 +/- 17.56` | `15.71%` | `0.0%` | `-0.524` | 4,265 | FINAL MAIN |
| Raw-action KAN, constraint removed | `4229.31 +/- 10.88` | `13.33%` | `35.1%` | `-0.541` | 4,265 | FINAL CONTROL |
| Raw-action paper FCNN | `3874.38 +/- 622.50` | `20.64%` | `0.0%` | `-0.339` | 2,105 | FINAL CONTROL |
| Omniscient base stock, same sweep | `4297.45` | `0.00%` | `0.0%` | `-0.903` | - | REFERENCE |

The certificate binds here. The unconstrained KAN orders more as stock rises on 35.1%
of products (45.3%, 8.7%, 51.3% by seed); the certified model is at 0% by construction. The violations are
widespread and not always small: 3.7% of swept steps, and the worst rise is
`0.109` mean-demand units, about 11% of a day's demand. The violation rate is architecture dependent: the
feed-forward arm violates on 0.0% of products. The FCNN arm is unstable: seed 2027 collapsed to `3155.76` (`42.43%` regret), as the same seed did at 400 epochs; the other seeds average `4233.69`, above the certified model, so its three-seed mean understates what a successful run achieves.

The certified model also earned less here. Paired over the 300 products (seed-averaged), MonoKAN minus the
unconstrained KAN is
`-22.89` per period, 95% CI `[-32.57, -13.20]`, paired t `p < 0.0001`,
Wilcoxon `p < 0.0001`; MonoKAN is ahead on 28% of products. The two networks were also initialised differently (MonoKAN with an open readout calibrated to the empirical newsvendor level, the KAN with a constant output at that level), so the gap is not purely the cost of the constraint; a rerun with identical initialisation would isolate it. The paired test treats the
trained models as given (rewards averaged over seeds first), so its interval reflects variation across products, not
retraining. Under the target-stock decoder at the same budget no profit difference to an MLP level network was
detected (`-0.08`, `p = 0.94`); there the subtraction already enforces the inventory direction.

No direct-quantity policy reacts to stock with more than 60% of the omniscient policy's
`-0.90` on the same sweep, and none beats the classical rules that see the same information, the last 32 observations re-estimated daily
(`1.25%` Gamma fit, `1.90%` empirical quantile). The matched
target-stock MonoKAN reaches `1.62%` regret at the same 1,000 products and 2,400 epochs: the decoder supplies the
ordering strength, the certificate only the direction.

A 400-epoch pilot of the same comparison (superseded) gave the same qualitative picture: 48.1% of products violating
for the unconstrained KAN, 0% for the certified model and the FCNN, and a cost of about 66 per period.

Interpretability, by the test of Maggiar et al. (2025, Figure 3): the order a direct-quantity policy places at zero
stock is its implied order-up-to level. Per unit of mean demand, which removes product scale, MonoKAN's implied level
correlates `0.49` with the omniscient level, against `0.93` for the classical Gamma fit on the same 32-day history; in
raw units scale alone makes every policy look good. The certificate fixes each input's direction, not the level.
Speed: `0.41` ms per decision for MonoKAN against `0.20` ms for the FCNN (median CPU time, one decision); 10,000
decisions at once take about `57` ms. Reproduced by `run_direct_quantity_comparison.py interpret`; the
order-versus-stock curves of the worst break come from `run_direct_quantity_comparison.py curves`.

## Target-Stock Evidence (Alternative Design)

| Experiment | Report role | Reward | Regret | Config | Status |
|---|---|---:|---:|---|---|
| Soft-quantile MonoKAN, weighted objective, 5k products | Recommended final policy. | `4285.53 +/- 0.19` | `0.80%` | `configs/paper_lost_sales_soft_quantile_monokan.yaml` | FINAL MAIN |
| Soft-quantile MonoKAN, weighted objective, 1k products | Same policy at the smaller training set. | `4285.30 +/- 0.54` | `1.04%` | same, `training_products: 1000` | FINAL ABLATION |
| Soft-quantile MonoKAN, profit objective | Same policy maximising total profit. | `4286.46 +/- 0.23` | `1.77%` | `configs/paper_lost_sales_soft_quantile_monokan_profit.yaml` | FINAL MAIN |
| Soft-quantile MonoKAN, expanding decoder | Matched partner for the MLP control. | `4286.21 +/- 0.43` | `1.62%` | profit config, 2400 epochs | FINAL CONTROL |
| Soft-quantile MLP level network | Cost of the monotonicity certificate, decoder and parameters held fixed. | `4286.29 +/- 0.80` | `2.23%` | profit config, 2400 epochs, `level_network: mlp`, `level_hidden_units: 40` | FINAL CONTROL |
| Soft-quantile MonoKAN, sliding decoder | Matched partner for the KAN control. | `4280.83` | `1.71%` | profit config, 2400 epochs, sliding window + running-mean decoder | FINAL CONTROL |
| Soft-quantile MLP level network, sliding decoder | Third arm of the sliding-decoder control. | `4278.75` | `2.12%` | same, `level_network: mlp` | FINAL CONTROL |
| Soft-quantile unconstrained KAN level network | Separates the constraint from the KAN representation. | `4274.10 +/- 2.31` | `2.41%` | same, `level_network: kan` | FINAL CONTROL |
| Unconstrained FCNN direct target stock | Shipped neural anchor. | `4174.41 +/- 4.36` | `9.98%` | `configs/paper_lost_sales_rolling_direct_target_dev_1k_normalized.yaml` | FINAL MAIN |
| DeepGrid7 MonoKAN direct target stock | Shipped monotone policy; the starting point. | `4111.59 +/- 3.31` | `21.40%` | same | FINAL MAIN |
| Unconstrained DeepGrid7 KAN direct target stock | Shows the shipped gap was representation, not monotonicity. | `4130.54 +/- 3.95` | `20.68%` | same, `KANDirectTargetStockDeepGrid7` | FINAL CONTROL |

Each control is only readable against the arm it was matched to, not against the
profit row above it. The MLP control shares the expanding decoder and a 2400-epoch
budget with its MonoKAN partner: the paired difference is `-0.08` per period,
95% CI `[-2.09, +1.93]`, `p = 0.94`, so the certificate costs no measurable profit.
The KAN control shares the sliding-window and running-mean decoder with its own
partner, where MonoKAN leads the unconstrained KAN by `+6.73` per period
(`p = 0.0001`). Comparing an arm across those two blocks confounds the decoder
with the level network.

## Target Stock: MonoKAN Against The Same KAN In Every Information Window

The core monotonicity test of the alternative design: the same soft-quantile policy with the certified MonoKAN level
network and with the identical KAN without the certificate (`monokan.level_network: kan`; same initialisation through
`calibrate_level`, same parameter count,
1,892). Final setup (5,000 products, weighted objective, 2,400 epochs) in all three information windows, three seeds,
the same 300 validation products. Regret:

| Information | Classical Gamma fit | Classical empirical quantile | KAN, no certificate | MonoKAN |
|---|---:|---:|---:|---:|
| Initial 32 lags, never updated (`monokan.roll_history: false`) | `1.35%` | `1.76%` | `2.37%` | `2.04%` |
| Last 32 observations (`monokan.window_mode: sliding`) | `1.25%` | `1.90%` | `1.96%` | `1.83%` |
| All demand so far (final config) | `0.18%` | `0.27%` | `0.87%` | `0.80%` |

MonoKAN has the lower regret in every window. Paired over the 300 products (rewards averaged over seeds), MonoKAN
minus KAN, reward per period:

| Setting | Difference | 95% CI | paired t | Wilcoxon | MonoKAN ahead | Reading |
|---|---:|---:|---:|---:|---:|---|
| Final setup, all demand | `-0.44` | `[-1.97, +1.08]` | `0.569` | `0.0001` | 61% | no significant mean difference; ahead on most products |
| Final setup, last 32 observations | `+1.81` | `[+0.16, +3.46]` | `0.032` | `< 0.0001` | 62% | small but consistent |
| Final setup, initial 32 lags | `+2.64` | `[+0.17, +5.10]` | `0.036` | `0.464` | 51% | small and borderline |
| 1,000 products, profit objective, last 32 observations | `+5.90` | `[+3.32, +8.48]` | `< 0.0001` | `< 0.0001` | 72% | ahead on both tests |

Rewards behind the table (MonoKAN / KAN): `4285.53` / `4285.97`, `4255.32` / `4253.51`, `4249.02` / `4246.38`, and
`4260.45` / `4254.55`; the last row's regrets are `2.26%` / `3.35%`. In no setting is MonoKAN significantly behind.
The earlier sliding-window, running-mean block above (`+6.73`, `p = 0.0001`) points the same way; it is kept as an
independent check, since its decoder differs from every row here.

### The Two Designs Under Matched Training And Information

1,000 training products, 2,400 epochs, profit objective, three seeds, the same 300 validation products, and both
designs see only the last 32 observations of demand:

| | Direct quantity | Target stock |
|---|---:|---:|
| Who enforces "more stock, never a larger order" | the certificate | the subtraction in `max(S - I, 0)` |
| Unconstrained arm: products that order more as stock rises | `35.1%` | `0%` (impossible) |
| Certified minus unconstrained KAN, reward per period | `-22.89` (`p < 0.0001`) | `+5.90` (`p < 0.0001`) |
| MonoKAN regret | `15.71%` | `2.26%` |

The certified model earned less where the certificate alone enforces the stock rule, and more where the structure
already does that work. This is consistent with a certificate whose cost grows with the work it does, but the designs
also differ in inputs, architecture and decoder, and the direct-quantity arms in initialisation, so this table does not
isolate the stock constraint. With all demand observed so far the matched
target-stock MonoKAN reaches `1.62%`.

## Classical Baselines

Evaluated on the same products and demand paths. Information access matters and must be matched when quoting these.

| Baseline | Information | Reward | Regret |
|---|---|---:|---:|
| Omniscient base stock | true Gamma parameters | `4297.45` | `0.00%` |
| Gamma fit | all demand observed so far | `4294.02` | `0.18%` |
| Empirical quantile | all demand observed so far | `4290.64` | `0.27%` |
| Running mean of rolling 32-window quantile | all demand observed so far | `4282.26` | `0.44%` |
| Gamma fit | last 32 observations, re-estimated daily | `4275.37` | `1.25%` |
| Gamma fit | initial 32 lags, never updated | `4274.91` | `1.35%` |
| Empirical quantile | initial 32 lags, never updated | `4264.52` | `1.76%` |
| Empirical quantile | last 32 observations, re-estimated daily | `4255.93` | `1.90%` |

The "initial 32 lags" rows are the paper's benchmark definition (one estimate, never updated); "last 32
observations" is what a policy in daily use sees and exactly the information of the direct-quantity design; "all
demand observed so far" is the information of the final target-stock decoder. `scripts/run_classical_windows.py`
reproduces every window cell. The Gamma fit is additionally told the demand family, so the empirical quantile is the
fairer partner for a learned policy.

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

Across the three tested sizes MonoKAN improved monotonically on both metrics, while the MLP peaked around `n = 100` to `300` and then degraded; three sizes and three seeds do not establish this beyond the tested range. Selected checkpoints sit near the end of the budget for both, so this is not undertraining.

The two metrics disagree at `n = 300`, so any reported crossover point depends on which is quoted.

## Evaluation Precision

Per-product reward has a standard deviation of `8,537` against a mean of `4,284`, so absolute means are far less precise than the seed spread suggests. Two different error bars must not be confused:

- **seed SEM**: training variability on a fixed product set. This is what `+/-` denotes elsewhere in this file.
- **product SEM**: sampling variability of the product population. At 300 validation products the 95% CI on the population mean is roughly `+/- 966`.

Paired comparisons largely remove this product-level noise, because every policy is evaluated on identical products and demand paths, so the product variance cancels (selection on the same validation products is a separate limitation): the paired SEM is `+/- 2.40` against differences of 1 to 10 reward per period.

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

The capacity, data-scale, and epoch ablations each ruled out one explanation for the shipped model's weakness. The explanation they were missing was the spline grid; see `docs/DEBUGGING.md`. Their configs, the temporal-summary config and the supervised diagnostic were removed from the tree after commit `226f001`.

The raw-action FCNN anchor is not comparable to the target-stock rows: it uses a different product draw and validation size, so its higher reward reflects a different product population, not a better policy.

## Reproduction

All commands run from the repository root; each target-stock command trains one seed (repeat for 2025, 2026, 2027).

| Evidence | Command |
|---|---|
| Direct quantity, all three arms | `python scripts/run_direct_quantity_comparison.py train --out outputs/direct_quantity`, then `summarize`, `interpret` and `curves` on that directory |
| Final target-stock policy | `python scripts/run_soft_quantile_monokan.py --seed 2025 --output-dir outputs/ts_monokan/seed_2025` |
| Same policy, KAN without the certificate | add `--set monokan.level_network=kan` |
| Last 32 observations only | add `--set monokan.window_mode=sliding` |
| Initial 32 lags, never updated | add `--set monokan.roll_history=false` |
| Matched to the direct-quantity training | `--config configs/paper_lost_sales_soft_quantile_monokan_profit.yaml --set epochs=2400 --set idp_benchmark.compact.reward_epochs=2400 --set idp_benchmark.compact.validation_interval=150 --set monokan.window_mode=sliding` |
| Regret and paired tests of saved runs | `python scripts/compare_target_stock_runs.py summary <run roots>` and `pair <first> <second>` |
| Classical rules in every information window | `python scripts/run_classical_windows.py` |
| Structural audit and interpretability | `scripts/run_direct_target_structural_audit.py`, `scripts/run_monokan_interpretability.py` |
| Historical DeepGrid7 and FCNN direct target stock | `python scripts/run_base_stock_output_experiment.py --config configs/paper_lost_sales_rolling_direct_target_dev_1k_normalized.yaml` |
