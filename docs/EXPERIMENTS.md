# Experiment Index

This index records the experiment paths that support the current MonoKAN project
story. It distinguishes reproducible development evidence from diagnostics and
incomplete runs. None of the entries below is a protected paper-scale result.

The outputs were produced from a working tree based on commit
`1bf315be788e0d61f2dabc9f980274a8ee7fb839`. Some direct-target, rolling, and
temporal runs used the implementation changes committed during the first-stage
cleanup; consult the provenance note for each experiment and the implementation
commit recorded at the end of this document.

## Evidence Status

- **CANONICAL DEVELOPMENT**: the primary reproducible development evidence for
  a predeclared question.
- **SUPPORTING**: a valid ablation or diagnostic that supports interpretation.
- **DEVELOPMENT**: valid but too small or insufficiently replicated for a main
  claim.
- **INVALID/DEBUG**: useful for diagnosing an implementation problem, but not
  valid policy-performance evidence.
- **INCOMPLETE**: the intended experiment did not finish.

## Experiment Inventory

| Experiment | Scientific question | Main result | Config | Seeds | Protocol summary | Output | Status | Provenance notes |
|---|---|---|---|---|---|---|---|---|
| Raw FCNN anchor | Does the paper-style rolling raw-action FCNN provide a credible reference anchor? | Reward/period `4580.54 +/- 4.09`, close to HistoricalDemandBaseStock `4580.38`. | `configs/paper_lost_sales_raw_fcnn_replication_5k.yaml` | 2025, 2026, 2027 | 5,000 training products, 1,000 epochs, horizon 100, rolling causal history, no imitation, no action cap, terminal salvage enabled. | `outputs/paper_raw_fcnn_replication_5k` | CANONICAL DEVELOPMENT | Development-scale replication, not the paper's 40,000/100,000 train/test protocol. Base SHA `1bf315b`. |
| Supervised capacity | Can compact and DeepGrid7 KAN/MonoKAN represent the empirical newsvendor target mapping independently of DirectBackprop? | DeepGrid7 MonoKAN improved validation target identification across the fixed diagnostic; the diagnostic separates representation from reward-only optimization. | `configs/paper_lost_sales_direct_target_capacity_medium.yaml` | 2025, 2026, 2027 | Same observable product split and empirical target labels; compact and DeepGrid7 KAN/MonoKAN; diagnostic only, not policy pretraining. | `outputs/supervised_target_diagnostic/capacity_canonical` | SUPPORTING | No future demand and no protected test products. Base SHA `1bf315b`; capacity factories are part of the cleanup implementation commit. |
| Compact vs DeepGrid7 | Does improved supervised capacity translate into DirectBackprop reward? | DeepGrid7 MonoKAN averaged about `4185.19`, above compact MonoKAN at about `4126.50`; DeepGrid7 KAN was less consistently improved. | `configs/paper_lost_sales_direct_target_capacity_medium.yaml` | 2025, 2026, 2027 | 1,000 products, batch 512, 500 epochs, static direct target stock, reward-only DirectBackprop, no persistent heuristic anchor or action cap. | `outputs/direct_target_capacity_medium` | CANONICAL DEVELOPMENT | HistoricalDemandBaseStock is `4264.52` under this protocol. Base SHA `1bf315b`; DeepGrid7 registry changes are in the cleanup implementation commit. |
| 1k vs 5k training scale | Does more training-product diversity improve the validated DeepGrid7 MonoKAN? | 5k DeepGrid7 MonoKAN reached `4192.54 +/- 1.10`, a modest improvement over the 1k result. | `configs/paper_lost_sales_direct_target_capacity_5k.yaml` | 2025, 2026, 2027 | Changes training products from 1,000 to 5,000 and batch from 512 to 2,500; architecture, optimizer, target semantics and 500-epoch budget remain fixed. | `outputs/direct_target_capacity_5k` | CANONICAL DEVELOPMENT | The batch change preserves approximately 50% pool coverage per update and is declared in the config. Base SHA `1bf315b`. |
| 500 vs 1,000 epochs | Is the remaining DeepGrid7 MonoKAN gap caused by insufficient optimization time? | 1,000 epochs reached about `4196.20` mean reward/period; improvement over 500 epochs was small and the historical gap remained. | `configs/paper_lost_sales_direct_target_capacity_5k_1000e.yaml` | 2025, 2026, 2027 | Same 5k protocol and DeepGrid7 architecture; only reward epochs change from 500 to 1,000. | `outputs/direct_target_capacity_5k_1000e` | CANONICAL DEVELOPMENT | Selected epochs were 925, 1000 and 925. Base SHA `1bf315b`. |
| Selective monotonicity | Does constraining an explicit demand-level feature instead of all history lags improve the direct-target policy? | Selective MonoKAN did not close the reward gap in the medium comparison. | `configs/paper_lost_sales_selective_monotonicity_medium.yaml` | 2025, 2026, 2027 | 1,000 products, 500 epochs; matched KAN/MonoKAN direct-target comparison. | `outputs/base_stock_output_paper_aligned/selective_identification_medium` | SUPPORTING | This YAML is semantically identical to the capacity-medium YAML; both paths are retained because historical outputs use their experiment-specific names. Base SHA `1bf315b`. |
| Residual order-up-to | Does a historical anchor plus an order-up-to decoder remove the raw-action clipping failure and support paired OOD analysis? | The clipping-dominated failure disappeared; KAN and MonoKAN were operationally near-tied, with MonoKAN structurally constrained but not reward-superior. | `configs/paper_lost_sales_factorial_small.yaml` | 2025-2034 | Four residual policies, 10 training seeds, eight fixed distribution shifts, paired product-level evaluation. | `outputs/monokan_final_distribution_shift_10seed` | SUPPORTING | Earlier project framing retained as an ablation, not the supervisor-preferred final direct-target formulation. Committed before base SHA `1bf315b`. |
| Normalized rolling vs static MonoKAN | Does recomputing target stock from causal rolling history improve over a static episode target? | Rolling DeepGrid7 MonoKAN `4108.17`; static `4105.89` in one seed. | `configs/paper_lost_sales_rolling_direct_target_dev_1k_normalized.yaml`; `configs/paper_lost_sales_static_direct_target_dev_1k_normalized.yaml` | 2025 | 1,000 products, 200 epochs, normalized observations, matched products and demand paths. | `outputs/direct_target_dev_1k_normalized/seed_2025/full_200e` | DEVELOPMENT | One-seed matched development evidence. Rolling/vectorized implementation is in the cleanup implementation commit. |
| Normalized rolling FCNN vs MonoKAN | Under identical normalized rolling direct-target semantics, how do FCNN and DeepGrid7 MonoKAN compare? | FCNN `4178.27 +/- 4.23`; DeepGrid7 MonoKAN `4108.48 +/- 5.90`. FCNN led by `69.78 +/- 1.70` reward/period. | `configs/paper_lost_sales_rolling_direct_target_dev_1k_normalized.yaml` | 2025, 2026, 2027 | Same 1k/200-epoch products, histories, economics, evaluation paths and direct-target decoder. | `outputs/direct_target_dev_1k_normalized/seed_2025/full_200e`; `seed_2026/full_200e`; `seed_2027/full_200e`; analysis in `outputs/direct_target_confirmation_3seed` | CANONICAL DEVELOPMENT | The FCNN advantage reproduced in all three seeds. Product generation and evaluation paths are fixed; initialization/training seeds vary. This is development-scale, not paper-scale evidence. |
| Temporal MonoKAN | Do five causal temporal summaries help DeepGrid7 MonoKAN use rolling history? | Temporal MonoKAN `4135.64 +/- 9.08`; gain over baseline MonoKAN `27.16 +/- 5.62`. It closed `38.98% +/- 8.40%` of the FCNN gap but remained `42.63 +/- 6.47` below FCNN. | `configs/paper_lost_sales_rolling_direct_target_temporal_dev_1k_normalized.yaml` | 2025, 2026, 2027 | Controlled change from raw lags to 32 lags plus mean-32/8/4, population std-32 and OLS trend-8; 1,000 products and 200 epochs. | `outputs/direct_target_temporal_dev_1k_normalized/seed_2025/full_200e`; `seed_2026/full_200e`; `seed_2027/full_200e`; analysis in `outputs/direct_target_confirmation_3seed` | CANONICAL DEVELOPMENT | The temporal gain was positive in all three seeds (`+33.38`, `+22.44`, `+25.65`). This supports a reproducible development-scale representation benefit, not paper-scale superiority or parity with FCNN. The model is monotone in its configured expanded coordinates, but `std_32` and `trend_8` are unconstrained functions of the raw lags, so it is not globally monotone in the original demand-history coordinates. |
| Raw-input normalization failure | Why did the first rolling/static DeepGrid7 experiment fail catastrophically? | Unnormalized observations were incompatible with spline grid `[-2, 4]`, causing extreme extrapolation; rewards are invalid for model comparison. | `configs/paper_lost_sales_rolling_direct_target_dev_1k.yaml`; `configs/paper_lost_sales_static_direct_target_dev_1k.yaml` | 2025 | 1,000 products, 200 epochs, `normalize_observations: false`. | `outputs/direct_target_dev_1k` | INVALID/DEBUG | Retained because saved metadata and regression tests reference these exact paths. See `docs/DEBUGGING.md`. |
| Raw-action DeepGrid7 MonoKAN | Can a matched raw-action MonoKAN reproduce the raw FCNN anchor? | The full run was not completed; the only artifact is a one-epoch MPS timing estimate of about 37 hours for 1,000 epochs. | `configs/paper_lost_sales_raw_monokan_deepgrid7_5k.yaml` | None completed | Intended 5,000 products, 1,000 epochs, rolling raw action, DeepGrid7 MonoKAN. | `outputs/raw_monokan_deepgrid7_5k` | INCOMPLETE | Runner/config/tests are retained for provenance but are not part of the final core workflow. Base SHA `1bf315b`. |

## Current Core And Supporting Story

The current core development comparison is:

1. paper-style rolling raw-action FCNN as the reference anchor;
2. direct-target FCNN;
3. DeepGrid7 MonoKAN, hard-monotone in each configured raw-lag and economics coordinate;
4. temporal-feature DeepGrid7 MonoKAN, hard-monotone only in its configured
   expanded-feature coordinates;
5. shared DirectBackprop, simulator, baselines and evaluation infrastructure.

Compact/deep capacity, training scale, optimization budget, rolling/static,
selective monotonicity and residual order-up-to results remain supporting
ablations. The predeclared three-seed confirmation is complete: temporal
features consistently improve DeepGrid7 MonoKAN, while FCNN remains consistently
better under this development protocol.

## Path Preservation

The rolling/static, raw/normalized, and selective/capacity config files are
intentionally not deduplicated. Saved metadata and historical reports refer to
their exact paths. Archive or consolidation should happen only after the final
report and confirmation protocol are frozen.

Implementation commit: `06553e3` (`feat: add direct-target capacity and rolling experiments`).

## End-To-End Monotonicity Scope

A read-only audit on the selected 2025--2027 checkpoints traced the deployed
mapping through normalization, target scaling and the order-up-to decoder. The
rolling policies use `S_t = m_0 r_theta(H_t, e)`, where `m_0` is the positive
mean of the product's initial 32-demand history and remains fixed during the
episode, followed by `q_t = max(S_t - I_t, 0)`.

For baseline DeepGrid7 MonoKAN, the construction certifies nondecreasing target
and order in every raw demand lag, selling price and lost-sales penalty;
nonincreasing target and order in purchase and holding costs; and nonincreasing
order in current inventory. Across three checkpoints, 300 existing validation
products, six trajectory periods and three quantile perturbation spans, the
audit found zero material violations. One of 518,400 raw-lag pairs differed by
`0.000122` target/order units against a strict float32 tolerance of `0.0001`;
the constrained edge-level maximum wrong increment was `2.98e-08`, so this is
treated as numerical accumulation rather than a structural reversal.

For temporal DeepGrid7 MonoKAN, the same economics and inventory guarantees
remain certified. Raw-demand monotonicity does not: changing one original lag
also changes unconstrained `std_32` and, for the latest eight lags, `trend_8`.
The end-to-end audit observed target reversals in 194,507 of 518,400 pairs
(`37.52%`) and final-order reversals in 194,494 pairs (`37.52%`), with a maximum
reversal of `52.80` units. Therefore, temporal MonoKAN must not be described as
globally monotone in original demand history even though its configured spline
coordinates satisfy their constraints. This audit changes the scope of the
structural claim, not the previously reported reward results.
