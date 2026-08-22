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
| Normalized rolling FCNN vs MonoKAN | Under identical normalized rolling direct-target semantics, how do FCNN and DeepGrid7 MonoKAN compare? | Rolling FCNN `4177.75`; rolling DeepGrid7 MonoKAN `4108.17`. | `configs/paper_lost_sales_rolling_direct_target_dev_1k_normalized.yaml` | 2025 | Same 1k/200-epoch products, histories, economics, evaluation paths and direct-target decoder. | `outputs/direct_target_dev_1k_normalized/seed_2025/full_200e` | DEVELOPMENT | One seed only; not evidence of a stable architecture ranking. |
| Temporal MonoKAN | Do five causal temporal summaries help DeepGrid7 MonoKAN use rolling history? | Reward `4141.55`, `+33.38` over the saved rolling MonoKAN and `36.20` below rolling FCNN. | `configs/paper_lost_sales_rolling_direct_target_temporal_dev_1k_normalized.yaml` | 2025 | Controlled change from raw lags to 32 lags plus mean-32/8/4, population std-32 and OLS trend-8; 1,000 products and 200 epochs. | `outputs/direct_target_temporal_dev_1k_normalized/seed_2025/full_200e` | DEVELOPMENT | **One-seed development evidence only.** It does not establish a cross-seed or paper-scale improvement. Temporal implementation is in the cleanup implementation commit. |
| Raw-input normalization failure | Why did the first rolling/static DeepGrid7 experiment fail catastrophically? | Unnormalized observations were incompatible with spline grid `[-2, 4]`, causing extreme extrapolation; rewards are invalid for model comparison. | `configs/paper_lost_sales_rolling_direct_target_dev_1k.yaml`; `configs/paper_lost_sales_static_direct_target_dev_1k.yaml` | 2025 | 1,000 products, 200 epochs, `normalize_observations: false`. | `outputs/direct_target_dev_1k` | INVALID/DEBUG | Retained because saved metadata and regression tests reference these exact paths. See `docs/DEBUGGING.md`. |
| Raw-action DeepGrid7 MonoKAN | Can a matched raw-action MonoKAN reproduce the raw FCNN anchor? | The full run was not completed; the only artifact is a one-epoch MPS timing estimate of about 37 hours for 1,000 epochs. | `configs/paper_lost_sales_raw_monokan_deepgrid7_5k.yaml` | None completed | Intended 5,000 products, 1,000 epochs, rolling raw action, DeepGrid7 MonoKAN. | `outputs/raw_monokan_deepgrid7_5k` | INCOMPLETE | Runner/config/tests are retained for provenance but are not part of the final core workflow. Base SHA `1bf315b`. |

## Current Core And Supporting Story

The current core development comparison is:

1. paper-style rolling raw-action FCNN as the reference anchor;
2. direct-target FCNN;
3. DeepGrid7 hard-monotone MonoKAN;
4. temporal-feature DeepGrid7 MonoKAN;
5. shared DirectBackprop, simulator, baselines and evaluation infrastructure.

Compact/deep capacity, training scale, optimization budget, rolling/static,
selective monotonicity and residual order-up-to results remain supporting
ablations. The temporal result must remain labelled as one-seed development
evidence until a predeclared confirmation is completed.

## Path Preservation

The rolling/static, raw/normalized, and selective/capacity config files are
intentionally not deduplicated. Saved metadata and historical reports refer to
their exact paths. Archive or consolidation should happen only after the final
report and confirmation protocol are frozen.

Implementation commit: `06553e3` (`feat: add direct-target capacity and rolling experiments`).
