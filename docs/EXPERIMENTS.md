# Final Experiment Index

This file indexes evidence that is relevant to the final IDP report. Historical exploratory branches are preserved in Git history, not in the current tree.

## Evidence Status

- **FINAL MAIN**: central evidence for the report's main comparison.
- **FINAL ABLATION**: retained evidence that supports interpretation of the main result.
- **DEBUGGING ONLY**: retained only to explain why an invalid run is excluded.

## Final Evidence

| Experiment | Report role | Main result | Config | Seeds | Output | Status |
|---|---|---:|---|---|---|---|
| Paper raw-action FCNN anchor | Shows the DirectBackprop and simulator pipeline can reproduce a strong paper-style FCNN anchor. | `4580.54 +/- 4.09` reward/period | `configs/paper_lost_sales_raw_fcnn_replication_5k.yaml` | 2025, 2026, 2027 | `outputs/paper_raw_fcnn_replication_5k` | FINAL ABLATION |
| Rolling target-stock FCNN vs DeepGrid7 MonoKAN | Main matched comparison under the deployed target-stock formulation. | FCNN `4178.27 +/- 4.23`; MonoKAN `4108.48 +/- 5.90` | `configs/paper_lost_sales_rolling_direct_target_dev_1k_normalized.yaml` | 2025, 2026, 2027 | `outputs/direct_target_dev_1k_normalized`; analysis in `outputs/direct_target_confirmation_3seed` | FINAL MAIN |
| Temporal DeepGrid7 MonoKAN | Tests whether causal temporal summaries improve reward. | `4135.64 +/- 9.08`; improvement over baseline MonoKAN `+27.16 +/- 5.62` | `configs/paper_lost_sales_rolling_direct_target_temporal_dev_1k_normalized.yaml` | 2025, 2026, 2027 | `outputs/direct_target_temporal_dev_1k_normalized`; analysis in `outputs/direct_target_confirmation_3seed` | FINAL ABLATION |
| Compact vs DeepGrid7 MonoKAN | Tests whether MonoKAN capacity contributes to the reward gap. | DeepGrid7 MonoKAN about `4185.19`; compact MonoKAN about `4126.50` | `configs/paper_lost_sales_direct_target_capacity_medium.yaml` | 2025, 2026, 2027 | `outputs/direct_target_capacity_medium` | FINAL ABLATION |
| 1k vs 5k training scale | Tests whether more training products improve DeepGrid7 MonoKAN. | 5k DeepGrid7 MonoKAN `4192.54 +/- 1.10` | `configs/paper_lost_sales_direct_target_capacity_5k.yaml` | 2025, 2026, 2027 | `outputs/direct_target_capacity_5k` | FINAL ABLATION |
| 500 vs 1,000 epochs | Tests whether optimization time explains the gap. | 1,000 epochs about `4196.20` | `configs/paper_lost_sales_direct_target_capacity_5k_1000e.yaml` | 2025, 2026, 2027 | `outputs/direct_target_capacity_5k_1000e` | FINAL ABLATION |
| Supervised target-fitting diagnostic | Separates target-map representational capacity from reward-only DirectBackprop. | DeepGrid7 improves target fitting relative to compact models. | `configs/paper_lost_sales_base_stock_small.yaml` | 2025, 2026, 2027 | `outputs/supervised_target_diagnostic/capacity_canonical` | FINAL ABLATION |
| Raw-input normalization failure | Explains an invalid catastrophic MonoKAN run. | Unnormalized observations fell outside spline grid `[-2, 4]`; not policy-performance evidence. | removed from current configs; see Git history | 2025 | `outputs/direct_target_dev_1k` | DEBUGGING ONLY |

## Final Formulation

The final target-stock policies use:

```text
r_theta = softplus(model(H_t, economics))
S_t = m_0 * r_theta
q_t = max(S_t - I_t, 0)
```

`m_0` is fixed from the product's initial demand history during rolling evaluation. Inventory is excluded from the target network and enters only through the decoder.

## Structural Claims

The baseline DeepGrid7 MonoKAN is the final certified monotone model. Its end-to-end deployed mapping is certified nondecreasing in all 32 raw demand lags, selling price, and lost-sales penalty; nonincreasing in purchase and holding costs; and nonincreasing in current inventory through `q_t = max(S_t - I_t, 0)`.

The temporal MonoKAN is not globally monotone in the original raw demand-history coordinates. The temporal summaries improve reward, but `std_32` and `trend_8` are unconstrained functions of the original demand lags; the audit observed about `37.52%` raw-demand monotonicity reversals for that variant.

## Outputs Not Deleted Yet

The following output roots are tied only to branches removed from the current code path and are safe to delete after the final report and presentation no longer need historical cross-checks:

- `outputs/monokan_final_distribution_shift_10seed`
- `outputs/monokan_distribution_shift_validation`
- `outputs/monokan_distribution_shift_multiseed`
- `outputs/monokan_distribution_shift_factorial_small_10seed`
- `outputs/monokan_optimization_sweep`
- `outputs/raw_monokan_deepgrid7_5k`
- `outputs/base_stock_output_paper_aligned/selective_identification_medium`
- `outputs/idp_benchmark`
