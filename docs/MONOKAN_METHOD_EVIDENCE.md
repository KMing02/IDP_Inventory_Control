# MONOkan Method And Evidence Note

This note records the current defensible story for MONOkan in the lost-sales
inventory-control experiments. It separates the inventory-control contribution
from the MonoKAN architecture paper.

## Method Framing

The raw-action MONOkan policy is not the right final framing for the lost-sales
problem. Although it enforces a monotone relationship between inventory and
raw order, it does not encode the stronger economic shape of an order-up-to
policy. In previous diagnostics, raw MONOkan learned nearly flat oversized
orders and the deployment cap determined most actions.

The corrected policy class is residual order-up-to:

```text
context -> target stock correction -> target stock S_hat
order = max(S_hat - inventory, 0), capped only by the deployment guard
```

This preserves the core economic geometry: order should decrease with inventory
with slope close to -1 while active, then become zero above the target stock.

The MONOkan contribution is the structured function class used for the context
correction. The implementation uses cubic Hermite KAN edges with projected
monotonic constraints, following the key idea of the MonoKAN paper:

```text
cubic Hermite spline edges
+ monotone control-point and derivative projection
+ positive downstream combinations
= certified partial monotonicity for selected inputs
```

The implementation is therefore best described as a MONOkan-inspired
structure-informed residual order-up-to policy, not as a reproduction of the
MonoKAN paper's supervised benchmarks.

## Current Medium Diagnostic

Command:

```bash
uv run python scripts/run_idp_benchmark.py --mode structure-ls --profile compact --config configs/paper_lost_sales_fcnn_alignment_medium.yaml --models PaperFCNNResidualOrderUpTo,KANResidualOrderUpTo,MonoKANResidualOrderUpTo --output-dir /private/tmp/current_corrected_residual_medium_diagnostics --progress
```

Stable copied artifacts:

```text
outputs/monokan_final_evidence/medium_diagnostics/summary.csv
outputs/monokan_final_evidence/medium_diagnostics/training_log.csv
outputs/monokan_final_evidence/medium_diagnostics/metadata.json
outputs/monokan_final_evidence/plots/controlled_inventory_history_multiplier_deployed_order.png
outputs/monokan_final_evidence/plots/controlled_inventory_history_multiplier_target_stock.png
outputs/monokan_final_evidence/plots/policy_raw_vs_deployed_curves.png
outputs/monokan_final_evidence/plots/policy_inventory_demand_contours.png
```

| Policy | Reward/period | Service | Lost-sales rate | Order-up-to MAE | Cap-hit | Raw monotone violation | Train seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PaperFCNNResidualOrderUpTo | 4603.599 | 0.923 | 0.077 | 19.943 | 0.026 | 0.000 | 115.501 |
| KANResidualOrderUpTo | 4603.266 | 0.928 | 0.072 | 20.184 | 0.026 | 0.000 | 203.678 |
| MonoKANResidualOrderUpTo | 4603.436 | 0.922 | 0.078 | 19.710 | 0.026 | 0.000 | 218.990 |
| NonOmniscientOptimal | 4605.353 | 0.922 | 0.078 | 19.722 | 0.026 | 0.000 | n/a |
| OmniscientOptimal | 4628.960 | 0.927 | 0.073 | 3.503 | 0.000 | 0.000 | n/a |
| HistoricalDemandBaseStock | 4580.380 | 0.917 | 0.083 | 26.196 | 0.010 | 0.000 | n/a |

Reading: all residual learned policies are close to the non-omniscient
base-stock benchmark. FCNN is slightly best among learned policies on this
single medium validation, while MONOkan has the best order-up-to MAE among the
learned residual models.

## Compact Multi-Seed Diagnostic

Command:

```bash
uv run python scripts/run_monokan_multiseed_validation.py --config configs/paper_lost_sales_factorial_small.yaml --output-root /private/tmp/current_corrected_residual_multiseed --seeds 2025,2026,2027 --models PaperFCNNResidualOrderUpTo,KANResidualOrderUpTo,MonoKANResidualOrderUpTo --vary-product-seed
```

Stable copied artifacts:

```text
outputs/monokan_final_evidence/multiseed_compact/aggregate_summary.csv
outputs/monokan_final_evidence/multiseed_compact/per_seed_summary.csv
```

| Policy | Seeds | Mean reward/period | Std reward/period | Mean service | Mean lost-sales rate | Mean inventory | Mean train seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OmniscientOptimal | 3 | 4258.635 | 1153.799 | 0.922 | 0.078 | 72.858 | n/a |
| NonOmniscientOptimal | 3 | 4223.460 | 1153.154 | 0.911 | 0.089 | 65.676 | n/a |
| KANResidualOrderUpTo | 3 | 4223.235 | 1151.530 | 0.918 | 0.082 | 68.816 | 10.190 |
| MonoKANResidualOrderUpTo | 3 | 4222.943 | 1151.335 | 0.917 | 0.083 | 68.732 | 11.481 |
| PaperFCNNResidualOrderUpTo | 3 | 4216.222 | 1149.848 | 0.913 | 0.087 | 66.427 | 5.412 |
| HistoricalDemandBaseStock | 3 | 4185.438 | 1132.788 | 0.904 | 0.096 | 54.860 | n/a |

Reading: in compact multi-seed validation, KAN and MONOkan residual policies
beat FCNN residual. MONOkan is almost tied with unconstrained KAN, with a small
reward cost for certified monotonic structure.

## Current Claim

The current evidence supports this claim:

```text
Raw MONOkan was architecturally misaligned for lost-sales ordering.
Residual order-up-to fixes the economic action geometry.
KAN-style residual policies are competitive with the non-omniscient base-stock benchmark.
MONOkan adds certified monotonic context structure and interpretable splines while retaining near-KAN performance.
The current evidence does not show that monotonicity consistently improves reward over unconstrained KAN or FCNN.
```

This is a stronger and more honest story than claiming MONOkan is simply a
higher-performing model. Its value is structure, interpretability, and
shape-certified behavior with competitive operational reward.

## Structural Diagnostics Added

The benchmark now exports residual target-stock structure diagnostics for the
final comparison policies:

```text
target_structure_diagnostics.csv
monokan_spline_certification.csv
```

`target_structure_diagnostics.csv` evaluates the learned target stock rather
than only the deployed order. It reports finite-difference violation rates,
autograd directional derivative extrema, and residual correction magnitudes for:

```text
demand_level: nondecreasing
price: nondecreasing
purchase_cost: nonincreasing
holding_cost: nonincreasing
lost_sale_penalty: nondecreasing
```

`monokan_spline_certification.csv` samples each constrained first-layer
MONOkan edge and reports wrong-direction interval rates and maximum
wrong-direction increments over the spline domain.

The intended four-model residual comparison is:

```bash
uv run python scripts/run_idp_benchmark.py --mode structure-ls --profile full --config configs/paper_lost_sales_fcnn_alignment_medium.yaml --models PaperFCNNResidualOrderUpTo,KANResidualOrderUpTo,ConstrainedMLPResidualOrderUpTo,MonoKANResidualOrderUpTo --output-dir outputs/monokan_residual_four_model_validation --progress
```

The 10-seed paired robustness command is configured as:

```bash
uv run python scripts/run_distribution_shift_multiseed.py --config configs/paper_lost_sales_fcnn_alignment_medium.yaml --output-root outputs/monokan_distribution_shift_multiseed_10seed --profile full --models PaperFCNNResidualOrderUpTo,KANResidualOrderUpTo,ConstrainedMLPResidualOrderUpTo,MonoKANResidualOrderUpTo --eval-products 500 --diagnostic-products 200 --inventory-states 41 --vary-product-seed --progress
```

This command is intentionally a robustness diagnostic, not the final
paper-replication run.

## Larger Fast Validation

Command:

```bash
uv run python scripts/run_idp_benchmark.py --mode structure-ls --profile full --config configs/paper_lost_sales_fcnn_alignment_medium.yaml --models PaperFCNNResidualOrderUpTo,KANResidualOrderUpTo,MonoKANResidualOrderUpTo --output-dir /private/tmp/current_corrected_residual_large_fast --fast-diagnostics --progress
```

Stable copied artifacts:

```text
outputs/monokan_final_evidence/large_fast_validation/summary.csv
outputs/monokan_final_evidence/large_fast_validation/training_log.csv
outputs/monokan_final_evidence/large_fast_validation/metadata.json
```

| Policy | Reward/period | Service | Lost-sales rate | Mean order | Mean inventory | Train seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PaperFCNNResidualOrderUpTo | 4632.423 | 0.922 | 0.078 | 91.586 | 83.124 | 120.932 |
| KANResidualOrderUpTo | 4631.465 | 0.925 | 0.075 | 92.210 | 86.176 | 213.507 |
| MonoKANResidualOrderUpTo | 4632.360 | 0.921 | 0.079 | 91.592 | 82.670 | 222.198 |
| NonOmniscientOptimal | 4634.080 | 0.921 | 0.079 | 91.583 | 87.107 | n/a |
| OmniscientOptimal | 4656.072 | 0.925 | 0.075 | 92.040 | 88.517 | n/a |
| HistoricalDemandBaseStock | 4612.075 | 0.917 | 0.083 | 90.980 | 74.389 | n/a |

Reading: the larger validation strengthens the "competitive and structurally
clean" claim. FCNN residual remains slightly ahead of MONOkan by only about
0.06 reward/period, while MONOkan is almost tied with FCNN and ahead of
unconstrained KAN on this split. None of these learned residual policies closes
the full gap to the non-omniscient benchmark, but all are far above the
historical empirical baseline.

## Distribution-Shift Diagnostic

Command:

```bash
uv run python scripts/run_distribution_shift_validation.py --config configs/paper_lost_sales_fcnn_alignment_medium.yaml --source-benchmark-dir /private/tmp/current_corrected_residual_large_fast --output-dir outputs/monokan_distribution_shift_validation --profile full --eval-products 500 --diagnostic-products 200 --inventory-states 41
```

This diagnostic trains no new model. It loads the saved residual policies from
the larger validation, keeps the original training-distribution feature
normalization, and evaluates them on shifted validation product generators.

Shifts:

```text
in_distribution
low_cv
high_cv
low_mean_demand
high_mean_demand
high_lost_sale_penalty
high_holding_cost
mixed_stress
```

Stable artifacts:

```text
outputs/monokan_distribution_shift_validation/summary_all_shifts.csv
outputs/monokan_distribution_shift_validation/robustness_summary.csv
outputs/monokan_distribution_shift_validation/<shift>/summary.csv
outputs/monokan_distribution_shift_validation/<shift>/operational_metrics.csv
outputs/monokan_distribution_shift_validation/<shift>/predictions.csv
```

Robustness aggregate over the 8 shifts:

| Policy | Mean reward/period | Worst reward/period | Mean regret vs non-omniscient | Worst regret vs non-omniscient | Mean order-up-to MAE | Mean cap-hit | Raw monotone violation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MonoKANResidualOrderUpTo | 4937.557 | 2301.707 | 7.342 | 29.520 | 30.815 | 0.0525 | 0.000 |
| KANResidualOrderUpTo | 4936.189 | 2301.457 | 8.710 | 28.805 | 31.674 | 0.0531 | 0.000 |
| PaperFCNNResidualOrderUpTo | 4934.103 | 2301.688 | 10.796 | 50.600 | 31.251 | 0.0531 | 0.000 |

Per-shift learned-policy winners:

| Shift | Best learned policy | Reward/period |
| --- | --- | ---: |
| in_distribution | PaperFCNNResidualOrderUpTo | 4604.130 |
| low_cv | MonoKANResidualOrderUpTo | 4528.614 |
| high_cv | PaperFCNNResidualOrderUpTo | 4082.059 |
| low_mean_demand | MonoKANResidualOrderUpTo | 2301.707 |
| high_mean_demand | PaperFCNNResidualOrderUpTo | 9208.422 |
| high_lost_sale_penalty | PaperFCNNResidualOrderUpTo | 4560.927 |
| high_holding_cost | PaperFCNNResidualOrderUpTo | 4189.098 |
| mixed_stress | MonoKANResidualOrderUpTo | 6029.190 |

Reading: this is the first evidence that MONOkan may help the robustness story.
FCNN wins more individual shifts, but MONOkan has the best average reward and
lowest average regret over all shifts because it avoids the large mixed-stress
regret seen in FCNN. This should be presented cautiously: it is a one-run
saved-model shift diagnostic, not a final multi-seed robustness benchmark.

## Multi-Seed Distribution-Shift Diagnostic

Command:

```bash
uv run python scripts/run_distribution_shift_multiseed.py --config configs/paper_lost_sales_fcnn_alignment_medium.yaml --output-root outputs/monokan_distribution_shift_multiseed --seeds 2025,2026,2027 --profile full --models PaperFCNNResidualOrderUpTo,KANResidualOrderUpTo,MonoKANResidualOrderUpTo --eval-products 500 --diagnostic-products 200 --inventory-states 41 --vary-product-seed --progress
```

This diagnostic trains one residual policy set per seed, evaluates each saved
policy set across all 8 distribution shifts, and aggregates over 24 seed-shift
cases.

Stable artifacts:

```text
outputs/monokan_distribution_shift_multiseed/summary_all_seeds_shifts.csv
outputs/monokan_distribution_shift_multiseed/robustness_summary_all_seeds.csv
outputs/monokan_distribution_shift_multiseed/per_shift_summary_all_seeds.csv
outputs/monokan_distribution_shift_multiseed/winner_counts.csv
outputs/monokan_distribution_shift_multiseed/seed_<seed>/trained_models/summary.csv
outputs/monokan_distribution_shift_multiseed/seed_<seed>/distribution_shift/summary_all_shifts.csv
```

Aggregate over 3 seeds and 8 shifts:

| Policy | Seed-shift cases | Mean reward/period | Std reward/period | Worst reward/period | Mean regret vs non-omniscient | Worst regret vs non-omniscient | Mean order-up-to MAE | Raw monotone violation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MonoKANResidualOrderUpTo | 24 | 4591.266 | 1840.372 | 2049.951 | 6.701 | 29.522 | 32.263 | 0.000 |
| PaperFCNNResidualOrderUpTo | 24 | 4589.739 | 1839.575 | 2050.531 | 8.229 | 50.600 | 32.246 | 0.000 |
| KANResidualOrderUpTo | 24 | 4588.209 | 1839.849 | 2048.280 | 9.759 | 36.180 | 33.321 | 0.000 |

Winner counts over 24 seed-shift cases:

| Policy | Wins |
| --- | ---: |
| PaperFCNNResidualOrderUpTo | 15 |
| MonoKANResidualOrderUpTo | 8 |
| KANResidualOrderUpTo | 1 |

Reading: the multi-seed shift result supports a careful robustness claim, not a
blanket accuracy claim. FCNN wins most individual seed-shift cases, especially
ordinary in-distribution and single-axis shifts. MONOkan has the best average
reward and lowest average and worst regret across all seed-shift cases because
it is stronger on the harder combined `mixed_stress` shift and remains
competitive elsewhere. This is the best current evidence that monotonic
structure can reduce severe shifted-distribution failure modes.
