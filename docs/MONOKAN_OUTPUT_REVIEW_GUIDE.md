# MONOkan Output Review Guide

This guide points to the outputs worth reviewing after the compact four-model
validation, 10-seed paired distribution-shift run, and evidence analysis.

## Start Here

Read the generated report first:

```text
outputs/analysis/MONOKAN_EVIDENCE_REPORT.md
```

Main conclusion from the current compact evidence:

```text
MonoKAN and unconstrained KAN are practically equivalent under the proposed
±5 reward/period margin, while KAN is statistically slightly higher by a very
small amount.
```

Numerically:

```text
MonoKAN - KAN mean delta: -0.259567 reward/period
95% hierarchical CI: [-0.590681, -0.056785]
Delta as percent of KAN reward: -0.005580%
Classification: practically_equivalent
```

This supports the cautious story:

```text
MonoKAN matches unconstrained KAN within a practical margin while adding
structure on the target-stock function.
```

It does not support:

```text
MonoKAN consistently outperforms KAN.
```

## Core Tables

### 1. KAN vs MonoKAN equivalence

```text
outputs/analysis/monokan_vs_kan_equivalence.csv
```

Use this for the main statistical statement.

Important columns:

- `mean_delta_reward`: average `MonoKAN - KAN`.
- `ci95_low`, `ci95_high`: hierarchical paired bootstrap interval.
- `mean_delta_pct_of_kan`: effect size relative to KAN reward.
- `equivalence_classification`: current practical-equivalence decision.
- `statistically_excludes_zero`: whether the CI excludes zero.
- `monokan_product_win_rate`, `monokan_product_loss_rate`, `product_tie_rate`.
- `worst_product_delta`, `worst_product_seed`, `worst_product_shift`.
- `worst_seed_shift_delta`, `worst_seed_shift_seed`, `worst_seed_shift_shift`.

### 2. KAN vs MonoKAN by shift

```text
outputs/analysis/monokan_vs_kan_by_shift.csv
```

Use this to see where MonoKAN helps or hurts.

Important columns:

- `shift`
- `mean_delta_reward`
- `median_delta_reward`
- `ci95_low`, `ci95_high`
- `mean_delta_pct_of_kan`
- `monokan_product_win_rate`
- `bottom_decile_delta`
- `worst_product_delta`

Interpretation:

- Positive delta means MonoKAN beats KAN.
- Negative delta means KAN beats MonoKAN.
- Current pattern: differences are very small; KAN is usually slightly ahead,
  but MonoKAN wins some seed-shift cells.

### 3. Target-stock monotonicity

```text
outputs/analysis/target_monotonicity_by_model_feature.csv
```

Use this for the structural claim.

This audits the complete target:

```text
S_theta = historical_base_stock + scaled_residual
```

not only the residual network.

Important columns:

- `policy_name`
- `seed`
- `shift`
- `feature`
- `expected_direction`
- `domain`: `local_in_domain` or `stress_extrapolation`
- `finite_diff_violation_rate`
- `finite_diff_max_wrong_direction`
- `autograd_wrong_sign_rate`
- `autograd_min_directional_derivative`
- `autograd_max_directional_derivative`
- `evaluated_states`

Features checked:

- `demand_level`: increasing
- `selling_price`: increasing
- `lost_sales_penalty`: increasing
- `purchase_cost`: decreasing
- `holding_cost`: decreasing

Important caution:

Autograd derivatives condition on the current batch anchor tensor, while finite
differences recompute the full target under perturbed product features. Use
finite differences as the stronger complete-target check.

### 4. Clipping and action geometry

```text
outputs/analysis/clipping_and_target_summary.csv
```

Use this to verify that residual order-up-to fixed the old raw-action failure.

Important columns:

- `raw_order_mean`, `raw_order_q10`, `raw_order_median`, `raw_order_q90`
- `deployed_order_mean`, `deployed_order_q10`, `deployed_order_median`, `deployed_order_q90`
- `cap_hit_rate`
- `mean_abs_raw_deployed_difference`
- `materially_changed_by_guard_rate`
- `target_stock_mean`, `target_stock_q10`, `target_stock_median`, `target_stock_q90`
- `order_up_to_zero_crossing_mean`

Interpretation:

- Low `cap_hit_rate` means the cap is mostly a feasibility guard.
- High `cap_hit_rate` would suggest clipping is again shaping behavior.
- Current residual models have much lower clipping than the old raw MONOkan
  failure mode.

### 5. Constrained MLP diagnosis

```text
outputs/analysis/constrained_mlp_diagnosis.csv
```

Use this to explain why the constrained MLP is not competitive in the current
run.

Current diagnosis:

```text
The constrained MLP is conservative: higher service level, lower lost-sales
rate, larger target/order-up-to error, and higher inventory/target proxies.
This suggests excessive shape restriction or scaling/conservatism. Gradient
norms are not logged, so optimization difficulty remains unresolved.
```

### 6. MonoKAN spline audit

```text
outputs/analysis/monokan_spline_audit.csv
```

Use this for the spline-level implementation check.

Important columns:

- `seed`
- `feature_index`
- `expected_direction`
- `violating_interval_fraction`
- `max_wrong_direction_increment`
- `left_extrapolation_violating_fraction`
- `right_extrapolation_violating_fraction`
- `first_layer_only`
- `complete_model_certified_from_this_alone`

Important caution:

Spline checks alone do not certify the complete model. They certify inspected
spline components and extrapolation behavior. The complete target-stock audit is
the stronger end-to-end diagnostic.

## Main Validation Run

Compact four-model validation:

```text
outputs/idp_benchmark/factorial_small_four_model/summary.csv
```

Key reward results:

```text
KANResidualOrderUpTo             3525.06
MonoKANResidualOrderUpTo         3525.00
NonOmniscientOptimal             3522.41
PaperFCNNResidualOrderUpTo       3516.17
HistoricalDemandBaseStock        3490.80
ConstrainedMLPResidualOrderUpTo  3424.60
OmniscientOptimal                3561.14
```

Also inspect:

```text
outputs/idp_benchmark/factorial_small_four_model/target_structure_diagnostics.csv
outputs/idp_benchmark/factorial_small_four_model/monokan_spline_certification.csv
outputs/idp_benchmark/factorial_small_four_model/training_log.csv
outputs/idp_benchmark/factorial_small_four_model/metadata.json
```

## Visualizations

Open these plots:

```text
outputs/idp_benchmark/factorial_small_four_model/plots/policy_inventory_demand_contours.png
outputs/idp_benchmark/factorial_small_four_model/plots/policy_raw_vs_deployed_curves.png
outputs/idp_benchmark/factorial_small_four_model/plots/controlled_inventory_history_multiplier_deployed_order.png
outputs/idp_benchmark/factorial_small_four_model/plots/controlled_inventory_history_multiplier_target_stock.png
```

How to read them:

- `policy_inventory_demand_contours.png`: deployed order as a function of
  inventory and historical demand scale.
- `policy_raw_vs_deployed_curves.png`: whether the feasibility guard materially
  changes raw proposals.
- `controlled_inventory_history_multiplier_deployed_order.png`: controlled
  deployed-order surface under historical-demand scaling.
- `controlled_inventory_history_multiplier_target_stock.png`: controlled
  target-stock surface. This is the most relevant plot for the MonoKAN story.

## Distribution-Shift Outputs

10-seed compact shift run:

```text
outputs/monokan_distribution_shift_factorial_small_10seed/robustness_summary_all_seeds.csv
outputs/monokan_distribution_shift_factorial_small_10seed/per_shift_summary_all_seeds.csv
outputs/monokan_distribution_shift_factorial_small_10seed/winner_counts.csv
outputs/monokan_distribution_shift_factorial_small_10seed/paired_product_reward_differences.csv
outputs/monokan_distribution_shift_factorial_small_10seed/paired_product_reward_summary.csv
```

High-level aggregate:

```text
KANResidualOrderUpTo             4651.68
MonoKANResidualOrderUpTo         4651.42
PaperFCNNResidualOrderUpTo       4642.17
ConstrainedMLPResidualOrderUpTo  4494.75
```

Winner counts:

```text
KANResidualOrderUpTo             33
MonoKANResidualOrderUpTo         28
PaperFCNNResidualOrderUpTo       17
ConstrainedMLPResidualOrderUpTo   2
```

Use the generated analysis tables rather than the raw paired CSV for final
interpretation, because the analysis tables use the direct KAN-vs-MonoKAN
hierarchical bootstrap.

## Proposed Frozen Protocol

Review:

```text
docs/FROZEN_EXPERIMENT_PROTOCOL.md
```

Before any final/paper-scale run, decide whether to accept:

- the four-model comparison;
- the shift set;
- the seed policy;
- the practical equivalence margin;
- the protected final evaluation rule.

## Recommended Thesis Framing

Use:

```text
We develop a structure-informed residual order-up-to policy. The order-up-to
decoder encodes the known inventory-response geometry, and MonoKAN constrains
selected comparative statics of the learned target stock. In compact paired
experiments, MonoKAN is practically equivalent to unconstrained KAN while
providing stronger structural guarantees.
```

Avoid:

```text
MonoKAN consistently outperforms KAN.
Monotonicity causes better robustness.
The compact run proves paper-scale performance.
```
