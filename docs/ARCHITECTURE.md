# Final Code Path

The current tree is organized around the final report, not historical experiment coverage.

## Core Data And Simulation

- `inventory_control.products`: Maggiar-style product generation with observable historical demand and hidden Gamma demand parameters.
- `inventory_control.paper_lost_sales`: lost-sales reward evaluation and base-stock baselines.
- `inventory_control.training.direct_backprop`: differentiable DirectBackprop rollouts, target-stock decoding, observation-window handling, and training loops.
  - `_paper_windowed_target_levels` dispatches between a sliding 32-period window and an expanding window over everything observed so far.
  - `_paper_smooth_rolling_targets` optionally averages the target over elapsed periods. A target recomputed each period jitters, and with `q_t = max(S_t - I_t, 0)` that jitter is asymmetric.
  - `per_product_reward_weighting` equalises each product's contribution to the gradient. The unweighted reward sum is dominated by a few large price-times-demand products.

## Retained Neural Policies

- `inventory_control.policies.neural.fcnn`: raw-action FCNN anchor and direct target-stock FCNN.
- `inventory_control.policies.neural.monokan`: Hermite-spline KAN/MonoKAN implementation, the soft-quantile target policy, and target-stock factories.
  - `HermiteKANLayer.adapt_grid_to_samples` fits knots to per-feature training quantiles; the fixed `[-2, 4]` grid leaves small-magnitude features inside a single interval.
  - `MonoKANSoftQuantileTargetPolicy` is the final policy. It decodes a target as a convex combination of the order statistics of observed demand, at a level produced by a monotone KAN over economics only. `level_network` swaps that level network for an unconstrained KAN or an MLP, which is how the cost of the certificate is measured with the decoder held fixed.
- `inventory_control.policies.neural.registry`: final report-facing model names and checkpoint filenames.

## Final Experiment Modules

- `inventory_control.base_stock_output_experiment`: general direct target-stock runner used for FCNN, compact KAN/MonoKAN, and DeepGrid7 MonoKAN.
- `inventory_control.rolling_direct_target_matched`: final matched protocol checks and raw-FCNN/direct-target result pairing.
- `inventory_control.supervised_target_diagnostic`: supervised empirical-newsvendor target-fitting diagnostic.
- `inventory_control.direct_target_structural_audit`: checkpoint-only monotonicity audit for target-stock policies.
- `inventory_control.monokan_interpretability`: partial-dependence evidence for a trained checkpoint, verifying each certified direction as it reports it.
- `inventory_control.evaluation.torch_policy`: batched torch-policy evaluator shared by final target-stock runs.

## Final CLI Scripts

The `scripts/` directory contains thin wrappers around those modules. Experiment logic should stay in `inventory_control/` so it remains importable and testable.
