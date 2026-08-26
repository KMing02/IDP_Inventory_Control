# Final Code Path

The current tree is organized around the final report, not historical experiment coverage.

## Core Data And Simulation

- `inventory_control.products`: Maggiar-style product generation with observable historical demand and hidden Gamma demand parameters.
- `inventory_control.paper_lost_sales`: lost-sales reward evaluation and base-stock baselines.
- `inventory_control.training.direct_backprop`: differentiable DirectBackprop rollouts, target-stock decoding, rolling history updates, and training loops.

## Retained Neural Policies

- `inventory_control.policies.neural.fcnn`: raw-action FCNN anchor and direct target-stock FCNN.
- `inventory_control.policies.neural.monokan`: Hermite-spline KAN/MonoKAN implementation and direct target-stock factories.
- `inventory_control.policies.neural.registry`: final report-facing model names and checkpoint filenames.

## Final Experiment Modules

- `inventory_control.base_stock_output_experiment`: general direct target-stock runner used for FCNN, compact KAN/MonoKAN, and DeepGrid7 MonoKAN.
- `inventory_control.rolling_direct_target_matched`: final matched protocol checks and raw-FCNN/direct-target result pairing.
- `inventory_control.supervised_target_diagnostic`: supervised empirical-newsvendor target-fitting diagnostic.
- `inventory_control.direct_target_structural_audit`: checkpoint-only monotonicity audit for target-stock policies.
- `inventory_control.evaluation.torch_policy`: batched torch-policy evaluator shared by final target-stock runs.

## Final CLI Scripts

The `scripts/` directory contains thin wrappers around those modules. Experiment logic should stay in `inventory_control/` so it remains importable and testable.
