# Architecture

This project is currently organized around the paper-style no-lead-time lost-sales benchmark and residual order-up-to neural policies.

## Main Data Flow

```text
product generation
-> historical demand and observable product features
-> feature preprocessing and normalization
-> historical base-stock anchor S_hist
-> residual policy network r_theta(z)
-> target stock S_theta(z)
-> order-up-to decoder q_theta(y,z)=max(S_theta(z)-y,0)
-> lost-sales rollout
-> reward, operational metrics, and structural diagnostics
```

`y` is current inventory. `z` is observable context: historical demand and product economics. Current inventory is intentionally excluded from residual target-stock networks so the shared decoder owns the known inventory response.

## Shared Components

- Product generation: `inventory_control.products.PaperLostSalesProductGenerator`.
- Lost-sales dynamics: `inventory_control.paper_lost_sales.paper_lost_sales_step`.
- Evaluation demand paths: `inventory_control.paper_lost_sales.paper_evaluation_demand_path`.
- Feature construction and normalization: currently `inventory_control.policies.neural.fcnn.compute_paper_fcnn_feature_stats` and batch/context builders in `inventory_control.training.direct_backprop`.
- Historical base-stock anchor: `inventory_control.paper_lost_sales.historical_demand_base_stock_policy`, `non_omniscient_gamma_base_stock_policy`, and residual-anchor helpers used by training/evaluation.
- Residual order-up-to decoding: `paper_lost_sales_target_level`, `paper_lost_sales_raw_order`, and neutral aliases `paper_lost_sales_policy_order`, `paper_lost_sales_policy_rollout`.
- DirectBackprop training and model selection: `inventory_control.training.direct_backprop.train_paper_lost_sales_policy`.
- Benchmark evaluation and result writing: `inventory_control.idp_benchmark.run_idp_benchmark`.
- Target-stock diagnostics: emitted by `run_idp_benchmark` as `target_structure_diagnostics.csv`.
- Distribution-shift evaluation: `scripts/run_distribution_shift_validation.py` and `scripts/run_distribution_shift_multiseed.py`.
- Model metadata: `inventory_control.policies.neural.registry`.

## Architecture-Specific Components

- FCNN residual: `inventory_control.policies.neural.fcnn.make_paper_residual_order_up_to_fcnn_model`.
- Unconstrained KAN residual: `inventory_control.policies.neural.monokan.make_paper_residual_order_up_to_kan_model`.
- Constrained monotonic MLP residual: `inventory_control.policies.neural.constrained_mlp.make_paper_residual_order_up_to_constrained_mlp_model`.
- MonoKAN residual: `inventory_control.policies.neural.monokan.make_paper_residual_order_up_to_monokan_model`.
- MonoKAN spline certification: `sample_monokan_first_layer_splines` and benchmark export `monokan_spline_certification.csv`.

The constrained MLP implementation must not import MonoKAN modules. It should only rely on the shared residual decoder contract and shared training/evaluation code.

## Policy API

Residual order-up-to policies are ordinary `torch.nn.Module` objects.

Expected forward signature:

```python
def forward(self, features: torch.Tensor) -> torch.Tensor:
    ...
```

Tensor semantics:

- For residual target-stock policies, `features` is context only: normalized historical demand followed by normalized price, purchase cost, holding cost, and lost-sales penalty.
- The model returns a scalar residual factor per product.
- The shared decoder computes `S_theta = S_hist + residual_scale * (residual - residual_center) * historical_mean_demand`.
- The shared action decoder computes `order = max(S_theta - inventory, 0)` and then applies the configured deployment capacity guard.

Required residual metadata on the model:

- `paper_action_parameterization = "residual_order_up_to"`
- `paper_residual_center`
- `paper_residual_scale`
- `paper_residual_anchor`

Training/evaluation mode:

- Implement standard PyTorch `train()` and `eval()` behavior.
- If the architecture needs post-step projection, expose `apply_monotonic_constraints(self)`.

Checkpoint expectations:

- Checkpoints are saved as `state_dict()` files under `outputs/.../models/`.
- Canonical filenames live in `inventory_control.policies.neural.registry`.
- Do not change checkpoint filenames without migration notes.

Inventory monotonicity vs context monotonicity:

- Inventory monotonicity is enforced by the shared order-up-to decoder for all residual models.
- Context monotonicity is architecture-specific. `ConstrainedMLPResidualOrderUpTo` and `MonoKANResidualOrderUpTo` enforce selected comparative statics of `S_theta(z)`.

## Adding A Residual Model

1. Implement the architecture-specific module under `inventory_control/policies/neural/`.
2. Exclude current inventory from the residual network input.
3. Set the residual metadata attributes listed above.
4. Add a factory function accepting `(config: dict, metadata: dict | None = None)`.
5. Add a `NeuralPolicySpec` entry in `inventory_control/policies/neural/registry.py`.
6. Add benchmark training support in `inventory_control/idp_benchmark.py` if the policy needs custom settings or seed handling.
7. Add a checkpoint filename to the registry and preserve it in future runs.
8. Include the model in CLI comparisons through `--models`.
9. Reuse `target_structure_diagnostics.csv` for shared target-stock checks.
10. Add architecture-specific diagnostics only inside the architecture module or a clearly named evaluation helper.
11. Add tests for factory construction, decoder semantics, monotonicity if claimed, checkpoint naming, and benchmark smoke output.

See `examples/add_custom_residual_policy.py` for a minimal runnable example that reuses the shared training and decoder path.
