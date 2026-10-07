# Code Map

The tree is organized around the final report. Experiment logic lives in `inventory_control/` so it stays importable
and testable; `scripts/` holds thin command-line wrappers.

## Setting

The synthetic lost-sales setting of Maggiar et al. (2025), "Structure-Informed Deep Reinforcement Learning for
Inventory Management": no lead time, lost sales, Gamma demand with hidden parameters, and DirectBackprop training,
i.e. gradients of the simulated profit flow through the simulator into the policy. A policy observes on-hand stock,
the demand history and the product economics, never the hidden demand parameters or future demand. Products are
drawn from the paper's Table 1 distribution:

```text
selling price p         ~ Exponential(mean=100)
purchase cost c         = p * U, U ~ Uniform(0, 1)
holding cost h          ~ Exponential(mean=5)
lost-sale penalty k     = 10 * U, U ~ Uniform(0, 1)
mean demand mu          ~ Exponential(mean=100)
demand CV nu            ~ Uniform(0, 1)
```

## Simulation, Products And Baselines

- `inventory_control.products`: product generation with observable history and hidden Gamma parameters.
- `inventory_control.paper_lost_sales`: the lost-sales step, the per-product evaluator, and the classical
  base-stock rules. `WindowedBaseStockPolicy` re-estimates a percentile or Gamma-fit level from the first 32 days,
  the last 32 days, or all demand so far, so learned policies are compared with rules that see the same data.
- `inventory_control.paper_configs`: config loading, validation and `dotted.key=value` overrides.
- `inventory_control.envs`, `inventory_control.evaluation`: generic simulators, metrics and the batched evaluator of
  torch policies used by every final run.

## Policies

- `inventory_control.policies.neural.monokan`: Hermite-spline KAN and MonoKAN.
  - `_project_edge_coefficients` projects every spline onto the Fritsch-Carlson monotone set after each optimizer
    step; this is the certificate.
  - `HermiteKANLayer.adapt_grid_to_samples` places knots at per-feature training quantiles.
  - `make_paper_raw_action_monokan_model_for_products` builds the main design: the order from 37 inputs (on-hand stock,
    32 demand lags, 4 economics), every direction certified. `monotone=False` gives the identical KAN without it.
  - `MonoKANSoftQuantileTargetPolicy` is the alternative design: an order-up-to target that is a smoothed quantile of
    observed demand at a level chosen by a small network over the economics. `level_network` swaps MonoKAN for an
    unconstrained KAN or an MLP with the decoder held fixed.
- `inventory_control.policies.neural.fcnn`: the paper's feed-forward network and its feature builder.
- `inventory_control.policies.neural.registry`: report-facing model names and checkpoint filenames.
- `inventory_control.training.direct_backprop`: differentiable rollouts and training loops, target-stock decoding,
  sliding and expanding observation windows, and per-product reward weighting.

## Experiments And Analysis

- `inventory_control.direct_quantity_experiment`: the main comparison (MonoKAN, the same KAN without the certificate,
  the paper FCNN), the stock sweep that counts rule violations, the implied order-up-to test and decision speed.
- `inventory_control.base_stock_output_experiment`: the target-stock runner behind the alternative design and the
  historical DeepGrid7 and FCNN direct target-stock runs.
- `inventory_control.classical_windows`: classical rules in all three information windows.
- `inventory_control.analysis`: regret, paired product-by-product tests, and summaries of saved runs.
- `inventory_control.direct_target_structural_audit`: checkpoint-only monotonicity audit of target-stock policies.
- `inventory_control.monokan_interpretability`: partial dependence of the alternative design's level network and its
  match with the analytic newsvendor fractile.

## Scripts

| Script | Purpose |
|---|---|
| `run_direct_quantity_comparison.py` | main design: `train`, `summarize`, `interpret`, `curves` |
| `run_soft_quantile_monokan.py` | alternative design, one seed; `--set` runs the controls |
| `compare_target_stock_runs.py` | regret of saved runs and paired comparisons between them |
| `run_classical_windows.py` | classical rules in every information window |
| `run_direct_target_structural_audit.py` | certificate audit from checkpoints |
| `run_monokan_interpretability.py` | partial dependence and newsvendor match |
| `run_base_stock_output_experiment.py` | general target-stock runner (historical DeepGrid7 / FCNN rows) |
| `run_paper_lost_sales_baselines.py` | the paper's fixed-history base-stock benchmarks |
| `train_paper_lost_sales_fcnn.py`, `evaluate_paper_lost_sales_fcnn.py` | paper-style raw-action FCNN replication |

## Configs

| Config | Used for |
|---|---|
| `paper_lost_sales.yaml` | base paper configuration |
| `paper_lost_sales_rolling_direct_target_dev_1k_normalized.yaml` | main design, classical windows, historical first results (1,000 products, 300 validation products) |
| `paper_lost_sales_soft_quantile_monokan.yaml` | final alternative design (5,000 products, weighted objective) |
| `paper_lost_sales_soft_quantile_monokan_profit.yaml` | alternative design at 1,000 products with the profit objective; matched comparisons |
| `paper_lost_sales_raw_fcnn_replication_5k.yaml` | replication of the paper's raw-action FCNN |
| `paper_lost_sales_debug.yaml` | tiny config for tests |
