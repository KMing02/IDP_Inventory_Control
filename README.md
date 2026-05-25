# Structure-Informed Neural Inventory Control Benchmark

This repository is a research codebase for structure-informed neural inventory control. The goal is to compare neural policy architectures later, including unconstrained MLPs, monotonic MLPs, MonoKAN, and optionally MLPs with monotonicity penalties.

This is not a generic reinforcement learning benchmark. The intended training direction is DirectBackprop-style differentiable policy optimization through inventory simulators, where domain structure is injected through inventory balance equations, lead-time pipeline states, monotonicity of order quantities with respect to inventory position, and classical heuristic baselines.

The current priority is replicating the lost-sales setting from Maggiar et al. (2025), "Structure-Informed Deep Reinforcement Learning for Inventory Management". Generic benchmark scenarios remain in the repository, but they are secondary until the lost-sales replication is stable.

The current implementation includes clean true simulators, demand processes, heuristic baselines, evaluation, plotting, tests, a paper-style lost-sales replication layer, and initial unconstrained FCNN training paths. Monotone MLP and MonoKAN policies are future work.

## Project Status

Current status:

- Core single-item lost-sales simulators are implemented and tested.
- Paper-style no-lead-time lost-sales product generation, reward evaluation, base-stock baselines, and FCNN training are implemented.
- FCNN evaluation is comparable against baselines using the same test products, seeds, horizon, and burn-in.
- Failure-analysis CSV generation is available for policy comparison and bottom-product inspection.

Current limitations:

- Paper lost-sales product generation follows Table 1 of Maggiar et al. for the synthetic LS setting, but the policy architecture is still a simplified FCNN rather than the paper's causal-CNN history encoder plus MLP.
- `OracleGammaBaseStock` is an oracle critical-quantile base-stock benchmark, not a globally optimal dynamic policy.
- Differentiable training currently exists for the FCNN path only.
- MonoMLP, MonoKAN, multi-echelon, perishability, and procurement-removal settings are not implemented yet.
- Generated outputs and model checkpoints are intentionally ignored by Git and are not required for tests.

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For runtime-only use without test/neural dependencies:

```bash
python -m pip install -e .
```

For neural-only dependencies when not using `.[dev]`:

```bash
python -m pip install -e ".[neural]"
```

## Environments

`LostSalesZeroLeadTimeEnv`

- State: on-hand inventory.
- Action: scalar order quantity.
- Orders arrive immediately.
- Demand not satisfied from available inventory is lost.

`LostSalesLeadTimeEnv`

- State: `[on_hand_inventory, pipeline_order_1, ..., pipeline_order_L]`.
- `pipeline_order_1` arrives next period.
- Action: scalar order quantity.
- New orders enter the last pipeline position.

`DualSourcingEnv`

- State: `[on_hand_inventory, regular_pipeline..., expedited_pipeline...]`.
- Action: `[regular_order_quantity, expedited_order_quantity]`.
- Regular supply is cheaper and slower.
- Expedited supply is faster and more expensive.

All environments use lost-sales dynamics:

```text
available = on_hand + arrivals
sales = min(available, demand)
lost_sales = max(demand - available, 0)
ending_inventory = max(available - demand, 0)
```

Costs include holding cost, lost-sales cost, order costs, and optional fixed order costs.

## Demand Processes

Implemented demand models:

- `PoissonDemand`
- `NormalDemand`, rounded and clipped at zero
- `NegativeBinomialDemand`
- `SeasonalPoissonDemand`
- `RegimeSwitchingPoissonDemand`
- `DeterministicDemand` for tests

## Baseline Policies

Implemented heuristic baselines:

- `BaseStockPolicy`: zero-lead-time order-up-to policy.
- `InventoryPositionBaseStockPolicy`: lead-time base-stock policy using on-hand plus pipeline inventory.
- `DualSourcingHeuristicPolicy`: configurable regular and expedited order-up-to heuristic.
- `ConstantOrderPolicy`
- `RandomPolicy`

The dual-sourcing heuristic is a baseline, not an optimality claim.

## Run Tests

```bash
python -m pytest
```

## Quick Smoke Test

Run a tiny end-to-end check using `configs/paper_lost_sales_debug.yaml`:

```bash
python scripts/run_quick_smoke_test.py
```

This command generates a tiny paper-style product set, evaluates historical/oracle base-stock baselines, trains a miniature FCNN for a few epochs, and evaluates it. It does not require any pre-existing files in `outputs/`.

## Reproducible Commands

From a clean checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
python scripts/run_quick_smoke_test.py
python scripts/run_paper_lost_sales_baselines.py
python scripts/train_paper_lost_sales_fcnn.py
python scripts/evaluate_paper_lost_sales_fcnn.py
python scripts/analyze_paper_lost_sales_fcnn_failures.py
```

## Reward And Cost Naming

The repository uses two separate conventions:

- Generic inventory simulators use cost minimization:
  - `total_cost`
  - `mean_total_cost`
  - `mean_cost_per_period`
  - lower is better

- Maggiar-style paper replication uses reward/profit maximization:
  - `total_reward`
  - `mean_total_reward`
  - `mean_reward_per_period`
  - higher is better

For paper-style neural training, the optimizer minimizes:

```text
training_loss = -mean_total_reward
```

so training loss going down corresponds to reward going up.

## Primary Replication Target: Maggiar Et Al. Lost-Sales Setting

The main replication config is:

```text
configs/paper_lost_sales.yaml
```

It defines the lost-sales history length, training/evaluation horizons, FCNN architecture, Gamma-demand product generation, and local runtime caps.

The main paper config uses the synthetic product distribution from Table 1 of Maggiar et al.:

```text
price p                 ~ Exponential(mean=100)
purchase cost c         = p * U, U ~ Uniform(0, 1)
holding cost h          ~ Exponential(mean=5)
lost-sale penalty k     = 10 * U, U ~ Uniform(0, 1)
mean demand mu          ~ Exponential(mean=100)
demand CV nu            ~ Uniform(0, 1)
```

The sampled `mu` and `nu` define hidden Gamma demand parameters. For each product, the generator samples one full demand trace. The first `H` values are stored as observable historical demand, and the future values are used by training/evaluation rollouts. Policies do not receive the hidden Gamma parameters or future demand trace.

Run paper-style no-lead-time LS baselines:

```bash
python scripts/run_paper_lost_sales_baselines.py
```

This writes:

```text
outputs/results/paper_lost_sales_baselines.csv
```

The two explicit baselines are:

- `OracleGammaBaseStock`: uses hidden true Gamma parameters and is an oracle.
- `HistoricalDemandBaseStock`: uses only historical demand observations and is non-omniscient.

Train the paper-style FCNN:

```bash
python scripts/train_paper_lost_sales_fcnn.py
```

Training uses observable product features only: full historical demand, current inventory, and product economics. Feature scaling is computed from training products only, the output is clipped by a configurable multiple of historical mean demand, and the saved checkpoint is selected by validation true reward. Training writes:

```text
outputs/results/paper_lost_sales_fcnn_training_log.csv
outputs/plots/paper_lost_sales_fcnn_training_curve.png
outputs/models/paper_lost_sales_fcnn.pt
```

Evaluate the saved FCNN:

```bash
python scripts/evaluate_paper_lost_sales_fcnn.py
```

This writes:

```text
outputs/results/paper_lost_sales_fcnn_eval.csv
```

Sweep stronger FCNN sizes:

```bash
python scripts/sweep_paper_lost_sales_fcnn.py
```

This compares `2x32`, `3x64`, and `3x128` FCNNs and writes:

```text
outputs/results/paper_lost_sales_fcnn_sweep.csv
```

Plot policy structure and monotonicity diagnostics:

```bash
python scripts/plot_paper_lost_sales_policy_structure.py
```

This writes:

```text
outputs/plots/paper_lost_sales_policy_structure.png
```

The main evaluation focus is FCNN performance, policy structure, and monotonicity diagnostics. Dual sourcing, perishability, and procurement-removal are later paper-setting extensions.

Analyze FCNN vs base-stock baselines:

```bash
python scripts/analyze_paper_lost_sales_fcnn_failures.py
```

This writes:

```text
outputs/results/paper_lost_sales_policy_comparison_summary.csv
outputs/results/paper_lost_sales_fcnn_worst_cases.csv
```

## Run Original Baselines

```bash
python scripts/run_baselines.py
```

This uses `configs/lost_sales_default.yaml` and writes:

```text
outputs/results/baseline_results.csv
```

## Tune Single Base-Stock Benchmark

```bash
python scripts/tune_base_stock.py
```

This grid-searches `S` over the configured range and writes:

```text
outputs/results/base_stock_grid.csv
```

## Run Benchmark Scenarios

```bash
python scripts/run_benchmark_baselines.py
```

This loads `configs/benchmark_scenarios.yaml`, evaluates heuristic baselines across zero-lead-time, lead-time, high-variance, seasonal, cost-regime, and dual-sourcing scenarios, and writes:

```text
outputs/results/benchmark_baselines.csv
```

The benchmark scenarios include:

- `lost_sales_zero_lead_time`
- `lost_sales_lead_time_L2`
- `lost_sales_lead_time_high_variance`
- `lost_sales_lead_time_seasonal`
- `lost_sales_high_stockout_cost`
- `lost_sales_high_holding_cost`
- `dual_sourcing_basic`
- `dual_sourcing_high_variance`

Fixed scenarios test robustness across hand-designed settings: different lead times, demand processes, and cost regimes are specified explicitly in the scenario YAML.

## Run Product Generalization Benchmark

```bash
python scripts/run_product_benchmark_baselines.py
```

This loads `configs/product_benchmark.yaml`, generates heterogeneous train, validation, and test product sets, tunes heuristic base-stock baselines per product, and writes:

```text
outputs/results/product_benchmark_baselines.csv
```

The product benchmark tests generalization across unseen products rather than only robustness across fixed hand-designed scenarios. Each generated product has hidden true simulator parameters, historical demand observations, cost parameters, lead-time information, and an initial inventory level.

Two base-stock benchmarks are reported:

- `HistoricalBaseStockPolicy`: non-omniscient tuning using only historical demand observations and observable product features. This is the realistic benchmark.
- `OracleBaseStockPolicy`: omniscient tuning using the product's hidden true simulator parameters and a larger Monte Carlo sample. This is an oracle reference, not a deployable policy.

Policies should receive practical observable features such as current inventory, pipeline orders, recent demand history, costs, and lead time. They should not receive the true demand parameters used internally by the simulator.

## Plot Benchmark Results

```bash
python scripts/plot_benchmark_baselines.py
```

Plots are saved to:

```text
outputs/plots/
```

## Train Minimal Neural Policy

The first neural path is a minimal zero-lead-time PyTorch setup:

```bash
python scripts/train_zero_lead_time_mlp.py
```

It trains an unconstrained MLP through a differentiable zero-lead-time simulator, then evaluates the trained policy on the true discrete simulator. This is only the first DirectBackprop-style smoke path; monotone MLP and MonoKAN policies are still future work.

## Train Product-Level Lead-Time MLP

```bash
python scripts/train_product_lead_time_mlp.py
```

This trains one unconstrained MLP across generated lead-time products using only historical demand summaries, costs, lead time, on-hand inventory, and pipeline state as inputs. It evaluates the same trained policy on train, validation, and test products in the true discrete lead-time simulator and writes:

```text
outputs/results/product_lead_time_mlp.csv
```

This is the first product-level neural generalization path. It is intentionally unconstrained; monotonic architectures should be compared against it later.

## Research Direction

The benchmark is designed for later neural policy experiments inspired by recent neural inventory-control work on Deep Inventory Management, dual-sourcing neural controllers, and structure-informed DRL. The next stage is to add differentiable policy optimization and compare unconstrained and monotone neural architectures against the heuristic baselines implemented here.

## License

License is currently undecided. See `LICENSE` for the placeholder note. Choose a real license before public release or external collaboration.
