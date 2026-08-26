# Structure-Informed Neural Inventory Control Benchmark

This repository is a research codebase for structure-informed neural inventory control. The current primary development workflow compares paper-style raw-action FCNN behavior with direct target-stock neural policies for a paper-style lost-sales benchmark.

This is not a generic reinforcement learning benchmark. The intended training direction is DirectBackprop-style differentiable policy optimization through inventory simulators, where domain structure is injected through inventory balance equations, lead-time pipeline states, monotonicity of order quantities with respect to inventory position, and classical heuristic baselines.

The current priority is replicating the lost-sales setting from Maggiar et al. (2025), "Structure-Informed Deep Reinforcement Learning for Inventory Management". Generic benchmark scenarios remain in the repository, but they are secondary until the lost-sales replication is stable.

The current implementation includes clean simulators, demand processes, heuristic baselines, paper-style lost-sales product generation, DirectBackprop training, direct target-stock policies, residual order-up-to policy decoders, structural diagnostics, distribution-shift evaluation, plotting, and tests.

## Project Status

Current status:

- Core single-item lost-sales simulators are implemented and tested.
- Paper-style no-lead-time lost-sales product generation, reward evaluation, base-stock baselines, and DirectBackprop training are implemented.
- The current main evidence compares a paper-style rolling raw-action FCNN anchor with rolling direct target-stock FCNN and DeepGrid7 MonoKAN policies.
- The direct target-stock policy predicts `S_t` and deploys `q_t = max(S_t - I_t, 0)`.
- The older residual order-up-to workflow remains implemented and documented as a supporting ablation, not the current final framing.
- Target-stock monotonicity diagnostics, MonoKAN spline certification, and distribution-shift evaluation are available.
- Evaluation is comparable across policies using the same products, demand paths, seeds, horizon, and burn-in.

Current limitations:

- Paper lost-sales product generation follows Table 1 of Maggiar et al. for the synthetic LS setting, and neural policies use the current observable feature pipeline.
- `OmniscientOptimal` is the exact Gamma critical-quantile base-stock benchmark and uses hidden true demand parameters.
- Current MonoKAN evidence is development-scale and behavioral; it is not a full paper replication or statistically established robustness result.
- Temporal-feature MonoKAN is a reward-improving secondary ablation, but it is not globally monotone with respect to original raw demand lags.
- Raw-action FCNN and raw-action MonoKAN scripts remain as legacy diagnostics; they are not the primary workflow.
- Multi-echelon, perishability, and procurement-removal settings are not implemented yet.
- Generated outputs and model checkpoints are intentionally ignored by Git and are not required for tests.

See also:

- `docs/EXPERIMENTS.md` for the current experiment index, provenance, and which outputs are canonical development evidence.
- `docs/DEBUGGING.md` for major resolved implementation issues and invalid/debug outputs.
- `docs/ARCHITECTURE.md` for module boundaries and reusable policy APIs.
- `docs/TEAM_WORKFLOW.md` for branch, test, and pull-request expectations.
- `docs/MONOKAN_METHOD_EVIDENCE.md` for current MONOkan evidence and limitations.

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

Run a tiny comparable residual benchmark for the supporting residual workflow:

```bash
python scripts/run_idp_benchmark.py --mode structure-ls --profile compact --config configs/paper_lost_sales_debug.yaml --models PaperFCNNResidualOrderUpTo,ConstrainedMLPResidualOrderUpTo,MonoKANResidualOrderUpTo --output-dir outputs/idp_benchmark/residual_smoke
```

## Reproducible Commands

From a clean checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
python scripts/run_quick_smoke_test.py
python scripts/run_idp_benchmark.py --mode structure-ls --profile compact --config configs/paper_lost_sales_debug.yaml --models PaperFCNNResidualOrderUpTo,ConstrainedMLPResidualOrderUpTo,MonoKANResidualOrderUpTo --output-dir outputs/idp_benchmark/residual_smoke
```

Older raw-action and residual training/evaluation scripts are retained for diagnostics and historical comparison. For the current direct target-stock evidence, start from `docs/EXPERIMENTS.md` and the `scripts/run_base_stock_output_experiment.py` and `scripts/run_rolling_direct_target_matched.py` entry points.

The evaluation scripts default to the validation split. The held-out test split should be used only for the final report:

```bash
python scripts/run_paper_lost_sales_baselines.py --split test --final-test
python scripts/evaluate_paper_lost_sales_fcnn.py --split test --final-test
python scripts/evaluate_paper_lost_sales_monokan.py --split test --final-test
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
outputs/results/paper_lost_sales_baselines_validation.csv
```

The two explicit baselines are:

- `OmniscientOptimal`: uses hidden true Gamma parameters and is an oracle.
- `NonOmniscientOptimal`: fits Gamma parameters from historical demand observations only.
- `HistoricalDemandBaseStock`: empirical historical quantile diagnostic, not the primary paper baseline.

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
outputs/results/paper_lost_sales_fcnn_eval_validation.csv
```

Train the adapted MonoKAN policy:

```bash
python scripts/train_paper_lost_sales_monokan.py
```

The current MonoKAN implementation adapts the MIT-licensed Hermite-spline monotonicity mechanism from the official MonoKAN repository. It constrains order quantity to be nonincreasing in current inventory before action scaling/clipping. This is implemented and unit-tested, but it should be treated as preliminary until benchmarked against FCNN and base-stock baselines.

Training writes:

```text
outputs/results/paper_lost_sales_monokan_training_log.csv
outputs/models/paper_lost_sales_monokan.pt
```

Evaluate the saved MonoKAN:

```bash
python scripts/evaluate_paper_lost_sales_monokan.py
```

This writes:

```text
outputs/results/paper_lost_sales_monokan_eval_validation.csv
```

Run a validation-only MonoKAN tuning sweep:

```bash
python scripts/sweep_paper_lost_sales_monokan.py
```

This writes:

```text
outputs/results/paper_lost_sales_monokan_sweep_validation.csv
```

## Colab Training

The local config is intentionally capped. For a larger validation run on Google Colab, use:

```bash
python -m pip install -e ".[dev]"
python scripts/run_colab_paper_lost_sales.py
```

This uses:

```text
configs/paper_lost_sales_colab.yaml
```

The Colab runner still evaluates validation only. It does not touch the held-out test split. Useful variants:

```bash
python scripts/run_colab_paper_lost_sales.py --models monokan --skip-baselines
python scripts/run_colab_paper_lost_sales.py --models fcnn --skip-sweeps
```

You can also run individual scripts with the Colab config:

```bash
PAPER_LOST_SALES_CONFIG=configs/paper_lost_sales_colab.yaml python scripts/train_paper_lost_sales_monokan.py
PAPER_LOST_SALES_CONFIG=configs/paper_lost_sales_colab.yaml python scripts/evaluate_paper_lost_sales_monokan.py
```

Only after model selection is frozen, run the final held-out test once:

```bash
PAPER_LOST_SALES_CONFIG=configs/paper_lost_sales_colab.yaml python scripts/run_paper_lost_sales_baselines.py --split test --final-test
PAPER_LOST_SALES_CONFIG=configs/paper_lost_sales_colab.yaml python scripts/evaluate_paper_lost_sales_fcnn.py --split test --final-test
PAPER_LOST_SALES_CONFIG=configs/paper_lost_sales_colab.yaml python scripts/evaluate_paper_lost_sales_monokan.py --split test --final-test
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

This raw-action analysis is retained for historical diagnostics. The main
current evaluation focus is the direct target-stock workflow indexed in
`docs/EXPERIMENTS.md`; the residual order-up-to workflow below is a supporting
ablation. Dual
sourcing, perishability, and procurement-removal are later paper-setting
extensions.

Analyze FCNN vs base-stock baselines:

```bash
python scripts/analyze_paper_lost_sales_fcnn_failures.py
```

This writes:

```text
outputs/results/paper_lost_sales_policy_comparison_summary_validation.csv
outputs/results/paper_lost_sales_fcnn_worst_cases_validation.csv
```

## IDP Monotonic Architecture Benchmark

The IDP benchmark is LS-first and paper-aligned. It starts with the no-lead-time
lost-sales setting from Maggiar et al. before considering any extensions. The
residual order-up-to workflow is retained as a structured ablation with a
historical base-stock anchor.

Primary residual order-up-to policies:

- `PaperFCNNResidualOrderUpTo`: residual target-stock MLP baseline.
- `KANResidualOrderUpTo`: unconstrained KAN residual target-stock baseline.
- `ConstrainedMLPResidualOrderUpTo`: monotonic MLP residual target-stock comparator.
- `MonoKANResidualOrderUpTo`: MonoKAN residual target-stock policy.

Legacy raw-action and diagnostic policies:

- `PaperFCNN`: the causal-CNN plus MLP architecture from the paper.
- `MonoKAN`: a Hermite-spline architecture with nonincreasing inventory response enforced by construction.
- `ConstrainedMonoMLP`: a C-MNN-style constrained monotone MLP with mixed monotone activations.
- `PositiveWeightMLP`: a simpler positive-weight monotone MLP used as an expressiveness control.
- `PenaltyMLP`: an optional penalty-based reference, not the primary method.
- `NonOmniscientOptimal`: Gamma method-of-moments base-stock benchmark using historical demand.
- `OmniscientOptimal`: exact Gamma critical-quantile base-stock benchmark using hidden parameters.

Run the deterministic compact benchmark:

```bash
python scripts/run_idp_benchmark.py --mode structure-ls --profile compact
```

Run the current four-model residual comparison:

```bash
python scripts/run_idp_benchmark.py --mode structure-ls --profile compact --config configs/paper_lost_sales_debug.yaml --models PaperFCNNResidualOrderUpTo,KANResidualOrderUpTo,ConstrainedMLPResidualOrderUpTo,MonoKANResidualOrderUpTo
```

Run the paper replication baseline:

```bash
python scripts/run_idp_benchmark.py --mode replicate-ls --profile paper_replication
```

The held-out test split remains protected:

```bash
python scripts/run_idp_benchmark.py --mode structure-ls --profile paper_replication --split test --final-test
```

Artifacts are written under `outputs/idp_benchmark/<mode>_<profile>_<split>/`:

```text
summary.csv
paper_table3_replication.csv
operational_metrics.csv
predictions.csv
training_log.csv
monokan_splines.csv
target_structure_diagnostics.csv
monokan_spline_certification.csv
metadata.json
models/
plots/
```

The primary report is `paper_table3_replication.csv`, which gives average
reward and gap to the omniscient optimal benchmark. Structural artifacts include
policy curves, order-up-to scatter plots, inventory-order violation rates,
target-stock context monotonicity diagnostics, residual correction magnitudes,
MonoKAN spline samples, MonoKAN spline certification, and constrained-MLP
architecture comparisons.

Distribution-shift diagnostics are available through:

```bash
python scripts/run_distribution_shift_validation.py --config configs/paper_lost_sales_fcnn_alignment_medium.yaml --source-benchmark-dir outputs/idp_benchmark/<trained_run>
python scripts/run_distribution_shift_multiseed.py --config configs/paper_lost_sales_fcnn_alignment_medium.yaml --output-root outputs/monokan_distribution_shift_multiseed_10seed --progress
```

These commands are diagnostics, not final paper-replication claims. Generated
results and checkpoints under `outputs/` should not be committed.
Perishability, multi-echelon, and real-world data are intentionally out of scope
until the LS replication is credible.

Open `notebooks/idp_monotonic_inventory_control.ipynb` for the visual analysis.
Its default `PROFILE="auto"` loads full validation artifacts when present and
otherwise runs and caches the compact benchmark. Execute it non-interactively:

```bash
python -m jupyter nbconvert --to notebook --execute notebooks/idp_monotonic_inventory_control.ipynb --output idp_monotonic_inventory_control_executed.ipynb --output-dir outputs
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

It trains an unconstrained MLP through a differentiable zero-lead-time simulator, then evaluates the trained policy on the true discrete simulator. The paper-style IDP benchmark above contains the soft-penalty MLP and strict MonoKAN comparison.

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

The benchmark supports differentiable policy optimization and a controlled comparison of unconstrained, soft-penalty, and architecturally monotone policies. Later extensions can apply the same comparison to lead-time, dual-sourcing, multi-echelon, and perishable settings.

## License

License is currently undecided. See `LICENSE` for the placeholder note. Choose a real license before public release or external collaboration.
