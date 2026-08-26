# Inductive Biases for Neural Inventory Control

This repository contains the code used for a TUM IDP project on structure-informed neural inventory control. The final report studies a Maggiar-style lost-sales setting and asks what changes when a flexible neural policy is replaced by a hard-monotone MonoKAN representation.

## Research Question

How does a certified monotone neural inventory-control policy perform relative to an unconstrained fully connected neural-network policy under the same DirectBackprop training and evaluation protocol?

The main comparison is intentionally matched: the policies see the same products, demand paths, training seeds, horizon, burn-in, simulator, reward function, and target-stock decoder.

## Final Formulation

The report-facing target-stock policies predict a positive target ratio from observable demand history and economics:

```text
r_theta = softplus(model(H_t, economics))
S_t = m_0 * r_theta
q_t = max(S_t - I_t, 0)
```

`m_0` is the positive scale computed from the initial demand history. Current inventory is excluded from the target network and enters only through the order-up-to decoder.

## Final Models

- `PaperFCNN`: paper-style rolling raw-action FCNN anchor used to check the DirectBackprop and simulator pipeline.
- `PaperFCNNDirectTargetStock`: unconstrained rolling target-stock FCNN.
- `MonoKANDirectTargetStockDeepGrid7`: main certified monotone DeepGrid7 MonoKAN policy.
- Temporal DeepGrid7 MonoKAN: secondary ablation using raw demand lags plus causal temporal summaries.

Compact MonoKAN and supervised target-fitting diagnostics are retained as supporting capacity evidence.

## Main Result

| Model | Mean reward per period |
|---|---:|
| Rolling target-stock FCNN | 4178.27 +/- 4.23 |
| Rolling DeepGrid7 MonoKAN | 4108.48 +/- 5.90 |
| Temporal DeepGrid7 MonoKAN | 4135.64 +/- 9.08 |

The baseline DeepGrid7 MonoKAN is certified monotone end to end in the intended raw demand-lag and economics coordinates, with final order nonincreasing in current inventory through the decoder.

The temporal variant improves reward, but it is not globally monotone in the original raw demand-history coordinates because `std_32` and `trend_8` are unconstrained functions of those lags.

## Running

Install the project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest
```

Validate final matched configs:

```bash
python scripts/run_rolling_direct_target_matched.py audit
python scripts/run_rolling_direct_target_matched.py audit-development
python scripts/run_rolling_direct_target_matched.py audit-temporal
```

Run the final target-stock development protocol for one seed:

```bash
python scripts/run_rolling_direct_target_matched.py run-development \
  --seed 2025 \
  --device cpu \
  --output-dir outputs/direct_target_dev_1k_normalized/seed_2025/full_200e
```

Run the retained capacity and supervised diagnostics:

```bash
python scripts/run_base_stock_output_experiment.py \
  --config configs/paper_lost_sales_direct_target_capacity_medium.yaml \
  --models MonoKANDirectTargetStock MonoKANDirectTargetStockDeepGrid7 \
  --output-dir outputs/direct_target_capacity_medium

python scripts/run_supervised_target_diagnostic.py \
  --config configs/paper_lost_sales_base_stock_small.yaml \
  --output-dir outputs/supervised_target_diagnostic/capacity_canonical
```

Run the checkpoint-only structural audit:

```bash
python scripts/run_direct_target_structural_audit.py \
  --config configs/paper_lost_sales_rolling_direct_target_dev_1k_normalized.yaml \
  --checkpoint-root outputs/direct_target_dev_1k_normalized \
  --output-dir outputs/direct_target_structural_audit
```

The large experiments are compute-heavy. Existing report-relevant outputs are intentionally kept under `outputs/` and are not committed.

## Structure

```text
configs/                         final and retained ablation configs
docs/                            final evidence, debugging, and methodology notes
inventory_control/envs/          inventory simulators and demand processes
inventory_control/paper_*.py     Maggiar-style products, evaluation, and config loading
inventory_control/policies/      classical policies and retained neural policies
inventory_control/training/      DirectBackprop training utilities
inventory_control/*diagnostic.py retained experiment and audit modules
scripts/                         thin CLI wrappers for final/report-facing runs
tests/                           correctness tests for retained scientific behavior
```

See `docs/EXPERIMENTS.md` for the final evidence index and output provenance.
