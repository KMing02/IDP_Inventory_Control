# Inductive Biases for Neural Inventory Control

This repository contains the code developed for the individual TUM IDP report
**From Monotonicity to Order-Up-To Structure: Inductive Biases for Neural Inventory Control**.
It studies neural policies for stationary, no-lead-time, single-item lost-sales
inventory systems across a heterogeneous population of products.

The final report is submitted separately. Its LaTeX source, bibliography,
compiled PDF, and report figure PDFs are intentionally not part of this code
repository.

The submitted scope is the FCNN/CMNN line of work. The obsolete MonoKAN
implementation that remains in the repository is not part of this report and is
maintained separately by the MonoKAN team. It is intentionally left untouched
until the repository owner approves its removal.

## Scientific scope

The final report compares increasingly specific inductive biases:

1. **PaperFCNN** -- an unconstrained direct-order policy. The repository name
   `PaperFCNN` denotes the FCNN derived from the paper-style benchmark.
2. **PenaltyMLP** -- the same direct-order FCNN trained with a sampled soft
   monotonicity penalty.
3. **PositiveWeightMLP** -- a simple hard-monotone direct-order baseline.
4. **ConstrainedMonoMLP (CMNN)** -- a direct-order constrained monotonic neural
   network with convex, concave, and bounded activation groups.
5. **TargetStockFCNN** -- a context network that predicts a positive,
   dimensionless target-stock multiple and uses the exact decoder
   `q = relu(S - inventory)`.
6. **Residual order-up-to policies** -- FCNN and CMNN-backbone context models
   that learn a bounded residual correction around a Historical Gamma
   target-stock anchor and use the same exact order-up-to decoder.

The analytical references are:

- **Historical Gamma** (`NonOmniscientOptimal` internally), fitted from each
  product's historical demand;
- **Omniscient Gamma** (`OmniscientOptimal` internally), fitted with access to
  the corresponding future-demand distribution for benchmark evaluation.

The central distinction is that hard monotonicity guarantees only that desired
order quantity is non-increasing in inventory. An exact order-up-to decoder
additionally enforces active-region slope `-1` and zero order above the target
stock. The report evaluates structural behavior separately from operational
reward.

## Repository layout

```text
configs/                    Frozen benchmark and experiment configurations
inventory_control/          Environments, products, policies, evaluation, training
scripts/                    Experiment, audit, and plotting entry points
tests/                      Unit and workflow tests
reports/
  data/                     Compact validation-only plotting inputs
  plot_target_stock_reward_dynamics.py  Plotting utility retained as code
outputs/                    Generated artifacts; intentionally ignored by Git
```

Key model implementations are in:

- `inventory_control/policies/neural/fcnn.py`
- `inventory_control/policies/neural/constrained_mlp.py`
- `inventory_control/training/direct_backprop.py`
- `inventory_control/paper_lost_sales.py`
- `inventory_control/idp_benchmark.py`

## Environment

Python 3.10 or newer is required. Install the project and test dependencies with:

```bash
python -m pip install -e ".[dev,neural]"
```

The final experiments were run locally on CPU in two recorded environments:

| Experiment family | Python | PyTorch | NumPy | pandas |
|---|---:|---:|---:|---:|
| Primary tuning, matched comparison, residual and convergence audits | 3.10.20 | 2.12.1+cpu | 2.2.6 | 2.3.3 |
| Target-stock and direct-q 120-update diagnostics | 3.12.7 | 2.12.1+cpu | 1.26.4 | 2.2.2 |

The dependency declarations are in `pyproject.toml`, `requirements.txt`,
`requirements-neural.txt`, and `requirements-dev.txt`. Per-run
`environment.json` files record the exact environment used for each completed
experiment.

## Frozen data and evaluation design

The main configuration is:

```text
configs/paper_lost_sales_medium_seed0_converged.yaml
```

The experiment runners separate training randomness from product generation,
validation-grid generation, and operational evaluation scenarios. The main
validation workflows use 128 training products, 32 validation products, the
fixed validation-grid seed `2000003`, operational seeds `40000--40031`, and the
shared deployment cap `F=4`.

Generated products and demand paths are deterministic from the configuration;
they are not committed as private input data. Each completed workflow records
data/configuration fingerprints so that compatible runs can be checked before
reuse.

## Main workflows

All commands below are intended for the **training and validation splits only**.
Generated artifacts are written under `outputs/` and are ignored by Git.

### Two-stage primary tuning

```bash
python scripts/tune_primary_neural_policies.py \
  --stage screening \
  --config configs/paper_lost_sales_medium_seed0_converged.yaml \
  --profile full --split validation --seeds 0 \
  --output-dir outputs/idp_tuning/primary_neural_balanced8_validation \
  --resume

python scripts/tune_primary_neural_policies.py \
  --stage confirmation \
  --config configs/paper_lost_sales_medium_seed0_converged.yaml \
  --profile full --split validation --seeds 1 2 \
  --output-dir outputs/idp_tuning/primary_neural_balanced8_validation \
  --resume
```

Use `--dry-run` (also accepted as `--manifest-only`) to inspect the tuning
manifest without training.

### Primary and structural validation analyses

| Script | Purpose |
|---|---|
| `scripts/run_primary_validation_followups.py` | Equal-capacity and sample-efficiency validation analyses |
| `scripts/run_primary_structural_ood_diagnostic.py` | Inventory-range structural diagnostics without operational evaluation |
| `scripts/run_policy_behavior_extrapolation_analysis.py` | Reconstructed training-state support, inventory-state extrapolation, and policy-behavior figures |
| `scripts/run_no_imitation_ablation.py` | Reward-only direct-policy diagnostic without imitation |
| `scripts/run_penalty_gradient_audit.py` | Controlled PenaltyMLP gradient audit |
| `scripts/run_cmnn_initialization_ablation.py` | Direct-q CMNN initialization diagnostic |

### Residual and target-stock analyses

| Script | Purpose |
|---|---|
| `scripts/run_residual_order_up_to_validation.py` | Historical-Gamma-anchored residual order-up-to comparison |
| `scripts/run_target_stock_validation.py` | Ten-seed reward-only TargetStockFCNN validation analysis |
| `scripts/run_target_stock_training_dynamics.py` | Standard/informed target-stock dynamics through 120 reward updates |
| `scripts/run_direct_q_120_update_diagnostic.py` | Matched direct-q FCNN 120-update diagnostic |
| `reports/plot_target_stock_reward_dynamics.py` | Rebuild the target-stock dynamics visualization from bundled validation CSVs only |

### Checkpoint provenance and convergence

| Script | Purpose |
|---|---|
| `scripts/run_primary_checkpoint_provenance_audit.py` | Reconstruct and verify the 20 archived primary FCNN/CMNN checkpoints |
| `scripts/run_primary_imitation_convergence_audit.py` | Fixed-checkpoint validation trajectory at reward updates 0, 15, 30, 60, 120, and 240 |

The provenance audit requires the frozen checkpoint manifest, matched-run
records, and archived checkpoints. These binary/result artifacts are not stored
in Git and must be supplied through the accompanying reproducibility archive.
The convergence audit uses the provenance outputs to verify update-0 and
update-15 states before extending the validation-only trajectories.

## Protected-test safety

The protected test was consumed exactly once under the frozen final protocol.
Its raw products, demand traces, predictions, per-product outputs, and generated
evaluation directory are deliberately excluded from this Git submission.

`scripts/run_final_protected_test_evaluation.py` is retained only to document
the frozen evaluation implementation and for unit testing of its safety guards.
Do not run it as part of ordinary setup, testing, or validation reproduction.

The separately submitted report contains the already-recorded aggregate
protected-test results. No other workflow documented here requires
protected-test access.

## Reproducibility artifacts

The Git repository contains source code, configurations, tests, and two compact
validation-only CSVs needed by the included target-stock plotting utility. It
does not contain the final report materials or the complete historical
experiment outputs.

A clean clone is sufficient to inspect the implementations, run unit tests, and
start new validation experiments from the frozen configurations. It is not, by
itself, sufficient to reproduce every historical table, checkpoint-provenance
decision, or figure. Exact reproduction of those completed results additionally
requires an external artifact archive containing the relevant outputs,
especially:

- `outputs/idp_tuning/primary_neural_balanced8_validation/`
- `outputs/idp_followup/matched_width_depth_10seed_validation/`
- `outputs/idp_followup/equal_capacity_validation/`
- `outputs/idp_followup/data_efficiency_validation/`
- `outputs/idp_followup/no_imitation_matched_width_validation/`
- `outputs/idp_followup/residual_order_up_to_validation/`
- `outputs/idp_followup/target_stock_reward_only_validation/`
- `outputs/idp_followup/target_stock_training_dynamics/`
- `outputs/idp_followup/direct_q_120_update_diagnostic/`
- `outputs/idp_followup/primary_checkpoint_provenance_audit/`
- `outputs/idp_followup/primary_imitation_assisted_convergence_validation/`
- `outputs/idp_followup/policy_behavior_extrapolation_validation/`

At minimum, preserve resolved configurations, aggregate/seed-level CSVs,
fingerprints, completion metadata, checkpoint hashes, and the checkpoints needed
by the provenance analysis. Large artifacts should be distributed separately
rather than committed to Git. The protected-test output directory must not be
included in that archive unless access is explicitly authorized by the project
supervisor.

## Tests and source checks

The normal test suite does not require running training experiments or accessing
the protected test:

```bash
python -m pytest
git diff --check
```

Individual workflow tests can be run with, for example:

```bash
python -m pytest tests/test_primary_neural_tuning.py
python -m pytest tests/test_primary_validation_followups.py
python -m pytest tests/test_policy_behavior_extrapolation_analysis.py
python -m pytest tests/test_residual_order_up_to.py
python -m pytest tests/test_target_stock_fcnn.py
python -m pytest tests/test_target_stock_validation.py
python -m pytest tests/test_target_stock_training_dynamics.py
python -m pytest tests/test_final_protected_test_runner.py
```

These are unit/workflow checks; they do not authorize rerunning the protected
evaluation.

## License and attribution

See `LICENSE` and `third_party_licenses/` for repository and third-party notices.
