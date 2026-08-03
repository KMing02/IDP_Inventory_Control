# Pre-Commit Summary

Date: 2026-08-02
Branch: `MonoKAN`

## Files Changed In Phase 2

- `.gitignore`
- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/TEAM_WORKFLOW.md`
- `docs/PDF_ASSET_DECISION.md`
- `docs/PRE_COMMIT_SUMMARY.md`
- `examples/add_custom_residual_policy.py`
- `inventory_control/policies/neural/registry.py`
- `inventory_control/policies/neural/__init__.py`
- `inventory_control/training/__init__.py`
- `inventory_control/training/direct_backprop.py`
- `inventory_control/idp_benchmark.py`
- `scripts/run_distribution_shift_validation.py`
- `scripts/run_distribution_shift_multiseed.py`
- `scripts/run_monokan_multiseed_validation.py`
- legacy script docstrings:
  - `scripts/analyze_paper_lost_sales_fcnn_failures.py`
  - `scripts/evaluate_paper_lost_sales_fcnn.py`
  - `scripts/evaluate_paper_lost_sales_monokan.py`
  - `scripts/plot_paper_lost_sales_policy_structure.py`
  - `scripts/sweep_paper_lost_sales_fcnn.py`
  - `scripts/sweep_paper_lost_sales_monokan.py`
  - `scripts/train_paper_lost_sales_fcnn.py`
  - `scripts/train_paper_lost_sales_monokan.py`
- `tests/test_distribution_shift_scripts.py`
- `tests/test_idp_benchmark.py`

Phase 1 and prior MONOkan work also remain in the dirty tree.

## Collaboration Blockers Resolved

- README now describes the residual order-up-to workflow, four-model comparison, diagnostics, shift evaluation, generated outputs, and limitations.
- `docs/ARCHITECTURE.md` documents data flow, shared vs architecture-specific modules, policy API, and how to add a residual model.
- `docs/TEAM_WORKFLOW.md` documents branch naming, ownership boundaries, tests, smoke commands, comparable experiment rules, and PR checklist.
- `examples/add_custom_residual_policy.py` demonstrates a minimal residual policy without copying the training loop or importing MonoKAN internals.
- `docs/PDF_ASSET_DECISION.md` records PDF sizes, likely ownership, redistribution uncertainty, and commit recommendations.
- Legacy raw-action scripts now clearly state that they are diagnostic/legacy paths.
- Distribution-shift tests now cover registry metadata, shift config construction, shift summary columns, paired product-level keys, one-seed/one-shift aggregation, and order-independent paired comparison.

## Compatibility Aliases Introduced

Architecture-neutral aliases were added while preserving old FCNN-named functions:

- `train_paper_lost_sales_policy` wraps `train_paper_lost_sales_fcnn`.
- `paper_lost_sales_policy_order` wraps `paper_lost_sales_fcnn_order`.
- `paper_lost_sales_policy_rollout` wraps `paper_lost_sales_fcnn_rollout`.

Existing imports of the old names continue to work.

## Registry Changes

Added `inventory_control/policies/neural/registry.py` with:

- canonical model names;
- factories;
- display/order positions;
- checkpoint filenames;
- architecture categories;
- seed offsets;
- residual comparison policy list.

Consumers updated:

- `inventory_control/idp_benchmark.py` uses the registry for trainable names, policy order, defaults, and checkpoint filenames.
- `scripts/run_distribution_shift_validation.py` uses the registry for factories, residual defaults, and checkpoint filenames.
- `scripts/run_distribution_shift_multiseed.py` and `scripts/run_monokan_multiseed_validation.py` use the registry residual default list.

Preserved checkpoint filenames:

- `paper_fcnn_residual_order_up_to.pt`
- `kan_residual_order_up_to.pt`
- `constrained_mlp_residual_order_up_to.pt`
- `monokan_residual_order_up_to.pt`

Preserved canonical output model names:

- `PaperFCNNResidualOrderUpTo`
- `KANResidualOrderUpTo`
- `ConstrainedMLPResidualOrderUpTo`
- `MonoKANResidualOrderUpTo`

## PDF Decisions

Ignored by `.gitignore` as local literature:

- `docs/1-s2.0-S0893608025011591-main.pdf`
- `docs/2507.22040v1.pdf`
- `docs/Structure-Informed Neural Policies for Lost-Sales Inventory Control.pdf`

Recommendation:

- Do not commit third-party PDFs unless redistribution rights are clear.
- Prefer citation/link notes and extracted summaries.

Existing tracked generated PDF:

- `output/pdf/idp_monotonic_inventory_report.pdf`

Recommendation:

- Leave untouched in this cleanup.
- Decide in a separate commit whether generated reports should remain tracked.

## Validation Commands And Results

No dedicated formatter/linter is configured in `pyproject.toml`; compile checks were run as the low-risk syntax validation.

Compile check:

```bash
uv run python -m py_compile inventory_control/policies/neural/registry.py inventory_control/idp_benchmark.py inventory_control/training/direct_backprop.py scripts/run_distribution_shift_validation.py scripts/run_distribution_shift_multiseed.py scripts/run_monokan_multiseed_validation.py examples/add_custom_residual_policy.py tests/test_distribution_shift_scripts.py
```

Result: passed.

Full test suite:

```bash
uv run --extra dev python -m pytest
```

Result: `136 passed in 15.70s`.

Targeted residual/distribution tests:

```bash
uv run --extra dev python -m pytest tests/test_paper_monokan.py tests/test_idp_benchmark.py tests/test_distribution_shift_scripts.py -q
```

Result: `45 passed in 13.63s`.

Quick smoke test:

```bash
uv run --extra dev python scripts/run_quick_smoke_test.py
```

Result:

```text
HistoricalDemandBaseStock: mean_total_reward=170.676
OracleGammaBaseStock: mean_total_reward=180.888
FCNN: mean_total_reward=126.393
training_loss_initial=5.190
training_loss_final=-619.335
quick_smoke_test=passed
```

Custom residual-policy example:

```bash
uv run --extra dev python examples/add_custom_residual_policy.py
```

Result:

```text
trained_epochs=2
example_order_mean=3.344726
custom_residual_policy_example=passed
```

Tiny residual benchmark:

```bash
uv run --extra dev python scripts/run_idp_benchmark.py --mode structure-ls --profile compact --config configs/paper_lost_sales_debug.yaml --models PaperFCNNResidualOrderUpTo,ConstrainedMLPResidualOrderUpTo,MonoKANResidualOrderUpTo --output-dir outputs/idp_benchmark/residual_smoke_phase2
```

Result:

- `PaperFCNNResidualOrderUpTo`: mean reward/period `253.334357`.
- `ConstrainedMLPResidualOrderUpTo`: mean reward/period `216.913190`.
- `MonoKANResidualOrderUpTo`: mean reward/period `243.725348`.
- Artifacts saved under ignored `outputs/idp_benchmark/residual_smoke_phase2`.

Determinism check:

```bash
uv run --extra dev python scripts/run_idp_benchmark.py --mode structure-ls --profile compact --config configs/paper_lost_sales_debug.yaml --models PaperFCNNResidualOrderUpTo,ConstrainedMLPResidualOrderUpTo,MonoKANResidualOrderUpTo --output-dir outputs/idp_benchmark/residual_smoke_phase2_repeat
```

Result:

- Model names identical.
- Summary columns identical.
- Numeric values deterministic except `training_seconds`.
- Checkpoints present:
  - `constrained_mlp_residual_order_up_to.pt`
  - `monokan_residual_order_up_to.pt`
  - `paper_fcnn_residual_order_up_to.pt`

Constrained MLP MonoKAN dependency check:

```bash
rg -n "monokan" inventory_control/policies/neural/constrained_mlp.py examples/add_custom_residual_policy.py tests/test_constrained_mlp.py
```

Result: no matches.

## Output And Schema Compatibility

Existing benchmark CSV schemas remain compatible:

- `summary.csv`
- `operational_metrics.csv`
- `predictions.csv`
- `training_log.csv`

New diagnostic CSVs are additional files:

- `target_structure_diagnostics.csv`
- `monokan_spline_certification.csv`

Distribution-shift CLI now also writes non-breaking metadata files:

- `run_metadata.json`
- `<shift>/resolved_shift_config.json`

## Issues Intentionally Deferred

- Moving target-stock diagnostics into `inventory_control/evaluation/`.
- Broad `idp_benchmark.py` restructuring.
- Full declarative benchmark registration for all training branches.
- Renaming old FCNN-named functions everywhere.
- Moving feature normalization out of `fcnn.py`.
- Centralizing result schema constants.
- Deleting or merging legacy scripts.
- Changing reward, product generation, feature definitions, capacity guards, residual scaling, schedules, or checkpoint-selection logic.
- Running 10-seed or paper-scale experiments.

## Remaining Dirty And Untracked Files

Branch status after cleanup:

- `MonoKAN` is ahead of `origin/MonoKAN` by 1 commit.
- No files were staged or committed.
- Generated `outputs/` and local literature PDFs are ignored.
- `uv.lock` remains untracked; decide separately whether to commit it.

Important untracked non-generated files:

- new docs under `docs/`;
- `examples/add_custom_residual_policy.py`;
- `inventory_control/policies/neural/registry.py`;
- new scripts under `scripts/`;
- `tests/test_distribution_shift_scripts.py`;
- new experiment configs.

## Recommended Commit Grouping

Commit 1: residual policy, decoder, diagnostics, and registry

- `inventory_control/policies/neural/constrained_mlp.py`
- `inventory_control/policies/neural/fcnn.py`
- `inventory_control/policies/neural/monokan.py`
- `inventory_control/policies/neural/registry.py`
- `inventory_control/policies/neural/__init__.py`
- `inventory_control/training/direct_backprop.py`
- `inventory_control/training/__init__.py`
- `inventory_control/idp_benchmark.py`
- `tests/test_paper_fcnn.py`
- `tests/test_paper_monokan.py`
- `tests/test_idp_benchmark.py`

Commit 2: scripts, configs, and distribution-shift tests

- `configs/paper_lost_sales_factorial_small.yaml`
- `configs/paper_lost_sales_fcnn_alignment_medium.yaml`
- `configs/paper_lost_sales*.yaml` one-line setting updates
- `scripts/run_distribution_shift_validation.py`
- `scripts/run_distribution_shift_multiseed.py`
- `scripts/run_monokan_multiseed_validation.py`
- `scripts/run_monokan_optimization_sweep.py`
- plotting scripts
- legacy script docstring updates
- `tests/test_distribution_shift_scripts.py`

Commit 3: collaboration docs and example

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/TEAM_WORKFLOW.md`
- `docs/PDF_ASSET_DECISION.md`
- `docs/PRE_COMMIT_CODEBASE_AUDIT.md`
- `docs/PRE_COMMIT_SUMMARY.md`
- method/evidence docs
- `examples/add_custom_residual_policy.py`
- `.gitignore`

Commit 4: dependency/reference policy decision, only if agreed

- `uv.lock`, if the team wants locked reproducibility.
- Third-party PDFs only if redistribution is approved; otherwise keep ignored.
