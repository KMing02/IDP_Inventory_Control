# Pre-Commit Codebase Audit

Date: 2026-08-02
Branch inspected: `MonoKAN`
Scope: Phase 1 audit only. No cleanup or behavior changes were made.

## 1. Current Repository Map

### Root

Responsibility:
- Project metadata, packaging, global docs, notebooks, and ignored/generated outputs.

Public entry points:
- `README.md`
- `pyproject.toml`
- `requirements*.txt`
- `scripts/run_idp_benchmark.py`
- `scripts/run_quick_smoke_test.py`

Shared components:
- Package metadata and dependency declaration.
- Repository-level usage docs.

Architecture-specific components:
- None intended at root.

Audit notes:
- `.gitignore` already ignores `outputs/`, checkpoints, virtualenvs, `.DS_Store`, `third_party/`, and extracted `idinn-main/`.
- `output/pdf/idp_monotonic_inventory_report.pdf` is tracked even though it is generated.
- `uv.lock` is present and untracked; the team should decide whether lockfile reproducibility matters.
- Local generated folders are physically present but ignored: `.venv/`, `outputs/`, `inventory_control.egg-info/`, `idinn-main/`.

### `inventory_control/envs/`

Responsibility:
- Generic inventory-control environments and demand distributions.

Public entry points:
- `inventory_control.envs.demand`
- `inventory_control.envs.lost_sales`

Shared components:
- Simulator state transition logic.
- Demand process implementations.

Architecture-specific components:
- None.

Audit notes:
- This is appropriately shared and does not depend on neural policies.

### `inventory_control/policies/`

Responsibility:
- Generic non-neural policy abstractions and classical policies.

Public entry points:
- `inventory_control.policies.base`
- `inventory_control.policies.classical`
- `inventory_control.policies.neural`

Shared components:
- Classical base-stock and simple policy code.

Architecture-specific components:
- Neural architectures are under `inventory_control/policies/neural/`.

Audit notes:
- Separation between generic policy concepts and neural policy implementations is mostly clean.

### `inventory_control/policies/neural/fcnn.py`

Responsibility:
- FCNN/CNN policy architectures and paper-feature construction.

Public entry points:
- `make_paper_fcnn_model`
- `make_paper_order_up_to_fcnn_model`
- `make_paper_residual_order_up_to_fcnn_model`
- `compute_paper_fcnn_feature_stats`
- `build_paper_lost_sales_features`
- `PaperFCNNPolicyAdapter`

Shared components:
- Feature normalization and feature vector construction are currently here, but they are shared by all paper neural policies.

Architecture-specific components:
- `FCNNPolicy`
- `PaperCausalCNNPolicy`
- `PaperContextMLPPolicy`

Audit notes:
- The file mixes FCNN architecture code with shared feature-normalization utilities. This causes MonoKAN and monotonic MLP code to import "FCNN" utilities for non-FCNN workflows.

### `inventory_control/policies/neural/constrained_mlp.py`

Responsibility:
- Sign-constrained and partially monotone MLP architectures.

Public entry points:
- `make_positive_weight_mlp_model`
- `make_constrained_monotone_mlp_model`
- `make_paper_residual_order_up_to_constrained_mlp_model`
- `ConstrainedMonotoneMLP`
- `PartialContextMonotoneMLP`

Shared components:
- None intended.

Architecture-specific components:
- Positive-weight MLP, constrained monotone MLP, context-monotone residual MLP.

Audit notes:
- The monotonic MLP implementation does not depend on MonoKAN modules.
- It does depend on the same residual decoder contract through model attributes such as `paper_action_parameterization`, `paper_residual_scale`, and `paper_residual_anchor`.
- Those attributes are informal rather than documented through a shared interface.

### `inventory_control/policies/neural/monokan.py`

Responsibility:
- MonoKAN/KAN architecture and spline-specific diagnostics.

Public entry points:
- `MonoKANPolicy`
- `make_paper_monokan_model`
- `make_paper_order_up_to_monokan_model`
- `make_paper_residual_order_up_to_kan_model`
- `make_paper_residual_order_up_to_monokan_model`
- `sample_monokan_first_layer_splines`
- `monokan_inventory_violation_rate`

Shared components:
- None should be shared beyond the generic residual decoder contract.

Architecture-specific components:
- Cubic Hermite KAN layer.
- Monotone spline projection.
- MonoKAN spline sampling.

Audit notes:
- `make_paper_order_up_to_monokan_model` uses `monotonic_vars={}`; this is a naming risk because it is not actually constrained.
- The residual MonoKAN factory is the scientifically relevant one.

### `inventory_control/training/direct_backprop.py`

Responsibility:
- Paper-style lost-sales training batch creation, target decoding, differentiable rollout, penalties, model selection, and training loop.

Public entry points:
- `make_paper_lost_sales_training_batch`
- `paper_lost_sales_context_features`
- `paper_lost_sales_raw_order`
- `paper_lost_sales_target_level`
- `paper_lost_sales_fcnn_order`
- `paper_lost_sales_fcnn_rollout`
- `train_paper_lost_sales_fcnn`

Shared components:
- Residual order-up-to decoder.
- Action-cap deployment rule.
- Differentiable lost-sales reward rollout.
- Imitation warm-start and penalty hooks.

Architecture-specific components:
- None should be architecture-specific, but the function names still say `fcnn`.

Audit notes:
- This module is central shared infrastructure, but it is named and documented as FCNN-specific.
- At about 792 lines, it is too large for easy collaboration.

### `inventory_control/paper_lost_sales.py`

Responsibility:
- Paper-style lost-sales simulator, baseline policies, gamma quantiles, and evaluation demand paths.

Public entry points:
- `paper_lost_sales_step`
- `evaluate_paper_lost_sales_policy`
- `historical_demand_base_stock_policy`
- `non_omniscient_gamma_base_stock_policy`
- `omniscient_gamma_base_stock_policy`
- `paper_evaluation_demand_path`

Shared components:
- Reward calculation.
- Classical baseline policies.
- Evaluation demand path generation.

Architecture-specific components:
- None.

Audit notes:
- This is appropriately shared and should remain independent of neural architectures.

### `inventory_control/products.py`

Responsibility:
- Product generation and paper-style heterogeneous product instances.

Public entry points:
- `PaperLostSalesProduct`
- `PaperLostSalesProductGenerator`
- `ProductGeneratorConfig`

Shared components:
- Product/data generation and stored future demand traces.

Architecture-specific components:
- None.

Audit notes:
- This is a clean shared layer.

### `inventory_control/idp_benchmark.py`

Responsibility:
- Unified paper-style benchmark orchestration.

Public entry points:
- `run_idp_benchmark`
- `IDPBenchmarkArtifacts`

Shared components:
- Model registration.
- Training orchestration.
- Evaluation orchestration.
- Prediction-grid diagnostics.
- Target-stock structure diagnostics.
- Result writing.
- Plot generation.

Architecture-specific components:
- Currently contains direct factory branches for FCNN, constrained MLP, KAN, and MonoKAN.
- Contains MonoKAN spline export/certification calls.

Audit notes:
- This is the biggest maintainability risk. It is about 1565 lines and mixes registry, training, evaluation, diagnostics, plotting, result schemas, and architecture-specific decisions.
- Distribution-shift scripts import private helpers from it, making it a hidden shared API.

### `inventory_control/evaluation/`

Responsibility:
- Generic evaluation helpers and older architecture audit utilities.

Public entry points:
- `evaluation.runner`
- `evaluation.metrics`
- `evaluation.tuning`
- `evaluation.architecture_audit`

Shared components:
- Generic evaluation/tuning code.

Architecture-specific components:
- `architecture_audit.py` includes MONOkan-related and raw/deployed monotonicity diagnostics.

Audit notes:
- Some current benchmark diagnostics could be moved here, but the module currently reflects earlier raw-action experiments.

### `inventory_control/neural/`

Responsibility:
- Older neural policy and differentiable simulator code for non-paper experiments.

Public entry points:
- `TorchMLPPolicy`
- `TorchPolicyAdapter`
- `train_zero_lead_time_policy`
- `train_lead_time_product_policy`

Shared components:
- Legacy generic neural inventory-control experiments.

Architecture-specific components:
- Generic MLP policies.

Audit notes:
- This appears separate from the paper lost-sales path. It is still tested, but it is not the main current research path.

### `scripts/`

Responsibility:
- CLI entry points, diagnostics, sweeps, plotting, report generation.

Public entry points:
- Primary: `run_idp_benchmark.py`, `run_quick_smoke_test.py`, `run_distribution_shift_validation.py`, `run_distribution_shift_multiseed.py`.
- Diagnostics: sweeps, one-product overfit, plotting, audit scripts.

Shared components:
- Some reusable logic currently lives in scripts, especially distribution-shift evaluation and paired aggregation.

Architecture-specific components:
- Several scripts are FCNN-only, MonoKAN-only, or legacy raw-action experiments.

Audit notes:
- Scripts need classification and deprecation markers so teammates do not run stale workflows.

### `configs/`

Responsibility:
- Experiment settings.

Public entry points:
- `paper_lost_sales.yaml`
- `paper_lost_sales_debug.yaml`
- `paper_lost_sales_fcnn_alignment_medium.yaml`
- `paper_lost_sales_factorial_small.yaml`

Shared components:
- Product-generation settings.
- Training/evaluation profile definitions.
- Model-specific architecture settings.

Architecture-specific components:
- `fcnn`, `monokan`, `positive_weight_mlp`, and `constrained_monotone_mlp` sections.

Audit notes:
- Config profiles are understandable, but values are repeated across files and scripts sometimes override them.

### `tests/`

Responsibility:
- Unit, smoke, and benchmark behavior tests.

Public entry points:
- `pytest`

Shared components:
- Simulator, product generation, training/evaluation consistency, benchmark smoke tests.

Architecture-specific components:
- `test_paper_fcnn.py`
- `test_paper_monokan.py`
- `test_constrained_mlp.py`

Audit notes:
- Current tests cover many correctness paths but not every new script path.

### `docs/`

Responsibility:
- Reference notes, method framing, supervisor reports, PDFs.

Public entry points:
- `MONOKAN_METHOD_EVIDENCE.md`
- `MAGGIAR_2025_CODEX_REFERENCE.md`
- `PAPER_ALIGNMENT_CHECKLIST.md`
- `STRUCTURE_INFORMED_MONOKAN_PLAN.md`

Shared components:
- Method and experiment rationale.

Architecture-specific components:
- MonoKAN-specific evidence note and external MonoKAN PDF.

Audit notes:
- Docs contain absolute `/private/tmp/...` paths from development runs.
- README is behind the current residual-four-model workflow.

## 2. Collaboration Assessment

Can a teammate add a monotonic MLP through a clear interface?
- Partially.
- They can implement a model under `inventory_control/policies/neural/constrained_mlp.py` and set the informal attributes used by the shared decoder.
- There is no formal `ResidualPolicySpec`, registry object, or documented interface yet.

Can they use the same residual order-up-to decoder?
- Yes, technically.
- The shared decoder is `paper_lost_sales_target_level` plus `paper_lost_sales_raw_order` in `inventory_control/training/direct_backprop.py`.
- The decoder is controlled by informal model attributes: `paper_action_parameterization`, `paper_residual_center`, `paper_residual_scale`, and `paper_residual_anchor`.

Can they use the same training loop?
- Yes.
- `train_paper_lost_sales_fcnn` works for FCNN, constrained MLP, KAN, and MonoKAN because it accepts any `torch.nn.Module`.
- The name is misleading because it is no longer FCNN-only.

Can they use the same evaluation products and demand paths?
- Yes.
- Product generation is centralized in `PaperLostSalesProductGenerator`.
- Evaluation demand paths use deterministic common random numbers through `paper_evaluation_demand_path`.

Can they use target-stock diagnostics?
- Yes after the latest changes.
- `run_idp_benchmark` now emits `target_structure_diagnostics.csv`.
- However, these diagnostics live inside `idp_benchmark.py`, not a reusable diagnostics module.

Can they run the same benchmark scripts?
- Yes for `scripts/run_idp_benchmark.py` and distribution-shift scripts.
- The README does not yet advertise the new four-model residual comparison.

Can they avoid importing MonoKAN internals?
- Mostly yes.
- The constrained MLP file does not import MonoKAN.
- The benchmark imports both architecture families directly, which is acceptable for orchestration but should be replaced by a registry before the codebase grows further.

Overall collaboration verdict:
- The project is workable for a teammate, but not yet clean.
- The main obstacle is not algorithmic coupling; it is infrastructure coupling through large shared modules, private helper imports, and informal model attributes.

## 3. Problems Found

### Issue 1: `idp_benchmark.py` is too large and owns too many responsibilities

Classification: maintainability issue.

File/function:
- `inventory_control/idp_benchmark.py`
- Especially `run_idp_benchmark`, model branches, diagnostics, plotting, and artifact writing.

Why it matters:
- A teammate adding a monotonic MLP must understand a 1500+ line orchestrator.
- Architecture-specific registration is implemented as repeated `if "ModelName" in requested_models` blocks.
- Reusable diagnostics are hard to discover and reuse outside this benchmark.

Proposed minimal fix:
- Introduce a small model registry mapping policy names to factory, checkpoint filename, default seed offset, config section, and settings adapter.
- Move target-structure diagnostics to `inventory_control/evaluation/target_structure.py`.

Blocks commit?
- Should not block a research checkpoint commit if documented.
- Should block a "clean collaboration infrastructure" commit.

### Issue 2: Shared decoder/training code is named FCNN

Classification: naming/documentation issue.

File/function:
- `inventory_control/training/direct_backprop.py`
- `train_paper_lost_sales_fcnn`
- `paper_lost_sales_fcnn_order`
- `paper_lost_sales_fcnn_rollout`

Why it matters:
- These functions are now shared by FCNN, constrained MLP, KAN, and MonoKAN.
- New contributors may duplicate training code because names imply FCNN-only use.

Proposed minimal fix:
- Add alias names such as `train_paper_lost_sales_policy`, `paper_lost_sales_policy_order`, and `paper_lost_sales_policy_rollout`.
- Keep old names as compatibility wrappers.

Blocks commit?
- No, but should be fixed soon.

### Issue 3: Policy interface is informal

Classification: maintainability issue.

File/function:
- `paper_lost_sales_target_level`
- factories in `fcnn.py`, `constrained_mlp.py`, `monokan.py`

Why it matters:
- Residual behavior depends on dynamic attributes attached to `nn.Module` instances.
- Missing or misspelled attributes can silently change decoding semantics.

Proposed minimal fix:
- Define a small dataclass or protocol such as `PaperPolicyMetadata`.
- Provide helper `attach_residual_order_up_to_metadata(model, ...)`.

Blocks commit?
- No, if tests cover registered models.

### Issue 4: Distribution-shift scripts import private benchmark helpers

Classification: maintainability issue.

File/function:
- `scripts/run_distribution_shift_validation.py`
- Private imports from `inventory_control.idp_benchmark`: `_model_config`, `_operational_table`, `_oracle_levels`, `_prediction_summary`, `_prediction_table`, `_profile_settings`.

Why it matters:
- Private helper changes can break scripts without obvious API warnings.
- This makes the scripts hard to reuse as stable research infrastructure.

Proposed minimal fix:
- Promote reusable evaluation helpers into `inventory_control/evaluation/paper_benchmark.py` or similar.
- Or document these scripts as research-only wrappers.

Blocks commit?
- No, but should be fixed before larger collaborative development.

### Issue 5: Model registration is duplicated

Classification: maintainability issue.

File/function:
- `POLICY_ORDER` and `TRAINABLE_POLICIES` in `idp_benchmark.py`.
- `MODEL_FACTORIES` and `MODEL_FILENAMES` in distribution-shift scripts.
- Checkpoint filename mapping inside `run_idp_benchmark`.

Why it matters:
- Adding a policy requires edits in multiple files.
- A teammate can easily add training support but forget shift-evaluation support or checkpoint naming.

Proposed minimal fix:
- Centralize policy registry and checkpoint filenames in one package module.

Blocks commit?
- No, but it is the most important collaboration cleanup.

### Issue 6: Target-stock diagnostics are shared but embedded in benchmark orchestrator

Classification: maintainability issue.

File/function:
- `_target_structure_diagnostics` in `inventory_control/idp_benchmark.py`.

Why it matters:
- The diagnostic is valuable for monotonic MLP and MonoKAN work.
- Keeping it private in the benchmark discourages reuse.

Proposed minimal fix:
- Move it to `inventory_control/evaluation/target_structure.py`.
- Export a stable function with documented inputs and output schema.

Blocks commit?
- No.

### Issue 7: `raw_violation_rate` remains misleading for residual policies

Classification: experiment-comparability risk.

File/function:
- `_prediction_summary` in `inventory_control/idp_benchmark.py`.

Why it matters:
- For residual order-up-to models, inventory monotonicity is mostly guaranteed by the decoder for all models.
- Reporting this as the main structural metric can falsely imply MONOkan has a unique monotonic advantage.

Proposed minimal fix:
- Rename or document the metric as `inventory_order_violation_rate`.
- In reports, emphasize `target_structure_diagnostics.csv` for residual models.

Blocks commit?
- No, if docs explain it.
- Should block any paper/table claim that uses it as MONOkan evidence.

### Issue 8: `MonoKANOrderUpTo` name is misleading

Classification: naming/documentation issue.

File/function:
- `make_paper_order_up_to_monokan_model` in `inventory_control/policies/neural/monokan.py`.

Why it matters:
- It uses `monotonic_vars={}` and is therefore not a constrained MonoKAN model.
- A teammate may interpret it as a certified monotonic architecture.

Proposed minimal fix:
- Rename to `KANOrderUpTo` or document as unconstrained order-up-to KAN.
- Keep compatibility alias if needed.

Blocks commit?
- Should block final presentation if left unexplained.
- Does not block intermediate research commit.

### Issue 9: Config values are repeated across paper configs

Classification: maintainability issue.

File/function:
- `configs/paper_lost_sales.yaml`
- `configs/paper_lost_sales_debug.yaml`
- `configs/paper_lost_sales_fcnn_alignment_medium.yaml`
- `configs/paper_lost_sales_factorial_small.yaml`

Why it matters:
- The same experiment defaults appear in multiple files with different local overrides.
- It is hard to tell which config is authoritative.

Proposed minimal fix:
- Add comments marking `debug`, `medium`, `factorial_small`, and `paper_lost_sales` roles.
- Longer-term: support config inheritance or write a resolved config into every run output.

Blocks commit?
- No.

### Issue 10: Scripts override config values

Classification: experiment-comparability risk.

File/function:
- `scripts/run_distribution_shift_validation.py`
- `scripts/run_distribution_shift_multiseed.py`

Why it matters:
- CLI flags override evaluation product count, diagnostic product count, inventory states, oracle samples, seeds, and shifts.
- This is acceptable, but all resolved values must be recorded.

Proposed minimal fix:
- Write `resolved_config.yaml` and `run_metadata.json` for every script run.
- The multi-seed script now preserves seed configs, but shift overrides should also be saved explicitly.

Blocks commit?
- No, but should block final long-run evidence if missing.

### Issue 11: Distribution-shift logic is reusable package logic inside scripts

Classification: maintainability issue.

File/function:
- `scripts/run_distribution_shift_validation.py`
- `scripts/run_distribution_shift_multiseed.py`

Why it matters:
- Shift definitions, paired reward differences, and aggregation are useful beyond CLI execution.
- Tests cannot easily import script functions without relying on script internals.

Proposed minimal fix:
- Move shift definitions and paired aggregation into `inventory_control/evaluation/distribution_shift.py`.
- Keep scripts as thin CLI wrappers.

Blocks commit?
- No.

### Issue 12: Quick smoke test is stale

Classification: naming/documentation issue.

File/function:
- `scripts/run_quick_smoke_test.py`

Why it matters:
- It only trains raw FCNN and does not cover the current residual-policy path.
- The team may assume quick smoke validates current MONOkan work when it does not.

Proposed minimal fix:
- Add a new quick residual benchmark command to docs.
- Optionally update the smoke script or add `run_quick_residual_smoke.py`.

Blocks commit?
- No.

### Issue 13: README is behind current workflow

Classification: documentation issue.

File/function:
- `README.md`

Why it matters:
- It documents the older policy set and artifact list.
- It does not mention `ConstrainedMLPResidualOrderUpTo`, target-stock diagnostics, spline certification, or paired multi-seed shift outputs.

Proposed minimal fix:
- Add `docs/ARCHITECTURE.md` and `docs/TEAM_WORKFLOW.md`.
- Update README to point to them.

Blocks commit?
- No for code checkpoint.
- Yes for team handoff readiness.

### Issue 14: Generated PDF is tracked

Classification: harmless technical debt.

File/function:
- `output/pdf/idp_monotonic_inventory_report.pdf`

Why it matters:
- Generated binary artifacts add repository weight and stale-report risk.
- The path is not under ignored `outputs/`.

Proposed minimal fix:
- Decide whether final report PDFs belong in git.
- If not, add `output/` to `.gitignore` and remove the tracked PDF in a separate cleanup commit.

Blocks commit?
- No, because it predates this work.

### Issue 15: Documentation contains local absolute paths

Classification: documentation issue.

File/function:
- `docs/MONOKAN_METHOD_EVIDENCE.md`

Why it matters:
- Paths like `/private/tmp/...` are not reproducible for teammates.

Proposed minimal fix:
- Replace local temp paths with `outputs/...` examples or explicitly label them as local scratch paths.

Blocks commit?
- No, but should be cleaned before sharing.

### Issue 16: Large PDFs are untracked but present

Classification: harmless technical debt.

File/function:
- `docs/1-s2.0-S0893608025011591-main.pdf`
- `docs/2507.22040v1.pdf`
- `docs/Structure-Informed Neural Policies for Lost-Sales Inventory Control.pdf`

Why it matters:
- PDFs may be useful references but can add size and licensing ambiguity.

Proposed minimal fix:
- Keep if allowed by team policy, or replace with citation notes and local-only instructions.
- If committed, verify redistribution rights.

Blocks commit?
- Potentially yes if licensing/redistribution is unclear.

### Issue 17: Test coverage does not include new distribution-shift CLIs

Classification: correctness risk.

File/function:
- `scripts/run_distribution_shift_validation.py`
- `scripts/run_distribution_shift_multiseed.py`

Why it matters:
- These scripts are central to the robustness story.
- Current validation has compile checks and manual runs, but not automated smoke tests for CLI arguments, checkpoint loading, shift aggregation, and paired output schemas.

Proposed minimal fix:
- Add tests for shift config construction, model filename registry, paired aggregation, and skip-existing behavior.
- Add one tiny CLI smoke if runtime is acceptable.

Blocks commit?
- Should block a final robustness-evidence commit.
- Does not block an intermediate instrumentation commit if documented.

### Issue 18: Result schemas are spread across ad hoc CSV writes

Classification: maintainability issue.

File/function:
- `idp_benchmark.py`
- distribution-shift scripts
- older training/evaluation scripts

Why it matters:
- Adding or renaming columns can silently break plotting/report scripts.

Proposed minimal fix:
- Define expected schema constants or small writer helpers in `inventory_control/evaluation/result_schemas.py`.

Blocks commit?
- No.

### Issue 19: Architecture-specific diagnostics leak into shared benchmark

Classification: maintainability issue.

File/function:
- `_monokan_spline_certification` in `idp_benchmark.py`
- `sample_monokan_first_layer_splines` import in `idp_benchmark.py`

Why it matters:
- Shared benchmark code directly knows MonoKAN internals.
- This is tolerable for one architecture but will scale poorly.

Proposed minimal fix:
- Let architectures expose optional diagnostic hooks, or put MonoKAN certification in `inventory_control/evaluation/monokan_diagnostics.py`.

Blocks commit?
- No.

### Issue 20: Current evidence docs mix old and new experiment states

Classification: documentation issue.

File/function:
- `docs/MONOKAN_METHOD_EVIDENCE.md`
- `docs/STRUCTURE_INFORMED_MONOKAN_PLAN.md`

Why it matters:
- The document includes old three-model results and newer four-model plan.
- Readers may confuse historical diagnostics with current proposed evidence.

Proposed minimal fix:
- Split into "Historical diagnostics" and "Current protocol".
- Mark which commands were run and which are proposed.

Blocks commit?
- No.

## 4. Duplication Audit

Policy input preprocessing:
- `build_paper_lost_sales_features` lives in `fcnn.py`.
- `_model_orders`, `_batched_torch_policy_metrics`, and training batch construction manually build equivalent batch/context tensors.
- Minimal fix: move paper feature normalization/context construction to `inventory_control/paper_features.py`.

Historical base-stock anchoring:
- `_residual_anchor_level` in `direct_backprop.py`.
- `_anchor_policy` in `idp_benchmark.py`.
- Baseline factories in `paper_lost_sales.py`.
- Minimal fix: centralize residual anchor selection in one shared module.

Residual order-up-to decoding:
- `paper_lost_sales_target_level` and `paper_lost_sales_uncapped_order` in `direct_backprop.py`.
- `_target_level_from_context` duplicated in `idp_benchmark.py` for diagnostics.
- Minimal fix: expose a context-aware target decoder from training or a policy-decoder module.

Action capacity guards:
- `paper_lost_sales_fcnn_order`.
- `paper_lost_sales_fcnn_rollout`.
- `_model_orders`.
- `_batched_torch_policy_metrics`.
- `PaperFCNNPolicyAdapter` and `PaperMonoKANPolicyAdapter`.
- Minimal fix: shared `apply_order_cap(order, inventory, historical_mean, max_order_factor)`.

Training loops:
- Main DirectBackprop loop is centralized.
- Older training scripts duplicate orchestration and output writing.
- Minimal fix: mark older FCNN/MONOKAN raw-action scripts as legacy or route them through `run_idp_benchmark`.

Checkpoint handling:
- Benchmark checkpoint filenames are in `idp_benchmark.py`.
- Distribution-shift checkpoint filenames are duplicated in `run_distribution_shift_validation.py`.
- Minimal fix: central `POLICY_REGISTRY`.

Evaluation rollout:
- Scalar policy evaluation lives in `paper_lost_sales.py`.
- Batched torch evaluation lives in `idp_benchmark.py`.
- Training differentiable rollout lives in `direct_backprop.py`.
- Minimal fix: document why three paths exist and test equivalence for representative policies.

Reward calculation:
- `paper_lost_sales_step` and `evaluate_paper_lost_sales_policy`.
- Differentiable reward in `paper_lost_sales_fcnn_rollout`.
- Batched reward in `_batched_torch_policy_metrics`.
- Minimal fix: keep separate NumPy/Torch implementations but maintain explicit parity tests.

Target-stock diagnostics:
- New diagnostics are private in `idp_benchmark.py`.
- Older architecture audit diagnostics live in `inventory_control/evaluation/architecture_audit.py`.
- Minimal fix: move target-stock diagnostics into evaluation package.

Model registration:
- Policy names appear in `POLICY_ORDER`, `TRAINABLE_POLICIES`, benchmark branches, checkpoint filename map, distribution shift maps, and script defaults.
- Minimal fix: central registry.

Result writing:
- Benchmark writes many CSVs directly.
- Distribution-shift scripts write their own summaries.
- Older scripts write fixed paths.
- Minimal fix: central result writer or schema constants.

## 5. Shared API Audit

Policy forward pass:
- Current interface: `torch.nn.Module.forward(features) -> Tensor`.
- Status: usable but not explicitly documented.

Target-stock prediction:
- Current interface: `paper_lost_sales_target_level(model, batch)`.
- Status: shared but buried in `direct_backprop.py`.

Residual prediction:
- Current interface: model outputs residual factor, interpreted by `paper_residual_center` and `paper_residual_scale`.
- Status: informal.

Order-up-to action decoding:
- Current interface: `paper_lost_sales_raw_order` and `paper_lost_sales_fcnn_order`.
- Status: shared behavior but misleading names.

Feature normalization:
- Current interface: `compute_paper_fcnn_feature_stats` and batch builders.
- Status: shared behavior in FCNN-named module.

Training:
- Current interface: `train_paper_lost_sales_fcnn`.
- Status: works for all models but needs alias/doc cleanup.

Evaluation:
- Current interface: `run_idp_benchmark`, scalar `evaluate_paper_lost_sales_policy`, private batched helpers.
- Status: public benchmark API exists; reusable lower-level evaluation API is private.

Structural diagnostics:
- Current interface: private `_target_structure_diagnostics` and `_monokan_spline_certification`.
- Status: should be promoted or documented.

Checkpoint save/load:
- Current interface: ad hoc filename maps.
- Status: needs registry.

Benchmark model registration:
- Current interface: scattered constants and branch blocks.
- Status: functional but not clean.

## 6. Script Audit

Active primary scripts:
- `scripts/run_idp_benchmark.py`: primary benchmark entry point.
- `scripts/run_quick_smoke_test.py`: quick legacy FCNN smoke; should be supplemented with residual smoke.
- `scripts/run_distribution_shift_validation.py`: active robustness diagnostic.
- `scripts/run_distribution_shift_multiseed.py`: active paired multi-seed robustness diagnostic.

Diagnostics:
- `scripts/run_monokan_multiseed_validation.py`: compact multiseed residual validation.
- `scripts/run_monokan_optimization_sweep.py`: MONOkan architecture sweep.
- `scripts/plot_controlled_policy_surfaces.py`: residual policy surface plots.
- `scripts/plot_policy_contours_from_predictions.py`: plots from benchmark predictions.
- `scripts/run_architecture_scientific_audit.py`: older compact architecture audit.
- `scripts/analyze_paper_lost_sales_fcnn_failures.py`: FCNN failure analysis.
- `scripts/overfit_paper_lost_sales_fcnn_one_product.py`: overfit diagnostic.
- `scripts/overfit_paper_lost_sales_monokan_one_product.py`: overfit diagnostic.

Deprecated or legacy scripts:
- `scripts/train_paper_lost_sales_fcnn.py`: older direct FCNN training workflow.
- `scripts/train_paper_lost_sales_monokan.py`: older raw MONOkan workflow.
- `scripts/evaluate_paper_lost_sales_fcnn.py`: older checkpoint-specific evaluator.
- `scripts/evaluate_paper_lost_sales_monokan.py`: older checkpoint-specific evaluator.
- `scripts/sweep_paper_lost_sales_fcnn.py`: older FCNN sweep.
- `scripts/sweep_paper_lost_sales_monokan.py`: older raw MONOkan sweep.
- `scripts/plot_paper_lost_sales_policy_structure.py`: older FCNN plot path.

Other project scripts:
- `scripts/run_baselines.py`
- `scripts/run_benchmark_baselines.py`
- `scripts/run_product_benchmark_baselines.py`
- `scripts/run_paper_lost_sales_baselines.py`
- `scripts/plot_baselines.py`
- `scripts/plot_benchmark_baselines.py`
- `scripts/train_product_lead_time_mlp.py`
- `scripts/train_zero_lead_time_mlp.py`
- `scripts/tune_base_stock.py`
- These appear to belong to earlier generic inventory-control work and should be kept but separated in docs from the current paper lost-sales path.

Duplicate wrappers:
- FCNN train/evaluate/sweep scripts duplicate functionality now better covered by `run_idp_benchmark.py`.
- MonoKAN train/evaluate/sweep scripts include old raw-action assumptions and need deprecation notes.

## 7. Config Audit

Common experiment settings:
- Shared values are repeated across paper configs: history length, horizon, burn-in, product-generation parameters, model sizes, action caps.
- This is manageable but error-prone.

Model-specific separation:
- `fcnn`, `monokan`, `positive_weight_mlp`, and `constrained_monotone_mlp` sections exist.
- Residual-specific settings such as `residual_scale` and `residual_anchor` appear under model sections but are not uniformly documented.

Profiles:
- `compact`, `full`, and `paper_replication` profiles are identifiable in `paper_lost_sales.yaml` and `paper_lost_sales_fcnn_alignment_medium.yaml`.
- `paper_lost_sales_debug.yaml` is used by tests.
- `paper_lost_sales_factorial_small.yaml` is a local experiment config and needs a header explaining its intended role.

Silent overrides:
- `run_distribution_shift_validation.py` overrides validation products, diagnostic products, inventory states, and oracle samples through CLI.
- `run_distribution_shift_multiseed.py` overrides training seed and optionally product-generation seed.
- `idp_benchmark._profile_settings` merges global and profile-specific settings.

Resolved config recording:
- `run_idp_benchmark` writes metadata including settings and feature stats.
- Distribution-shift scripts write result CSVs, but shift overrides and full resolved shift configs are not written per shift.
- Multi-seed now preserves effective seed configs, but this should be verified in tests.

## 8. Test Audit

Simulator correctness:
- Covered by `tests/test_lost_sales_env.py`.
- Paper-style reward/decomposition covered by `tests/test_paper_lost_sales.py`.

Product generation:
- Covered by `tests/test_products.py` and paper product tests in `tests/test_paper_lost_sales.py`.

Training/evaluation consistency:
- Covered by `tests/test_paper_fcnn.py` and `tests/test_idp_benchmark.py`.
- DirectBackprop and evaluation path consistency have targeted tests.

Policy architecture:
- FCNN covered by `tests/test_paper_fcnn.py`.
- MonoKAN covered by `tests/test_paper_monokan.py`.
- Constrained MLP covered by `tests/test_constrained_mlp.py` and new residual tests.

Monotonicity diagnostics:
- Raw/deployed inventory monotonicity covered.
- Context target-stock monotonicity export now has benchmark smoke coverage.
- Need lower-level unit tests for target-stock diagnostic values and autograd signs.

Spline certification:
- `sample_monokan_first_layer_splines` has a reconstruction test.
- New benchmark export is tested.
- Need direct unit tests for `monokan_spline_certification` detecting violations if projection is broken.

Paired experiment validity:
- Product/evaluation common random numbers are tested in paper lost-sales tests.
- Paired multi-seed aggregation is not yet unit tested.

CLI/script smoke tests:
- Baseline scripts have some tests.
- Main benchmark has smoke tests through `run_idp_benchmark`.
- New distribution-shift scripts do not have CLI smoke tests.

Untested critical paths:
- `scripts/run_distribution_shift_validation.py` CLI and checkpoint loading.
- `scripts/run_distribution_shift_multiseed.py` skip-existing behavior and paired output schemas.
- Resolved-config persistence for shifted runs.
- README-documented commands after current workflow changes.
- Full test suite after the most recent additions.

## 9. Git Hygiene

Generated outputs/checkpoints:
- `.gitignore` ignores `outputs/`, `*.pt`, `*.pth`, and `*.ckpt`.
- `outputs/` is not tracked.
- `output/pdf/idp_monotonic_inventory_report.pdf` is tracked.

Large accidental files:
- Large PDFs are untracked in `docs/`.
- `idinn-main.zip` and `idinn-main/` are ignored and untracked.
- No generated model checkpoints appear staged because nothing is staged.

Local absolute paths:
- `docs/MONOKAN_METHOD_EVIDENCE.md` contains `/private/tmp/...` development paths.
- No `/Users/linkeming/...` paths were found in tracked/untracked project docs/scripts by `rg`.

Secrets:
- No obvious secrets were found by quick text search.

Commented-out dumps:
- No obvious large commented-out experimental dumps were found in changed files.
- Config TODO strings exist and should either remain explicit or move to docs.

Licensing/citation:
- Project has `LICENSE`.
- `third_party_licenses/MonoKAN_LICENSE` exists.
- `third_party/MonoKAN` is ignored.
- PDF redistribution status needs team decision before committing PDFs.

## Phase 1 Blockers Before Commit

Blockers for a clean collaboration-ready commit:
- `README.md` is stale relative to the residual-four-model workflow.
- No `docs/ARCHITECTURE.md` or `docs/TEAM_WORKFLOW.md` yet.
- New distribution-shift scripts lack automated tests.
- Policy registration and checkpoint filenames are duplicated across benchmark and scripts.
- Large untracked PDFs need a licensing decision before inclusion.

Non-blockers for an intermediate research checkpoint:
- Oversized `idp_benchmark.py` and `direct_backprop.py`.
- Misleading `fcnn` names in shared training helpers, if aliases/documentation are added soon.
- Ignored local generated outputs.
- Existing tracked generated PDF, if not touched in this commit.

## Recommended Cleanup

Minimal cleanup before committing:
- Add `docs/ARCHITECTURE.md`.
- Add `docs/TEAM_WORKFLOW.md`.
- Add a minimal example showing how to implement a residual order-up-to policy.
- Update README to point to the architecture/workflow docs.
- Add deprecation notes to legacy raw FCNN/MONOKAN scripts.
- Add tests for distribution-shift aggregation and output schemas.
- Add `.gitignore` coverage for `output/` if generated PDFs should not be tracked.

Medium cleanup after the next commit:
- Create a central model registry.
- Move target-stock diagnostics into `inventory_control/evaluation/`.
- Add shared aliases for policy training/rollout/order decoding.
- Move paper feature construction out of `fcnn.py`.
- Centralize result schema constants.

## Recommended Commit Boundaries

Commit 1: residual policy and diagnostics infrastructure
- `inventory_control/policies/neural/constrained_mlp.py`
- `inventory_control/policies/neural/fcnn.py`
- `inventory_control/policies/neural/monokan.py`
- `inventory_control/training/direct_backprop.py`
- `inventory_control/idp_benchmark.py`
- `inventory_control/policies/neural/__init__.py`
- related tests

Commit 2: experiment scripts and configs
- `configs/paper_lost_sales_factorial_small.yaml`
- `configs/paper_lost_sales_fcnn_alignment_medium.yaml`
- `scripts/run_distribution_shift_validation.py`
- `scripts/run_distribution_shift_multiseed.py`
- `scripts/run_monokan_multiseed_validation.py`
- `scripts/run_monokan_optimization_sweep.py`
- plotting scripts

Commit 3: documentation and workflow
- `docs/MAGGIAR_2025_CODEX_REFERENCE.md`
- `docs/PAPER_ALIGNMENT_CHECKLIST.md`
- `docs/STRUCTURE_INFORMED_MONOKAN_PLAN.md`
- `docs/MONOKAN_METHOD_EVIDENCE.md`
- `docs/PRE_COMMIT_CODEBASE_AUDIT.md`
- future `docs/ARCHITECTURE.md`
- future `docs/TEAM_WORKFLOW.md`
- README updates

Commit 4: external references, only if approved
- PDFs in `docs/`, subject to redistribution/citation decision.
- `uv.lock`, subject to team decision on lockfile policy.
