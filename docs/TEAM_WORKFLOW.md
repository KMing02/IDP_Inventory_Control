# Team Workflow

Use this workflow for collaboration on the residual order-up-to lost-sales benchmark.

## Branches

- Use short topic branches: `monomlp/<topic>`, `monokan/<topic>`, `benchmark/<topic>`, or `docs/<topic>`.
- Do not mix architecture tuning, simulator changes, and report-writing in one branch.
- Before committing, run `git status --short --branch` and inspect untracked files.

## Ownership Boundaries

- Shared inventory infrastructure: `products.py`, `paper_lost_sales.py`, `paper_configs.py`, benchmark orchestration, training/decoder utilities.
- Monotonic MLP architecture: `inventory_control/policies/neural/constrained_mlp.py` and related tests.
- KAN and MonoKAN architecture: `inventory_control/policies/neural/monokan.py` and spline diagnostics.
- Experiments and reporting: `configs/`, `scripts/`, `docs/`, and generated artifacts under `outputs/`.

Monotonic MLP code should not import MonoKAN-specific modules.

## Tests

Run the full test suite before a commit:

```bash
python -m pytest
```

Run targeted architecture tests during development:

```bash
python -m pytest tests/test_constrained_mlp.py tests/test_paper_monokan.py tests/test_paper_fcnn.py tests/test_idp_benchmark.py -q
```

Run distribution-shift script tests:

```bash
python -m pytest tests/test_distribution_shift_scripts.py -q
```

## Smoke Benchmarks

Quick legacy FCNN smoke:

```bash
python scripts/run_quick_smoke_test.py
```

Tiny comparable residual benchmark:

```bash
python scripts/run_idp_benchmark.py --mode structure-ls --profile compact --config configs/paper_lost_sales_debug.yaml --models PaperFCNNResidualOrderUpTo,ConstrainedMLPResidualOrderUpTo,MonoKANResidualOrderUpTo --output-dir outputs/idp_benchmark/residual_smoke
```

Do not treat compact smoke results as final evidence.

## Comparable Experiments

For fair comparisons, keep these identical across models:

- product split and product-generation seed;
- evaluation demand paths;
- feature preprocessing and normalization;
- residual anchor;
- residual scale;
- capacity guard;
- DirectBackprop objective;
- training products, batch size, horizon, epochs, learning rate, validation interval;
- checkpoint-selection rule;
- evaluation split, horizon, burn-in, and seed start.

Use `metadata.json` in each run output to verify resolved benchmark settings.

## Pull-Request Checklist

- No generated outputs, plots, model checkpoints, or local caches are included unless explicitly intended.
- No silent changes to reward formulas, product generation, data splits, action decoding, capacity guards, or evaluation seeds.
- Any output-schema change is documented and tested.
- Any new experiment script has tests for arguments or callable logic, expected output files, and result schema.
- Any new model has factory, registry, checkpoint-name, and benchmark smoke coverage.
- Any monotonicity claim has a structural diagnostic or proof path.
- README or docs point users to the current workflow.

## Generated Files

Generated outputs belong under `outputs/` and should not be committed:

```text
outputs/
*.pt
*.pth
*.ckpt
```

Local literature PDFs should not be committed unless redistribution rights are clear. Prefer citation notes or URLs in docs.
