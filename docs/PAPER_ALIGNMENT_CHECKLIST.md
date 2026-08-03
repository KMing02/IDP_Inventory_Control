# Paper Alignment Checklist

Source PDF: `docs/2507.22040v1.pdf`

Use this checklist when reviewing or modifying the paper-style lost-sales path.

## Core Formula Checks

- Reward:
  ```text
  p * min(d, y + q)
  - c * q
  - b * max(d - y - q, 0)
  - h * max(y + q - d, 0)
  ```
- Transition:
  ```text
  y_next = max(y + q - d, 0)
  ```
- Base-stock critical ratio:
  ```text
  (p + b - c) / (p + b - c + h)
  ```
- Base-stock action:
  ```text
  q = max(S - y, 0)
  ```

## Information Constraints

- Learned policies may use historical demand and product economics.
- Learned policies must not use hidden Gamma shape/scale, true demand mean, or true coefficient of variation.
- Omniscient benchmark may use true Gamma parameters.
- Non-omniscient benchmark may fit Gamma parameters from historical demand.

## Default Paper Scale

- History length: `H = 32`
- Training horizon: `T = 100`
- Evaluation horizon: `520`
- Burn-in: `20`
- Training products: `40,000`
- Evaluation/test products: `100,000`
- Batch size: `2,500`
- Epochs: `1,000`
- Optimizer: `Adam`
- Learning rate: `0.001`

## Local Run Labels

- `debug` or `compact`: smoke test only.
- `validation`: model selection and diagnostics.
- `test` with `--final-test`: final held-out evaluation only after protocol is frozen.
- Do not cite compact results as paper replication evidence.

## Diagnostics To Report

- Average reward per period.
- Gap to omniscient optimal.
- Gap to non-omniscient optimal.
- Service level and lost-sales rate.
- Mean inventory and mean order quantity.
- Raw and deployed monotonicity violation rates.
- Order-up-to MAE/RMSE against omniscient and non-omniscient targets where applicable.
- Cap-hit/saturation rate for neural policies.
- Training vs evaluation preprocessing parity.

## MonoKAN-Specific Cautions

- MonoKAN is not the paper's structure-informed regularization method.
- Architectural monotonicity does not guarantee correct order magnitude.
- If MonoKAN raw orders require clipping to look reasonable, the model is not calibrated.
- If bounded MonoKAN under-orders, inspect warm-start length, checkpoint selection, and reward fine-tuning before blaming monotonicity.
- Compare MonoKAN under both parameter-matched and paper-faithful protocols before making architecture claims.

## Relevant Repo Files

- `inventory_control/paper_lost_sales.py`
- `inventory_control/products.py`
- `inventory_control/paper_evaluation.py`
- `inventory_control/training/direct_backprop.py`
- `inventory_control/policies/neural/fcnn.py`
- `inventory_control/policies/neural/monokan.py`
- `scripts/run_paper_lost_sales_baselines.py`
- `scripts/train_paper_lost_sales_fcnn.py`
- `scripts/evaluate_paper_lost_sales_fcnn.py`
- `scripts/train_paper_lost_sales_monokan.py`
- `scripts/evaluate_paper_lost_sales_monokan.py`
- `scripts/run_idp_benchmark.py`
