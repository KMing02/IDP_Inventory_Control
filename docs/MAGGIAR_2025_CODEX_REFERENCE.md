# Maggiar Et Al. 2025 Codex Reference

Source PDF: `docs/2507.22040v1.pdf`

Paper: Alvaro Maggiar, Sohrab Andaz, Akhil Bagaria, Carson Eisenach, Dean Foster, Omer Gottesman, Dominique Perrault-Joncas, "Structure-Informed Deep Reinforcement Learning for Inventory Management", arXiv:2507.22040v1, July 29, 2025.

Purpose of this note: give Codex a compact reference for the paper-specific inventory setup used by this repository. For exact wording, equations, figures, and tables, consult the PDF.

## Main Takeaways

- The repo's primary replication target is the paper's no-lead-time, multi-period lost-sales setting in Section 4.2.
- The paper trains a DirectBackprop-style policy across products, not one policy per product.
- The DRL agent should use practical observable information: historical demand and product economics, not hidden Gamma parameters.
- The main classical references for the no-lead-time lost-sales case are:
  - Omniscient optimal: knows the true Gamma distribution parameters.
  - Non-omniscient optimal: knows the Gamma/base-stock structure, but estimates Gamma parameters from historical demand.
  - DRL: learns from historical demand and economics without being told the demand family or base-stock structure.
- In the default no-lead-time lost-sales experiment, the paper reports DRL matching the non-omniscient optimal benchmark and coming within roughly 0.5 percent of the omniscient benchmark.
- The paper's structure-informed method is penalty-based targeted regularization over structural derivative inequalities, not a MonoKAN architecture. MonoKAN is this repo's architectural experiment inspired by the same structural motivation.

## Section Map

- Section 3: Exo-MDP formulation and DirectBackprop setup.
- Section 4.1: Generated product population, training hyperparameters, and evaluation protocol.
- Section 4.2: No-lead-time lost-sales model, benchmark, experiments, results, and policy-structure diagnostics.
- Section 4.3: Lost sales with positive lead times.
- Section 5: Structural properties, derivative constraints, calibration metrics, and structure-informed regularization.
- Appendix: DirectBackprop pseudocode.

## Product Generation

From Table 1:

- Price `p`: exponential with mean `p_m = 100`.
- Purchase cost `c`: `p * U_1`, where `U_1` is uniform on `[0, 1]`.
- Lost-sale penalty `b` or `k`: `k_m * U_4`, with `k_m = 10`.
- Holding cost `h`: exponential with mean `h_m = 5`.
- Mean demand `mu`: exponential with mean `mu_m = 100`.
- Demand coefficient of variation `nu`: uniform on `[0, 1]`.
- Demand distribution: Gamma, parameterized from `mu` and `nu`.
- Historical demand window: `H = 32` by default.
- Training path horizon: `T = 100` periods after the history window.
- Default training product count: `N = 40,000`.
- Default test/evaluation product count: `100,000`.

Repo mapping:

- Product generation: `inventory_control/products.py`
- Configs: `configs/paper_lost_sales.yaml`, `configs/paper_lost_sales_colab.yaml`, `configs/paper_lost_sales_debug.yaml`

## No-Lead-Time Lost-Sales Model

State:

- Endogenous state is on-hand inventory `y_t`.
- Time-series exogenous input is recent demand history `(d_{t-H}, ..., d_{t-1})`.
- Static exogenous product features are economics `(p, c, h, b)`.

Action:

- `q_t`: quantity purchased at the beginning of period `t`.
- With no lead time, the order is available immediately.

Transition:

```text
y_{t+1} = max(y_t + q_t - d_t, 0)
```

Reward:

```text
R_t =
  p * min(d_t, y_t + q_t)
  - c * q_t
  - b * max(d_t - y_t - q_t, 0)
  - h * max(y_t + q_t - d_t, 0)
```

Repo mapping:

- Step/evaluation: `inventory_control/paper_lost_sales.py`
- Differentiable rollout: `inventory_control/training/direct_backprop.py`

## Critical Fractile And Base-Stock Benchmark

For no-lead-time lost sales, the myopic/base-stock benchmark uses:

```text
underage cost c_u = p - c + b
overage cost c_o = h
F_D(y*) = c_u / (c_u + c_o)
q_t = max(y* - y_t, 0)
```

Implementation consequence:

```text
critical_ratio = (p + b - c) / (p + b - c + h)
```

This is the formula Codex should check when touching `paper_critical_quantile`.

Repo mapping:

- `paper_critical_quantile` in `inventory_control/paper_lost_sales.py`
- Regression tests in `tests/test_paper_lost_sales.py`

## Training Protocol

Default paper hyperparameters from Table 2:

- `H = 32`
- `T = 100`
- Discount factor `gamma = 1`
- Causal CNN dilations: `[1, 2, 4, 8, 16]`
- CNN channels: `8`
- MLP layers: `2`
- MLP neurons per layer: `32`
- Activation: `ELU`
- Optimizer: `Adam`
- Learning rate: `0.001`
- Training sample size: `40,000`
- Batch size: `2,500`
- Epochs: `1,000`

The paper initializes endogenous inventory randomly during training. For the no-lead-time lost-sales case, the paper samples initial inventory between zero and roughly twice mean demand.

Repo mapping:

- Training batch and rollout: `inventory_control/training/direct_backprop.py`
- FCNN model: `inventory_control/policies/neural/fcnn.py`
- Training script: `scripts/train_paper_lost_sales_fcnn.py`

## Evaluation Protocol

The paper evaluates stationary settings with:

- `100,000` independent products.
- `520` simulated periods.
- First `20` periods discarded as burn-in.
- Average reward computed over the remaining `500` periods.
- Endogenous inventory initialized to zero for evaluation.

Repo mapping:

- Shared evaluation counts/seeds: `inventory_control/paper_evaluation.py`
- Baseline/evaluation scripts:
  - `scripts/run_paper_lost_sales_baselines.py`
  - `scripts/evaluate_paper_lost_sales_fcnn.py`
  - `scripts/evaluate_paper_lost_sales_monokan.py`

## Paper Results To Anchor Against

No-lead-time lost-sales default experiment, Table 3:

- DRL reward: `4548.95`
- Non-omniscient optimal reward: `4548.95`
- Omniscient optimal reward: `4567.58`

Interpretation:

- The paper's learned DRL policy should be very close to the non-omniscient Gamma benchmark in the paper-scale setting.
- Compact local smoke tests should not be treated as paper-level evidence.
- If compact results are far from these relationships, inspect scale, training horizon, product count, checkpointing, feature preprocessing, and critical-ratio implementation before drawing architectural conclusions.

## Structural Policy Expectations

For no-lead-time lost sales:

- The optimal policy is order-up-to/base-stock.
- Purchase quantity should be nonincreasing in current inventory.
- In the active region below the order-up-to level, the slope should be approximately `-1`.
- Above the order-up-to level, purchase quantity should be zero.

For lead-time and other extensions:

- The paper motivates derivative-based structural constraints.
- Structure-informed learning in the paper adds penalties for violations of derivative inequalities during training.
- The paper explicitly notes that monotonicity alone is not the whole structure; sensitivity magnitudes and relative sensitivities can matter.

Repo-specific caution:

- MonoKAN is not the paper's method. It is this repo's architecture-enforced monotonicity experiment.
- Do not claim paper replication success for MonoKAN unless it is evaluated under the paper protocol and compared fairly.

## Codex Checklist Before Changing Paper Logic

- Confirm whether the change targets the paper's no-lead-time lost-sales setting or a repo extension.
- Preserve practical information constraints: learned policy should not consume hidden Gamma parameters.
- Keep critical ratio as `(p + b - c) / (p + b - c + h)`.
- Keep reward as revenue minus purchase, lost-sale, and holding costs.
- Keep training and evaluation feature preprocessing aligned.
- Report validation vs test split explicitly.
- Do not use compact/debug runs as final paper claims.
