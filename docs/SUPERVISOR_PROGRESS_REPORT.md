# Structure-Informed Neural Inventory Control: Progress Report

## Project Goal

This project studies whether inventory structure, especially monotonicity, improves neural inventory-control policies.

The current policy comparison includes:

- Historical-demand base-stock
- Oracle Gamma base-stock
- Unconstrained MLP
- MLP with a monotonicity penalty
- Strictly monotone MonoKAN

The neural policies are trained through differentiable lost-sales rollouts using DirectBackprop-style reward optimization rather than generic reinforcement learning.

> **Terminology note:** Branch `1` currently implements a soft monotonicity-penalty MLP, not a strictly monotone MLP architecture. We should confirm whether this is what the project intends to call "Mono-MLP."

## Work Completed

The repository currently includes:

- Zero-lead-time lost-sales simulator
- Lead-time lost-sales simulator
- Dual-sourcing simulator
- Gamma and other demand processes
- Product-level train, validation, and test generation
- Hidden true demand parameters and observable historical demand
- Paper-style profit calculation:

  ```text
  revenue - purchase cost - holding cost - lost-sales penalty
  ```

- Historical and oracle base-stock benchmarks
- FCNN and MonoKAN policies
- DirectBackprop training
- Validation-based checkpoint selection
- Common product demand traces across policies
- Monotonicity and policy-curve diagnostics
- Product-level failure analysis
- Action-cap sensitivity and calibration diagnostics

## Current Experimental Data

The latest comparable validation experiment used:

- 256 training products
- 100 validation products
- Historical demand length `H = 32`
- Training horizon `T = 100`
- Evaluation horizon `520`
- Burn-in `20`
- Gamma demand
- Hidden mean demand and coefficient of variation
- Fixed per-product future demand traces
- 20 imitation epochs
- 50 reward-optimization epochs
- Validation products only; the held-out test set has not been used

The current paper-style product generator samples:

- Price: exponential distribution with mean 100
- Purchase cost: uniform fraction of price
- Holding cost: exponential distribution with mean 5
- Lost-sales penalty: uniform between 0 and 10
- Mean demand: exponential distribution with mean 100
- Demand coefficient of variation: uniform between 0 and 1

These distributions should be confirmed against the intended setup from Maggiar et al. (2025).

## Current Neural Architectures

The common benchmark currently uses:

| Model | Architecture | Trainable parameters |
|---|---|---:|
| UnconstrainedMLP | Flat-history MLP, two hidden layers with 16 ELU units | 897 |
| PenaltyMLP | Same MLP with a soft paired-state monotonicity penalty | 897 |
| MonoKAN | Hermite-spline MonoKAN, two hidden layers with 16 units | 8,673 |

The models are not parameter-matched: MonoKAN currently has approximately ten times as many trainable parameters as the MLP.

## Current Validation Results

The latest bounded-action experiment used the same neural action cap:

```text
maximum inventory level = 2 * historical mean demand
```

The neural action was parameterized as:

```text
order = sigmoid(model logit) * feasible order cap
```

Results:

| Policy | Mean total reward | Service level | Mean inventory | Mean order quantity |
|---|---:|---:|---:|---:|
| HistoricalDemandBaseStock | 1.700M | 0.707 | 19.82 | 68.03 |
| OracleGammaBaseStock | 1.736M | 0.706 | 18.93 | 68.22 |
| UnconstrainedMLP | 1.438M | 0.752 | 12.40 | 63.41 |
| PenaltyMLP | 1.438M | 0.752 | 12.40 | 63.41 |
| MonoKAN | **2.050M** | **0.960** | **95.87** | **90.73** |

Product-level validation comparisons:

- MonoKAN beats HistoricalDemandBaseStock on 62% of products.
- MonoKAN beats OracleGammaBaseStock on 64% of products.
- UnconstrainedMLP beats HistoricalDemandBaseStock on 37% of products.
- UnconstrainedMLP beats OracleGammaBaseStock on 42% of products.
- PenaltyMLP has effectively the same reward and product-level performance as UnconstrainedMLP in this experiment.

These are preliminary validation results, not final test results.

## Calibration Investigation

The original MonoKAN policy produced extremely large raw order quantities:

```text
mean raw order approximately 27,500
```

The deployment cap reduced these predictions to feasible actions. This meant that the cap, rather than the learned output magnitude, largely determined policy behavior.

A bounded-action parameterization was therefore introduced:

```text
order = sigmoid(model logit) * feasible order cap
```

This removed the numerically extreme raw orders. Under a shared cap of two times historical mean demand:

```text
MonoKAN mean physical order: approximately 93
```

However, MonoKAN still selected approximately 99% of the available cap across most policy-curve states. A smaller spline initialization and negative output-bias initialization did not change the final selected policy.

Reducing the cap from four to two times historical mean demand:

- Increased MonoKAN mean reward from approximately 1.744M to 2.050M
- Reduced mean inventory from approximately 279 to 96
- Reduced service level from 0.997 to 0.960

This suggests that the cap of four allowed excessive stockholding and that a cap of two produced a healthier reward/inventory trade-off.

## Current Interpretation

The current evidence suggests:

> Strict monotonicity successfully controls policy direction but does not, by itself, guarantee economically calibrated order magnitudes.

MonoKAN currently achieves strong mean reward and service level, but carries substantially more inventory than the base-stock policies and remains sensitive to the action cap.

The bounded-action formulation fixes the raw numerical explosion, but MonoKAN still learns a policy close to "order as much as permitted." This appears to be learned policy behavior under the current reward and benchmark configuration rather than a basic lost-sales simulator error.

The monotonicity-penalty MLP also requires careful interpretation:

- Before bounded actions, it still showed frequent raw monotonicity violations.
- Under bounded actions, deployed actions can appear monotone partly because the feasible order cap decreases with inventory.
- Therefore, deployed violation rate alone is insufficient.
- Monotonicity should also be measured on model logits or predicted order fractions while holding the action cap fixed.

## Current Limitations

- Results use validation products only and should not be treated as final test performance.
- Only one main random seed has been evaluated.
- The local experiment uses 256 training products rather than the paper-scale 40,000 products.
- Training uses 50 reward epochs rather than the paper-style 1,000 epochs.
- MonoKAN and MLP parameter counts are not matched.
- The oracle Gamma base-stock policy is an informed heuristic benchmark, not necessarily a globally optimal dynamic policy.
- Product-distribution parameters still require confirmation against the primary paper.
- Action-cap choice materially affects conclusions.
- The current soft-penalty MLP is not a strictly monotone architecture.

## Questions for the Supervisor

1. Should the primary benchmark be an exact reduced-scale replication of the paper, or is a broader benchmark expected?
2. Are there recommended public datasets or established inventory-control benchmarks?
3. Should we use only synthetic Gamma-demand products, or also include real demand data?
4. Should zero-lead-time lost sales remain the primary problem, or are lead-time and dual-sourcing results required?
5. Which product-distribution parameters from the paper must be replicated exactly?
6. Should neural model capacity be parameter-matched across MLP and MonoKAN?
7. Is the intended "Mono-MLP" a strictly monotone architecture or an MLP with a soft monotonicity penalty?
8. Should the action cap be treated as part of the benchmark, tuned using validation products, or avoided through another policy parameterization?
9. Which metrics should determine success: reward alone, or reward together with inventory, service level, monotonicity, and cap saturation?
10. What training scale is expected given the available computational resources?
11. Is the oracle Gamma base-stock policy sufficient as an informed baseline, or is a dynamic-programming or other optimal-control benchmark expected?
12. How many random training seeds are required for a credible architecture comparison?

## Proposed Next Steps

Pending supervisor feedback:

1. Freeze the benchmark definition, product distribution, action constraints, and evaluation metrics.
2. Decide whether the comparison should use parameter-matched architectures or paper-specified architectures.
3. Clarify whether the MLP comparison requires strict architectural monotonicity or only a soft penalty.
4. Run several validation seeds under the frozen benchmark.
5. Use the held-out test products only once after model and hyperparameter selection are complete.

## Repository Status

The calibration experiments and generated results currently live in a separate branch-1 worktree and have not yet been committed to the main project branch. They should be reviewed and merged only after the benchmark and comparison protocol have been agreed upon.
