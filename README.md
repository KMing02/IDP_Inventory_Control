# Inductive Biases for Neural Inventory Control

This repository contains the code used for a TUM IDP project on structure-informed neural inventory control. It studies a Maggiar-style lost-sales setting and asks what a hard-monotone MonoKAN representation costs, and what it buys, relative to an unconstrained neural policy under an identical training and evaluation protocol.

## Research Question

How does a certified monotone neural inventory-control policy perform relative to an unconstrained policy under the same DirectBackprop training and evaluation protocol, and what does the monotonicity certificate cost?

The comparison is matched: same products, demand paths, training seeds, horizon, burn-in, simulator, reward function, decoder, and parameter count. Only the constrained component differs.

## Final Formulation

The final policy orders up to a smoothed empirical quantile of the demand it has observed:

```text
u       = sigmoid(MonoKAN(price, purchase cost, holding cost, lost-sale penalty))
S_t     = z_(1) + sum_k sigmoid((u - t_k) / tau) * (z_(k) - z_(k-1))
q_t     = max(S_t - I_t, 0)
```

`z` are the ascending order statistics of every demand observed up to `t`, and `t_k` are fixed positions in `[0, 1]`. The sigmoid weights are non-negative and telescope to one, so `S_t` is a convex combination of order statistics.

This places the structure in the decoder rather than in a large constrained network, which is what made the certificate cheap:

- every order statistic is nondecreasing in every raw demand, and the weights are non-negative, so `S_t` is nondecreasing in all observed demand;
- the order statistics are sorted, so `S_t` increases with `u`, and the monotone level network supplies the certified economics directions;
- the decoder is exactly scale-equivariant in demand, so the network never has to represent a scale-free ratio.

Only `u` and a scalar smoothing width are learned. The level network sees economics only, never inventory, and never the newsvendor fractile.

## Main Result

Three seeds, identical products and demand paths. Regret is the mean per-product shortfall against the omniscient base-stock policy; it is scale free, and far more sensitive than a mean over products whose rewards span orders of magnitude.

| Policy | Reward / period | Regret | Params |
|---|---:|---:|---:|
| Omniscient base stock (uses hidden Gamma parameters) | 4297.45 | 0.00% | — |
| **Soft-quantile MonoKAN, weighted objective** | 4285.53 +/- 0.19 | **0.80%** | 1,892 |
| **Soft-quantile MonoKAN, profit objective** | **4286.46 +/- 0.23** | 1.77% | 1,892 |
| Soft-quantile MLP control (unconstrained, matched) | 4286.29 +/- 0.80 | 2.23% | 1,884 |
| Classical Gamma-fit base stock (32 lags) | 4274.91 | 1.35% | — |
| Classical empirical-quantile base stock (32 lags) | 4264.52 | 1.76% | — |
| Unconstrained FCNN direct target stock | 4174.41 +/- 4.36 | 9.98% | 2,073 |
| DeepGrid7 MonoKAN direct target stock | 4111.59 +/- 3.31 | 21.40% | 15,297 |

The weighted row is the shipped 5,000-product setting; the same policy trained on 1,000 products reaches `4285.30 +/- 0.54` and `1.04%` regret.

The two soft-quantile rows are the same policy under different objectives and are Pareto-optimal against each other: the weighted objective wins on 60% of products (Wilcoxon `p = 0.0006`) while total profit is a statistical tie (`p = 0.33`).

## What The Certificate Costs

Against a parameter-matched MLP inside the identical decoder, trained for the same 2,400 epochs:

| Level network | Reward | Regret | Training |
|---|---:|---:|---:|
| MonoKAN (certified monotone) | 4286.21 +/- 0.43 | 1.62% | 4,671 s |
| MLP (unconstrained) | 4286.29 +/- 0.80 | 2.23% | 2,488 s |

Paired over 300 shared products the difference is `-0.08`, 95% CI `[-2.09, +1.93]`, `p = 0.94`. **Certified monotonicity is free in total profit and better on typical products, at 1.9x the training time.** Constraining an otherwise identical KAN *improves* it (`+6.73`, `p = 0.0001`): the constraint acts as a regularizer.

## Known Limits

- The level network reaches `corr = 0.982` with the analytic critical fractile at the shipped 5,000-product setting (`0.970` at 1,000). A supervised fit of the same architecture reaches `0.989`, and an MLP reaches `0.997`, so the residual is a small remaining reward-signal gap plus a KAN representation ceiling. The fractile is a ratio, and an additive-spline layer represents ratios poorly.
- Enlarging the KAN makes this *worse*, not better: the supervised fit degrades from RMSE `0.026` (2x8, grid 7) to `0.033` (3x16, grid 12).
- Monotonicity in every individual lag structurally excludes any spread statistic, so a certified policy cannot implement a Gamma-fit style estimator. Note the Gamma fit is itself not monotone in every lag, so it is not a legal comparator for a certified policy.
- Given the same expanding information, a classical empirical quantile reaches `4290.64` / `0.27%`. The learned policy does not beat a well-specified classical estimator at matched information; it is competitive with one.
- The monotone model is **not** more sample efficient. At 100 training products it is worse than the parameter-matched MLP on both metrics, and it overtakes only at 1,000. What the constraint buys is reliable scaling: MonoKAN improves monotonically with data while the MLP degrades past `n = 300`. See `docs/EXPERIMENTS.md`.

## Running

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

Train and evaluate the final policy for one seed:

```bash
python scripts/run_soft_quantile_monokan.py \
  --config configs/paper_lost_sales_soft_quantile_monokan.yaml \
  --seed 2025 --device cpu \
  --output-dir outputs/soft_quantile_monokan/seed_2025
```

Add `--models MonoKANSoftQuantileTargetStock PaperFCNNDirectTargetStock` to train the shipped FCNN anchor alongside it.

Produce the interpretability evidence from a trained checkpoint:

```bash
python scripts/run_monokan_interpretability.py \
  --config configs/paper_lost_sales_soft_quantile_monokan.yaml \
  --checkpoint outputs/soft_quantile_monokan/seed_2025/monokan_soft_quantile_target_stock.pt \
  --output-dir outputs/soft_quantile_monokan/interpretability
```

Audit the monotonicity certificate from persisted checkpoints:

```bash
python scripts/run_direct_target_structural_audit.py \
  --config configs/paper_lost_sales_soft_quantile_monokan.yaml \
  --checkpoint-root outputs/soft_quantile_monokan \
  --output-dir outputs/soft_quantile_structural_audit
```

The historical DeepGrid7 comparison is preserved and still runs; see `docs/EXPERIMENTS.md`.

## Structure

```text
configs/                         final and retained ablation configs
docs/                            evidence index, debugging record, methodology notes
inventory_control/envs/          inventory simulators and demand processes
inventory_control/paper_*.py     Maggiar-style products, evaluation, and config loading
inventory_control/policies/      classical policies and neural policies
inventory_control/training/      DirectBackprop training utilities
inventory_control/*diagnostic.py retained experiment and audit modules
inventory_control/monokan_interpretability.py  partial-dependence evidence
scripts/                         thin CLI wrappers for report-facing runs
tests/                           correctness tests for retained scientific behavior
```

See `docs/EXPERIMENTS.md` for the evidence index and `docs/DEBUGGING.md` for defects that materially affected earlier results.
