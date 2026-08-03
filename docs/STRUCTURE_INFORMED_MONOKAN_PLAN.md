# Structure-Informed MONOkan Plan

This note summarizes the literature-framing PDF added at:

`docs/Structure-Informed Neural Policies for Lost-Sales Inventory Control.pdf`

## Core Conclusion

The current raw-action MONOkan result is not evidence that monotonicity is useless. It is evidence that monotonicity of the raw purchase quantity is too weak for the no-lead-time lost-sales benchmark.

The stronger known structure is an order-up-to policy:

```text
order = max(S(context) - inventory, 0)
```

Raw-action monotonicity only enforces:

```text
inventory increases -> order does not increase
```

That still permits saturated policies that ask for too much stock and rely on deployment clipping.

## Observed Failure Mode

The medium MONOkan diagnostic showed:

```text
MonoKAN reward/period: 4346.97
PaperFCNN reward/period: 4495.38
Historical heuristic: 4580.38
Non-omniscient optimal: 4605.35
raw_order_mean: 196.70
deployed_order_mean: 49.98
cap_hit_rate: 98.36%
```

This means the deployed decision was usually determined by:

```text
deployed_order = min(raw_order, max_inventory_level - inventory)
```

rather than by a calibrated learned action.

## Cap Audit

The current `max_inventory_level = max_order_factor * historical_mean_demand` behaves as an action and inventory ceiling in the implementation. The main reference benchmark should be audited carefully because the no-lead-time paper formulation is an order-up-to problem and may not intend this ceiling as a physical warehouse capacity.

Current repo audit result:

```text
No explicit capacity or action-cap constraint is documented in docs/MAGGIAR_2025_CODEX_REFERENCE.md for the no-lead lost-sales benchmark.
The cap appears to be an implementation guard/deployment bound, not a modeled paper constraint.
```

Use this rule when interpreting experiments:

```text
If the cap is a real modeled capacity, it belongs in the problem statement.
If the cap is only a numerical guard, it should almost never bind.
```

The observed `98.36%` MONOkan cap-hit rate is therefore a red flag: either MONOkan is relying on a real capacity constraint, or the numerical guard is materially changing the benchmark.

Until a physical capacity is intentionally added to the problem statement, cap-hit rate should be treated as a benchmark-distortion diagnostic. A good paper-faithful learned policy should usually learn a calibrated target stock below the guard.

## Implemented Architecture Variants

The benchmark now supports the following trainable policy classes:

```text
PaperFCNN
PaperFCNNOrderUpTo
PaperFCNNResidualOrderUpTo
MonoKAN
MonoKANOrderUpTo
MonoKANResidualOrderUpTo
```

The target-stock variants use the same DirectBackprop rollout path but decode actions differently:

```text
raw action:
  order = model(inventory, context)

order-up-to:
  S = model(context)
  order = max(S - inventory, 0)

residual order-up-to:
  S = S_historical + residual_model(context)
  order = max(S - inventory, 0)
```

## Recommended Factorial Ablation

Run the same training budget, products, demand paths, and evaluation seeds across:

```text
FCNN:    raw action, order-up-to, residual order-up-to
MONOkan: raw action, order-up-to, residual order-up-to
```

This separates the benefit of target-stock parameterization from the benefit of the MONOkan backbone.

## Claim Boundary

Safe claim:

```text
For the no-lead lost-sales benchmark, order-up-to parameterization encodes stronger known inventory structure than raw-action monotonicity.
```

Unsafe claim:

```text
MONOkan is automatically optimal, robust, or interpretable because it is monotone.
```
