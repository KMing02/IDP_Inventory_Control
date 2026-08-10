# ChatGPT Research Review Brief: Structure-Informed MonoKAN Inventory Policies

This brief is intended to be copied into ChatGPT or a literature-review assistant
for an external research-story critique.

## Project Context

We are working on a team research project about structure-informed neural
policies for lost-sales inventory control.

The current benchmark uses:

- paper-style lost-sales simulation;
- DirectBackprop training through differentiable rollout;
- heterogeneous products with historical demand and economic features;
- residual order-up-to policies;
- target-stock structural diagnostics;
- paired multi-seed distribution-shift evaluation.

The motivating question is whether monotonic neural structure, specifically
MonoKAN-style constrained spline networks, can improve or strengthen neural
inventory-control policies.

## Current Primary Policy Formulation

The current main policy is residual order-up-to:

```text
S_theta(z) = S_hist(z) + alpha * r_theta(z)
q_theta(y, z) = max(S_theta(z) - y, 0)
```

where:

- `y` is current inventory;
- `z` contains demand history summaries and economic product features;
- `S_hist(z)` is a historical/non-omniscient base-stock anchor;
- `r_theta(z)` is a learned residual;
- `q_theta` is the deployed order.

Current inventory is intentionally excluded from the residual target-stock
network. Inventory response is handled by the shared order-up-to decoder.

This replaced an earlier raw-action MonoKAN formulation, where the network
directly predicted order quantities. That raw-action setup produced oversized
orders that were clipped by the deployment cap, so the executed policy was
dominated by clipping rather than learned structure.

## Main Four-Model Comparison

The current intended comparison is:

| Model | Decoder | Anchor | Context monotonicity |
| --- | --- | --- | --- |
| `PaperFCNNResidualOrderUpTo` | order-up-to | historical | none |
| `KANResidualOrderUpTo` | order-up-to | historical | none |
| `ConstrainedMLPResidualOrderUpTo` | order-up-to | historical | enforced |
| `MonoKANResidualOrderUpTo` | order-up-to | historical | enforced |

This separates four effects:

- action parameterization: raw action versus order-up-to;
- anchoring: free target versus residual historical anchor;
- representation: MLP versus KAN spline basis;
- constraint: unconstrained versus monotone architecture.

Raw-action models are now treated as legacy diagnostics, not the proposed final
policy class.

## Current Reward Evidence

### Compact Four-Model Validation

Output directory:

```text
outputs/idp_benchmark/factorial_small_four_model
```

Key reward per period:

```text
KANResidualOrderUpTo             3525.06
MonoKANResidualOrderUpTo         3525.00
NonOmniscientOptimal             3522.41
PaperFCNNResidualOrderUpTo       3516.17
HistoricalDemandBaseStock        3490.80
ConstrainedMLPResidualOrderUpTo  3424.60
OmniscientOptimal                3561.14
```

Interpretation: KAN and MonoKAN are nearly tied. Constrained MLP is much worse
and appears conservative.

### 10-Seed Compact Paired Distribution-Shift Run

Output directory:

```text
outputs/monokan_distribution_shift_factorial_small_10seed
```

Aggregate reward per period:

```text
KANResidualOrderUpTo             4651.68
MonoKANResidualOrderUpTo         4651.42
PaperFCNNResidualOrderUpTo       4642.17
ConstrainedMLPResidualOrderUpTo  4494.75
```

Winner counts across seed-shift cells:

```text
KANResidualOrderUpTo             33
MonoKANResidualOrderUpTo         28
PaperFCNNResidualOrderUpTo       17
ConstrainedMLPResidualOrderUpTo   2
```

Direct KAN-versus-MonoKAN paired analysis:

```text
Mean paired delta, MonoKAN - KAN: -0.259567 reward/period
Hierarchical bootstrap 95% CI: [-0.590681, -0.056785]
Delta as percent of mean KAN reward: -0.005580%
Practical equivalence margin: +/- 5 reward/period
Classification: practically_equivalent
Statistical interval excludes zero: True
MonoKAN product win/loss/tie rates: 0.437 / 0.409 / 0.154
Worst product-level delta: -385.123125
Worst seed-shift mean delta: -6.213671
```

The confidence interval keeps the eight named shifts fixed, resamples training
seeds, and resamples paired products within each selected seed-shift cell.
Shifts are aggregated with equal weight.

Interpretation: KAN is statistically slightly higher, but the effect size is
tiny and inside the proposed practical equivalence margin.

## Current Structural Evidence

The target-stock diagnostic evaluates the complete target:

```text
S_theta = historical_base_stock + scaled_residual
```

not only the residual network.

Checked comparative statics:

```text
demand_level: increasing
selling_price: increasing
lost_sales_penalty: increasing
purchase_cost: decreasing
holding_cost: decreasing
```

Diagnostics include:

- finite-difference violation rate;
- wrong-sign autograd derivative rate;
- minimum and maximum directional derivatives;
- maximum wrong-direction derivative;
- in-domain and stress/extrapolation domains;
- residual correction magnitude relative to the anchor.

Important caveat: finite differences recompute the full target and are the
stronger complete-target check. Autograd derivatives condition on the current
batch anchor tensor.

MonoKAN spline certification separately checks constrained spline intervals and
left/right extrapolation behavior. This is useful implementation evidence, but
it does not alone certify the complete model.

## Current Visualization Reading

Key plots:

```text
outputs/idp_benchmark/factorial_small_four_model/plots/controlled_inventory_history_multiplier_target_stock.png
outputs/idp_benchmark/factorial_small_four_model/plots/controlled_inventory_history_multiplier_deployed_order.png
outputs/idp_benchmark/factorial_small_four_model/plots/policy_raw_vs_deployed_curves.png
outputs/idp_benchmark/factorial_small_four_model/plots/policy_inventory_demand_contours.png
```

What they show:

- The order-up-to decoder makes all residual models monotone in current
  inventory: higher inventory gives lower deployed order.
- KAN also looks smooth and near monotone in the controlled plots. This is not
  surprising because the order-up-to decoder plus historical anchor already
  imposes strong operational structure.
- The meaningful MonoKAN distinction is not deployed-order monotonicity in
  inventory, but target-stock monotonicity with respect to selected context
  variables.
- The constrained MLP appears more conservative: higher target stock, higher
  orders, lower lost-sales rate, and worse reward.

## Current Defensible Claim

The strongest honest claim is:

```text
We develop a structure-informed residual order-up-to policy for lost-sales
inventory control. The order-up-to decoder encodes the known inventory-response
geometry, while MonoKAN constrains selected comparative statics of the learned
target stock. In compact paired experiments, MonoKAN is practically equivalent
to unconstrained KAN while providing stronger structural auditability.
```

Supported:

- Raw-action MonoKAN was misaligned because clipping dominated executed
  actions.
- Residual order-up-to fixed the action-geometry problem.
- KAN and MonoKAN are nearly indistinguishable in reward under compact paired
  distribution-shift evaluation.
- MonoKAN provides explicit structural constraints and spline-level audit
  evidence.

Not yet supported:

- MonoKAN consistently outperforms KAN.
- Monotonicity causes better reward or robustness.
- Paper-scale performance.
- Universal distribution-shift robustness.
- Complete end-to-end certification solely from spline checks.

## Main Concern

Using order-up-to is theoretically reasonable for inventory control, but it
also means inventory monotonicity is already handled by the policy decoder.
Therefore, MonoKAN cannot claim credit for inventory-response monotonicity.

The MonoKAN contribution shifts to:

```text
performance-preserving structure on the target-stock context function
```

rather than:

```text
higher reward from monotone raw actions
```

The open question is whether this contribution is strong enough for the thesis
or paper framing.

## Proposed Frozen Protocol

Before final evaluation, freeze:

- the four models listed above;
- residual order-up-to decoder and historical anchor;
- training budget;
- seed definitions;
- eight shift definitions:
  - `in_distribution`;
  - `low_cv`;
  - `high_cv`;
  - `low_mean_demand`;
  - `high_mean_demand`;
  - `high_lost_sale_penalty`;
  - `high_holding_cost`;
  - `mixed_stress`;
- checkpoint selection rule;
- primary metric: paired `MonoKAN - KAN` reward;
- secondary metrics: regret, win/loss/tie, worst-decile delta, target-stock
  monotonicity, spline checks, clipping rate;
- equivalence margin;
- protected final evaluation rule.

## Questions For ChatGPT / Literature Reviewer

Please critique this research story:

1. Is the residual order-up-to formulation a valid and defensible way to combine
   operations-research structure with MonoKAN-style monotone function classes?
2. Is the contribution still meaningful if most performance improvement comes
   from order-up-to action geometry rather than MonoKAN itself?
3. Is "practically equivalent reward plus stronger structural auditability" a
   strong enough thesis claim?
4. What exact additional evidence would make this story stronger without tuning
   models after seeing results?
5. Should the main comparison be KAN versus MonoKAN, or should FCNN and
   constrained MLP remain equally central?
6. How should we phrase the role of monotonicity if unconstrained KAN already
   appears nearly monotone in controlled visualizations?
7. What are the risks of calling this "physics-informed" versus
   "structure-informed" or "operations-informed"?
8. What final experiment should be run once, after freezing the protocol, to
   make the evidence scientifically credible?
9. Are there missing ablations that are essential before final evaluation?
10. How should the negative/near-zero reward result for MonoKAN versus KAN be
    framed honestly?

## Recommended Answer Format

Please answer in four sections:

1. `Story Strength`: whether the current claim is strong, weak, or salvageable.
2. `Necessary Changes`: what must change before final evaluation.
3. `Additional Evidence`: what to run or analyze next.
4. `Suggested Thesis Claim`: the most defensible one-paragraph claim.

