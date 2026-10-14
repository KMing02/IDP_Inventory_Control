# End-to-End Inventory Control via Monotonic Kolmogorov–Arnold Networks

Code for an Interdisciplinary Project (IDP) at the TUM Chair of Logistics and Supply Chain Management, by Qi Zhang and
Keming Lin, examined by Prof. Dr. Stefan Minner. The final report is [`report/IDP_Final_Report.pdf`](report/IDP_Final_Report.pdf).

## Question

A neural network can learn an ordering policy end to end, but it can also learn decisions no manager would accept,
such as ordering more because there is already more stock. MonoKAN, a Kolmogorov–Arnold network whose spline edges
are projected onto monotone shapes, *guarantees* such common-sense directions by construction. The project asks what
that guarantee does and what it costs, by comparing MonoKAN with an unconstrained KAN of the same architecture under
the same training and evaluation protocol, in the lost-sales setting of Maggiar et al. (2025).

## Two Designs

- **Direct quantity (main design).** The network decides the order itself from 37 inputs: on-hand stock, the last 32
  days of demand, and price, purchase cost, holding cost and lost-sale penalty (1 + 32 + 4). MonoKAN certifies the
  direction of every input, including "more stock, never a larger order".
- **Target stock (alternative design).** A small MonoKAN reads only the four economic inputs and chooses a demand
  percentile; a fixed formula turns it into an order-up-to level `S` over observed demand and orders
  `q = max(S - I, 0)`. The subtraction already enforces the stock rule, so the certificate only covers the economics.
  Maggiar et al. suggest this route but call it fragile: it only works because the best policy here is known to be
  order-up-to.

## Results

Regret is the mean per-product shortfall against the omniscient policy that knows the true demand distribution
(0% is perfect); every comparison is paired over the same 300 validation products and three seeds.

**Main design: the guarantee does real work, and its profit cost depends on the start.** Without the certificate,
the same KAN orders more when stock rises in every training run, on 9 to 51% of products; MonoKAN never does. Started
from the same initialisation, MonoKAN earned 14.11 less per period than that KAN when both started with a constant
output (95% CI 5.90 to 22.32, p = 0.0008), about 0.3% of profit, and 548 more when both started with an open readout,
from which the unconstrained KAN barely learned to react to stock. The first runs mixed the two starts (MonoKAN open,
KAN constant) and showed a gap of 22.89. All direct-quantity networks lower their order by at most 0.54 units per unit
of stock, against 0.90 for the optimal policy, and none beats a classical rule with the same 32-day window. Evaluation,
unlike training, caps stock after ordering at four times the initial mean demand; without the cap the constant-start
gap is -15.29 and the open-start gap +749, so no conclusion depends on it.

| Direct quantity | Start | Profit / period | Regret | Products ordering more as stock rises |
|---|---|---:|---:|---:|
| MonoKAN (certified) | constant | 4215.20 | 14.60% | 0% |
| Same KAN, no certificate | constant | 4229.31 | 13.33% | 35.1% |
| MonoKAN (certified) | open | 4206.43 | 15.71% | 0% |
| Same KAN, no certificate | open | 3658.28 | 48.78% | 23.6% |
| Paper FCNN | own | 3874.38 | 20.64% | 0% |
| Classical rules, last 32 days | - | 4255.93 – 4275.37 | 1.25 – 1.90% | 0% |

**Alternative design: the structure does the work, and no profit cost of the guarantee was detected.** The final policy reaches 0.80%
regret (0.78% on 3,000 products) with 1,892 parameters. In every information window it beats the same KAN without the
certificate on regret; against a good classical rule with the same information it is competitive, not better.

| Information the policy sees | Gamma fit* | Empirical quantile | KAN, no certificate | MonoKAN |
|---|---:|---:|---:|---:|
| First 32 days, never updated (the paper's benchmark) | 1.35% | 1.76% | 2.37% | 2.04% |
| Last 32 days, updated daily | 1.25% | 1.90% | 1.96% | 1.83% |
| All demand seen so far (final policy) | 0.18% | 0.27% | 0.87% | 0.80% |

\* also told that demand is Gamma distributed.

**Matched comparison.** With 1,000 training products and both designs seeing only the last 32 days, the certified
model earns 14.11 less per period in the main design (both networks from the constant start, p = 0.0008) and 5.90
*more* in the alternative (p < 0.0001), and regret is 14.60% against 2.26%. This is consistent with a guarantee that
costs more where it has to enforce the stock rule by itself, but the two designs differ in more than the constraint,
so the comparison does not isolate the stock constraint.

`docs/EXPERIMENTS.md` has every number with its confidence interval, the controls, the negative results and the
command that reproduces it.

## Repository

```text
inventory_control/              package: simulator, products, policies, training, experiments, analysis
  policies/neural/monokan.py    Hermite-spline KAN/MonoKAN, both policy designs
  direct_quantity_experiment.py main design
  base_stock_output_experiment.py  target-stock runner (alternative design)
  classical_windows.py          classical rules in every information window
  analysis.py                   regret and paired product-by-product tests
configs/                        final configs (see docs/ARCHITECTURE.md)
scripts/                        command-line entry points
tests/                          correctness tests
docs/                           evidence index, code map, record of defects that affected results
report/                         final report (PDF), the script that builds it, stored main-design runs (data/)
```

## Running

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; on Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"     # or: python -m pip install -r requirements.txt
python -m pytest
```

Main design (each `train` call can run in its own process, e.g. one per model):

```bash
python scripts/run_direct_quantity_comparison.py train --models monokan,kan,fcnn --seeds 2025,2026,2027 --epochs 2400 --out outputs/direct_quantity
python scripts/run_direct_quantity_comparison.py summarize outputs/direct_quantity
python scripts/run_direct_quantity_comparison.py interpret outputs/direct_quantity
```

Alternative design, one seed, and the same network without the certificate:

```bash
python scripts/run_soft_quantile_monokan.py --seed 2025 --output-dir outputs/ts_monokan/seed_2025
python scripts/run_soft_quantile_monokan.py --seed 2025 --output-dir outputs/ts_kan/seed_2025 --set monokan.level_network=kan
python scripts/compare_target_stock_runs.py pair outputs/ts_monokan outputs/ts_kan
```

Classical rules in all three information windows:

```bash
python scripts/run_classical_windows.py
```

Rebuild the final report (figures, HTML and PDF; needs Chrome or Edge installed):

```bash
python report/build_report.py
```

The full list of commands, one per reported table, is at the end of `docs/EXPERIMENTS.md`. Everything runs on CPU;
a 2,400-epoch target-stock run took between 5 and 55 minutes per seed on a desktop CPU, depending on the
observation window.
