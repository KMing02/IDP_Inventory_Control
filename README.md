# End-to-End Inventory Control via Monotonic Kolmogorov–Arnold Networks

Code for an Interdisciplinary Project (IDP) at the TUM Chair of Logistics and Supply Chain Management, by Qi Zhang and
Keming Lin, examined by Prof. Dr. Stefan Minner. The final report is [`report/IDP_Final_Report.pdf`](report/IDP_Final_Report.pdf).

The approved submission PDF and its matching, authoritative LaTeX source are preserved in
[`report/revised/`](report/revised/). See [`report/REVISED_REPORT_README.md`](report/REVISED_REPORT_README.md)
for compilation and [`report/RESULTS_PROVENANCE.md`](report/RESULTS_PROVENANCE.md) for the supplied evidence and its limits.
The older HTML report generator is retained as historical tooling; it does not generate the approved report.

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
  percentile; a structured decoder with learned smoothing and gain produces an order-up-to level `S` over observed demand and orders
  `q = max(S - I, 0)`. The subtraction already enforces the stock rule, so the certificate only covers the economics.
  Maggiar et al. suggest this route but call it fragile: it only works because the best policy here is known to be
  order-up-to.

## Results

Regret is the mean per-product percentage shortfall against the omniscient base-stock policy that knows the true
demand distribution. It includes only products with positive reference profit (279 of the 300 validation products).
The reference is not a pathwise clairvoyant upper bound. Comparisons use three training seeds and common evaluation
products and demand paths within each comparison; not all historical comparator outputs are supplied.

**Main design: the guarantee does real work, and its profit cost depends on the start.** Without the certificate,
the same KAN orders more when stock rises in every training run, on 9 to 51% of products; MonoKAN never does. Started
from matched initialization schemes, MonoKAN earned 14.11 less per period than that KAN when both started with a constant
output (95% CI for MonoKAN minus KAN: -22.32 to -5.90, p = 0.0008), about 0.3% of profit, and 548 more when both started with an open readout,
from which the unconstrained KAN barely learned to react to stock. The first runs mixed the two starts (MonoKAN open,
KAN constant) and showed a gap of 22.89. Average fitted stock-response magnitudes are at most 0.54, against 0.90
for the reference policy on the same sweep; these averages are not pointwise derivative bounds. None beats a classical
rule with the same 32-day window. Evaluation, unlike training, limits orders to `max(4*m0 - I, 0)`, where `m0` is the
initial historical mean floored at one; it cannot remove inventory already above that level. Without the cap the constant-start
gap is -15.29 and the open-start gap +749, so no conclusion depends on it.

| Direct quantity | Start | Profit / period | Regret | Products ordering more as stock rises |
|---|---|---:|---:|---:|
| MonoKAN (certified) | constant | 4215.20 | 14.60% | 0% |
| Same KAN, no certificate | constant | 4229.31 | 13.33% | 35.1% |
| MonoKAN (certified) | open | 4206.43 | 15.71% | 0% |
| Same KAN, no certificate | open | 3658.28 | 48.78% | 23.6% |
| Paper FCNN | own | 3874.38 | 20.64% | 0% |
| Classical rules, last 32 days | - | 4255.93 – 4275.37 | 1.25 – 1.90% | 0% |

**Alternative design: the decoder supplies inventory monotonicity, and no significant mean-profit difference from the matched KAN was detected.** The final policy reaches 0.80%
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
*more* in the alternative (p < 0.0001), and regret is 14.60% against 2.26%. This supports the combined policy design,
not a causal claim about the number of constrained inputs: architecture, initialization, checkpoint frequency and
the evaluation guard also differ across the two designs.

`docs/EXPERIMENTS.md` is the historical experiment index. The approved report and `report/RESULTS_PROVENANCE.md`
take precedence for interpretation and evidence availability. No significant difference is not proof of equivalence;
the 3,000-product evaluation contains the original validation set and is not an independent protected test.

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
report/                         approved PDF, matching LaTeX source (revised/), historical tooling and saved evidence
```

## Running

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; on Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"     # or: python -m pip install -r requirements.txt
python -m pytest
```

The previously observed environment is recorded in [`TESTED_ENVIRONMENT.md`](TESTED_ENVIRONMENT.md).
It is not a complete dependency lockfile or a fresh-install guarantee.

### Inspect saved results without training

```bash
python scripts/compare_target_stock_runs.py summary report/data/soft_quantile_monokan
```

Expected rounded final MonoKAN result: reward 4285.53 per period and regret 0.80%; seed regrets are
0.88%, 0.68% and 0.84%. The archived CSVs are unchanged copies of the three submitted product-metric files.
Main-design per-product rewards and paired summaries are in `report/data/direct_quantity/`.
The checkpoints and training logs supplied in the submission ZIP are not tracked in Git.

### Optional retraining (expensive)

Use new output directories and do not overwrite saved evidence. Each target-stock command trains one seed;
repeat for 2025, 2026 and 2027 in separate seed directories.

Main design (each `train` call can run in its own process, e.g. one per model):

```bash
python scripts/run_direct_quantity_comparison.py train --models monokan,kan,kan_open,monokan_constant,fcnn --seeds 2025,2026,2027 --epochs 2400 --out outputs/direct_quantity
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

### Compile the approved report

```bash
cd report/revised
pdflatex main.tex
pdflatex main.tex
```

Alternatively upload `report/revised/` to Overleaf and select `main.tex`, or run `tectonic main.tex` there.
Compilation produces `main.pdf`, not an automatic replacement of the approved `report/IDP_Final_Report.pdf`.
**Do not run `report/build_report.py` to rebuild the approved report:** it uses the older HTML narrative and
writes to that same approved-PDF path. It is retained only for provenance.

The historical command index is at the end of `docs/EXPERIMENTS.md`; not every historical table can be recomputed
from the supplied raw files. Everything runs on CPU;
a 2,400-epoch target-stock run took between 5 and 55 minutes per seed on a desktop CPU, depending on the
observation window.
