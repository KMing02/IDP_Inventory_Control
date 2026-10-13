"""Every number the final report quotes, in one place, with its source.

The numbers were transcribed from the experiment outputs, not read from files at build time. Provenance per block:

- DIRECT, DIRECT_PAIRED, KAN_*, FCNN_*, OMNISCIENT_SLOPE: ``run_direct_quantity_comparison.py train`` then ``summarize``
  (code of commit 0f2b8fa, 2,400 epochs, seeds 2025-2027), recorded in docs/EXPERIMENTS.md at 226f001.
  DIRECT_IMPLIED_CORR, CLASSICAL_IMPLIED_CORR, DIRECT_MS*: ``... interpret`` on the same checkpoints.
  The per-run result files and checkpoints were not kept; the curves of Figure 2b were taken from the chart of the
  final presentation (data/violation_curves.json).
- CLASSICAL: ``run_classical_windows.py`` (reproduced exactly at f1e2f20).
- TS_MONOKAN, TS_KAN, TS_PAIRED, MATCHED: ``run_soft_quantile_monokan.py`` with the --set overrides listed in
  Appendix A, compared with ``compare_target_stock_runs.py``. Only the final MonoKAN run (expanding window) is kept
  under outputs/soft_quantile_monokan; its 0.80% regret and 4285.53 profit are reproduced by ``summary``.
- FINAL_*, DEVELOPMENT, MLP_CONTROL, EARLIER_KAN_CONTROL, SAMPLE_*, CONFIRMATION_3000, NEWSVENDOR_CORR, TS_MS,
  PRODUCT_REWARD_*: docs/EXPERIMENTS.md (target-stock evidence, What Moved The Result, Sample Efficiency, Evaluation
  Precision) and data/*.csv for Figure 6.
- PAPER_*: Maggiar et al. (2025), Tables 3 and 4.

Rewards are profit per period averaged over the 300 validation products and three seeds; regret is in percent.
"""

# ---------------------------------------------------------------- main design: direct quantity (1,000 products)
DIRECT = {
    "monokan": {"label": "MonoKAN (certified)", "reward": 4206.43, "reward_sd": 17.56, "regret": 15.71,
                "violation": 0.0, "slope": -0.524, "params": 4265},
    "kan": {"label": "Same KAN, no certificate", "reward": 4229.31, "reward_sd": 10.88, "regret": 13.33,
            "violation": 35.1, "slope": -0.541, "params": 4265},
    "fcnn": {"label": "Paper FCNN", "reward": 3874.38, "reward_sd": 622.50, "regret": 20.64,
             "violation": 0.0, "slope": -0.339, "params": 2105},
}
OMNISCIENT_REWARD = 4297.45
OMNISCIENT_SLOPE = -0.903
KAN_VIOLATION_BY_SEED = {2025: 45.3, 2026: 8.7, 2027: 51.3}
KAN_VIOLATING_STEPS = 3.7  # percent of swept stock steps
KAN_WORST_RISE = 10.9  # percent of one day's mean demand
FCNN_COLLAPSED = {"seed": 2027, "reward": 3155.76, "regret": 42.43}
FCNN_HEALTHY_REWARD = 4233.69
# MonoKAN minus the same KAN, paired over products (seed-averaged rewards)
DIRECT_PAIRED = {"mean": -22.89, "ci": (-32.57, -13.20), "p_t": "< 0.0001", "p_w": "< 0.0001", "ahead": 28}
# interpretability (Maggiar et al. Figure 3 test, per unit of mean demand) and speed
DIRECT_IMPLIED_CORR = 0.49
CLASSICAL_IMPLIED_CORR = 0.93
DIRECT_MS = {"monokan": 0.41, "fcnn": 0.20}
DIRECT_MS_10K = 57

# ---------------------------------------------------------------- classical rules in three information windows
WINDOWS = ("fixed", "rolling", "expanding")
WINDOW_LABELS = {
    "fixed": "First 32 days, never updated",
    "rolling": "Last 32 days, updated daily",
    "expanding": "All demand seen so far",
}
CLASSICAL = {
    "gamma": {"fixed": (4274.91, 1.35), "rolling": (4275.37, 1.25), "expanding": (4294.02, 0.18)},
    "percentile": {"fixed": (4264.52, 1.76), "rolling": (4255.93, 1.90), "expanding": (4290.64, 0.27)},
}

# ---------------------------------------------------------------- alternative design: target stock
FINAL_REGRET = 0.80
FINAL_REWARD = 4285.53
FINAL_REGRET_BY_SEED = (0.88, 0.68, 0.84)
FINAL_REGRET_3000 = 0.78
LEVEL_PARAMS = 1892
# final setup (5,000 products, weighted objective) in each window: (reward, regret)
TS_MONOKAN = {"fixed": (4249.02, 2.04), "rolling": (4255.32, 1.83), "expanding": (4285.53, 0.80)}
TS_KAN = {"fixed": (4246.38, 2.37), "rolling": (4253.51, 1.96), "expanding": (4285.97, 0.87)}
# MonoKAN minus KAN, paired over products
TS_PAIRED = {
    "expanding": {"mean": -0.44, "ci": (-1.97, 1.08), "p_t": "0.569", "p_w": "0.0001", "ahead": 61,
                  "reading": "no significant mean difference; ahead on most products"},
    "rolling": {"mean": 1.81, "ci": (0.16, 3.46), "p_t": "0.032", "p_w": "< 0.0001", "ahead": 62,
                "reading": "small but consistent"},
    "fixed": {"mean": 2.64, "ci": (0.17, 5.10), "p_t": "0.036", "p_w": "0.464", "ahead": 51,
              "reading": "small and borderline"},
    "matched": {"mean": 5.90, "ci": (3.32, 8.48), "p_t": "< 0.0001", "p_w": "< 0.0001", "ahead": 72,
                "reading": "ahead on both tests"},
}
# matched to the main design: 1,000 products, profit objective, 2,400 epochs, last 32 days
MATCHED = {"monokan": (4260.45, 2.26), "kan": (4254.55, 3.35)}
MATCHED_EXPANDING_REGRET = 1.62
# earlier controls (docs/EXPERIMENTS.md): MLP level network, and a KAN block with a running-mean decoder
MLP_CONTROL = {"mean": -0.08, "ci": (-2.09, 1.93), "p_t": "0.94"}
EARLIER_KAN_CONTROL = {"mean": 6.73, "ci": (3.38, 10.08), "p_t": "0.0001"}
# development path of the alternative design (regret, three seeds each, cumulative)
DEVELOPMENT = [
    ("First MonoKAN design (DeepGrid7, fixed grid)", 21.40),
    ("Spline knots placed at data quantiles", 9.28),
    ("Sorted demand lags", 8.71),
    ("Corrected monotone projection, 400 epochs", 7.06),
    ("Soft-quantile target-stock decoder", 3.81),
    ("Level-dependent gain, 2,400 epochs", 2.50),
    ("All demand seen so far (expanding window)", 1.77),
    ("Per-product reward weighting", 1.04),
    ("5,000 training products (final)", 0.80),
]
FIRST_FCNN_REGRET = 9.98  # unconstrained FCNN direct target stock at the starting point
NEWSVENDOR_CORR = 0.982
TS_MS = {"monokan": 0.47, "mlp": 0.06}
# sample efficiency: regret by number of training products, MonoKAN vs parameter-matched MLP level network
SAMPLE_N = (100, 300, 1000)
SAMPLE_REGRET = {"MonoKAN": (2.61, 1.62, 1.00), "MLP": (2.33, 1.37, 2.44)}
SAMPLE_REWARD = {"MonoKAN": (4279.35, 4282.60, 4284.08), "MLP": (4285.05, 4284.38, 4278.53)}
# confirmation on 3,000 validation products: (reward, regret)
CONFIRMATION_3000 = [
    ("Omniscient base stock", 4693.34, 0.00),
    ("Soft-quantile MonoKAN, 5,000 products", 4679.59, 0.78),
    ("Soft-quantile MonoKAN, 1,000 products", 4678.87, 1.28),
    ("Classical Gamma fit, first 32 days", 4671.74, 1.46),
    ("Classical empirical quantile, first 32 days", 4651.87, 2.34),
]
PRODUCT_REWARD_MEAN, PRODUCT_REWARD_SD = 4284, 8537

# ---------------------------------------------------------------- the paper (Maggiar et al., 2025)
PAPER_TABLE3_GAP = 0.41  # DRL vs omniscient, average profit over 100,000 test products
PAPER_LEAD_TIME_GAP = (0.79, 5.68)  # order-up-to policies behind DRL, lead times 2 to 7 (Table 4)
