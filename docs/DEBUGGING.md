# Major Resolved Implementation Issues

This document records only defects that materially affected the validity or the interpretation of saved experiments. It is not an inventory of every development attempt.

## Monotone Projection Admitted Small Violations

`_project_edge_coefficients` enforces the Fritsch-Carlson condition that makes each cubic Hermite segment monotone. It wrote the per-interval conditions as two separate slice assignments, `coef[:, :-1, 1]` and then `coef[:, 1:, 1]`.

Every interior knot derivative is shared by the two intervals that meet at that knot, so the second assignment could restore a derivative the first had just corrected. The projected spline was then not monotone on that segment.

Measured on 400 random unevenly spaced grids: the old projection left visibly non-monotone segments on `0.01%` of edges with a worst wrong-direction increment of `5.6e-4`; the corrected projection sits at float32 epsilon, `1.9e-6`, with none.

The defect predates the current work and stayed hidden because the shipped model's splines barely bent, so the Fritsch-Carlson norm rarely exceeded its bound. Data-adapted knots made the splines bend and exposed it.

**No published number is invalidated.** The complete-target audit read `0.0` violations before and after, because the leak was tiny relative to the sum over 576 first-layer edges and the monotone layers downstream. Retraining the best configuration under the corrected projection changed reward by `+1.80`, well inside seed noise. What the fix buys is that the certificate now holds for the right reason rather than by accident.

Regression coverage: `tests/test_paper_monokan.py::test_projection_is_monotone_on_uneven_knots`.

## Fixed Spline Grid Was Incompatible With The Data

Every Hermite edge used a fixed `[-2, 4]` grid at every depth, while the observed features did not live there:

- `63.8%` of normalized demand lags fell inside the *first* of seven intervals;
- the `[-2, 0]` half of the grid is unreachable, since demand is non-negative;
- `3.6%` of lags fell outside the grid entirely, up to `25x` the scale;
- normalized holding cost spanned `27.7%` of a single interval and lost-sales penalty `11.3%`, so those two splines could not bend at all;
- layer activations drifted right with depth, leaving `5.8%` outside the grid by the third layer.

The consequence was measurable, not theoretical: the shipped DeepGrid7 MonoKAN's target ratio moved by `1.9e-06` across a six-fold change in observed demand. It had learned a near-constant policy that barely read its own history.

Fitting knots to per-feature training quantiles — the Hermite analogue of the grid update in Liu et al. (2024) — cut regret from `21.40%` to `9.28%`. Knot placement never changes the sign of a spline's slope, and the projection already handled uneven spacing, so the certificate is unaffected.

Enabled by `adaptive_grid: true`. Off by default, so historical configs reproduce unchanged.

## Normalization And Spline-Grid Incompatibility

The first rolling direct-target development experiment passed raw, unnormalized observations to DeepGrid7 MonoKAN with the same fixed `[-2, 4]` grid. Typical raw magnitudes were far outside it, the model relied on extensive linear extrapolation, and the run produced pathological target stocks.

Affected output: `outputs/direct_target_dev_1k`. Status: **INVALID/DEBUG**. It may be used to explain the failure mode but must not be cited as policy-performance evidence.

The normalized controls that replaced it are the report-facing evidence. Note that normalization alone did not resolve the grid problem; it only moved the features from far outside the grid to badly crowded inside it. See the section above.

## Metric Choice Concealed A Failure

Mean reward per period is dominated by a handful of large price-times-demand products. The shipped MonoKAN trailed the classical empirical-quantile baseline by `3.6%` on that metric while leaving `21.4%` of achievable profit on a typical product.

Both metrics are now reported everywhere. Any comparison quoting only mean reward should be treated as incomplete.

## Baseline Information Access Must Be Matched

The repository's classical baselines use only the initial 32-lag history and never update. Policies that re-estimate from demand observed during the episode therefore hold an information advantage, not merely a modelling one.

Giving the classical estimator the same expanding information moves it from `4264.52` to `4290.64`, which reverses any claim of beating it. Adaptive classical variants are recorded in `docs/EXPERIMENTS.md` and should accompany any such comparison.
