# Major Resolved Implementation Issues

This document records only debugging findings that materially affect the
validity of saved experiments. It is not an inventory of every development
attempt.

## Normalization And Spline-Grid Incompatibility

The first rolling/static direct-target development experiment passed raw,
unnormalized demand and economics observations to DeepGrid7 MonoKAN while its
spline grid was fixed to `[-2, 4]`. Typical raw feature magnitudes were far
outside that domain. The model therefore relied on extensive linear
extrapolation, produced pathological target stocks, and yielded invalid reward
results.

Affected output:

`outputs/direct_target_dev_1k`

Status: **INVALID/DEBUG**. It may be used to explain the failure mode, but it
must not be used as policy-performance evidence.

The corresponding normalized development controls use
`normalize_observations: true`. They restored feature/grid compatibility, sane
initial target stocks, finite gradients, and stable training behavior:

- `outputs/direct_target_dev_1k_normalized`
- `outputs/direct_target_temporal_dev_1k_normalized`

The raw and normalized config paths remain in place because saved metadata and
regression tests reference them directly.
