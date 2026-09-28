# Four-movie training motion diagnostic

Computed by `scripts/analyze-independent-training-motion.py` using only the
four recorded fitting movies. Selection and target-embryo labels were not read.
This is a diagnostic, not an authorized inference change.

The 2,691 annotated, nondivision, consecutive-frame links have median physical
displacement `(0, 0, 0)` micrometers. Pooled per-axis standard deviations are
`(2.144598, 1.176439, 1.461546)` micrometers in Z/Y/X order. Motion-length
percentiles are 1.816805 (50th), 4.755034 (90th), 5.773897 (95th), and 8.574654
(99th) micrometers.

There is substantial movie variation: 95th-percentile lengths range from
3.656250 to 7.835466 micrometers; individual movies also have directional drift.
A globally isotropic cutoff would discard real movements. A soft anisotropic
motion prior is a reasonable next hypothesis if the fixed association-only
optimization fails, but these GT-only statistics omit localization noise.
Any use on detected nodes must account for that noise and be evaluated on
complete selection movies without changing the running experiment.
