# Image registration anchor: training functionality first

Hypothesis motivated by the completed source error decomposition: a global
bias in owned dense flow may misplace the Gaussian association search region.
Reuse the existing project-owned bounded phase correlation and projection
refinement, not a new external network. Prior v3 contextual work used this
estimator as a feature; this experiment instead considers a direct global
flow-offset correction. No claim that global motion explains every local miss.

Candidate rule, before any new results: for target-frame detections, add
`reliability * (-forward_image_shift - median(backward_flow))` to each backward
flow vector. Reliability is the existing estimator's image-derived value.
No tuned blend, coordinate changes, pruning, added candidates or distance-gate
expansion. Preserve all local pairwise flow differences. Use native ZYX scale
(1.625,0.40625,0.40625) micrometres. Synthetic tests must establish the sign.

First run is CPU-only, offline, on frames0/1/2 of original training movies
6bba_f1fde7e0 and6bba_23af9eeb. Read images only, verify split hash, synthetic
shift/sign, native shapes and finite registration outputs. Save forward and
reverse shifts, NCC, reliability, inverse consistency and input hashes.
No source-eight or target images/labels in this probe. One-hour hard cap.

After functionality, compare anchored versus original motion residuals on
the previously cached matched ordinary training links in these transitions.
Require lower pooled squared physical residual and no per-movie worsening
before a larger image cache. A tiny training result is a feasibility screen,
not accuracy validation. Do not launch a larger job automatically.

Any subsequent complete source comparison must retain the original frozen
source gate and also improve versus FOCUS-flow and calibrated residual controls,
with unchanged nodes and no loss of known true divisions. Remaining65 target
movies stay closed. No submission authorization from this probe.
