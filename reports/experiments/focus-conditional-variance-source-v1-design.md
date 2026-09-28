# Conditional uncertainty: full-source comparison

Proceed only after actual complete14fold verification and final model export.
Use the frozen conditional mean plus newly fitted per-target diagonal variance;
only the uncertainty differs from conditional-motion-source-v1. All input
coordinates remain unchanged; no node pruning, time remapping or false nodes.
Retain unnormalized Gaussian proximity weights, null-4.5, posterior>.5,
one parent and at most2children; deterministic ordering unchanged. No fit or
parameter search on source. This graph policy is not the normalized Gaussian
density used in the training screen; a density pass does not imply graph gain.

Use exactly the same8complete100frame exposed source movies. Persist every
candidate graph before opening sourceGT. Verify all raw/flow/parent provenance,
then replay parent, FOCUS-flow and constant-variance conditional-motion control
scores under the pinned patched official scorer. Fresh graph/GT per arm.

Require all previous parent/FOCUS-flow source gates unchanged, plus strictly
better combined score/rawedgeJaccard than constant-variance conditional motion,
preserving its5true divisions. Retain per-movie loss<=0.02 and worst-movie
guards. Report pooled, per-movie, embryo and worst-movie results. A pass is
still exposed-source development evidence, not independent validation or
authorization to submit. No new target movie reads. CPU-only/no cloud changes.
