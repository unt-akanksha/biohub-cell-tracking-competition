# Selection-only localization refinement — next-step diagnosis

## Evidence

`6bba_57b7cc1e` is the limiting selection movie across all independently
adapted detector families:

| Detector | Pooled recall | Limiting-movie recall | Limiting count ratio |
|---|---:|---:|---:|
| Spotiflow PU | 0.779240 | 0.486438 | near calibrated target |
| SpatialDINO PU | 0.885638 | 0.611212 | near calibrated target |
| SpatialDINO selective distillation | 0.885906 | 0.602170 | 0.998942 |
| LSM-FM feature-24 PU | 0.884206 | 0.622061 | 0.999196 |

For feature-24 LSM-FM, the limiting movie has 65,689 predicted nodes against
65,511 estimated nodes, yet only 1,032 of 1,659 annotated nodes match and the
mean matched distance is 2.64 micrometers. The failure is therefore not global
density calibration. It is dominated by peak placement and duplicate-versus-
miss allocation in a high-density movie.

## Next experiment if feature-36 still misses

Run a checkpoint-frozen, inference-only postprocessing sweep. Infer each movie
once and preserve the same confidence-ranked peaks and count calibration. Test
only global coordinate refiners:

1. probability centroid, radius 1 / power 2 (current control);
2. probability centroid, radius 2 / powers 2 and 4;
3. joint probability × locally background-subtracted raw-intensity centroid,
   radii 1 and 2;
4. a local quadratic peak fit where the 3×3×3 Hessian is negative definite,
   falling back to the control elsewhere.

Select one global strategy on the eight selection movies using the existing
`0.80` pooled and `0.65` worst-movie gates, with an added rule that no other
movie may regress by more than 0.01 recall from the control. Do not choose a
different strategy per movie. Open the four acceptance movies only after the
strategy and thresholds are frozen.

This path requires no new teacher labels, no public prediction copy, and only a
short validation GPU run. It directly attacks the observed localization error
instead of spending another long run on density or same-teacher loss changes.

## Frozen implementation update

The packaged strategy set also includes one log-probability quadratic fit.
The detector is trained against Gaussian-like heatmaps, whose logarithm is a
quadratic around an isolated peak; this gives a theory-matched sub-voxel fit
while preserving the exact peak identities and confidence values. A synthetic
off-grid Gaussian test recovers its center to below `1e-4` voxel error.

The private runtime staging directory is
`biohub-lsm-fm-image-text-refinement-runtime-v1`. Its source manifest SHA-256
is `73682dbeb5ae4a0e97e103f3214adb4cd77f5f1ff1ad161e18d8590dc6eaedf2`.
Eight focused tests pass. The runtime is staged only: it has not been uploaded
or launched, and it contains no submission path.
