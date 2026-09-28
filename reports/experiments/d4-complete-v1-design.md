# Complete-D4 candidate on the current public base

Frozen 2026-09-16, before any GPU launch or submission. **No quality gain is
established here.** This document pins what changes, what does not, and what
evidence would justify promotion.

## What the two requested notebooks actually are

Both notebooks the user asked about attach the identical three Pilkwang model
datasets and share the same inference code. The materialized predictor is
byte-identical across both and across the older `sjlee101/biohub-lf-dctta`:

| Notebook | SHA-256 | Differences |
| --- | --- | --- |
| [`flexonafft/biohub-harmonic-fusion`](https://www.kaggle.com/code/flexonafft/biohub-harmonic-fusion) | `e378e723…fb44a` | none vs the 2026-09-10 local reference |
| [`sjlee101/biohub-lf-dctta020-sectta1-sister16`](https://www.kaggle.com/code/sjlee101/biohub-lf-dctta020-sectta1-sister16) | `998b7bc9…9cdb` | 3 constants (below) |

Cell-0/cell-4 constant differences between the two:

| Constant | harmonic | dctta020 |
| --- | ---: | ---: |
| `BIOHUB_SAFE_DIV_SISTER_MAX_UM` | 14.0 | 16.0 |
| `BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD` | 0.25 | 0.20 |
| `BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT` | 0.75 | 1.0 |

`biohub-harmonic-fusion` is **configuration-identical** to the `biohub-lf-dctta`
base already measured locally on 2026-09-10. That earlier paired complete-movie
result therefore applies to this base directly rather than by analogy.

Neither notebook is an independent model family, and both declare
`leaderboard_feedback_used_for_configuration = True`. Their constants are not
adopted as our selection evidence; they are inherited unchanged as the base.

## The defect

Both sources build an eight-pass D4 test-time-augmentation group whose final
"anti-diagonal" view is `rot90(x, 1, dims=(-2,-1)).transpose(-1,-2)`. For
`A[i,j]`, `rot90` gives `B[i,j] = A[j, N-1-i]`, and transposing gives
`C[i,j] = A[i, N-1-j]` — exactly `flip(A, dims=(-1,))`, the horizontal flip
already present as view 2.

The advertised eight-view average therefore contains **seven unique views**,
with horizontal reflection double-weighted at 2/8 and anti-diagonal reflection
absent. An independent NumPy proof over square, rectangular and 6-D layouts
confirms 7 unique legacy views versus 8 corrected, exact inverses in both arms,
and group-average equivariance error ≤ 1e-10 corrected versus 8.9–21.3 legacy.

This affects both detectors, both edge-feature averages, and the DeepCenter
veto — which the public source itself annotates as "the only gate whose
threshold moves the score in both directions".

## The correction

`rot90(x, 2, dims=(-2,-1)).transpose(-1,-2)` forward, `rot90(x.transpose(-1,-2),
-2, dims=(-2,-1))` inverse. The anti-diagonal reflection is an involution, so
this round-trips exactly.

**Eight rotation arguments change. Nothing else.** Six in the dynamically
materialized predictor (2 forward, 4 inverse) and two in the DeepCenter cell.
Model calls per stage stay at eight; no model, checkpoint, threshold, fusion
weight, post-process constant or graph-topology contract is touched.

A byte-span test restores the six recorded argument spans in the 51 KB
materialized predictor and asserts byte equality with the legacy source, so any
other change anywhere in that file would fail the build.

Cell-level diff of artifact versus base: 9 of 12 cells byte-identical; cell 0
and cell 4 have blocks appended with **zero** lines removed; cell 5 changes
exactly the two DeepCenter rotation lines.

## Project-authored deployment guards

- **Two-device fail-closed.** The public notebook silently falls back to one GPU,
  which roughly doubles wall time against the platform's 12-hour cap. Required
  by the runbook for competition submission kernels.
- **`BIOHUB_VALIDATOR_ENABLE=0` in production mode.** The public notebook runs an
  in-notebook post-process sweep that selects a configuration by a proxy score
  on held-out TRAIN movies. A submission must not carry a proxy-selected
  override, so the declared base configuration is the one that runs. This also
  removes the sweep's runtime.
- **Validation mode** enables the held-out validator with `N_PER_TYPE=12` but
  empties `PP_CANDIDATES`, so the base configuration is scored and nothing is
  selected.

## Prior measured result on this exact base

From `public-d4-full-movie-v1-result.md` (2026-09-10, patched official metric,
four complete movies, both arms with fixed weights and gates):

| Complete movie | Original | Corrected | Change |
| --- | ---: | ---: | ---: |
| 44b6_24264f12 | 0.895361 | 0.923089 | +0.027728 |
| 44b6_81c256f0 | 0.987744 | 0.989007 | +0.001263 |
| 6bba_23af9eeb | 1.004163 | 0.998102 | −0.006062 |
| 6bba_f1fde7e0 | 0.889318 | 0.888963 | −0.000355 |
| **Official pooled** | **0.944839** | **0.947619** | **+0.002781** |

Raw edge Jaccard 0.941720 → 0.944365. Counts TP/FP/FN 1325/44/38 → 1324/39/39:
the gain is fewer false edges, not more true edges.

**The frozen movie and embryo non-regression gates FAIL.** Embryo 44b6 gains
0.016012; embryo 6bba loses 0.003052. That failure is why this was never
promoted, and it is not waived here.

### Honest limits of that evidence

- Four movies, two per embryo. The 6bba regression is within what two movies can
  produce by chance; it is neither confirmed nor refuted at this sample size.
- The public checkpoints saw TRAIN movies, so these are exposed diagnostics, not
  independent cross-validation. They cannot establish generalization.
- No annotated positive divisions exist in that four-movie subset, so positive
  division recovery is untested.
- 0.947619 is a local diagnostic number. It is not a leaderboard prediction.

## Why the correction is nonetheless the right next candidate

It is a geometry defect, not a tuned constant. Nothing about it adapts to data,
to labels, or to leaderboard feedback, so exposure contamination affects it far
less than it would a threshold sweep. The counterfactual is also known from
public metadata rather than from our own tuning: this base family's own authors
report 0.947, and our best verified submission is 0.946.

## Budget

Kaggle quota at build time: 27.26 h remaining of 30.00 h, refresh
2026-09-19T00:00:00Z. Protected reserve 8.00 h.

The public author's linked full run for this family is 7,126.602 s (1.98 h) with
the validator enabled; production mode removes the validator and the sweep.
Declaring a 4.00 h wall maximum on two devices reserves 8.00 GPU-h and leaves
19.26 h, which clears the reserve. The internal watchdog must stop earlier than
the platform's 12 h cap.

Submissions used today: 0 of 5.

## Promotion gate

A leaderboard number from one submission is a measurement, not a promotion. A
promotion still requires complete-movie scoring under the patched official
metric with per-embryo and worst-movie reporting. If validation mode is run
first, the paired comparison must hold the configuration fixed across both arms
and report every movie, including regressions.

Do not threshold-sweep this candidate, route by embryo identity, or relax the
non-regression gates to make it pass.

## Artifacts

| Slug | Base | Mode | Notebook SHA-256 |
| --- | --- | --- | --- |
| `biohub-d4-harmonic-production-v1` | harmonic | production | `fd961d0f685a…` |
| `biohub-d4-dctta020-production-v1` | dctta020 | production | `17d876296b19…` |
| `biohub-d4-harmonic-validation-v1` | harmonic | validation | `c6ae1eccf958…` |

Builder `scripts/build-d4-complete-v1.py` regenerates each deterministically
from the pinned base and support-pack SHAs; drift in either fails closed.
34 focused tests pass. No GPU, Kaggle launch or submission has occurred.
