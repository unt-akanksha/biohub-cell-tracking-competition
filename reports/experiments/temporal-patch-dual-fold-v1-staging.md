# Temporal patch dual-fold v1 staging

Date: 2026-08-27

Status: implemented and locally verified; reserved for the cloud handoff, not
launched on Kaggle.

## Why this is a new candidate

The public 0.927 comparator and the active reciprocal Trackastra experiment are
predominantly geometry-driven. Frozen SpatialDINO patch similarity previously
made zero safe corrections, which is evidence against weakening its swap gates,
not evidence that local image appearance is useless. The missing experiment is
a representation trained directly on adjacent Biohub cells.

This lane implements an independent 19,218,498-parameter 3D residual encoder per
fold. It samples 17-cubed patches over the same 16-micrometer physical field of
view from both pooled isotropic synthetic images and anisotropic real movies.
Multi-positive contrastive supervision treats two true daughters as positives
instead of forcing one to be a negative; a separate sparse division head is
trained with correction for the inflated synthetic division prior. Checkpoint
selection uses a full-model optimizer-step exponential moving average (decay
0.997), reducing sensitivity to one noisy minibatch or validation instant.
The prior correction is class-conditional: synthetic positives are downweighted
from the 4.07% synthetic rate toward the 0.26% real rate, while synthetic
negatives retain approximately unit weight. This avoids the earlier failure
mode where scaling the whole BCE nearly erased non-division supervision.

No public Kaggle implementation, public prediction identity, CELLECT source, or
external checkpoint is copied. CELLECT and the 2026 microscopy-embedding paper
informed only the modeling hypothesis.

## Leakage and selection boundaries

Two isolated GPU workers train reciprocal embryo folds. Each worker uses 1,900
synthetic movies plus 96 real movies from the opposite embryo prefix. Twelve
unopened target-prefix movies select its checkpoint, and a different twelve are
reserved for later appearance-weight calibration. The four already opened
processed-acceptance movies are excluded from both stages.

The radius builder fails closed if a labeled edge falls outside the declared
32-micrometer candidate neighborhood. It also rejects batches without a hard
negative, so an apparently good contrastive loss cannot come from trivial
single-choice rows.

After both models pass their absolute retrieval gates, a separate calibration
step must compare a small predeclared appearance/division blend grid whose
zero/zero setting is the exact Trackastra control. Unlike a retrieval-only
proxy, selection now executes the frozen clean graph linker and measures pooled
organizer-aligned score (adjusted edge Jaccard plus 0.1 division Jaccard), using
the organizer's lineage-aware division matching and a per-movie regression
floor. Because calibration predicts edges over the exact ground-truth node
inventory, adjusted edge Jaccard equals edge Jaccard in this clean stage. The trained source
division logit shifts link confidence for that source, allowing a genuine
second daughter to clear frozen thresholds without changing candidate rank.
Only after freezing both weights may the four processed
movies be materialized once and scored by the pinned exact evaluator. No
leaderboard feedback participates in any decision.

## Runtime boundary

This is deliberately staged for the user-provided cloud instance. The active
Kaggle training run is projected to leave about 20.45 GPU-hours, so launching
this roughly ten-hour reciprocal image experiment on Kaggle would violate the
reserve handoff rule. Any future competition inference remains exactly two-GPU,
whole-movie sharded, and separately authorized.

## Verification

The updated focused temporal/Trackastra suite passes 32 tests across the normal
and pinned exact-scorer environments. It covers physical
resampling, division-aware multi-positive loss, candidate-radius failure,
transition construction, model output normalization, arbitrary node-ID
alignment, exact zero/zero fallback, source-division alignment, actual
second-daughter recovery, the frozen processed exact gate, runtime-package
integrity, and two-GPU whole-movie sharding.

The rebuilt portable archive is
`.biohub/staging/biohub-temporal-patch-runtime-v1-heavy-ema-prior-20260827.zip`
(93,560 bytes, SHA-256
`f90ccba02f5c7bfbc77a46b41263d0eb302f0716e2e42818736f96662781ffb4`).
An independent extraction verified all 25 manifest-bound files; the embedded
verifier reported manifest SHA-256
`f3a7e3f87e864e1d9cd1001eb9814665c11ea312797a099789bb0caff4dbc0fe`,
required GPU count 2, and no submission command.

A separate end-to-end gradient smoke test used two visibly different synthetic
3D cells, their shifted children, and one distractor. Over 40 CPU optimizer
steps, multi-positive contrastive loss fell from `1.0668325` to `0.00001049`;
the two sources selected target columns `[0, 1]` with correct-pair cosine scores
`0.9896` and `0.9962`. This confirms that physical crop extraction, the encoder,
and the association objective form a learnable path rather than merely passing
shape checks.

The complete repository suite subsequently passed 495 tests with zero failures;
two Windows tests were skipped only because unprivileged symlink creation is
unavailable. For timeout resistance, final whole-movie LPT sharding now weights
both Trackastra frame-pair products and the added per-node 3D encoding work; its
node cost was increased to 12,288 after scaling the encoder to 19.2M parameters.
