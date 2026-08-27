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

This lane implements an independent 7,498,890-parameter 3D residual encoder per
fold. It samples 17-cubed patches over the same 16-micrometer physical field of
view from both pooled isotropic synthetic images and anisotropic real movies.
Multi-positive contrastive supervision treats two true daughters as positives
instead of forcing one to be a negative; a separate sparse division head is
trained with correction for the inflated synthetic division prior.

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
step must compare a small predeclared blend grid whose zero weight is the exact
Trackastra control. Only after freezing that weight may the four processed
movies be materialized once and scored by the pinned exact evaluator. No
leaderboard feedback participates in any decision.

## Runtime boundary

This is deliberately staged for the user-provided cloud instance. The active
Kaggle training run is projected to leave about 20.45 GPU-hours, so launching
this roughly ten-hour reciprocal image experiment on Kaggle would violate the
reserve handoff rule. Any future competition inference remains exactly two-GPU,
whole-movie sharded, and separately authorized.

## Verification

The focused temporal/Trackastra suite passes 38 tests. These cover physical
resampling, division-aware multi-positive loss, candidate-radius failure,
transition construction, model output normalization, arbitrary node-ID
alignment, exact zero-weight fallback, the frozen processed exact gate, and
two-GPU whole-movie sharding.
