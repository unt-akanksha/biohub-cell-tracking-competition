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
step must compare a small predeclared appearance/division blend grid whose
zero/zero setting is the exact Trackastra control. Unlike a retrieval-only
proxy, selection now executes the frozen clean graph linker and measures pooled
edge and division Jaccard with a per-movie regression floor. The trained source
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

The updated focused temporal/Trackastra suite passes 29 tests across the normal
and pinned exact-scorer environments. It covers physical
resampling, division-aware multi-positive loss, candidate-radius failure,
transition construction, model output normalization, arbitrary node-ID
alignment, exact zero/zero fallback, source-division alignment, actual
second-daughter recovery, the frozen processed exact gate, runtime-package
integrity, and two-GPU whole-movie sharding.

The rebuilt portable archive is
`.biohub/staging/biohub-temporal-patch-runtime-v1-division-aware-20260827.zip`
(92,116 bytes, SHA-256
`ca57cba93f5b45a679cd0ca1d1e9a14fff45e397e247ee74db28bddf90cbd1c1`).
An independent extraction verified all 25 manifest-bound files; the embedded
verifier reported manifest SHA-256
`bd61b0945fce4e5408f077a57f428dc9075a6c649ead49784238c240aae0d6c9`,
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
both Trackastra frame-pair products and the added per-node 3D encoding work.
