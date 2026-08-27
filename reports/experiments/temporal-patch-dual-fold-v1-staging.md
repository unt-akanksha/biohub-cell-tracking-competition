# Temporal patch dual-fold v1 staging

Date: 2026-08-27

Status: implemented and locally verified; reserved for the cloud handoff, not
launched on Kaggle. Its frozen geometry source is the verified predeclared
Trackastra initialization because the reciprocal fine-tune failed its clean
real-data gate.

## Why this is a new candidate

The public 0.927 comparator and the completed reciprocal Trackastra experiment are
predominantly geometry-driven. Frozen SpatialDINO patch similarity previously
made zero safe corrections, which is evidence against weakening its swap gates,
not evidence that local image appearance is useless. The missing experiment is
a representation trained directly on adjacent Biohub cells.

This lane implements an independent 19,221,954-parameter 3D residual encoder per
fold. It samples 17-cubed patches over the same 16-micrometer physical field of
view from both pooled isotropic synthetic images and anisotropic real movies.
Each node is represented by aligned `t-1`, `t`, and `t+1` channels with
deterministic boundary clamping. This exposes motion and mitotic morphology to
both the identity and division heads while leaving the expensive 64-channel
feature-map width unchanged.
All-positive supervised contrastive learning averages the log-probability of
every true child. This prevents one easy daughter from hiding a missed second
daughter in the loss; a separate sparse division head is trained with
correction for the inflated synthetic division prior. Checkpoint
selection uses a full-model optimizer-step exponential moving average (decay
0.997), reducing sensitivity to one noisy minibatch or validation instant.
The prior correction is class-conditional: synthetic positives are downweighted
from the 4.07% synthetic rate toward the 0.26% real rate, while synthetic
negatives retain approximately unit weight. This avoids the earlier failure
mode where scaling the whole BCE nearly erased non-division supervision.
Gradient accumulation is also driven by successful batches rather than loop
attempt numbers, and a partial final window is explicitly rescaled. Invalid
transitions can therefore no longer silently shrink or enlarge an optimizer
update.

Calibration now also compares the target-fold encoder against an equal mean of
both reciprocal encoders. Because the global split keeps every calibration
movie out of both models' training inventories, this is a clean ensemble test,
not in-sample stacking. `target_only` wins exact ties to preserve runtime; the
two-encoder path is selected only when it clears the same pooled-gain and
per-movie regression gates. In that path, both encoders consume the same
already-resampled physical patch tensor, avoiding a second grid-sampling and
image-extraction pass per node.

Temporal contexts also use a bounded rolling frame cache. The two contexts in
one training transition load their four unique frames once instead of loading
the shared middle pair twice. Whole-movie inference reuses each decompressed
Zarr frame across neighboring contexts, then evicts frames that cannot be used
again. Predictions are unchanged, while the fixed wall clock can cover more
optimizer work and final inference is less exposed to I/O timeout.

No public Kaggle implementation, public prediction identity, CELLECT source, or
external checkpoint is copied. CELLECT and the 2026 microscopy-embedding paper
informed only the modeling hypothesis.

The daughter-complete loss follows the all-positive formulation in
[Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362). Its use of
local morphology alongside a separate division signal is also consistent with
the 2025 primary study on
[contrastive cell-division detection and tracking](https://doi.org/10.1186/s12859-025-06344-5).

## Leakage and selection boundaries

Two isolated GPU workers train reciprocal embryo folds. Each worker uses 1,900
synthetic movies plus 96 real movies from the opposite embryo prefix. Twelve
unopened target-prefix movies select its checkpoint, and a different twelve are
reserved for later appearance-weight calibration. The four already opened
processed-acceptance movies are excluded from both stages.

The two workers share one deterministic partition for each embryo prefix:
12 checkpoint movies, 12 calibration movies, then up to 96 training movies
(all remaining eligible movies if fewer are available). This is
stricter than independently salted fold lists: neither reciprocal model can
train on a movie used to select or calibrate the other, which also keeps a
future two-model ensemble evaluation honest.

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

This is deliberately staged for the user-provided cloud instance. The completed
Kaggle training run left 21.29 GPU-hours, so launching this roughly ten-hour
reciprocal image experiment on Kaggle would violate the reserve handoff rule.
Any future competition inference remains exactly two-GPU, whole-movie sharded,
and separately authorized.

The package now includes a fail-closed Trackastra ingestion verifier. The
adapted-model mode still requires both on-disk worker terminals to equal the
aggregate evidence and pass the original gains. The explicitly enabled control
mode instead requires `best_step: 0`, exact initial/best metric equality, the
same model hash in both folds, the corrected-geometry manifest, exact available
real split sizes (96 and 69 training movies), and no CSV, ZIP, or
submission-named artifact. This admits only the initialization that was
predeclared as a candidate, never a rejected fine-tuned state.

## Verification

The updated temporal/Trackastra checks and the full environment-split
repository suite pass across the normal and pinned exact-scorer environments.
Coverage includes physical
resampling, temporal boundary clamping and shared-channel grids, all-daughter
supervised-contrastive loss, partial-accumulation normalization,
candidate-radius failure,
transition construction, model output normalization, arbitrary node-ID
alignment, exact zero/zero fallback, source-division alignment, actual
second-daughter recovery, the frozen processed exact gate, runtime-package
integrity, and two-GPU whole-movie sharding.

The rebuilt portable archive is
`.biohub/staging/biohub-temporal-patch-runtime-v1-heavy-temporal3-framecache-strictingest-ema-ensemble-daughterloss-runtimeguard-t4x2-controlsource-20260827.zip`
(103,920 bytes, SHA-256
`7ac8a2ecf5887f0fb5a65250cc0c9292e8aba54445bea3691a86db02b4ca8c3e`).
An independent extraction verified all 26 manifest-bound files; the embedded
verifier reported manifest SHA-256
`5eea2355619210ce54c115f979eb4a08ab23c18b638e98d3f2b1410abd55381b`,
required GPU count 2, and no submission command.

A separate end-to-end gradient smoke test used two visibly different synthetic
3D cells, their shifted children, and one distractor. Over 40 CPU optimizer
steps, multi-positive contrastive loss fell from `1.0668325` to `0.00001049`;
the two sources selected target columns `[0, 1]` with correct-pair cosine scores
`0.9896` and `0.9962`. This confirms that physical crop extraction, the encoder,
and the association objective form a learnable path rather than merely passing
shape checks.

The cumulative environment-split repository suite now passes 522 unique tests
with zero failures
when each group runs in its declared environment; two Windows tests are skipped
only because unprivileged symlink creation is unavailable. The ordinary
environment passed 466 tests after excluding the scorer-only files, and all 56
locked-scorer tests passed in the pinned evaluator environment. For timeout
resistance, final whole-movie LPT sharding now weights
both Trackastra frame-pair products and the added per-node 3D encoding work; its
node cost was increased to 12,288 after scaling the encoder to 19.2M parameters.
Both final candidate builders now enforce a 36,000-second inference ceiling,
leaving at least 7,200 seconds of Kaggle's 12-hour GPU notebook limit for
setup, final assembly, artifact persistence, and shutdown. This ceiling cannot
be raised through a command-line override. Final-kernel metadata is also
fail-closed to Kaggle's `NvidiaTeslaT4` T4 x2 shape with TPU and internet
disabled; runtime separately requires exactly two visible CUDA devices.
Timed-out workers in every dual-GPU runtime stage are terminated together,
given one shared 15-second grace period, then force-killed and reaped if
necessary so they cannot consume the next stage's budget or the reserved
finalization window after the orchestrator exits.
