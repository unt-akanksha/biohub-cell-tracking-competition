# Biohub research refresh — 2026-08-27

## Scope and integrity

This refresh is restricted to clean modeling. Metric exploits, duplicate-track
schemes, fake forks, leaderboard-driven parameter selection, and exact public
code replication remain excluded. The active experiment is private,
training-only, and cannot create or submit a competition archive.

## Current community signals

The live Kaggle discussion index was re-read on 2026-08-27. The newest threads
continue to separate the post-patch frontier from metric-hack scores and focus
on training, edge quality, the public synthetic corpus, ground-truth acquisition
jumps, and submission/scoring timeouts:

- Competition discussions:
  https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion
- Ground-truth frozen frames and global jumps:
  https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/724283
- Consecutive-frame failure behind otherwise valid zero-score submissions:
  https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/728613
- Metric-hack discussion retained only as a prohibition/evaluator warning:
  https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/723655

Two implications are actionable. Frame-global motion augmentation is justified,
and saving runtime by linking skipped frames is invalid because official edges
must connect `t` to `t+1`. The active trainer implements the former. All owned
linkers and the future dual-GPU submission assembly validate the latter.

## Relevant model research

- Trackastra learns division-aware pairwise associations over a full
  spatiotemporal detection context. It remains the lower-risk architecture for
  the current corrected-synthetic run:
  https://arxiv.org/abs/2405.15700
- HOCT is an edge-centric higher-order transformer designed to avoid the
  division/path entanglement of node-centric graph representations:
  https://arxiv.org/abs/2607.11754
- CELLECT uses adjacent 3D frames to predict a center confidence map, a
  64-channel voxel embedding, and a division map. This supports adding learned
  local image appearance once geometry-only association plateaus:
  https://www.nature.com/articles/s41592-025-02886-x
- Recent cell-linking evidence reports gains from self-supervised/foundation
  image embeddings over shallow geometry features:
  https://openreview.net/pdf?id=4OLrHi9ROr
- Ultrack jointly reasons over uncertain segmentations and temporal tracks,
  which is relevant if the frozen detector topology becomes the limiting term:
  https://www.nature.com/articles/s41592-025-02778-0

## Active experiment decision

`trackastra-dual-fold-synthetic-v1` addresses a previously unnoticed coordinate
error: the public temporal synthetic labels are native Y/X coordinates even
though sequence images are pooled. The old geometry path divided Y/X by four
and then applied the real-data Trackastra scale, shrinking transverse motion by
four. The new path validates pooled bounds and restores native Y/X before model
scaling.

Each of the two T4 workers trains a 27,456,880-parameter model. Reciprocal
leave-one-embryo-out folds use 95% corrected synthetic replay and 5% real replay,
with frame-global drift/jump augmentation. Fixed threshold-free ranking on
unopened opposite-embryo movies selects checkpoints; the official initialization
is retained when an adapted state does not pass the minimum-gain/non-regression
gate.

## Cloud continuation if geometry plateaus

The next heavy lane should not repeat the rejected 289-parameter HOCT linear
probe. It should test one of the following in order:

1. Add learned adjacent-frame 3D appearance embeddings to the accepted
   reciprocal Trackastra candidate while keeping whole-movie cross-embryo
   evaluation.
2. Backpropagate through the full 6.25M-parameter JIT HOCT backbone on corrected
   synthetic graphs, with the original checkpoint retained as a candidate and
   conservative Biohub replay.
3. Train a two-frame 3D U-Net affinity/division head on the fully labeled
   synthetic volumes, then use its displacement/embedding output only to resolve
   ambiguous competing parents in the frozen strong detector topology.

None of these lanes may reuse the processed four-movie acceptance labels for
hyperparameter selection. After configuration freeze, one exact processed
official-score comparison decides whether a candidate survives.

## Runtime rule for every future submission

Every submission notebook must fail closed unless exactly two CUDA devices are
visible. Complete movies are assigned once using measured pair-inference cost;
one isolated process owns each GPU. Assembly is forbidden until the two shard
reports prove exact, disjoint movie coverage, every edge is consecutive in time,
and the candidate differs from the public baseline. No frame-stride shortcut is
allowed.
