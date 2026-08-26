# LSM-FM association integration — 2026-08-26

## Decision

Use the public `0.927` notebook only as a frozen score comparator. The next
candidate is a new hybrid: independently trained LSM-FM nodes are linked by
the organizers' official TemporalUNet3D node-transformer implementation. It
must pass complete-movie held-out scoring before any test inference or
submission is permitted.

This is not an exact public-notebook replica. Its detector identities,
probabilities, density threshold, and sub-voxel coordinates come from our
feature-24/feature-36 checkpoints and a selection rule frozen before the
acceptance movies are opened. Public predictions are not copied.

## Source and license audit

- Official source: `royerlab/kaggle-cell-tracking-competition`
- Pinned commit: `075fc5f5a52d11077f9dc2b074644618f26939e2`
- Source license: BSD-3-Clause
- License-file SHA-256:
  `4cd604324b1a9f0c420786a92501430a7ff9189f46bdcdfc485aa43bfc16973b`
- Primary checkpoint input:
  `pilkwang/biohub-tracking-support-pack-50ep-v1`, Kaggle metadata license
  `CC0-1.0`, checkpoint SHA-256
  `12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771`
- Independent seed checkpoint input:
  `pilkwang/biohub-temporal-unet3d-seed314159-v1`, Kaggle metadata license
  `CC0-1.0`, checkpoint SHA-256
  `9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f`

The production runtime must include the BSD notice. The two downloaded Kaggle
metadata records are frozen under
`.biohub/cache/metadata/association-assets/` for the local audit.

## Implemented bridge

`research/lsm_fm_detection/association_bridge.py` makes one narrow,
fail-closed substitution:

1. predict the complete movie with the cleanly selected LSM-FM candidate;
2. select one global probability threshold from fixed uniform frames and the
   organizer-provided `estimated_number_of_nodes` metadata;
3. pass rounded LSM coordinates to the official feature indexer and edge
   transformer;
4. verify that association preserved node count and frame order; and
5. restore the precise LSM sub-voxel coordinates before graph construction.

The last step prevents the detector's localization gain from being destroyed
by the official linker's integer feature lookup. Only coordinates are restored;
edge identities and probabilities remain the linker's outputs.

## Next clean gate

After `lsm-fm-ensemble-validation-v1` terminates, freeze its selected detector
candidate. Evaluate a single predeclared association configuration on the four
complete acceptance movies already used for the clean public comparator.

Promotion requires all of the following:

- complete official graph metric, not node recall alone;
- pooled proxy score above the frozen `0.9294432421` comparator;
- no embryo-prefix regression greater than `0.01`;
- exact node/frame coverage and valid biological topology;
- output hash different from the public comparator;
- no leaderboard evidence, test predictions, or submission artifacts.

If it fails, retain the LSM detector family but do not call it a submission
candidate. Diagnose edge recall at matched LSM nodes before spending more GPU.

## Two-GPU production contract

Any later submission run must see exactly two CUDA devices, shard whole movies
deterministically across both T4s, and validate disjoint exact movie coverage
before CSV assembly. There is no single-GPU fallback. This requirement applies
to the submission kernel; held-out scientific experiments may remain
single-GPU when that is more quota-efficient.
