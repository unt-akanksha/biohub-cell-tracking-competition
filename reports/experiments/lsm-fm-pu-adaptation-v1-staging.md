# LSM-FM PU adaptation v1 staging

## Decision

Launch a training-and-clean-validation experiment. Do not create or submit a
competition prediction.

This is a new detector family rather than another public-notebook replica or a
retuning of the retired SpatialDINO teacher-distillation path. The scientific
change is a geometry-matched, fully 3D light-sheet foundation model with a new
Biohub-owned one-channel detection head.

## Rationale

SpatialDINO selective distillation v2 preserved projected cell count but failed
the limiting localization gate: pooled selection recall was 0.885906, while
`6bba_57b7cc1e` reached only 0.602170 versus the immutable 0.65 worst-movie
gate. The projected count ratio on that movie was already 0.998942, so another
count or same-teacher loss adjustment is not the best next test.

The LSM-FM image-only student was pretrained on single-channel 3D light-sheet
volumes downsampled to isotropic 64-cubed patches, exactly matching the Biohub
training geometry. All transferable tensors are loaded; only its 512-channel
pretext head is replaced.

## Provenance

- Paper: https://arxiv.org/abs/2605.26026
- Official repository: https://github.com/AdinaScheinfeld/lsm_fm_public_repo
- Official model record: https://doi.org/10.5281/zenodo.20146516
- Source checkpoint SHA-256: `ef0f3d100f9a9aaa5d9a48bb9b07e7f1b0e0d1cc690cdcd308b1e0446bd01634`
- Safely stripped student SHA-256: `d287049e5f86ad1db7350cdf30acf2309c9dca34be8570c398f330f346afcfb0`
- Weight license: CC-BY-4.0
- Architecture implementation: MONAI 1.5.1, Apache-2.0
- Public Kaggle predictions copied: no
- Public Kaggle training code copied: no

The official repository lacks a license file at audited commit
`cb772551652c3ee762048936e396743fe9a65e10`, so its finetuning code is not
copied. The experiment uses the separately licensed weights with MONAI and the
project's own PU trainer/evaluator.

## Candidate and training

- Detector parameters: 15,702,979
- Pretrained student parameters: 16,656,946
- Decoder warm-up trainable parameters: 13,531,513
- Deep phase total trainable parameters: 15,544,993
- Seed: 20260827
- Training movies: all 187 non-validation movies
- Steps: 768 target, 192 minimum
- Unfreeze: final two Swin stages at step 192
- Learning rates: 2e-4 decoder, 2e-6 encoder
- Frozen teachers: the two independently seeded TemporalUNets
- Selective soft distillation: disabled
- Memory control: weak and strong 64-cubed activation graphs are serialized;
  the detached weak response is the consistency target for the strong view
- Accelerator: one T4 for this training experiment
- Maximum declared runtime: 2.00 hours

The user's two-GPU requirement applies to any later full competition
submission. This training-only notebook cannot create a submission artifact.

## Validation gates

The split and gates are unchanged:

- 8 clean selection movies
- 4 untouched acceptance movies, opened only after selection passes
- pooled selection recall at least 0.80
- worst selection movie recall at least 0.65
- no public leaderboard selection
- acceptance promotion must preserve the existing pooled, embryo-prefix, and
  per-movie regression gates

## Runtime verification

Private Kaggle dataset:
`indarkarhana/biohub-lsm-fm-pu-runtime-v1@version2`

Kaggle expands uploaded ZIP resources. The runtime therefore binds both the
top-level archive and all 428 extracted MONAI members. The independently
downloaded version verified 15 other top-level files plus those 428 members.

- Runtime manifest SHA-256: `fe6077dd0618f4fd2e6b026d88bf18323a66b27b7cebd5bb9431b2e1d3d6735c`
- Preflight report SHA-256: `7da695cf355e9336851e512e6e88fd1fdbef40f715d5c312124bd4ef401f4a21`
- Focused tests: 36 passed
- Full 64-cubed CPU forward/backward optimizer step: finite
- Internet: disabled
- TPU: disabled
- Submission API/file: absent and guarded

## Stop rule

If selection fails, retire the family without opening acceptance. If promotion
passes, build a separate non-replica, two-GPU, whole-movie-sharded submission
candidate; do not submit it without explicit authorization.
