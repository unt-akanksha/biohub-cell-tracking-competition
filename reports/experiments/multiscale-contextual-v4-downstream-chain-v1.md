# Multiscale contextual v4 downstream chain v1

Date: 2026-08-28

## Outcome

The separate project-authored `temporal_multiscale_contextual_pair_fusion_v4`
lane now has a complete guarded path from verified ZebraHub pretraining through
reciprocal Biohub transfer, calibration, processed materialization, exact CPU
acceptance, transition-balanced final inference, and verified Kaggle submission.
The v3 candidate remains first in the queue. The v4 lane cannot launch
pretraining until the v3 submission receipt exists and cannot advance any later
stage unless the preceding immutable verifier passes.

## Scientific boundary

- Parameters per fold: 46,386,607 (2.2357x contextual v3).
- Initialization: every v3 tensor strict-loaded; new axial residual adapters
  start at zero and preserve the v3 prediction function before optimization.
- Optimization source: ZSNS004 only.
- Selection source: fixed ZSNS005 developmental windows.
- Audit source: disjoint fixed ZSNS005 developmental windows opened only after
  checkpoint selection.
- ZSNS001 is not reopened for v4; its one-shot external role remains confined
  to v3, avoiding repeated test-set consultation.
- Both v4 folds must pass external selection and audit improvement gates, then
  reciprocal real-fold improvement and synthetic-retention gates.
- Calibration includes the exact zero-weight Trackastra control.
- Final promotion requires the four-movie exact processed gate, identical node
  rows, changed edge sets, bounded per-movie regression, and pooled improvement.
- Public predictions, public leaderboard selection, metric hacks, and replica
  submission are excluded.

## Kaggle execution contract

- Every GPU stage requires exactly two Nvidia T4 devices with internet and TPU
  disabled.
- User authorization changed the Kaggle reserve from 8.00 hours to 0.00 hours.
  A stage still waits until enough live quota exists for its declared ceiling.
- Calibration kernel: `indarkarhana/biohub-multiscale-calibration-v4`.
- Processed kernel: `indarkarhana/biohub-multiscale-processed-v4`.
- Final kernel: `indarkarhana/biohub-multiscale-submission-candidate-v4`.
- Final runtime dataset:
  `indarkarhana/biohub-multiscale-contextual-final-v4`, version 1.
- Final runtime manifest SHA-256:
  `221ab4359e84706a306cd88b41e2790ebd1a4bad71d2a36ef4a1873499a570b4`.
- Transition-balanced inference source SHA-256:
  `575f375a261b34cdda67709c4de46a6b64772ec347089e606e7c7dedae4716f7`.

## Controller chain

The already-armed v4 pretraining and transfer controllers are followed by four
new hash-bound background controllers:

1. verify v4 transfer output and launch reciprocal calibration;
2. verify calibration output and launch processed materialization;
3. run exact CPU scoring, publish its private evidence, run final preflight,
   and launch transition-balanced inference;
4. verify the completed kernel output and submit only the exact accepted,
   non-replica candidate.

Current controller SHA-256 values:

- calibration: `a0e9bbbd0598b4dbd6a4814272e264ed22796d68d69f9c65c7cd4467a43abe88`
- processed: `9093b6cfc9cd474aac4a4403185006f9861ead3b8edd2068bb71d7d8381bac1c`
- final: `b940cd1c36acd3e1ea70448e1b6bde2c9dc1b3b4e09e7bf140e7e79dc1cbb3fa`
- verified submit: `b3cecc3dea3e76d30d5b5ad8c016fdcd26f26ce8f1c426e41fdacc379118c197`

## Reliability repair

The v3 final preflight incorrectly referenced a nonexistent
`candidate_report` entry. It now binds the already verified
`materialization_result` for non-replica provenance. The v3 exact-evidence
dataset upload was also changed to execute from the dataset directory, avoiding
the Kaggle CLI Windows upload-marker path failure observed while publishing the
v4 final runtime. The affected v3 final controller was safely restarted before
any final-stage terminal existed.

## Verification

- 43 focused architecture, verifier, kernel, sharding, exact-acceptance, and
  submission-transport tests passed.
- Six downstream-specific tests passed, including parsing all four background
  controller transformations without launching them.
- The private final runtime was redownloaded from Kaggle and its 39-file frozen
  manifest passed `verify_runtime.py`.
- All three v4 kernel slugs are below Kaggle's 50-character limit.
- Python compilation and `git diff --check` passed. Ruff was unavailable in the
  local environment and therefore was not run.

## Final-runtime compute profile

The frozen clean-base submission inventory was profiled across all four
possible reciprocal calibration outcomes (one or two appearance encoders for
each embryo prefix). Conv/Linear multiply-accumulates were counted on the exact
17x17x17 physical patch used at inference:

- contextual v3: 3.315712192 GMAC per node;
- multiscale v4: 3.751869492 GMAC per node;
- v4/v3 compute ratio: 1.131542569;
- v4/v3 parameter ratio: 2.235740377.

Thus v4 supplies 2.24x parameter capacity for only about 1.13x per-node MACs.
Using the measured v4 cost to evaluate the already-published scheduler, the
worst two-GPU projected load ratio is 1.006967x across all four possible future
blend outcomes. A MAC-adjusted scheduler would improve that only to 1.003808x,
below the predeclared 1.01x mutation threshold. The frozen production runtime
is therefore retained rather than versioned for a negligible scheduling
change. Evidence is stored in
`artifacts/profiles/temporal-multiscale-runtime-v1.json`; profiling used CPU,
read no leaderboard, and performed no submission.
