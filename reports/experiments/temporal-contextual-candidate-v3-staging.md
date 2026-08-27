# Temporal contextual candidate v3 staging

Status: final two-GPU candidate notebook staged and tested; intentionally not
pushed, run, or submitted because exact processed acceptance does not yet
exist.

This is a gated inference artifact, not a public-notebook replica. It retains
the strong frozen detector nodes but replaces association decisions with the
project-authored contextual v3 model only after three independent clean gates:
external-embryo pretraining/acceptance, reciprocal competition transfer and
calibration, and exact processed CPU acceptance.

## Runtime and timeout repair

- Visible GPUs required: exactly `2`
- Sharding unit: complete movie; a movie is never split between GPUs
- Load estimate: candidate-pair contextual work plus temporal encoder work,
  including the second encoder only for a selected reciprocal-mean fold
- Inference hard stop: `36,000` seconds
- Notebook budget: `43,200` seconds
- Reserved setup/finalization time: `7,200` seconds
- Worker shutdown grace: `15` seconds, followed by forced reap
- Internet: disabled

This directly addresses the preceding candidate timeout: both T4s are required,
movies are balanced by contextual workload rather than movie count, and the
inference process cannot consume the final two notebook hours.

## Fail-closed evidence

The notebook cannot start inference without one hash-bound JSON object that
reports:

- `status: accepted`;
- `evaluation_kind: exact_processed_dual_fold_acceptance`;
- `exact_processed_gate_passed: true`;
- family `trackastra_contextual_pair_fusion_blend` / model
  `temporal_contextual_pair_fusion_v3`;
- no leaderboard selection; and
- no earlier competition submission.

The private evidence dataset
`indarkarhana/biohub-temporal-contextual-exact-acceptance-v3` must not be
created until that evidence genuinely exists. The candidate loader then
rechecks both Trackastra and appearance checkpoint hashes, architecture
metadata, accepted blend weights, and the association configuration.

## Non-replica and submission boundary

- Frozen base CSV SHA-256:
  `33c179b0449b9cdd186f06a653cddc8cf12359f008982f6713cdf30784a52e6a`
- Frozen raw-confidence graph tree SHA-256:
  `559332597da65f161f1b0b116e10fc86c7ff35eb31fe48937e080889b909a43e`
- Nodes must be preserved exactly
- Every movie must appear in exactly one GPU shard
- At least one association edge must change
- Candidate CSV hash must differ from the base
- Output: `/kaggle/working/submission.csv`
- Upload/competition submit command: absent

The base supplies detector nodes and raw confidence evidence; it does not make
this candidate our public-code replica. The independently trained contextual
models produce the candidate associations, and identical public edges are an
explicit runtime failure.

## Frozen package

- Kernel:
  `indarkarhana/biohub-temporal-contextual-submission-candidate-v3`
- Notebook SHA-256:
  `6e54ac2af540b4c3745db8589e0d8809fb2ac8e00f834e2172a8f147df14dbb6`
- Metadata SHA-256:
  `3c232d084b6a35e7524621a2baf4c5f3d9f0397378c8b22e3edb4021feb95d8a`

Execution belongs on the user-provided two-GPU cloud unless a fresh Kaggle
quota guard proves the full declared window still leaves at least eight Kaggle
GPU hours. Producing the CSV does not authorize uploading it; a competition
submission remains a separate explicit user decision.
