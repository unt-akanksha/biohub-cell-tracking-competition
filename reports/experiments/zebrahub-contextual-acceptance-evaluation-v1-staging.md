# ZebraHub contextual acceptance evaluation v1 staging

Status: staged, private runtime published and remotely verified, not launched.

This is the one-shot third-embryo gate for the project-authored contextual v3
family. It can run only after both pretraining folds pass their fixed ZSNS005
selection and audit gates and their checkpoint hashes are frozen.

The evaluator reconstructs each fold's exact seeded initialization and compares
it with the hash-bound final checkpoint on all 16 frozen ZSNS001 shards. Each
fold must gain at least `0.01` composite, strictly improve top-1 and MRR, avoid
division-top-2 regression, and preserve the exact row/division/transition
inventory. Both folds must pass. ZSNS001 cannot redirect checkpoint or
hyperparameter selection and cannot be reused for tuning.

## Isolation and budget

- Required GPUs: exactly `2`, one independent fold per GPU
- Kaggle machine shape: `NvidiaTeslaT4`
- Evaluator hard stop: `3,300` seconds
- Notebook watchdog: `3,600` seconds
- Internet: disabled
- Competition input: absent
- Competition submission: unavailable
- Public predictions: unavailable
- Public leaderboard selection: unavailable
- Launch state: not launched while `zebrahub-contextual-pretrain-v1` is active

## Frozen evidence

- Acceptance dataset:
  `indarkarhana/biohub-zebrahub-contextual-acceptance-v1`, version 1
- Acceptance manifest SHA-256:
  `cbbf670dde160e5a927ed84bb9e2a7313abe4506f4798f6afa00680fc8e7c6d0`
- Published inventory SHA-256:
  `e32bc686e14222e43acb8d6247351e286eae8ed6fdb1f4ab5087e55fb0c79667`
- Trainer-record inventory SHA-256:
  `05ad8b3195aa2786ffb8a2ffcb247d118d1e5cf96026264e0534c468979e6e0e`
- Evaluation runtime:
  `indarkarhana/biohub-zebrahub-contextual-acceptance-runtime-v1`, version 2
- Runtime manifest SHA-256:
  `bb9cd258a51f45141086104bba8f16c28a9cbdc55bc5c4cdaac5b04b7d07ad44`
- Runtime files: `15`
- Runtime verified bytes: `281,181`
- Remote re-download:
  `.biohub/cache/dataset-redownloads/biohub-zebrahub-contextual-acceptance-runtime-v1-version2`
- Notebook SHA-256:
  `054ecd49c4dbdb0a16ec546d2ebe53acf65f36b19f9503587ad31fe5aa70252b`
- Kernel metadata SHA-256:
  `691e8fc2c4d1e7ea27df2a2dc483d1159237714c38c3555b6ef0c913b68d8a0e`

The first relative-path dataset publication command failed locally before
creation because of a Kaggle CLI temporary-path bug. The absolute-path retry
created private version 1. Registration then caught a 65-character manifest
constant before launch. Version 2 corrects that fail-closed binding to the
actual 64-character local/remote acceptance manifest. A full CLI re-download
of version 2 reproduced the exact runtime manifest and all 15 content hashes.
Only runtime version 2 is admissible for the acceptance launch; version 1 is
retained solely as failed pre-launch provenance.

No GPU run, competition submission, public prediction, or leaderboard result
was used while staging this gate.
