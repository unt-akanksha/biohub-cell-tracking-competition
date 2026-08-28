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
  `indarkarhana/biohub-zebrahub-contextual-acceptance-runtime-v1`, version 3
- Runtime manifest SHA-256:
  `6aefc98b953a15b853731bc970a9734ddcab08a51938bcc9769f29fa6bba1f29`
- Runtime files: `15`
- Runtime verified bytes: `281,596`
- Remote re-download:
  `.biohub/cache/dataset-redownloads/biohub-zebrahub-contextual-acceptance-runtime-v1-version3`
- Notebook SHA-256:
  `41835ade5603bb7f64c3703d2eb5568198b3d4ce26e43bef13c708b68a3bc874`
- Kernel metadata SHA-256:
  `f7a2915c445cfa58e5f2e1361e0b33a40f328bfe5e0dc2702ca802781f8ea094`

The first relative-path dataset publication command failed locally before
creation because of a Kaggle CLI temporary-path bug. The absolute-path retry
created private version 1. Registration then caught a 65-character manifest
constant before launch. Version 2 corrected that fail-closed binding to the
actual 64-character local/remote acceptance manifest. Version 3 additionally
promotes compact autocast contextual logits to the sparse output dtype before
index scatter. A full CLI re-download of version 3 reproduced the exact runtime
manifest and all 15 content hashes. Only runtime version 3 is admissible for
the acceptance launch; version 1 is retained as failed pre-launch provenance
and version 2 is excluded because its contextual scatter was not autocast
dtype-safe.

No GPU run, competition submission, public prediction, or leaderboard result
was used while staging this gate.

The first guarded kernel-creation request was rejected by Kaggle before a
version existed because the provisional remote slug and title were each 51
characters. Both now use the 41-character
`biohub-zebrahub-contextual-acceptance-v1`; the scientific run ID, notebook
bytes, runtime, inputs, gates, and budget are unchanged. The downstream
transfer source was rebound to that shortened remote slug.

The next live creation attempt returned HTTP 409 before a kernel version was
created because that provisional kernel slug was identical to the attached
acceptance dataset slug. The kernel alone is now named
`indarkarhana/biohub-zsns001-contextual-gate-v1`, with title
`Biohub ZSNS001 Contextual Gate v1`. The acceptance dataset retains its frozen
original slug. This repair changes no notebook code, data, model, gate, or
budget; dependent kernel-source metadata is rebound to the distinct kernel
slug.
