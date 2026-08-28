# Temporal contextual Kaggle-only topology v1

Date: 2026-08-28

Status: implemented and locally verified; no Kaggle GPU job, kernel push, or
competition submission was performed.

## Finding

The final contextual-v3 notebook was attached to
`indarkarhana/biohub-temporal-contextual-transfer-output-v3`, a private dataset
intended to transport cloud-trained checkpoints. Under the active Kaggle-only
instruction, transfer instead completes as the private code-kernel output
`indarkarhana/biohub-temporal-contextual-transfer-v3`. The old topology would
therefore require an unnecessary download and dataset re-upload and would make
the guarded final kernel fail if that cloud-specific dataset did not exist.

## Repair

- The final notebook now attaches the Kaggle transfer kernel directly.
- Its input discovery still requires exactly one completed contextual-v3
  terminal, both fold checkpoints, both-fold improvement, and strict checkpoint
  verification.
- The upload wrapper independently requires the same exact kernel and dataset
  source inventories locally and remotely.
- The artifact stager now publishes only the hash-bound exact-acceptance JSON;
  contextual checkpoints remain in the Kaggle transfer output.
- The final preflight now binds the downloaded Kaggle transfer launcher,
  checkpoints, processed materialization launcher/CSV, and exact CPU acceptance.
  A cloud benchmark is no longer required.
- The complete processed benchmark must cover the four predeclared movies,
  change edges, use exactly two GPUs, avoid ground-truth access during
  materialization, and match the hashes in exact accepted evidence.

## Frozen source topology

Datasets:

- `indarkarhana/biohub-temporal-contextual-transfer-runtime-v1`
- `indarkarhana/biohub-temporal-contextual-exact-acceptance-v3`
- `pilkwang/biohub-tracking-support-pack-50ep-v1`

Kernel outputs:

- `indarkarhana/biohub-clean-0-927-reproduction-v1`
- `indarkarhana/biohub-trackastra-dual-fold-synthetic-v1`
- `indarkarhana/biohub-temporal-contextual-transfer-v3`

Competition input remains
`biohub-cell-tracking-during-development`; Internet and TPU remain disabled and
the machine shape remains two T4 GPUs.

## Local evidence

- Contextual training-to-submission contract suite: `80 passed`.
- Kaggle-only topology, stager, preflight, and upload-boundary suite:
  `16 passed`.
- Expanded final-path suite after the topology repair: `29 passed`.
- Final combined contextual regression after all repairs: `89 passed`.
- Final notebook SHA-256:
  `5dad76f56003be6f84e381dbf1a735cb2e091dcedf8f238edb184fd0091c03af`.
- Final metadata SHA-256:
  `429ddae35e0633069a5ac44151cd6a1e4782ccd130b1f315d448eb43be8276fe`.
- Builder SHA-256:
  `74aa08ab17171884a3a24f854ff4fa3a63f8ba0f3d6f43fe005aaab68d583b37`.
- Preflight builder SHA-256:
  `423d255816bc6c7bc197f183c50ad1cb01ea252970821c2640e91522059baa64`.
- Acceptance stager SHA-256:
  `1bd2319310c2e24c637b0d2b0d909fe11f756af7d2e5e450cafc614fee4f98fa`.
- Upload wrapper SHA-256:
  `f6122f465c9c39abd6f5e4dbbc476b00fa3c7dd2fe1d256826d22e41ea1be043`.

The third-embryo acceptance data remains unopened for model selection. This
repair changes transport and verification only; it does not alter the frozen
model, checkpoints, calibration grid, processed metric, or submission
authorization boundary.
