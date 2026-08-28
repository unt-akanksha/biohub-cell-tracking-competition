# Temporal contextual transfer v3 staging

Status: private transfer runtime version 4 remotely verified; training staged
for the user-provided cloud, not launched and not submitted.

This is the competition-specific continuation of the project-authored
contextual v3 family. It is not a public-notebook replica: the temporal 3D
encoder, candidate transition features, outgoing/incoming edge-set pooling,
reciprocal parent objective, transfer gate, and orchestration are local project
code. No public weights or predictions enter the appearance path.

The transfer cannot start until `zebrahub-contextual-pretrain-v1` passes both
fixed ZSNS005 gates and `zebrahub-contextual-acceptance-evaluation-v1` passes
once on frozen ZSNS001. The latter only accepts or rejects the family and
cannot redirect the checkpoint or tune this recipe.

## Transfer recipe

- GPUs: exactly `2`, one reciprocal embryo fold per GPU
- Parameters: `20,747,761` per fold
- Initialization: hash-bound external checkpoint for the matching fold
- Competition steps: `20,000` per fold
- Real replay: `0.60`
- Synthetic replay: `0.40`
- Learning rate: `5e-5` to `5e-7` cosine decay
- EMA decay: `0.997`
- Fixed real validation movies: `12` per target embryo
- Unopened calibration reserve: `12` movies per target embryo
- Opened four-movie processed acceptance: excluded from training
- Worker wall limit: `36,000` seconds
- Orchestrator hard stop: `37,800` seconds
- Notebook watchdog: `39,600` seconds
- Internet: disabled
- Submission command: absent

The lower learning rate and heavier real replay are deliberate transfer
settings. They adapt the same-organism ZebraHub representation to competition
imaging while retaining 40% synthetic replay against catastrophic forgetting.

## Evidence gate

Each selected fine-tuned checkpoint must satisfy all of the following relative
to its exact pretrained initialization on an unchanged inventory:

- real composite gain at least `0.005`;
- real top-1 gain strictly positive;
- real MRR gain strictly positive;
- real division-top-2 gain nonnegative; and
- no synthetic composite, top-1, MRR, or division-top-2 regression worse than
  `0.01`.

Both folds must pass. The downloaded output verifier independently recomputes
every gain, checks the initialization hashes and split inventories, strict-loads
both checkpoints, and rejects any competition artifact in the training output.

## Frozen package

- Runtime dataset:
  `indarkarhana/biohub-temporal-contextual-transfer-runtime-v1`
- Only admissible version: `4`
- Excluded versions: `1` (nested source omitted), `2` (pre-freeze experiment
  package), `3` (autocast-unsafe sparse contextual-logit scatter)
- Runtime manifest SHA-256:
  `cbe5fe27639155746c95a98d91702d5fbe595172b058e0e9db330374ecfff25d`
- Runtime files: `36`
- Runtime verified bytes: `550,438`
- Remote re-download:
  `.biohub/cache/dataset-redownloads/biohub-temporal-contextual-transfer-runtime-v1-version4`
- Kernel:
  `indarkarhana/biohub-temporal-contextual-transfer-v3`
- Notebook SHA-256:
  `17f0f25562c4a0878d9070ea0b238867bf190b1b9b754899bf0d395dee86f353`
- Kernel metadata SHA-256:
  `bbfbc009c9ffc9a18330f8556d88c7cea04542824550cf1a9c96922871f5f322`

The runtime version-4 re-download reproduced its exact manifest and all 36
content hashes after materializing the packaged Trackastra source directory.
Version 4 carries only the mixed-precision compatibility repair: autocast
contextual logits are differentiably promoted to the FP32 sparse output dtype
before index scatter. No recipe, split, seed, gate, or selection policy was
changed. No Kaggle GPU, leaderboard result, or competition submission was used
to stage this transfer lane.

The acceptance kernel input now resolves through the shortened remote slug
`indarkarhana/biohub-zebrahub-contextual-acceptance-v1`. This is a metadata-only
repair for Kaggle's identifier boundary; the transfer notebook remains byte
identical at that repair point.

Before launch, the transfer-side acceptance verifier was hardened without
changing training. It now binds the completed launcher to the aggregate
terminal by SHA-256, binds the aggregate rows to both exact fold terminals,
checks all frozen ZSNS001 inventory hashes, independently recomputes each
composite/top-1/MRR/division-top-2 gate, verifies the pretraining terminal and
worker hashes, and strict-loads both 20,747,761-parameter checkpoints. A copied,
mutated, incomplete, or internally inconsistent acceptance output therefore
cannot enter competition transfer. The builder SHA-256 is
`c9623a50442ce5f26e3e4a9cc5be60ba597940329d7a327eae5e00e578ee8fd3`;
its test SHA-256 is
`3df845747b554a7041f96c90cb7f374c605731a07e41d6bae76e10fbcfb9e37c`.
The metadata, runtime version, model, data splits, seeds, optimization recipe,
gates, GPU contract, and absence of a submission command are unchanged.
