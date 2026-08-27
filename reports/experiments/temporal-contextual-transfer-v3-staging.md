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
  `7ca20f9097d14dba745a3eb0aededc561e691e904cb85274d91219a282c4ee61`
- Kernel metadata SHA-256:
  `0d2afe024afb1da581ba2b4860188b48c5617ceda3aa49ff55875ff30b3fb5c6`

The runtime version-4 re-download reproduced its exact manifest and all 36
content hashes after materializing the packaged Trackastra source directory.
Version 4 carries only the mixed-precision compatibility repair: autocast
contextual logits are differentiably promoted to the FP32 sparse output dtype
before index scatter. No recipe, split, seed, gate, or selection policy was
changed. No Kaggle GPU, leaderboard result, or competition submission was used
to stage this transfer lane.
