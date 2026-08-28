# Temporal contextual v3 cloud handoff

Date: 2026-08-27

Status: downstream two-GPU workflow staged and tested. It must run sequentially
after the active ZebraHub pretraining and frozen third-embryo acceptance pass.
No command below uploads a competition submission.

The preferred transfer entry point is now the fail-closed host runner
`scripts/run-temporal-contextual-cloud-transfer.py`. It verifies the exact
runtime manifest, requires exactly two visible GPUs, hash-binds the repository
acceptance-verifier closure, independently verifies the complete accepted-v3
chain, launches the frozen transfer recipe, strict-loads both output
checkpoints, and writes `cloud_launcher_terminal.json`. It cannot authorize or
create a competition submission.

The repository closure now binds all `27` Python files under
`research/temporal_contrastive`, including the isolated future-division
prototype, at tree SHA-256
`9c1db520a098d70c746de2afbb4e1037796731cee9366c5aa0bd4d5f16d2b9c2`.

## Immutable assets

- ZebraHub pretraining run: `zebrahub-contextual-pretrain-v1`
- Frozen ZebraHub acceptance run:
  `zebrahub-contextual-acceptance-evaluation-v1`
- Transfer runtime dataset:
  `indarkarhana/biohub-temporal-contextual-transfer-runtime-v1`, version `4`
- Runtime manifest SHA-256:
  `cbe5fe27639155746c95a98d91702d5fbe595172b058e0e9db330374ecfff25d`
- Parameters per reciprocal fold: `20,747,761`
- Required visible GPUs for every GPU stage: exactly `2`
- Internet during all GPU stages: disabled
- Kaggle quota reserve: at least `8` hours

Set host paths without changing the recipes:

```bash
export BIOHUB_RUNTIME=/workspace/biohub-temporal-contextual-transfer-runtime-v1
export BIOHUB_COMPETITION=/workspace/biohub-cell-tracking-during-development
export BIOHUB_SYNTHETIC=/workspace/biohub-physical-synthetic-sequences-v1
export BIOHUB_PRETRAIN=/workspace/zebrahub-contextual-pretrain-v1
export BIOHUB_ZSNS_ACCEPT=/workspace/zebrahub-contextual-acceptance-evaluation-v1
export BIOHUB_TRACKASTRA=/workspace/trackastra-dual-fold-synthetic-v1
export BIOHUB_RUN=/workspace/runs/temporal-contextual-pair-fusion-v3
```

## 0. Integrity and prerequisites

```bash
python "$BIOHUB_RUNTIME/verify_runtime.py" \
  --root "$BIOHUB_RUNTIME" \
  --require-gpus

python "$BIOHUB_RUNTIME/verify_trackastra_output.py" \
  --root "$BIOHUB_TRACKASTRA" \
  --allow-pretrained-control
```

Stop unless both commands verify their exact manifests/checkpoints, exactly two
GPUs, coherent reciprocal evidence, and no competition artifact. The transfer
kernel separately requires both pretraining folds to pass fixed ZSNS005
selection/audit gates and the frozen ZSNS001 family gate.

## 1. Competition transfer

Preferred automated command from an exact copy of this repository:

```bash
python scripts/run-temporal-contextual-cloud-transfer.py \
  --repository-root /workspace/Biohub \
  --runtime-root "$BIOHUB_RUNTIME" \
  --competition-dir "$BIOHUB_COMPETITION" \
  --synthetic-root "$BIOHUB_SYNTHETIC" \
  --pretraining-root "$BIOHUB_PRETRAIN" \
  --acceptance-root "$BIOHUB_ZSNS_ACCEPT" \
  --output-dir "$BIOHUB_RUN/appearance"
```

Runner SHA-256:
`758fd46dc59ba8f0ef11f10b2b2167ce42ac76d62ab1e0b191be8be5ebee4ee8`.
Its test SHA-256 is
`ba7215a51c369bb836b43dc06a90ade6e6517f41571e1d3d661f91b70e406b17`;
all three focused fail-closed tests pass locally without GPU use, and the
20-test combined cloud/context/division/acceptance regression set passes.

Use the staged private notebook
`indarkarhana/biohub-temporal-contextual-transfer-v3`, or run its exact command:

```bash
python "$BIOHUB_RUNTIME/train_dual_fold_contextual_pair_fusion.py" \
  --orchestrate \
  --competition-dir "$BIOHUB_COMPETITION" \
  --synthetic-root "$BIOHUB_SYNTHETIC" \
  --initial-model-root "$BIOHUB_PRETRAIN" \
  --output-dir "$BIOHUB_RUN/appearance" \
  --steps 20000 \
  --seed 41027 \
  --base-channels 64 \
  --embedding-channels 256 \
  --ema-decay 0.997 \
  --real-replay-probability 0.60 \
  --learning-rate 0.00005 \
  --minimum-learning-rate 0.0000005 \
  --minimum-real-composite-gain 0.005 \
  --maximum-synthetic-metric-regression 0.01 \
  --validation-every 1000 \
  --real-train-movies 96 \
  --real-validation-movies 12 \
  --real-calibration-movies 12 \
  --max-wall-seconds 36000 \
  --orchestrator-hard-stop-seconds 37800 \
  --finalization-reserve-seconds 1200
```

The effective reciprocal seeds are `45427` and `47627`. Stop unless both folds
report `finetuning_gate_passed: true` and the aggregate reports
`both_folds_improved: true`. Independently strict-load and recompute the result:

```bash
python "$BIOHUB_RUNTIME/verify_appearance_output.py" \
  --root "$BIOHUB_RUN/appearance" \
  --expected-family temporal_contextual_pair_fusion_v3 \
  --strict-checkpoint
```

Each fold needs real composite gain at least `0.005`, strictly positive real
top-1/MRR gains, nonnegative real division-top-2 gain, and no synthetic metric
regression worse than `0.01`.

## 2. Clean reserved-movie calibration

Preferred automated cloud command after the transfer runner completes:

```bash
python scripts/run-temporal-contextual-cloud-calibration.py \
  --runtime-root "$BIOHUB_RUNTIME" \
  --competition-dir "$BIOHUB_COMPETITION" \
  --trackastra-root "$BIOHUB_TRACKASTRA" \
  --appearance-root "$BIOHUB_RUN/appearance" \
  --output-dir "$BIOHUB_RUN/calibration"
```

The runner re-verifies the runtime, both strict appearance checkpoints, the
Trackastra control, and the transfer launcher's exact source-tree binding. It
requires exactly two visible GPUs and independently enforces each fold's
`0.001` pooled gain and `-0.002` worst-movie floor before it writes
`cloud_calibration_launcher_terminal.json`; it has no processed-acceptance or
submission command. Runner SHA-256:
`c4e7b578378005a5aa5fe0473765c12df45c6a03a1ce5f0b0e7c2129004a6edc`.
Its test SHA-256 is
`e0ae687cb334285069b6238bf2d629c3ed91398da1b2ab2089e287e8edbf480b`;
all four focused tests and the 24-test combined regression set pass.

```bash
python "$BIOHUB_RUNTIME/calibrate_dual_fold_blend.py" \
  --orchestrate \
  --appearance-output-root "$BIOHUB_RUN/appearance" \
  --trackastra-output-root "$BIOHUB_TRACKASTRA" \
  --competition-dir "$BIOHUB_COMPETITION" \
  --trackastra-dir "$BIOHUB_RUNTIME/trackastra_source" \
  --output-dir "$BIOHUB_RUN/calibration" \
  --max-tokens 512 \
  --candidate-radius 80 \
  --node-batch-size 64 \
  --max-wall-seconds 18000 \
  --orchestrator-hard-stop-seconds 19800
```

Stop unless both folds improve over the exact zero-weight Trackastra control.
The gate requires pooled composite gain at least `0.001` and worst-movie delta
at least `-0.002` across 12 reserved movies per embryo. Processed acceptance
must remain unopened.

## 3. One-shot processed materialization

Run exactly once after calibration freezes all weights:

```bash
python scripts/run-temporal-contextual-cloud-processed.py \
  --runtime-root "$BIOHUB_RUNTIME" \
  --competition-dir "$BIOHUB_COMPETITION" \
  --trackastra-root "$BIOHUB_TRACKASTRA" \
  --appearance-root "$BIOHUB_RUN/appearance" \
  --calibration-root "$BIOHUB_RUN/calibration" \
  --processed-control-csv "$BIOHUB_PROCESSED_CONTROL" \
  --raw-graph-root "$BIOHUB_PROCESSED_RAW" \
  --output-dir "$BIOHUB_RUN/processed"
```

This preferred cloud runner verifies the full transfer-to-calibration hash
chain, both frozen input artifact hashes, exactly two visible GPUs, all four
predeclared movies, changed edges, and the absence of ground-truth access,
selection, exact scoring, and submission. It only authorizes the pinned local
CPU gate. Runner SHA-256:
`83682e930b09aeaf9531d54ce1c30bfbb3bb7a78dbd1d5ef335b25b9d5c2f23f`.
Its test SHA-256 is
`21051992d4fae5ef37bf3dc68f5506e1bc511d3adfae1fb85fb4eae00cdaa54e`;
all five focused tests and the 29-test combined regression set pass.

```bash
export BIOHUB_PROCESSED_CONTROL=/workspace/processed_validation/processed_validation.csv
export BIOHUB_PROCESSED_RAW=/workspace/processed_validation/raw_graphs

python "$BIOHUB_RUNTIME/dual_fold_appearance_processed_acceptance.py" \
  --orchestrate \
  --processed-control-csv "$BIOHUB_PROCESSED_CONTROL" \
  --raw-graph-root "$BIOHUB_PROCESSED_RAW" \
  --competition-dir "$BIOHUB_COMPETITION" \
  --trackastra-output-root "$BIOHUB_TRACKASTRA" \
  --appearance-output-root "$BIOHUB_RUN/appearance" \
  --calibration-terminal "$BIOHUB_RUN/calibration/calibration_terminal.json" \
  --trackastra-dir "$BIOHUB_RUNTIME/trackastra_source" \
  --output-dir "$BIOHUB_RUN/processed" \
  --hard-stop-seconds 19800
```

The control CSV must hash to
`6613545843ebd743dac66b5a0598702faaa5b3c0870566e55fa60250a009615b`
and the raw graph tree to
`559332597da65f161f1b0b116e10fc86c7ff35eb31fe48937e080889b909a43e`.
Stop unless two whole-movie GPU shards finish, the candidate changes edges and
has a different CSV hash, ground truth remains unread, and the result is still
unauthorized for submission.

## 4. Pinned exact CPU gate

Transfer the processed output back to this workspace and run from the repository
root, using a new output path:

```powershell
.biohub\evaluation-venv\Scripts\python.exe `
  scripts\run-temporal-contextual-exact-acceptance.py `
  --processed-root .biohub\cloud-results\temporal-contextual-pair-fusion-v3\processed `
  --control-csv .biohub\cache\kernel-outputs\hoct-multibackbone-probe-v1\processed_validation\processed_validation.csv `
  --truth-dir .biohub\cache\competition-truth\public-node-acceptance-v1 `
  --scorer-lock config\official-scorer.lock.json `
  --organizer-checkout .biohub\vendor\kaggle-cell-tracking-competition `
  --tracksdata-checkout .biohub\vendor\tracksdata `
  --output reports\experiments\temporal-contextual-pair-fusion-v3-exact-acceptance.json
```

The preferred runner binds the processed cloud launcher to the materialization
and CSV hashes before invoking the pinned scorer, then requires all five exact
checks. Runner SHA-256:
`0f77b8dff7706403d5f421d28b75bcf24bd234ec80f0b06e6fa2e4e59a0debb5`.
Its test SHA-256 is
`4f806c9ef7c5ccb50f2aacdd0740d4a5dde1c237601dea1b0deb62471669996b`.
The exact scorer was also repaired to consume the centralized contextual-v3
architecture contract (including the exact `20,747,761` parameter count)
instead of rejecting that new family. The 16 host-runner tests and 20 pinned
evaluation-environment scorer/model/materializer tests pass.

The equivalent direct scorer command is:

```powershell
.biohub\evaluation-venv\Scripts\python.exe -m `
  research.trackastra_graph.score_dual_fold_processed_candidate `
  --control-csv .biohub\cache\kernel-outputs\hoct-multibackbone-probe-v1\processed_validation\processed_validation.csv `
  --candidate-csv .biohub\cloud-results\temporal-contextual-pair-fusion-v3\processed\processed_candidate.csv `
  --truth-dir .biohub\cache\competition-truth\public-node-acceptance-v1 `
  --scorer-lock config\official-scorer.lock.json `
  --organizer-checkout .biohub\vendor\kaggle-cell-tracking-competition `
  --tracksdata-checkout .biohub\vendor\tracksdata `
  --materialization-result .biohub\cloud-results\temporal-contextual-pair-fusion-v3\processed\materialization_result.json `
  --output reports\experiments\temporal-contextual-pair-fusion-v3-exact-acceptance.json
```

Accept only `status: accepted`, positive pooled exact gain, identical node
recall, every movie above the `-0.002` regression floor, changed edge sets, and
exact contextual architecture/checkpoint evidence. Only then create the private
acceptance-evidence dataset.

## 5. Two-GPU whole-movie final candidate

Before the Kaggle code notebook is published, stage the cloud checkpoints and
accepted exact evidence as separate private datasets:

```bash
python scripts/stage-temporal-contextual-kaggle-artifacts.py \
  --appearance-root "$BIOHUB_RUN/appearance" \
  --acceptance-evidence /workspace/evidence/temporal-contextual-pair-fusion-v3-exact-acceptance.json \
  --staging-root /workspace/kaggle-artifacts
```

The stager refuses any non-accepted evidence or checkpoint-hash mismatch and
performs no Kaggle write itself. The final notebook now consumes
`indarkarhana/biohub-temporal-contextual-transfer-output-v3` as a private
dataset instead of incorrectly expecting the cloud transfer to exist as a
Kaggle kernel output. Stager SHA-256:
`2659eec86a02e0882fe40053107a53e75db9e2b5ee10f22584a0c1804c7199d9`;
test SHA-256:
`ceeb17f113ce13fd0929f2efe3e7f5a626b9cf06fea2662308a8b21c8ce9f4b3`.

Benchmark and verify the exact final recipe on the two-GPU cloud host first:

```bash
python scripts/run-temporal-contextual-cloud-candidate.py \
  --runtime-root "$BIOHUB_RUNTIME" \
  --competition-dir "$BIOHUB_COMPETITION" \
  --trackastra-root "$BIOHUB_TRACKASTRA" \
  --appearance-root "$BIOHUB_RUN/appearance" \
  --acceptance-evidence /workspace/evidence/temporal-contextual-pair-fusion-v3-exact-acceptance.json \
  --base-submission /workspace/test_control/submission.csv \
  --base-graph-root /workspace/test_control/raw_graphs \
  --output-dir "$BIOHUB_RUN/final"
```

The runner re-verifies exact acceptance, both checkpoint families, the frozen
base CSV and raw-graph tree, exactly two GPUs, whole-movie coverage, unchanged
nodes, changed edges, and the final CSV hash. Only then does it write
`ready_for_submission_upload: true`; it contains no upload command. Runner
SHA-256:
`d694543cc26073126fd5f9fcc01d0a5da8f814a83765b0deec0f3d2993053145`;
test SHA-256:
`42dc744914cd805d455f0bab60de5cbc8027c6659e88d1a83466c72710b98564`.
The combined final-inference and artifact-transport suite passes 19 tests.

Use the staged notebook
`indarkarhana/biohub-temporal-contextual-submission-candidate-v3`, or run:

```bash
python "$BIOHUB_RUNTIME/dual_fold_appearance_submission.py" \
  --orchestrate \
  --base-submission /workspace/test_control/submission.csv \
  --base-graph-root /workspace/test_control/raw_graphs \
  --image-root "$BIOHUB_COMPETITION/test" \
  --trackastra-44b6-dir "$BIOHUB_TRACKASTRA/target_44b6" \
  --trackastra-6bba-dir "$BIOHUB_TRACKASTRA/target_6bba" \
  --appearance-44b6-model "$BIOHUB_RUN/appearance/target_44b6/appearance_model.pt" \
  --appearance-6bba-model "$BIOHUB_RUN/appearance/target_6bba/appearance_model.pt" \
  --acceptance-evidence /workspace/evidence/temporal-contextual-pair-fusion-v3-exact-acceptance.json \
  --trackastra-dir "$BIOHUB_RUNTIME/trackastra_source" \
  --output-dir "$BIOHUB_RUN/final" \
  --hard-stop-seconds 36000
```

Exactly two GPUs balance complete movies by contextual pair and encoder cost.
The `36,000`-second inference ceiling leaves `7,200` seconds for notebook setup
and finalization. The candidate must preserve every node, cover every movie
once, change edges, and differ from the base hash. The resulting CSV is local
evidence until exact acceptance passes. The user supplied prospective upload
authorization on 2026-08-27: after every frozen gate passes, upload the accepted
hash-bound CSV without another prompt. The upload step itself does not require
a GPU.

For this code competition, submit the exact completed Kaggle notebook version,
not the cloud CSV directly. The fail-closed boundary first verifies and
downloads the current private notebook output:

```powershell
python scripts\submit-temporal-contextual-kernel.py `
  --kernel-dir kaggle\biohub-temporal-contextual-submission-candidate-v3 `
  --kernel-version <completed-version> `
  --download-dir .biohub\candidate-downloads\contextual-v3-<completed-version> `
  --receipt .biohub\submission-receipts\contextual-v3-<completed-version>.json
```

Only after that dry verification succeeds, invoke the identical command with
`--execute`. The wrapper rechecks the current owned kernel version and COMPLETE
status, exact private data sources, two-GPU/internet-off metadata, downloaded
candidate/report/launcher hashes, non-replica edge changes, the five-per-day
limit, and the narrowly scoped prospective authorization event. It then uses
Kaggle's code-kernel `--kernel/--version` submission path and writes a receipt.
Wrapper SHA-256:
`896a1e7719289dce310ec2f792787d99789211b627818f3bd8ba08061db8be3c`;
test SHA-256:
`328c6ea140465c9ba459e442fc7a4f956d994097594b3738133765c6859ba7af`.

After the two private datasets are published and the cloud benchmark is copied
under this workspace, generate the fresh guarded-launch preflight:

```powershell
python scripts\build-temporal-contextual-submission-candidate-preflight.py `
  --appearance-root .biohub\kaggle-artifacts\biohub-temporal-contextual-transfer-output-v3 `
  --acceptance-root .biohub\kaggle-artifacts\biohub-temporal-contextual-exact-acceptance-v3 `
  --cloud-candidate-root .biohub\cloud-results\temporal-contextual-pair-fusion-v3\final `
  --runtime-root .biohub\cache\dataset-redownloads\biohub-temporal-contextual-transfer-runtime-v1-version4
```

The preflight re-hashes every staged checkpoint, strict-loads the accepted
models, binds exact acceptance to the cloud candidate and final CSV, reruns the
focused boundary tests, and emits every check required for a 12-hour guarded
launch. Its cloud elapsed evidence can support a later measured-runtime
amendment without weakening the eight-hour reserve. Builder SHA-256:
`9cf66591ad4ec54d786e9b4bb5e7d9a777de4d0aced8fd8f4ee740a1d739248b`;
test SHA-256:
`1aca00925cf1e86c6c6cc292c7a0a02dcb4c51bf9db5c9c2ddc4342e3dc3d966`.
