# Temporal patch cloud handoff v1

Date: 2026-08-27

Status: ready after the active Kaggle reciprocal Trackastra run reaches a
terminal state. This runbook creates evidence and a candidate only; it contains
no Kaggle submission command.

## Bound runtime

- Archive: `.biohub/staging/biohub-temporal-patch-runtime-v1-heavy-temporal3-framecache-strictingest-ema-ensemble-daughterloss-20260827.zip`
- Archive bytes: `101048`
- Archive SHA-256: `16fa051b565971b5cd5b4b40f737501161bdd938665152a38cc0506a935f3325`
- Extracted manifest SHA-256: `cc749ed7c319bcb2bdee5d9987ea2984898f198c92a3fad81161f14e41ce6d10`
- Required visible GPUs: exactly 2
- Internet during model execution: not required
- Submission command included: false

The cloud machine must receive four inputs without renaming their internal
contents: the extracted runtime, the Biohub competition directory, the
corrected synthetic sequence dataset, and the terminal output from
`trackastra-dual-fold-synthetic-v1`.

## Stage 0: environment and integrity

Set these paths on the cloud host:

```bash
export BIOHUB_RUNTIME=/workspace/biohub-temporal-patch-runtime-v1
export BIOHUB_COMPETITION=/workspace/biohub-cell-tracking-during-development
export BIOHUB_SYNTHETIC=/workspace/biohub-physical-synthetic-sequences-v1
export BIOHUB_TRACKASTRA=/workspace/trackastra-dual-fold-synthetic-v1
export BIOHUB_RUN=/workspace/runs/temporal-patch-dual-fold-v1
```

Verify the archive outside the extracted directory, extract it into a new empty
directory, then run:

```bash
python "$BIOHUB_RUNTIME/verify_runtime.py" \
  --root "$BIOHUB_RUNTIME" \
  --require-gpus

python "$BIOHUB_RUNTIME/verify_trackastra_output.py" \
  --root "$BIOHUB_TRACKASTRA"
```

Stop if the runtime verifier does not report 26 files, two GPUs, the bound
manifest hash, and no submission command. Also stop unless the Trackastra
verifier reports `status: verified`, two hash-bound improved folds, corrected
native synthetic geometry, and no competition artifacts.

## Stage 1: reciprocal heavy appearance training

```bash
python "$BIOHUB_RUNTIME/train_dual_fold_patch.py" \
  --orchestrate \
  --competition-dir "$BIOHUB_COMPETITION" \
  --synthetic-root "$BIOHUB_SYNTHETIC" \
  --output-dir "$BIOHUB_RUN/appearance" \
  --steps 30000 \
  --base-channels 64 \
  --embedding-channels 256 \
  --ema-decay 0.997 \
  --real-train-movies 96 \
  --real-validation-movies 12 \
  --real-calibration-movies 12 \
  --max-wall-seconds 36000 \
  --orchestrator-hard-stop-seconds 37800
```

Stop unless `appearance/training_terminal.json` reports `gpu_count: 2`, both
folds trained beyond step zero, and no submission. Each fold must report
19,221,954 parameters, three input channels with temporal offsets `[-1, 0, 1]`,
EMA checkpoints, class-conditional division-prior correction, the all-positive
supervised-contrastive loss policy, successful-batch gradient accumulation,
and the global disjoint real-movie split policy.

## Stage 2: clean blend and reciprocal-ensemble calibration

```bash
python "$BIOHUB_RUNTIME/calibrate_dual_fold_blend.py" \
  --orchestrate \
  --appearance-output-root "$BIOHUB_RUN/appearance" \
  --trackastra-output-root "$BIOHUB_TRACKASTRA" \
  --competition-dir "$BIOHUB_COMPETITION" \
  --trackastra-dir "$BIOHUB_RUNTIME/trackastra_source" \
  --output-dir "$BIOHUB_RUN/calibration" \
  --max-wall-seconds 18000 \
  --orchestrator-hard-stop-seconds 19800
```

This stage compares `target_only` with `reciprocal_mean`, appearance weights
`0, .05, .10, .20, .35`, and division weights `0, .05, .10, .20`. The exact
zero/zero Trackastra control is mandatory. Stop unless both folds improve the
organizer-aligned clean score by at least `.001`, respect the `-.002`
per-movie floor, and report no processed-label or leaderboard use.

## Stage 3: one-shot processed materialization

This stage is allowed exactly once after Stage 2 freezes every weight. Set:

```bash
export BIOHUB_PROCESSED_CONTROL=/workspace/processed_validation/processed_validation.csv
export BIOHUB_RAW_GRAPHS=/workspace/processed_validation/raw_graphs
```

Then run:

```bash
python "$BIOHUB_RUNTIME/dual_fold_appearance_processed_acceptance.py" \
  --orchestrate \
  --processed-control-csv "$BIOHUB_PROCESSED_CONTROL" \
  --raw-graph-root "$BIOHUB_RAW_GRAPHS" \
  --competition-dir "$BIOHUB_COMPETITION" \
  --trackastra-output-root "$BIOHUB_TRACKASTRA" \
  --appearance-output-root "$BIOHUB_RUN/appearance" \
  --calibration-terminal "$BIOHUB_RUN/calibration/calibration_terminal.json" \
  --trackastra-dir "$BIOHUB_RUNTIME/trackastra_source" \
  --output-dir "$BIOHUB_RUN/processed"
```

The materializer may read the four images and frozen control nodes/edges, but
not their truth. It must change at least one edge, preserve every node exactly,
use two GPUs, and emit `authorized_for_submission: false`.

## Stage 4: pinned CPU exact gate

Transfer the materialization directory back to this workspace. Run
the following command from the repository root. The module form is required so
the pinned interpreter can import the repository packages correctly.

```powershell
.biohub\evaluation-venv\Scripts\python.exe -m `
  research.trackastra_graph.score_dual_fold_processed_candidate `
  --control-csv `
    .biohub\cache\kernel-outputs\hoct-multibackbone-probe-v1\processed_validation\processed_validation.csv `
  --candidate-csv `
    .biohub\cloud-results\temporal-patch-dual-fold-v1\processed\processed_candidate.csv `
  --truth-dir `
    .biohub\cache\competition-truth\public-node-acceptance-v1 `
  --scorer-lock config\official-scorer.lock.json `
  --organizer-checkout `
    .biohub\vendor\kaggle-cell-tracking-competition `
  --tracksdata-checkout .biohub\vendor\tracksdata `
  --materialization-result `
    .biohub\cloud-results\temporal-patch-dual-fold-v1\processed\materialization_result.json `
  --output `
    reports\experiments\temporal-patch-dual-fold-v1-exact-acceptance.json
```

The output path must not already exist because the gate refuses to overwrite
evidence. Accept only if `passed: true`: pooled score improves, node recall is
identical, every movie stays above the `-.002` regression floor, and the
candidate is not an edge replica.

If the exact gate accepts, the two-GPU whole-movie candidate builder may be run
to create a local `submission.csv`. Uploading that CSV to Kaggle remains a
separate action requiring explicit user authorization.
