# Temporal pair-fusion cloud handoff v2

Date: 2026-08-27

Status: implementation verified and ready as the predeclared fallback if
`temporal-patch-dual-fold-v1` fails before the processed gate. Do not launch v1
and v2 speculatively in parallel. This runbook creates evidence and a local
candidate only; it contains no Kaggle submission command.

## Bound runtime

- Archive: `.biohub/staging/biohub-temporal-pair-fusion-runtime-v2-heavy-temporal3-candidatepair-ema-t4x2-controlsource-coherent-20260827.zip`
- Archive bytes: `116550`
- Archive SHA-256: `87c98fecec484fdbbeb2f2aa334793439e6a39f750ef163b12c6d923106edcb8`
- Extracted manifest SHA-256: `2266fbd7b64b88a786aa7951292caf2fa11579e4e1febb3adfa6b5f583ae217d`
- Manifest-bound source files: `29`
- Required visible GPUs for every GPU stage: exactly `2`
- Final inference hard stop: `36000` seconds
- Kaggle notebook finalization reserve: at least `7200` seconds
- Internet during execution: disabled/not required
- Submission command included: `false`

The host needs the extracted runtime, Biohub competition directory, corrected
synthetic sequences, and the coherent terminal output from
`trackastra-dual-fold-synthetic-v1`.

## Stage 0: integrity

```bash
export BIOHUB_RUNTIME=/workspace/biohub-temporal-pair-fusion-runtime-v2
export BIOHUB_COMPETITION=/workspace/biohub-cell-tracking-during-development
export BIOHUB_SYNTHETIC=/workspace/biohub-physical-synthetic-sequences-v1
export BIOHUB_TRACKASTRA=/workspace/trackastra-dual-fold-synthetic-v1
export BIOHUB_RUN=/workspace/runs/temporal-patch-pair-fusion-v2

python "$BIOHUB_RUNTIME/verify_runtime.py" \
  --root "$BIOHUB_RUNTIME" \
  --require-gpus

python "$BIOHUB_RUNTIME/verify_trackastra_output.py" \
  --root "$BIOHUB_TRACKASTRA" \
  --allow-pretrained-control
```

Stop unless the first command verifies 29 files, the bound manifest, exactly
two GPUs, and no submission command. Stop unless the second verifies one
coherent source policy, byte-identical worker/aggregate evidence, both model
hashes, and no competition artifact.

## Stage 1: reciprocal pair-fusion training

```bash
python "$BIOHUB_RUNTIME/train_dual_fold_pair_fusion.py" \
  --orchestrate \
  --competition-dir "$BIOHUB_COMPETITION" \
  --synthetic-root "$BIOHUB_SYNTHETIC" \
  --output-dir "$BIOHUB_RUN/appearance" \
  --steps 30000 \
  --base-channels 64 \
  --embedding-channels 256 \
  --pair-chunk-size 4096 \
  --embedding-loss-weight 0.25 \
  --ema-decay 0.997 \
  --real-train-movies 96 \
  --real-validation-movies 12 \
  --real-calibration-movies 12 \
  --max-wall-seconds 36000 \
  --orchestrator-hard-stop-seconds 37800
```

Stop unless `appearance/training_terminal.json` reports family
`temporal_pair_fusion_v2`, 20,869,325 parameters per fold, `gpu_count: 2`,
both best steps above zero, exact pair widths `1029 -> 1024 -> 512 -> 128 ->
1`, EMA checkpoints, the fixed 0.25 embedding auxiliary loss, and no public
prediction, leaderboard, or submission use.

## Stage 2: reserved-movie calibration

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

The fixed grid retains exact zero appearance/division control. Stop unless the
terminal run is `temporal-patch-pair-fusion-blend-v2`, both folds clear the
`.001` pooled-gain and `-.002` per-movie gates, and every reciprocal model and
Trackastra source hash remains bound.

## Stage 3: one-shot processed materialization

Run this exactly once, only after Stage 2 freezes the selected weights:

```bash
export BIOHUB_PROCESSED_CONTROL=/workspace/processed_validation/processed_validation.csv
export BIOHUB_RAW_GRAPHS=/workspace/processed_validation/raw_graphs

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

The result must identify `trackastra_pair_fusion_blend`, use two GPUs and
whole-movie shards, preserve every node, change at least one edge, read no
ground truth, and emit `authorized_for_submission: false`.

## Stage 4: pinned CPU exact gate

Transfer the processed directory back to this workspace and run from the
repository root:

```powershell
.biohub\evaluation-venv\Scripts\python.exe -m `
  research.trackastra_graph.score_dual_fold_processed_candidate `
  --control-csv .biohub\cache\kernel-outputs\hoct-multibackbone-probe-v1\processed_validation\processed_validation.csv `
  --candidate-csv .biohub\cloud-results\temporal-patch-pair-fusion-v2\processed\processed_candidate.csv `
  --truth-dir .biohub\cache\competition-truth\public-node-acceptance-v1 `
  --scorer-lock config\official-scorer.lock.json `
  --organizer-checkout .biohub\vendor\kaggle-cell-tracking-competition `
  --tracksdata-checkout .biohub\vendor\tracksdata `
  --materialization-result .biohub\cloud-results\temporal-patch-pair-fusion-v2\processed\materialization_result.json `
  --output reports\experiments\temporal-patch-pair-fusion-v2-exact-acceptance.json
```

The output path must be new. Accept only `status: accepted` with an exact
pooled gain, identical node recall, every movie above the `-.002` regression
floor, different edge sets, and exact v2 architecture evidence.

## Stage 5: local candidate construction after acceptance

The accepted evidence may drive final inference only on exactly two GPUs with
whole-movie sharding and the 36,000-second hard stop. The builder creates
`submission.csv` locally but cannot upload it:

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
  --acceptance-evidence /workspace/evidence/temporal-patch-pair-fusion-v2-exact-acceptance.json \
  --trackastra-dir "$BIOHUB_RUNTIME/trackastra_source" \
  --output-dir "$BIOHUB_RUN/final" \
  --hard-stop-seconds 36000
```

Uploading the resulting CSV remains a separate user-authorized action. At
every two-GPU deadline, both workers terminate together, receive the fixed
15-second grace period, and any survivor is force-killed and reaped.
