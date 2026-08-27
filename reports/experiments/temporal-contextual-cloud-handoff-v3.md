# Temporal contextual v3 cloud handoff

Date: 2026-08-27

Status: downstream two-GPU workflow staged and tested. It must run sequentially
after the active ZebraHub pretraining and frozen third-embryo acceptance pass.
No command below uploads a competition submission.

## Immutable assets

- ZebraHub pretraining run: `zebrahub-contextual-pretrain-v1`
- Frozen ZebraHub acceptance run:
  `zebrahub-contextual-acceptance-evaluation-v1`
- Transfer runtime dataset:
  `indarkarhana/biohub-temporal-contextual-transfer-runtime-v1`, version `3`
- Runtime manifest SHA-256:
  `193478079a0d3f83c1307416c60ed5ad7c74a840fafc5e046ef7f30e2db4b3c1`
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
evidence only; uploading it requires separate explicit authorization.
