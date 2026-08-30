#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/biohub
data_root=/home/ubuntu/biohub-real-division-data/biohub_real_division_patches_v1
sweep_root=/home/ubuntu/biohub-results/competition-real-division-seed-sweep-v1
output_root=/home/ubuntu/biohub-results/competition-real-division-seed-ensemble-v1
python_bin=/home/ubuntu/venv/bin/python
terminal="$sweep_root/seed_sweep_terminal.json"
poll_seconds=60
maximum_polls=720

test ! -e "$output_root"
for required in \
  "$workspace/research/temporal_contrastive/evaluate_real_division_seed_ensemble.py" \
  "$python_bin"; do
  test -e "$required"
done

poll=0
while ! test -f "$terminal"; do
  poll=$((poll + 1))
  if test "$poll" -ge "$maximum_polls"; then
    echo "Timed out waiting for the overnight seed sweep" >&2
    exit 3
  fi
  sleep "$poll_seconds"
done

mkdir "$output_root"
set +e
CUDA_VISIBLE_DEVICES=0 "$python_bin" \
  "$workspace/research/temporal_contrastive/evaluate_real_division_seed_ensemble.py" \
  --data-root "$data_root" \
  --sweep-root "$sweep_root" \
  --output "$output_root/seed_ensemble_terminal.json" \
  --batch-size 64 \
  >"$output_root/evaluation.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" >"$output_root/evaluation.exit-code"
test -f "$output_root/seed_ensemble_terminal.json"
sha256sum \
  "$output_root/seed_ensemble_terminal.json" \
  "$output_root/evaluation.log" \
  >"$output_root/SHA256SUMS"
exit "$status"
