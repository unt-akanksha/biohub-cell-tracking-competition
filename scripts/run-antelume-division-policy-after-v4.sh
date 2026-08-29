#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/biohub
data_root=/home/ubuntu/biohub-data/biohub-zebrahub-contextual-shards-v1
output_root=/home/ubuntu/biohub-pretrain-output-20260829
result_root=/home/ubuntu/biohub-results/division-policy-v1
log_root=/home/ubuntu/biohub-logs
python_bin=/home/ubuntu/venv/bin/python
kaggle_cli=/home/ubuntu/venv/bin/kaggle
kernel_ref=indarkarhana/biohub-zebrahub-multiscale-pretrain-v1

exec 9>/home/ubuntu/biohub-division-policy.lock
if ! flock -n 9; then
  echo "Another Antelume division-policy controller already holds the lock" >&2
  exit 2
fi
if [[ "$(readlink -f "$workspace")" != /home/ubuntu/biohub ]]; then
  echo "Refusing unexpected Biohub workspace" >&2
  exit 2
fi
for required in \
  "$workspace/research/temporal_contrastive/calibrate_division_recovery_policy.py" \
  "$workspace/research/temporal_contrastive/verify_multiscale_pretraining_output.py" \
  "$data_root/DATASET_MANIFEST.json" \
  "$python_bin" \
  "$kaggle_cli"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required Antelume policy input: $required" >&2
    exit 2
  fi
done
mkdir -p "$result_root" "$log_root"

poll=0
while true; do
  poll=$((poll + 1))
  status=$($kaggle_cli kernels status "$kernel_ref" 2>&1)
  printf '%s poll=%d %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$poll" "$status" \
    >> "$log_root/antelume-division-policy-controller.log"
  if [[ "$status" == *COMPLETE* ]]; then
    break
  fi
  if [[ "$status" == *ERROR* || "$status" == *CANCEL* ]]; then
    echo "V4 pretraining failed: $status" >&2
    exit 3
  fi
  sleep 300
done

if [[ -e "$output_root" ]]; then
  echo "Refusing to reuse V4 pretraining download root: $output_root" >&2
  exit 3
fi
mkdir "$output_root"
$kaggle_cli kernels output "$kernel_ref" -p "$output_root" --force

$python_bin \
  "$workspace/research/temporal_contrastive/verify_multiscale_pretraining_output.py" \
  --root "$output_root" \
  > "$result_root/pretraining-verification.json"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader \
  > "$result_root/gpu-before-calibration.txt"
$python_bin \
  "$workspace/research/temporal_contrastive/calibrate_division_recovery_policy.py" \
  --data-root "$data_root" \
  --pretraining-root "$output_root" \
  --output "$result_root/policy.json" \
  --patch-batch-size 32 \
  > "$result_root/calibration.stdout.log" \
  2> "$result_root/calibration.stderr.log"

sha256sum \
  "$result_root/pretraining-verification.json" \
  "$result_root/policy.json" \
  > "$result_root/SHA256SUMS"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$result_root/.complete"
echo "Antelume external division-policy calibration complete"
