#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/biohub
data_archive=/home/ubuntu/biohub-synthetic16-temporal-localizer-v1.tar.gz
data_parent=/home/ubuntu/biohub-synthetic16-temporal-localizer-v1
output_root=/home/ubuntu/biohub-results/synthetic16-temporal-node-localizer-v1
result_archive=/home/ubuntu/biohub-synthetic16-temporal-node-localizer-v1-results.tar.gz
archive_sha256=__SYNTHETIC16_ARCHIVE_SHA256__
seeds=(41021 41029 41039 41047)

exec 9>/home/ubuntu/biohub-temporal-localizer-v1.lock
if ! flock -n 9; then
  echo "Another temporal localizer run holds the lock" >&2
  exit 2
fi
echo "$archive_sha256  $data_archive" | sha256sum -c -
for required in \
  "$workspace/research/synthetic_pretrain/data.py" \
  "$workspace/research/temporal_contrastive/patch_model.py" \
  "$workspace/research/temporal_localization/model.py" \
  "$workspace/research/temporal_localization/train_synthetic_localizer.py"; do
  test -f "$required"
done
test ! -e "$data_parent"
test ! -e "$output_root"
mkdir -p "$data_parent" "$output_root" /home/ubuntu/biohub-results
tar -xzf "$data_archive" -C "$data_parent"

python_bin=""
for candidate in \
  /home/ubuntu/venv/bin/python \
  /opt/conda/envs/pytorch/bin/python \
  /opt/pytorch/bin/python \
  /usr/bin/python3; do
  if test -x "$candidate" && "$candidate" -c 'import numpy, torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
    python_bin="$candidate"
    break
  fi
done
if test -z "$python_bin"; then
  echo "No CUDA-enabled AWS Python environment was found" >&2
  exit 3
fi
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
test "$gpu_count" -eq 4
nvidia-smi --query-gpu=name --format=csv,noheader | grep -vi A10G && exit 4 || true
cd "$workspace"
"$python_bin" -m py_compile \
  research/synthetic_pretrain/data.py \
  research/temporal_contrastive/patch_model.py \
  research/temporal_localization/model.py \
  research/temporal_localization/train_synthetic_localizer.py

pids=()
for gpu_index in 0 1 2 3; do
  member_root="$output_root/gpu_${gpu_index}"
  mkdir "$member_root"
  CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" \
    research/temporal_localization/train_synthetic_localizer.py \
    --synthetic-root "$data_parent" \
    --output-dir "$member_root" \
    --seeds "${seeds[$gpu_index]}" \
    --steps 20000 \
    --batch-size 16 \
    --validation-examples 1024 \
    --audit-examples 1024 \
    --validation-batch-size 24 \
    --validation-every 1000 \
    --log-every 100 \
    --learning-rate 2e-4 \
    --minimum-learning-rate 2e-6 \
    --weight-decay 0.03 \
    --ema-decay 0.997 \
    --member-max-wall-seconds 37800 \
    --total-max-wall-seconds 39600 \
    --finalization-reserve-seconds 900 \
    --required-gpu-name A10G \
    >"$member_root/training.log" 2>&1 &
  pids+=("$!")
done

failed=0
for gpu_index in 0 1 2 3; do
  status=0
  wait "${pids[$gpu_index]}" || status=$?
  printf '%s\n' "$status" >"$output_root/gpu_${gpu_index}/training.exit-code"
  if test "$status" -ne 0; then
    failed=1
  fi
done
printf '%s\n' "$failed" >"$output_root/aggregate.exit-code"
cd /home/ubuntu/biohub-results
find synthetic16-temporal-node-localizer-v1 -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum >"$output_root/SHA256SUMS"
tar -czf "$result_archive" -C /home/ubuntu/biohub-results \
  synthetic16-temporal-node-localizer-v1
sha256sum "$result_archive" >"$result_archive.sha256"

# Bound idle cost if the local harvest controller disappears overnight. The
# EBS volume remains recoverable because instance-initiated shutdown means stop.
for _poll in $(seq 1 15); do
  if test -f /home/ubuntu/biohub-temporal-localizer-v1-harvest-complete; then
    break
  fi
  sleep 120
done
sudo shutdown -h now
exit "$failed"
