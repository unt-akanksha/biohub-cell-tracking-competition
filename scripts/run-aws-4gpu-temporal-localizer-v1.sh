#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/biohub
data_archive=/home/ubuntu/biohub-synthetic256-temporal-localizer-v1.tar.gz
data_parent=/home/ubuntu/biohub-synthetic256-temporal-localizer-v1
real_archive=/home/ubuntu/biohub-real-localization-shards-v1.tar.gz
real_root=/home/ubuntu/biohub-real-localization-shards-v1
development_archive=/home/ubuntu/biohub-temporal-localizer-development-v1.tar.gz
development_root=/home/ubuntu/biohub-temporal-localizer-development-v1
output_root=/home/ubuntu/biohub-results/synthetic256-real-replay-temporal-node-localizer-v2
result_archive=/home/ubuntu/biohub-synthetic256-real-replay-temporal-node-localizer-v2-results.tar.gz
archive_sha256=__SYNTHETIC256_ARCHIVE_SHA256__
real_archive_sha256=__REAL_LOCALIZATION_ARCHIVE_SHA256__
real_manifest_sha256=__REAL_LOCALIZATION_MANIFEST_SHA256__
development_archive_sha256=__DEVELOPMENT_ARCHIVE_SHA256__
seeds=(41021 41029 41039 41047)

exec 9>/home/ubuntu/biohub-temporal-localizer-v1.lock
if ! flock -n 9; then
  echo "Another temporal localizer run holds the lock" >&2
  exit 2
fi
echo "$archive_sha256  $data_archive" | sha256sum -c -
echo "$real_archive_sha256  $real_archive" | sha256sum -c -
echo "$development_archive_sha256  $development_archive" | sha256sum -c -
for required in \
  "$workspace/research/synthetic_pretrain/data.py" \
  "$workspace/research/temporal_contrastive/patch_model.py" \
  "$workspace/research/temporal_localization/consensus.py" \
  "$workspace/research/temporal_localization/inference.py" \
  "$workspace/research/temporal_localization/model.py" \
  "$workspace/research/temporal_localization/score_real_development_probe.py" \
  "$workspace/research/temporal_localization/train_synthetic_localizer.py"; do
  test -f "$required"
done
test ! -e "$data_parent"
test ! -e "$real_root"
test ! -e "$development_root"
test ! -e "$output_root"
mkdir -p "$data_parent" "$real_root" "$development_root" "$output_root" /home/ubuntu/biohub-results
tar -xzf "$data_archive" -C "$data_parent"
tar -xzf "$real_archive" -C "$real_root"
tar -xzf "$development_archive" -C "$development_root"

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
if ! "$python_bin" -c 'import scipy, tracksdata, zarr' >/dev/null 2>&1; then
  pip_scope=()
  if "$python_bin" -c 'import site; assert site.ENABLE_USER_SITE' >/dev/null 2>&1; then
    pip_scope=(--user)
  fi
  "$python_bin" -m pip install --disable-pip-version-check --no-input \
    "${pip_scope[@]}" 'zarr<4' scipy tracksdata
fi
"$python_bin" -c 'import scipy, tracksdata, zarr'
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
test "$gpu_count" -eq 4
nvidia-smi --query-gpu=name --format=csv,noheader | grep -vi A10G && exit 4 || true
"$python_bin" - "$output_root/runtime-environment.json" <<'PY'
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch

payload = {
    "schema_version": 1,
    "python": platform.python_version(),
    "python_executable": sys.executable,
    "packages": {
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "torch", "tracksdata", "zarr")
    },
    "cuda_version": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "gpu_count": torch.cuda.device_count(),
    "gpu_names": [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ],
}
Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
cd "$workspace"
"$python_bin" -m py_compile \
  research/synthetic_pretrain/data.py \
  research/temporal_contrastive/patch_model.py \
  research/temporal_localization/consensus.py \
  research/temporal_localization/inference.py \
  research/temporal_localization/model.py \
  research/temporal_localization/score_real_development_probe.py \
  research/temporal_localization/train_synthetic_localizer.py

pids=()
for gpu_index in 0 1 2 3; do
  member_root="$output_root/gpu_${gpu_index}"
  mkdir "$member_root"
  CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" \
    research/temporal_localization/train_synthetic_localizer.py \
    --synthetic-root "$data_parent" \
    --real-shard-root "$real_root" \
    --real-shard-manifest-sha256 "$real_manifest_sha256" \
    --output-dir "$member_root" \
    --seeds "${seeds[$gpu_index]}" \
    --steps 40000 \
    --batch-size 16 \
    --validation-examples 1024 \
    --audit-examples 1024 \
    --division-validation-examples 512 \
    --division-audit-examples 512 \
    --real-validation-examples 512 \
    --real-audit-examples 512 \
    --real-division-validation-examples 256 \
    --real-division-audit-examples 256 \
    --real-replay-probability 0.25 \
    --division-critical-per-batch 4 \
    --validation-batch-size 24 \
    --validation-every 2000 \
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
development_status=5
set +e
CUDA_VISIBLE_DEVICES=0,1,2,3 "$python_bin" \
  research/temporal_localization/score_real_development_probe.py \
  --probe-root "$development_root/probe" \
  --control-root "$development_root/control" \
  --truth-root "$development_root/truth" \
  --results-root "$output_root" \
  --output "$output_root/real-development-probe.json" \
  --batch-size 16 \
  >"$output_root/real-development-probe.log" 2>&1
development_status=$?
set -e
printf '%s\n' "$development_status" >"$output_root/real-development-probe.exit-code"
cd /home/ubuntu/biohub-results
find synthetic256-real-replay-temporal-node-localizer-v2 -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum >"$output_root/SHA256SUMS"
tar -czf "$result_archive" -C /home/ubuntu/biohub-results \
  synthetic256-real-replay-temporal-node-localizer-v2
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
