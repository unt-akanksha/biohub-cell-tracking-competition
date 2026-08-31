#!/usr/bin/env bash
set -euo pipefail

run_root=/home/ubuntu/biohub-temporal-localizer-v2
workspace="$run_root/workspace"
data_archive="$run_root/input/biohub-synthetic256-temporal-localizer-v1.tar.gz"
data_parent="$run_root/data/synthetic256"
real_archive="$run_root/input/biohub-real-localization-shards-v1.tar.gz"
real_root="$run_root/data/real-replay"
development_archive="$run_root/input/biohub-temporal-localizer-development-v1.tar.gz"
development_root="$run_root/data/development"
output_root="$run_root/results/synthetic256-real-replay-temporal-node-localizer-v2"
result_archive="$run_root/biohub-synthetic256-real-replay-temporal-node-localizer-v2-results.tar.gz"
archive_sha256=__SYNTHETIC256_ARCHIVE_SHA256__
real_archive_sha256=__REAL_LOCALIZATION_ARCHIVE_SHA256__
real_manifest_sha256=__REAL_LOCALIZATION_MANIFEST_SHA256__
development_archive_sha256=__DEVELOPMENT_ARCHIVE_SHA256__
workspace_archive_sha256=__WORKSPACE_ARCHIVE_SHA256__
seeds=(41021 41029 41039 41047)

exec 9>"$run_root/run.lock"
if ! flock -n 9; then
  echo "Another Antelume temporal localizer run holds the lock" >&2
  exit 2
fi
echo "$archive_sha256  $data_archive" | sha256sum -c -
echo "$real_archive_sha256  $real_archive" | sha256sum -c -
echo "$development_archive_sha256  $development_archive" | sha256sum -c -
echo "$workspace_archive_sha256  $run_root/input/workspace.tar.gz" | sha256sum -c -
test ! -e "$workspace"
test ! -e "$data_parent"
test ! -e "$real_root"
test ! -e "$development_root"
test ! -e "$output_root"
mkdir -p "$workspace" "$data_parent" "$real_root" "$development_root" "$output_root"
tar -xzf "$run_root/input/workspace.tar.gz" -C "$workspace"
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
  echo "No CUDA-enabled Antelume Python environment was found" >&2
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
test "$gpu_count" -eq 1
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi A10G
"$python_bin" - "$output_root/runtime-environment.json" <<'PY'
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch

payload = {
    "schema_version": 1,
    "execution_policy": "four independent members trained sequentially on one Antelume A10G",
    "python": platform.python_version(),
    "python_executable": sys.executable,
    "packages": {
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "torch", "tracksdata", "zarr")
    },
    "cuda_version": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "gpu_count": torch.cuda.device_count(),
    "gpu_names": [torch.cuda.get_device_name(0)],
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

failed=0
for gpu_index in 0 1 2 3; do
  member_root="$output_root/gpu_${gpu_index}"
  mkdir "$member_root"
  status=0
  CUDA_VISIBLE_DEVICES=0 "$python_bin" \
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
    --warmup-steps 1000 \
    --weight-decay 0.03 \
    --ema-decay 0.997 \
    --member-max-wall-seconds 11700 \
    --total-max-wall-seconds 12600 \
    --finalization-reserve-seconds 600 \
    --required-gpu-name A10G \
    >"$member_root/training.log" 2>&1 || status=$?
  printf '%s\n' "$status" >"$member_root/training.exit-code"
  if test "$status" -ne 0; then
    failed=1
  fi
done
printf '%s\n' "$failed" >"$output_root/aggregate.exit-code"

development_status=5
CUDA_VISIBLE_DEVICES=0 "$python_bin" \
  research/temporal_localization/score_real_development_probe.py \
  --probe-root "$development_root/probe" \
  --control-root "$development_root/control" \
  --truth-root "$development_root/truth" \
  --results-root "$output_root" \
  --output "$output_root/real-development-probe.json" \
  --batch-size 16 \
  >"$output_root/real-development-probe.log" 2>&1 || development_status=$?
printf '%s\n' "$development_status" >"$output_root/real-development-probe.exit-code"

cd "$run_root/results"
find synthetic256-real-replay-temporal-node-localizer-v2 -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum >"$output_root/SHA256SUMS"
tar -czf "$result_archive" synthetic256-real-replay-temporal-node-localizer-v2
sha256sum "$result_archive" >"$result_archive.sha256"
printf '%s\n' "completed" >"$run_root/run.complete"
exit "$failed"
