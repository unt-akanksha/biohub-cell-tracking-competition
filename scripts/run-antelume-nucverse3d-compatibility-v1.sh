#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/biohub
run_root=/home/ubuntu/biohub-nucverse3d-compatibility-v1
input_root="$run_root/input"
package_root="$run_root/python-packages"
result_root="$run_root/results"
screen="$input_root/screen_pretrained.py"
onnx_model="$input_root/resunet_combined_scaled_1000.fixed.onnx"
onnx_wheel="$input_root/onnx-1.17.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
onnx2torch_wheel="$input_root/onnx2torch-1.5.15-py3-none-any.whl"
real_root=/home/ubuntu/biohub-temporal-localizer-v2/data/real-replay
manifest="$real_root/real_localization_shard_manifest.json"
python_bin=/home/ubuntu/venv/bin/python
v3_complete=/home/ubuntu/biohub-peak-rank-detector-v1/capacity-pu-v3/run.complete
result_archive=/home/ubuntu/biohub-nucverse3d-compatibility-v1-results.tar.gz

screen_sha256=9d696b327472d420852ca3122e89a9558f725ac2b2c1a13bed5e8e33ab9121a4
onnx_sha256=ca16e1b26d21ae522d68aba384ee7121f5ba2de28a122c291d1e1e627601e871
onnx_wheel_sha256=4f3fb5cc4e2898ac5312a7dc03a65133dd2abf9a5e520e69afb880a7251ec97a
onnx2torch_wheel_sha256=123258e0f147e07b259cf845c8113c5634b8260e52c5fb26b7d507226732d9e5
manifest_sha256=5ccd52b96a36db7dff5f4ae482bb7e7cdd28f49c024b047d55fa7c575504a697
minimum_free_bytes=1300000000

exec 9>"$run_root/run.lock"
if ! flock -n 9; then
  echo "Another NucVerse3D compatibility run holds the lock" >&2
  exit 3
fi
test ! -e "$result_root"
test ! -e "$result_archive"
test ! -e "$result_archive.sha256"
echo "$screen_sha256  $screen" | sha256sum -c -
echo "$onnx_sha256  $onnx_model" | sha256sum -c -
echo "$onnx_wheel_sha256  $onnx_wheel" | sha256sum -c -
echo "$onnx2torch_wheel_sha256  $onnx2torch_wheel" | sha256sum -c -
echo "$manifest_sha256  $manifest" | sha256sum -c -

# Preserve the sequential A10G evidence chain.  This independent public model
# receives GPU time only after every precommitted peak-rank member has finished.
while test ! -f "$v3_complete"; do sleep 30; done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -q '[0-9]'; do
  sleep 30
done
available_bytes="$(df --output=avail -B1 "$run_root" | tail -n 1 | tr -d ' ')"
if test "$available_bytes" -lt "$minimum_free_bytes"; then
  echo "Insufficient disk for NucVerse3D screen: $available_bytes bytes" >&2
  exit 9
fi
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi A10G

if test ! -f "$package_root/onnx/__init__.py" \
  || test ! -f "$package_root/onnx2torch/__init__.py"; then
  test ! -e "$package_root"
  mkdir -p "$package_root"
  PIP_NO_CACHE_DIR=1 "$python_bin" -m pip install \
    --no-index --no-deps --target "$package_root" \
    "$onnx_wheel" "$onnx2torch_wheel"
fi
PYTHONPATH="$package_root:$workspace" "$python_bin" -c \
  'import onnx, onnx2torch, torch, torchvision; assert onnx.__version__ == "1.17.0"'
"$python_bin" -m py_compile "$screen"

mkdir -p "$result_root"
cd "$run_root"
set +e
PYTHONPATH="$package_root:$workspace" CUDA_VISIBLE_DEVICES=0 \
  timeout --signal=TERM --kill-after=30s 7200s \
  "$python_bin" "$screen" \
    --onnx "$onnx_model" \
    --real-root "$real_root" \
    --manifest "$manifest" \
    --output "$result_root/optimization-screen.json" \
    --phase optimization \
    --per-embryo 8 \
    --maximum-points-per-example 4 \
    --device cuda \
    >"$result_root/screen.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" >"$result_root/screen.exit-code"
find "$result_root" -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum >"$result_root/SHA256SUMS"
cd "$run_root"
tar -czf "$result_archive" results
sha256sum "$result_archive" >"$result_archive.sha256"
printf '%s\n' completed >"$run_root/run.complete"
exit "$status"

