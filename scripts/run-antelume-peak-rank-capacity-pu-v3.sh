#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/biohub
run_root=/home/ubuntu/biohub-peak-rank-detector-v1/capacity-pu-v3
input_root="$run_root/input"
result_parent="$run_root/results"
result_root="$result_parent/synthetic256-real-conservative-pu-depth-robust-capacity-peak-rank-v3"
synthetic_root=/home/ubuntu/biohub-temporal-localizer-v2/data/synthetic256
real_root=/home/ubuntu/biohub-temporal-localizer-v2/data/real-replay
trainer="$input_root/train_synthetic_real_detector.py"
base_trainer="$workspace/research/peak_rank_detection/train_synthetic_real_detector.py"
python_bin=/home/ubuntu/venv/bin/python
result_archive=/home/ubuntu/biohub-peak-rank-capacity-pu-v3-results.tar.gz

v1_result_parent=/home/ubuntu/biohub-peak-rank-detector-v1/results
v1_archive=/home/ubuntu/biohub-peak-rank-detector-v1-results.tar.gz
v1_ack=/home/ubuntu/biohub-peak-rank-detector-v1/harvest.verified
v2_result_parent=/home/ubuntu/biohub-peak-rank-detector-v1/depth-pu-v2/results
v2_archive=/home/ubuntu/biohub-peak-rank-depth-pu-v2-results.tar.gz
v2_complete=/home/ubuntu/biohub-peak-rank-detector-v1/depth-pu-v2/run.complete
v2_ack=/home/ubuntu/biohub-peak-rank-detector-v1/depth-pu-v2/harvest.verified

real_manifest_sha256=5ccd52b96a36db7dff5f4ae482bb7e7cdd28f49c024b047d55fa7c575504a697
trainer_sha256=2b7023b4335a48520f107c3147b2e86abb616fedc6f92d68c0118a39dd5cc861
base_trainer_sha256=f64dc9da4b8babab667d3ab216a98759fded2e806032f5c33decc30e31028880
model_sha256=da1eae6cb7dac7f3175c0a672403b590a19ea0ff2bbfb9e07dff17fd7347c637
objectives_sha256=9340af271c96512f58192917b2763c386b7f32d9560e24ce266ada006cbfc787
synthetic_data_sha256=8e108e0650f3565ea4c05f0e747fb04d3c9fdc09922b2ef077f753bc203160be
minimum_free_bytes=1300000000

safe_remove_biohub_tree() {
  local requested="$1"
  local expected="$2"
  local resolved_requested
  local resolved_expected
  resolved_requested="$(realpath -m -- "$requested")"
  resolved_expected="$(realpath -m -- "$expected")"
  test "$resolved_requested" = "$resolved_expected"
  case "$resolved_requested" in
    /home/ubuntu/biohub-peak-rank-detector-v1/results|/home/ubuntu/biohub-peak-rank-detector-v1/depth-pu-v2/results)
      rm -rf -- "$resolved_requested"
      ;;
    *)
      echo "Refusing unsafe Biohub cleanup target: $resolved_requested" >&2
      exit 8
      ;;
  esac
}

verified_local_harvest() {
  local archive="$1"
  local acknowledgement="$2"
  test -s "$archive"
  test -s "$archive.sha256"
  test -s "$acknowledgement"
  sha256sum -c "$archive.sha256" >/dev/null
  test "$(awk '{print $1}' "$acknowledgement")" = "$(awk '{print $1}' "$archive.sha256")"
}

exec 9>"$run_root/run.lock"
if ! flock -n 9; then
  echo "Another capacity peak detector run holds the lock" >&2
  exit 3
fi
test ! -e "$result_parent"
test ! -e "$result_archive"
test ! -e "$result_archive.sha256"
echo "$trainer_sha256  $trainer" | sha256sum -c -
echo "$base_trainer_sha256  $base_trainer" | sha256sum -c -
echo "$model_sha256  $workspace/research/peak_rank_detection/model.py" | sha256sum -c -
echo "$objectives_sha256  $workspace/research/peak_rank_detection/objectives.py" | sha256sum -c -
echo "$synthetic_data_sha256  $workspace/research/synthetic_pretrain/data.py" | sha256sum -c -
echo "$real_manifest_sha256  $real_root/real_localization_shard_manifest.json" | sha256sum -c -

# Preserve the single-GPU ownership chain.  This member cannot begin until v2
# has completed and both predecessor archives were hash-verified locally.
while test ! -f "$v2_complete"; do sleep 30; done
while test ! -f "$v1_ack" || test ! -f "$v2_ack"; do sleep 30; done
verified_local_harvest "$v1_archive" "$v1_ack"
verified_local_harvest "$v2_archive" "$v2_ack"
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -q '[0-9]'; do
  sleep 30
done

# Only Biohub-generated predecessor copies are removed, after local verified
# receipts exist.  RSNA paths and processes are outside every cleanup target.
safe_remove_biohub_tree "$v1_result_parent" /home/ubuntu/biohub-peak-rank-detector-v1/results
safe_remove_biohub_tree "$v2_result_parent" /home/ubuntu/biohub-peak-rank-detector-v1/depth-pu-v2/results
rm -f -- "$v1_archive" "$v1_archive.sha256" "$v2_archive" "$v2_archive.sha256"

available_bytes="$(df --output=avail -B1 "$run_root" | tail -n 1 | tr -d ' ')"
if test "$available_bytes" -lt "$minimum_free_bytes"; then
  echo "Insufficient disk for capacity run: $available_bytes bytes" >&2
  exit 9
fi
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi A10G

mkdir -p "$result_parent"
cd "$run_root"
"$python_bin" -m py_compile "$trainer" "$base_trainer" \
  "$workspace/research/peak_rank_detection/model.py" \
  "$workspace/research/peak_rank_detection/objectives.py"
set +e
PYTHONPATH="$workspace" CUDA_VISIBLE_DEVICES=0 "$python_bin" "$trainer" \
  --synthetic-root "$synthetic_root" \
  --real-root "$real_root" \
  --real-manifest-sha256 "$real_manifest_sha256" \
  --output-root "$result_root" \
  --steps 2000 \
  --validation-every 1000 \
  --log-every 50 \
  --learning-rate 2e-4 \
  --minimum-learning-rate 2e-6 \
  --weight-decay 2e-4 \
  --ema-decay 0.999 \
  --real-frequency 4 \
  --widths 128,256,512,1024 \
  --depths 3,3,9,3 \
  --seed 2607157 \
  --max-wall-seconds 25200 \
  >"$run_root/training.log" 2>&1
status=$?
set -e
mkdir -p "$result_root"
cp "$run_root/training.log" "$result_root/training.log"
printf '%s\n' "$status" >"$result_root/training.exit-code"
cd "$result_parent"
find synthetic256-real-conservative-pu-depth-robust-capacity-peak-rank-v3 \
  -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum >"$result_root/SHA256SUMS"
tar -czf "$result_archive" synthetic256-real-conservative-pu-depth-robust-capacity-peak-rank-v3
sha256sum "$result_archive" >"$result_archive.sha256"
printf '%s\n' completed >"$run_root/run.complete"
exit "$status"
