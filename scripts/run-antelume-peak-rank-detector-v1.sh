#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/biohub
run_root=/home/ubuntu/biohub-peak-rank-detector-v1
input_root="$run_root/input"
result_parent="$run_root/results"
result_root="$result_parent/synthetic256-real-positive-temporal-peak-rank-v1"
synthetic_root=/home/ubuntu/biohub-temporal-localizer-v2/data/synthetic256
real_root=/home/ubuntu/biohub-temporal-localizer-v2/data/real-replay
trainer="$input_root/train_synthetic_real_detector.py"
python_bin=/home/ubuntu/venv/bin/python
result_archive=/home/ubuntu/biohub-peak-rank-detector-v1-results.tar.gz

real_manifest_sha256=5ccd52b96a36db7dff5f4ae482bb7e7cdd28f49c024b047d55fa7c575504a697
trainer_sha256=9310a87086b313362c135b360ef175ab6b87033653c3700b3bfb0184a313fa05
model_sha256=da1eae6cb7dac7f3175c0a672403b590a19ea0ff2bbfb9e07dff17fd7347c637
objectives_sha256=9340af271c96512f58192917b2763c386b7f32d9560e24ce266ada006cbfc787
synthetic_data_sha256=8e108e0650f3565ea4c05f0e747fb04d3c9fdc09922b2ef077f753bc203160be

exec 9>"$run_root/run.lock"
if ! flock -n 9; then
  echo "Another peak-ranking detector run holds the lock" >&2
  exit 3
fi
test ! -e "$result_parent"
test ! -e "$result_archive"
test ! -e "$result_archive.sha256"
echo "$trainer_sha256  $trainer" | sha256sum -c -
echo "$model_sha256  $workspace/research/peak_rank_detection/model.py" | sha256sum -c -
echo "$objectives_sha256  $workspace/research/peak_rank_detection/objectives.py" | sha256sum -c -
echo "$synthetic_data_sha256  $workspace/research/synthetic_pretrain/data.py" | sha256sum -c -
echo "$real_manifest_sha256  $real_root/real_localization_shard_manifest.json" | sha256sum -c -

# The recovered graph ensemble has first claim on the shared A10G.  Any later
# unrelated GPU process (including RSNA) is never signalled by this runner and
# causes the separately deployed Biohub yield guard to pause this trainer.
while test ! -f /home/ubuntu/biohub-graph-context-recovery-v1/run.complete; do
  sleep 30
done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -q '[0-9]'; do
  sleep 30
done
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi A10G

mkdir -p "$result_parent"
cd "$workspace"
"$python_bin" -m py_compile \
  "$trainer" \
  research/peak_rank_detection/model.py \
  research/peak_rank_detection/objectives.py \
  research/synthetic_pretrain/data.py
set +e
PYTHONPATH="$workspace" CUDA_VISIBLE_DEVICES=0 "$python_bin" "$trainer" \
  --synthetic-root "$synthetic_root" \
  --real-root "$real_root" \
  --real-manifest-sha256 "$real_manifest_sha256" \
  --output-root "$result_root" \
  --steps 12000 \
  --validation-every 1000 \
  --log-every 50 \
  --learning-rate 2e-4 \
  --minimum-learning-rate 2e-6 \
  --weight-decay 2e-4 \
  --ema-decay 0.999 \
  --real-frequency 4 \
  --widths 96,192,384,768 \
  --depths 3,3,9,3 \
  --seed 1041729 \
  --max-wall-seconds 25200 \
  >"$run_root/training.log" 2>&1
status=$?
set -e
mkdir -p "$result_root"
cp "$run_root/training.log" "$result_root/training.log"
printf '%s\n' "$status" >"$result_root/training.exit-code"
cd "$result_parent"
find synthetic256-real-positive-temporal-peak-rank-v1 -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum >"$result_root/SHA256SUMS"
tar -czf "$result_archive" synthetic256-real-positive-temporal-peak-rank-v1
sha256sum "$result_archive" >"$result_archive.sha256"
printf '%s\n' completed >"$run_root/run.complete"
exit "$status"
