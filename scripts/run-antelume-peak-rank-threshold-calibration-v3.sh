#!/usr/bin/env bash
set -euo pipefail

calibrator_sha256="${1:?calibrator sha256 required}"
evaluator_sha256="${2:?evaluator sha256 required}"

workspace=/home/ubuntu/biohub
run_root=/home/ubuntu/biohub-peak-rank-detector-v1/capacity-pu-v3
result_parent="$run_root/results"
result_name=synthetic256-real-conservative-pu-depth-robust-capacity-peak-rank-v3
result_root="$result_parent/$result_name"
archive=/home/ubuntu/biohub-peak-rank-capacity-pu-v3-results.tar.gz
synthetic_root=/home/ubuntu/biohub-temporal-localizer-v2/data/synthetic256
python_bin=/home/ubuntu/venv/bin/python
calibrator="$workspace/research/peak_rank_detection/calibrate_detection_threshold.py"
evaluator="$workspace/research/peak_rank_detection/evaluate_peak_rank_detector.py"
terminal="$result_root/terminal.json"
checkpoint="$result_root/peak_rank_detector.pt"
calibration="$result_root/threshold_calibration.json"
complete="$run_root/threshold-calibration.complete"

exec 9>"$run_root/threshold-calibration.lock"
if ! flock -n 9; then
  echo "Another V3 threshold calibration holds the lock" >&2
  exit 3
fi
test ! -e "$complete"
echo "$calibrator_sha256  $calibrator" | sha256sum -c -
echo "$evaluator_sha256  $evaluator" | sha256sum -c -

while test ! -f "$run_root/run.complete"; do sleep 30; done
test -f "$terminal"
status="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$terminal")"
if test "$status" = accepted_at_audit; then
  test -f "$checkpoint"
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | grep -q '[0-9]'; do
    sleep 30
  done
  set +e
  PYTHONPATH="$workspace" CUDA_VISIBLE_DEVICES=0 "$python_bin" "$calibrator" \
    --synthetic-root "$synthetic_root" \
    --checkpoint "$checkpoint" \
    --training-terminal "$terminal" \
    --output "$calibration" \
    --tta-modes none,zflip2,rot4,d4 \
    --max-wall-seconds 3600 \
    >"$result_root/threshold-calibration.log" 2>&1
  calibration_status=$?
  set -e
  printf '%s\n' "$calibration_status" >"$result_root/threshold-calibration.exit-code"
  test "$calibration_status" -eq 0
  test -s "$calibration"

  cd "$result_parent"
  find "$result_name" -type f ! -name SHA256SUMS -print0 \
    | sort -z | xargs -0 sha256sum >"$result_root/SHA256SUMS"
  rm -f -- "$archive.partial" "$archive.sha256.partial"
  tar -czf "$archive.partial" "$result_name"
  sha256sum "$archive.partial" >"$archive.sha256.partial"
  mv "$archive.partial" "$archive"
  sed "s/\.partial//" "$archive.sha256.partial" >"$archive.sha256"
  rm -f -- "$archive.sha256.partial"
  printf '%s\n' calibrated >"$complete.partial"
else
  printf '%s\n' "training_$status" >"$complete.partial"
fi
mv "$complete.partial" "$complete"
