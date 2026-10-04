#!/usr/bin/env bash
set -euo pipefail

original_root=/home/ubuntu/biohub-temporal-localizer-v2
original_results="$original_root/results/synthetic256-real-replay-temporal-node-localizer-v2"
workspace="$original_root/workspace"
data_parent="$original_root/data/synthetic256"
real_root="$original_root/data/real-replay"
development_root="$original_root/data/development"
run_root=/home/ubuntu/biohub-temporal-localizer-rescue-v1
input_root="$run_root/input"
result_parent="$run_root/results"
output_root="$result_parent/synthetic256-real-replay-temporal-node-localizer-v2"
result_archive="$run_root/biohub-synthetic256-real-replay-temporal-node-localizer-v2-results.tar.gz"
planner="$input_root/plan_member_rescue.py"
terminal_path="$run_root/rescue-terminal.json"
planner_sha256=__PLANNER_SHA256__
real_manifest_sha256=5ccd52b96a36db7dff5f4ae482bb7e7cdd28f49c024b047d55fa7c575504a697

mkdir -p "$input_root"
exec 9>"$run_root/run.lock"
if ! flock -n 9; then
  echo "Another temporal-localizer rescue holds the lock" >&2
  exit 2
fi
echo "$planner_sha256  $planner" | sha256sum -c -
test -d "$original_results"
test -d "$workspace"
test -d "$data_parent"
test -d "$real_root"
test -d "$development_root"
test ! -e "$result_parent"
test ! -e "$result_archive"
test ! -e "$result_archive.sha256"

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

json_field() {
  "$python_bin" - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = payload[sys.argv[2]]
print("" if value is None else value)
PY
}

write_terminal() {
  "$python_bin" - "$terminal_path" "$1" "$2" "$planner" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
status = sys.argv[2]
detail_path = Path(sys.argv[3])
planner = Path(sys.argv[4])
detail = json.loads(detail_path.read_text(encoding="utf-8"))
payload = {
    "schema_version": 1,
    "run_id": "temporal-localizer-member-rescue-v1",
    "status": status,
    "rescue_policy": detail,
    "planner_sha256": hashlib.sha256(planner.read_bytes()).hexdigest(),
    "member_or_policy_ranking_performed": False,
    "competition_test_data_read": False,
    "public_code_copied": False,
    "public_predictions_copied": False,
    "public_leaderboard_used_for_selection": False,
    "authorized_for_submission": False,
}
temporary = output.with_suffix(".json.partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(output)
PY
}

wait_for_stable_idle_gpu() {
  local idle_polls=0
  while test "$idle_polls" -lt 5; do
    if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
      idle_polls=0
    else
      idle_polls=$((idle_polls + 1))
    fi
    sleep 30
  done
}

# The original four-seed run owns its evidence first.  The rescue never races
# it or changes its archive.
while test ! -f "$original_root/run.complete"; do
  if ! pgrep -f '[r]un-antelume-1gpu-temporal-localizer-v2.sh|[t]rain_synthetic_localizer.py' >/dev/null; then
    echo "Original localizer stopped without terminal evidence" >&2
    exit 4
  fi
  sleep 60
done
test -f "$original_root/biohub-synthetic256-real-replay-temporal-node-localizer-v2-results.tar.gz"
test -f "$original_root/biohub-synthetic256-real-replay-temporal-node-localizer-v2-results.tar.gz.sha256"
(
  cd "$original_root"
  sha256sum -c biohub-synthetic256-real-replay-temporal-node-localizer-v2-results.tar.gz.sha256
)

original_plan="$run_root/original-plan.json"
"$python_bin" "$planner" --results-root "$original_results" --output "$original_plan"
if test "$(json_field "$original_plan" status)" = "sufficient_members"; then
  write_terminal "not_required" "$original_plan"
  printf '%s\n' completed >"$run_root/run.complete"
  exit 0
fi

# Graph-context recovery has the next precommitted GPU turn.  Rescue starts
# only after that independent lane exits, and only after five idle polls.
while pgrep -f '[r]un-antelume-graph-context-division-recovery-v1.sh|[t]rain_graph_context_division_sweep.py|[s]core_graph_context_division_development_probe.py' >/dev/null; do
  sleep 60
done
wait_for_stable_idle_gpu
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi A10G

mkdir -p "$result_parent"
cp -a "$original_results" "$output_root"
for name in \
  SHA256SUMS \
  aggregate.exit-code \
  real-development-probe.json \
  real-development-probe.log \
  real-development-probe.exit-code; do
  if test -e "$output_root/$name"; then
    mv "$output_root/$name" "$output_root/original-$name"
  fi
done

cd "$workspace"
"$python_bin" -m py_compile \
  "$planner" \
  research/synthetic_pretrain/data.py \
  research/temporal_contrastive/patch_model.py \
  research/temporal_localization/consensus.py \
  research/temporal_localization/inference.py \
  research/temporal_localization/model.py \
  research/temporal_localization/score_real_development_probe.py \
  research/temporal_localization/train_synthetic_localizer.py

final_plan="$run_root/final-plan.json"
failed=0
while true; do
  "$python_bin" "$planner" --results-root "$output_root" --output "$final_plan"
  plan_status=$(json_field "$final_plan" status)
  if test "$plan_status" = "sufficient_members"; then
    break
  fi
  if test "$plan_status" = "fixed_rescue_pool_exhausted"; then
    failed=1
    break
  fi
  test "$plan_status" = "train_next_fixed_seed"
  seed=$(json_field "$final_plan" next_seed)
  gpu_index=$(json_field "$final_plan" next_gpu_index)
  member_root="$output_root/gpu_${gpu_index}"
  mkdir "$member_root"
  wait_for_stable_idle_gpu
  status=0
  CUDA_VISIBLE_DEVICES=0 "$python_bin" \
    research/temporal_localization/train_synthetic_localizer.py \
    --synthetic-root "$data_parent" \
    --real-shard-root "$real_root" \
    --real-shard-manifest-sha256 "$real_manifest_sha256" \
    --output-dir "$member_root" \
    --seeds "$seed" \
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
    --member-max-wall-seconds 16500 \
    --total-max-wall-seconds 17400 \
    --finalization-reserve-seconds 900 \
    --required-gpu-name A10G \
    >"$member_root/training.log" 2>&1 || status=$?
  printf '%s\n' "$status" >"$member_root/training.exit-code"
done

if test "$failed" -ne 0; then
  write_terminal "fixed_rescue_pool_exhausted" "$final_plan"
  printf '%s\n' failed >"$run_root/run.failed"
  exit 5
fi

development_status=0
wait_for_stable_idle_gpu
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
printf '%s\n' "$development_status" >"$output_root/aggregate.exit-code"

"$python_bin" "$planner" --results-root "$output_root" --output "$final_plan"
test "$(json_field "$final_plan" status)" = "sufficient_members"
write_terminal "completed" "$final_plan"
cd "$result_parent"
find synthetic256-real-replay-temporal-node-localizer-v2 -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum >"$output_root/SHA256SUMS"
tar -czf "$result_archive" synthetic256-real-replay-temporal-node-localizer-v2
sha256sum "$result_archive" >"$result_archive.sha256"
printf '%s\n' completed >"$run_root/run.complete"
exit "$development_status"
