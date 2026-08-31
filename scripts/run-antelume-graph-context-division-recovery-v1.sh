#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/biohub
run_root=/home/ubuntu/biohub-graph-context-recovery-v1
input_root="$run_root/input"
result_parent="$run_root/results"
result_root="$result_parent/competition-graph-context-division-sweep-v1"
model_root="$result_root/models"
source_models=/home/ubuntu/biohub-results/competition-graph-context-division-sweep-v1/models
data_root=/home/ubuntu/biohub-graph-context-relational-data-v1/biohub_graph_context_relational_patches_v1
development_inventory=/home/ubuntu/biohub-graph-context-development-inventory-v1.json
probe_cache=/home/ubuntu/biohub-probe-data/cache/competition-division-probe-frames-v1
initial_44=/home/ubuntu/biohub-results/competition-real-division-gate-head-v1/target_44b6/division_model.pt
initial_6=/home/ubuntu/biohub-results/competition-real-division-gate-head-v1/target_6bba/division_model.pt
trainer="$input_root/train_graph_context_division_sweep.py"
python_bin=/home/ubuntu/venv/bin/python
result_archive=/home/ubuntu/biohub-graph-context-recovery-v1-results.tar.gz

manifest_sha256=2e5c4c46b11ff1480194c29e88c0e16b0ebf2f0fd65b2fc29548d26fee599bd9
development_inventory_sha256=c8883e77abcf76c5a837c5fd0b21afdfeb2e51cb8e550e3a81f69d70562f311
initial_44_sha256=ad369d5c122a13a8763a92a94ac3e67548cd19628b0e3500fc8db5cae1201133
initial_6_sha256=cf2ba21a8b696216e4ae4bc59a2531c44f0ca47fc1d090d3fffc441f71607b75
trainer_sha256=530504f1653451c66bf89f0098f747caa3c5aef497e3952e4fe7a3fe6aa264c4
model_source_sha256=7ca45bffc2e52cf26223b968c01cbf5b03d5dc0966ffeee17d2f287a0ff4ada4
inference_source_sha256=190d7ad1e7fedbad13a22bedeb0d2f9ae77c76c44df0bba49386e87715aff08e
probe_source_sha256=82c58823ef9be1b971cfd9e8030c5d7f720dad5c6e466385779a0c9d86dcf52f

recovered_members=(
  seed-613111-init-1
  seed-613111-init-2
  seed-713117-init-1
  seed-713117-init-2
  seed-813121-init-1
)

exec 9>"$run_root/run.lock"
if ! flock -n 9; then
  echo "Another graph-context recovery holds the lock" >&2
  exit 3
fi
test ! -e "$result_parent"
test ! -e "$result_archive"
test ! -e "$result_archive.sha256"
echo "$trainer_sha256  $trainer" | sha256sum -c -
echo "$manifest_sha256  $data_root/graph_context_relational_patch_manifest.json" | sha256sum -c -
echo "$development_inventory_sha256  $development_inventory" | sha256sum -c -
echo "$initial_44_sha256  $initial_44" | sha256sum -c -
echo "$initial_6_sha256  $initial_6" | sha256sum -c -
echo "$model_source_sha256  $workspace/research/temporal_contrastive/graph_context_division_model.py" | sha256sum -c -
echo "$inference_source_sha256  $workspace/research/temporal_contrastive/graph_context_division_inference.py" | sha256sum -c -
echo "$probe_source_sha256  $workspace/research/temporal_contrastive/score_graph_context_division_development_probe.py" | sha256sum -c -

# The localizer owns the A10G until its runner exits. A missing completion file
# after that process ends is not allowed to strand the independent recovery.
while test ! -f /home/ubuntu/biohub-temporal-localizer-v2/run.complete; do
  if ! pgrep -f 'run-antelume-1gpu-temporal-localizer-v2.sh|train_synthetic_localizer.py' >/dev/null; then
    break
  fi
  sleep 60
done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; do
  sleep 60
done
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi A10G

mkdir -p "$model_root"
for member in "${recovered_members[@]}"; do
  source_root="$source_models/$member"
  test -f "$source_root/worker_terminal.json"
  test -f "$source_root/selection_history.json"
  test -f "$source_root/graph_context_model.pt"
  test ! -e "$source_root/audit_terminal.json"
  cp -al "$source_root" "$model_root/$member"
done

"$python_bin" - "$result_root/recovery-manifest.json" "$trainer" "${recovered_members[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
trainer = Path(sys.argv[2])
members = sys.argv[3:]
source_root = Path("/home/ubuntu/biohub-results/competition-graph-context-division-sweep-v1/models")
records = []
for member in members:
    root = source_root / member
    terminal = json.loads((root / "worker_terminal.json").read_text(encoding="utf-8"))
    checkpoint = root / "graph_context_model.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != terminal.get("model_sha256"):
        raise ValueError(f"recovered checkpoint changed: {member}")
    records.append({"member": member, "model_sha256": digest})
payload = {
    "schema_version": 1,
    "run_id": "competition-graph-context-division-recovery-v1",
    "status": "five_completed_members_recovered",
    "recovered_member_count": len(records),
    "recovered_members": records,
    "partial_checkpoint_resumed": False,
    "audit_opened_before_recovery": False,
    "trainer_sha256": hashlib.sha256(trainer.read_bytes()).hexdigest(),
    "competition_test_data_read": False,
    "public_leaderboard_used_for_selection": False,
    "authorized_for_submission": False,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

cd "$workspace"
"$python_bin" -m py_compile \
  "$trainer" \
  research/temporal_contrastive/graph_context_division_model.py \
  research/temporal_contrastive/graph_context_division_inference.py \
  research/temporal_contrastive/score_graph_context_division_development_probe.py
set +e
PYTHONPATH="$workspace" CUDA_VISIBLE_DEVICES=0 "$python_bin" "$trainer" \
  --data-root "$data_root" \
  --initial-model "$initial_44" \
  --initial-model "$initial_6" \
  --output-root "$model_root" \
  --seeds 613111,713117,813121,913127 \
  --steps 20000 \
  --batch-size 10 \
  --validation-batch-size 20 \
  --validation-every 250 \
  --log-every 100 \
  --learning-rate 6e-5 \
  --minimum-learning-rate 3e-7 \
  --backbone-lr-multiplier 0.15 \
  --weight-decay 2e-4 \
  --ema-decay 0.995 \
  --resume-completed \
  >"$result_root/training.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" >"$result_root/training.exit-code"

probe_status=4
if test "$status" -eq 0; then
  set +e
  CUDA_VISIBLE_DEVICES=0 "$python_bin" \
    research/temporal_contrastive/score_graph_context_division_development_probe.py \
    --inventory "$development_inventory" \
    --cache-root "$probe_cache" \
    --results-root "$model_root" \
    --output "$result_root/graph_context_development_probe.json" \
    --batch-size 10 \
    >"$result_root/development-probe.log" 2>&1
  probe_status=$?
  set -e
fi
printf '%s\n' "$probe_status" >"$result_root/development-probe.exit-code"

cd "$result_parent"
find competition-graph-context-division-sweep-v1 -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum >"$result_root/SHA256SUMS"
tar -czf "$result_archive" competition-graph-context-division-sweep-v1
sha256sum "$result_archive" >"$result_archive.sha256"
printf '%s\n' completed >"$run_root/run.complete"
if test "$status" -ne 0; then
  exit "$status"
fi
exit "$probe_status"
