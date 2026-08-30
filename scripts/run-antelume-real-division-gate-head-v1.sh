#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/biohub
data_parent=/home/ubuntu/biohub-real-division-data
archive="$data_parent/biohub_real_division_patches_v1.tar.gz"
data_root="$data_parent/biohub_real_division_patches_v1"
probe_root=/home/ubuntu/biohub-probe-data/cache/competition-division-probe-frames-v1
probe_inventory=/home/ubuntu/biohub-probe-data/analysis/competition-division-probe-inventory.json
initial_44=/home/ubuntu/biohub-results/focused-division-gate-v1/target_44b6/division_model.pt
initial_6=/home/ubuntu/biohub-results/focused-division-gate-v2/target_44b6/division_model.pt
output_root=/home/ubuntu/biohub-results/competition-real-division-gate-head-v1
probe_output=/home/ubuntu/biohub-results/competition-real-division-gate-head-v1-probe.json
log_root=/home/ubuntu/biohub-logs
python_bin=/home/ubuntu/venv/bin/python

archive_sha256=46db151310808fa455e63a4c8791f14121125ba6bda0d46b5aa2b1d7913990e0
manifest_sha256=943717472518b917175312bd4ada9e12660d31d3ebf0bf7afd5672ab40442e1e

exec 9>/home/ubuntu/biohub-real-division-gate-head-v1.lock
if ! flock -n 9; then
  echo "Another real division-gate run already holds the lock" >&2
  exit 2
fi
if [[ "$(readlink -f "$workspace")" != /home/ubuntu/biohub ]]; then
  echo "Refusing unexpected Biohub workspace" >&2
  exit 2
fi
for required in \
  "$workspace/research/temporal_contrastive/train_real_division_gate.py" \
  "$workspace/research/score_competition_division_probe.py" \
  "$archive" \
  "$probe_root/probe_cache_manifest.json" \
  "$probe_inventory" \
  "$initial_44" \
  "$initial_6" \
  "$python_bin"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing real division input: $required" >&2
    exit 2
  fi
done
if [[ "$(sha256sum "$archive" | cut -d' ' -f1)" != "$archive_sha256" ]]; then
  echo "Real division archive hash changed" >&2
  exit 2
fi
if [[ -e "$output_root" || -e "$probe_output" ]]; then
  echo "Refusing to reuse real division output" >&2
  exit 3
fi
mkdir -p "$data_parent" "$log_root"
if [[ ! -d "$data_root" ]]; then
  temporary="$data_parent/.extract-head-v1-$$"
  mkdir "$temporary"
  tar -xzf "$archive" -C "$temporary"
  if [[ "$(sha256sum "$temporary/biohub_real_division_patches_v1/real_division_patch_manifest.json" | cut -d' ' -f1)" != "$manifest_sha256" ]]; then
    echo "Extracted real division manifest hash changed" >&2
    exit 2
  fi
  mv "$temporary/biohub_real_division_patches_v1" "$data_root"
  rmdir "$temporary"
fi
if [[ "$(sha256sum "$data_root/real_division_patch_manifest.json" | cut -d' ' -f1)" != "$manifest_sha256" ]]; then
  echo "Mounted real division manifest hash changed" >&2
  exit 2
fi
if [[ "$(sha256sum "$initial_44" | cut -d' ' -f1)" == "$(sha256sum "$initial_6" | cut -d' ' -f1)" ]]; then
  echo "Real division initial checkpoints are not independent" >&2
  exit 2
fi
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -ne 1 ]]; then
  echo "Real division training requires exactly one Antelume GPU" >&2
  exit 2
fi
if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi 'A10G'; then
  echo "Real division training requires the Antelume A10G" >&2
  exit 2
fi

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader \
  > "$log_root/competition-real-division-head-v1-gpu-before.txt"
CUDA_VISIBLE_DEVICES=0 "$python_bin" \
  "$workspace/research/temporal_contrastive/train_real_division_gate.py" \
  --data-root "$data_root" \
  --target-44b6-initial-model "$initial_44" \
  --target-6bba-initial-model "$initial_6" \
  --output-root "$output_root" \
  --train-mode head_only \
  --steps 2000 \
  --batch-size 64 \
  --validation-batch-size 96 \
  --validation-every 100 \
  --log-every 50 \
  --learning-rate 1e-4 \
  --minimum-learning-rate 1e-6 \
  --weight-decay 1e-4 \
  --ema-decay 0.995

CUDA_VISIBLE_DEVICES=0 "$python_bin" \
  "$workspace/research/score_competition_division_probe.py" \
  --inventory "$probe_inventory" \
  --cache-root "$probe_root" \
  --model-path "$output_root/target_44b6/division_model.pt" \
  --model-path "$output_root/target_6bba/division_model.pt" \
  --training-terminal "$output_root/real_division_gate_terminal.json" \
  --output "$probe_output" \
  --batch-size 64 \
  --required-gpu-name A10G

"$python_bin" - "$probe_output" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
policy = payload.get("policy_evaluation", {})
if not (
    payload.get("authorized_for_competition_graph_evaluation") is True
    and policy.get("status") == "accepted"
    and policy.get("authorized_for_submission") is False
):
    raise SystemExit("Real division held-out policy did not pass")
PY

sha256sum \
  "$archive" \
  "$data_root/real_division_patch_manifest.json" \
  "$output_root/real_division_gate_terminal.json" \
  "$output_root/target_44b6/division_model.pt" \
  "$output_root/target_6bba/division_model.pt" \
  "$probe_output" \
  > "$output_root/SHA256SUMS"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$output_root/.accepted"
echo "Antelume real division head v1 passed the held-out policy gate"
