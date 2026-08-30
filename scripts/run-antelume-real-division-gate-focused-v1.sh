#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/biohub
data_root=/home/ubuntu/biohub-real-division-data/biohub_real_division_patches_v1
initial_44=/home/ubuntu/biohub-results/competition-real-division-gate-head-v1/target_44b6/division_model.pt
initial_6=/home/ubuntu/biohub-results/competition-real-division-gate-head-v1/target_6bba/division_model.pt
output_root=/home/ubuntu/biohub-results/competition-real-division-gate-focused-v1
python_bin=/home/ubuntu/venv/bin/python

initial_44_sha256=ad369d5c122a13a8763a92a94ac3e67548cd19628b0e3500fc8db5cae1201133
initial_6_sha256=cf2ba21a8b696216e4ae4bc59a2531c44f0ca47fc1d090d3fffc441f71607b75
manifest_sha256=943717472518b917175312bd4ada9e12660d31d3ebf0bf7afd5672ab40442e1e

exec 9>/home/ubuntu/biohub-real-division-gate-focused-v1.lock
if ! flock -n 9; then
  echo "Another focused real division run holds the lock" >&2
  exit 2
fi
for pair in \
  "$initial_44_sha256  $initial_44" \
  "$initial_6_sha256  $initial_6" \
  "$manifest_sha256  $data_root/real_division_patch_manifest.json"; do
  echo "$pair" | sha256sum -c -
done
for required in \
  "$workspace/research/temporal_contrastive/train_real_division_gate.py" \
  "$python_bin"; do
  test -e "$required"
done
test ! -e "$output_root"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi A10G

cd "$workspace"
CUDA_VISIBLE_DEVICES=0 "$python_bin" \
  research/temporal_contrastive/train_real_division_gate.py \
  --data-root "$data_root" \
  --target-44b6-initial-model "$initial_44" \
  --target-6bba-initial-model "$initial_6" \
  --output-root "$output_root" \
  --train-mode focused \
  --steps 3000 \
  --batch-size 48 \
  --validation-batch-size 64 \
  --validation-every 100 \
  --log-every 50 \
  --learning-rate 2e-5 \
  --minimum-learning-rate 2e-7 \
  --weight-decay 1e-4 \
  --ema-decay 0.995

sha256sum \
  "$output_root/real_division_gate_terminal.json" \
  "$output_root/target_44b6/division_model.pt" \
  "$output_root/target_6bba/division_model.pt" \
  > "$output_root/SHA256SUMS"
