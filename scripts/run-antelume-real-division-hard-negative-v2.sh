#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/biohub
data_root=/home/ubuntu/biohub-real-division-data/biohub_real_division_hard_negative_patches_v2
archive=/home/ubuntu/biohub-real-division-data/biohub_real_division_hard_negative_patches_v2.tar.gz
initial_44=/home/ubuntu/biohub-results/focused-division-gate-v1/target_44b6/division_model.pt
initial_6=/home/ubuntu/biohub-results/focused-division-gate-v2/target_44b6/division_model.pt
output_root=/home/ubuntu/biohub-results/competition-real-division-hard-negative-v2
audit_output=/home/ubuntu/biohub-results/competition-real-division-hard-negative-v2-audit.json
python_bin=/home/ubuntu/venv/bin/python

archive_sha256=b148a16eef851380c82185b70209e581ac3f7f6c2c2a22362747a0b6d144a605
manifest_sha256=bfa974f7c5cfa7b2271b7f65128ac9896d458b3e1af150efb238cc5937a4d251
initial_44_sha256=c22f716b174654ad06d2c8ca692edc950422488a21bb7b0dafc78c36e042cdab
initial_6_sha256=b08bbc9795d41ca657910f57dd8a75cd527db98a9f06c880884b8c113c4b6426

exec 9>/home/ubuntu/biohub-real-division-hard-negative-v2.lock
if ! flock -n 9; then
  echo "Another hard-negative division run holds the lock" >&2
  exit 2
fi
for pair in \
  "$archive_sha256  $archive" \
  "$manifest_sha256  $data_root/real_division_hard_negative_manifest.json" \
  "$initial_44_sha256  $initial_44" \
  "$initial_6_sha256  $initial_6"; do
  echo "$pair" | sha256sum -c -
done
for required in \
  "$workspace/research/temporal_contrastive/train_real_division_gate_v2.py" \
  "$workspace/research/temporal_contrastive/score_real_division_gate_v2_audit.py" \
  "$python_bin"; do
  test -e "$required"
done
test ! -e "$output_root"
test ! -e "$audit_output"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi A10G

cd "$workspace"
CUDA_VISIBLE_DEVICES=0 "$python_bin" \
  research/temporal_contrastive/train_real_division_gate_v2.py \
  --data-root "$data_root" \
  --target-44b6-initial-model "$initial_44" \
  --target-6bba-initial-model "$initial_6" \
  --output-root "$output_root" \
  --train-mode focused \
  --steps 4000 \
  --batch-size 48 \
  --validation-batch-size 96 \
  --validation-every 100 \
  --log-every 50 \
  --learning-rate 2e-5 \
  --minimum-learning-rate 2e-7 \
  --weight-decay 1e-4 \
  --ema-decay 0.995 \
  --required-gpu-name A10G

CUDA_VISIBLE_DEVICES=0 "$python_bin" \
  research/temporal_contrastive/score_real_division_gate_v2_audit.py \
  --data-root "$data_root" \
  --training-terminal "$output_root/real_division_hard_negative_terminal.json" \
  --model-path "$output_root/target_44b6/division_model.pt" \
  --model-path "$output_root/target_6bba/division_model.pt" \
  --output "$audit_output" \
  --batch-size 96 \
  --required-gpu-name A10G

sha256sum \
  "$archive" \
  "$data_root/real_division_hard_negative_manifest.json" \
  "$output_root/real_division_hard_negative_terminal.json" \
  "$output_root/target_44b6/division_model.pt" \
  "$output_root/target_6bba/division_model.pt" \
  "$audit_output" \
  > "$output_root/SHA256SUMS"
