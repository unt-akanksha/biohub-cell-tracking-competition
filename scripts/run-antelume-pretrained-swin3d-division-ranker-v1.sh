#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/biohub
data_root=/home/ubuntu/biohub-real-division-data/biohub_real_division_patches_v1
output_root=/home/ubuntu/biohub-results/competition-pretrained-swin3d-division-ranker-v1
python_bin=/home/ubuntu/venv/bin/python
manifest_sha256=943717472518b917175312bd4ada9e12660d31d3ebf0bf7afd5672ab40442e1e

exec 9>/home/ubuntu/biohub-pretrained-swin3d-division-ranker-v1.lock
if ! flock -n 9; then
  echo "Another pretrained Swin3D division run holds the lock" >&2
  exit 2
fi
echo "$manifest_sha256  $data_root/real_division_patch_manifest.json" | sha256sum -c -
for required in \
  "$workspace/research/temporal_contrastive/train_pretrained_swin3d_division_ranker.py" \
  "$python_bin"; do
  test -e "$required"
done
test ! -e "$output_root"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi A10G

cd "$workspace"
CUDA_VISIBLE_DEVICES=0 "$python_bin" \
  research/temporal_contrastive/train_pretrained_swin3d_division_ranker.py \
  --data-root "$data_root" \
  --output-root "$output_root" \
  --steps 1500 \
  --batch-size 8 \
  --validation-batch-size 24 \
  --validation-every 150 \
  --log-every 25 \
  --learning-rate 2e-5 \
  --minimum-learning-rate 2e-7 \
  --weight-decay 5e-3 \
  --ema-decay 0.995 \
  --required-gpu-name A10G

sha256sum \
  "$output_root/swin3d_division_ranker.pt" \
  "$output_root/swin3d_division_ranker_terminal.json" \
  > "$output_root/SHA256SUMS"
