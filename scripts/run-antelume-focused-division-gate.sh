#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/biohub
data_root=/home/ubuntu/biohub-data/biohub-zebrahub-contextual-shards-v1
localization_root=/home/ubuntu/biohub-data/biohub-division-localization-shards-v1
parent_root=/home/ubuntu/biohub-recovered-v4/zebrahub_multiscale_contextual_pretrain_v1
output_root=/home/ubuntu/biohub-results/focused-division-gate-v1
policy_root=/home/ubuntu/biohub-results/division-policy-v1
log_root=/home/ubuntu/biohub-logs
python_bin=/home/ubuntu/venv/bin/python

exec 9>/home/ubuntu/biohub-focused-division-gate.lock
if ! flock -n 9; then
  echo "Another focused division-gate run already holds the lock" >&2
  exit 2
fi
if [[ "$(readlink -f "$workspace")" != /home/ubuntu/biohub ]]; then
  echo "Refusing unexpected Biohub workspace" >&2
  exit 2
fi
for required in \
  "$workspace/research/temporal_contrastive/train_focused_division_gate.py" \
  "$data_root/train" \
  "$localization_root/selection" \
  "$localization_root/audit" \
  "$parent_root/pretraining_terminal.json" \
  "$python_bin"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing focused division input: $required" >&2
    exit 2
  fi
done
if [[ -e "$output_root" ]]; then
  echo "Refusing to reuse focused division output: $output_root" >&2
  exit 3
fi
if [[ -e "$policy_root/policy.json" ]]; then
  echo "Refusing to overwrite an existing published division policy" >&2
  exit 3
fi
mkdir -p "$policy_root" "$log_root"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader \
  > "$log_root/focused-division-gpu-before.txt"
CUDA_VISIBLE_DEVICES=0 "$python_bin" \
  "$workspace/research/temporal_contrastive/train_focused_division_gate.py" \
  --data-root "$data_root" \
  --localization-root "$localization_root" \
  --parent-root "$parent_root" \
  --output-root "$output_root" \
  --steps 3000 \
  --batch-size 48 \
  --validation-batch-size 64 \
  --validation-every 250 \
  --log-every 50 \
  --learning-rate 2e-4 \
  --minimum-learning-rate 2e-6 \
  --weight-decay 1e-5 \
  --ema-decay 0.995 \
  --patch-batch-size 64

cp "$output_root/division-recovery-policy.json" "$policy_root/policy.json"
sha256sum \
  "$output_root/focused_division_gate_terminal.json" \
  "$output_root/division-recovery-policy.json" \
  "$output_root/target_44b6/division_model.pt" \
  "$output_root/target_6bba/division_model.pt" \
  > "$output_root/SHA256SUMS"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$output_root/.complete"
echo "Antelume focused division gate accepted and published"
