#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/biohub
archive=/home/ubuntu/biohub-graph-context-relational-patches-v1.tar.gz
data_parent=/home/ubuntu/biohub-graph-context-relational-data-v1
data_root="$data_parent/biohub_graph_context_relational_patches_v1"
output_root=/home/ubuntu/biohub-results/competition-graph-context-division-sweep-v1
prior_terminal=/home/ubuntu/biohub-results/competition-relational-division-sweep-v1/models/relational_division_sweep_terminal.json
initial_44=/home/ubuntu/biohub-results/competition-real-division-gate-head-v1/target_44b6/division_model.pt
initial_6=/home/ubuntu/biohub-results/competition-real-division-gate-head-v1/target_6bba/division_model.pt
python_bin=/home/ubuntu/venv/bin/python
development_inventory=/home/ubuntu/biohub-graph-context-development-inventory-v1.json
probe_cache=/home/ubuntu/biohub-probe-data/cache/competition-division-probe-frames-v1
archive_sha256=__GRAPH_CONTEXT_ARCHIVE_SHA256__
manifest_sha256=__GRAPH_CONTEXT_MANIFEST_SHA256__
development_inventory_sha256=c8883e77abcf76c5a837c5fd0b21afdfefb2e51cb8e550e3a81f69d70562f311
initial_44_sha256=ad369d5c122a13a8763a92a94ac3e67548cd19628b0e3500fc8db5cae1201133
initial_6_sha256=cf2ba21a8b696216e4ae4bc59a2531c44f0ca47fc1d090d3fffc441f71607b75
poll_seconds=60
maximum_polls=1440

exec 9>/home/ubuntu/biohub-graph-context-division-sweep-v1.lock
if ! flock -n 9; then
  echo "Another graph-context division sweep holds the lock" >&2
  exit 2
fi
for pair in \
  "$archive_sha256  $archive" \
  "$initial_44_sha256  $initial_44" \
  "$initial_6_sha256  $initial_6"; do
  echo "$pair" | sha256sum -c -
done
for required in \
  "$workspace/research/temporal_contrastive/graph_context_division_model.py" \
  "$workspace/research/temporal_contrastive/graph_context_division_inference.py" \
  "$workspace/research/temporal_contrastive/score_graph_context_division_development_probe.py" \
  "$workspace/research/temporal_contrastive/train_graph_context_division_sweep.py" \
  "$workspace/research/temporal_contrastive/relational_division_model.py" \
  "$workspace/research/temporal_contrastive/train_relational_division_sweep.py" \
  "$workspace/research/temporal_contrastive/train_real_division_gate.py" \
  "$workspace/research/temporal_contrastive/multiscale_contextual_pair_fusion.py" \
  "$development_inventory" \
  "$probe_cache/probe_cache_manifest.json" \
  "$python_bin"; do
  test -e "$required"
done
test ! -e "$output_root"
test ! -e "$data_parent"

poll=0
while ! test -f "$prior_terminal"; do
  poll=$((poll + 1))
  if test "$poll" -ge "$maximum_polls"; then
    echo "Timed out waiting for the relational sweep terminal" >&2
    exit 3
  fi
  sleep "$poll_seconds"
done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; do
  sleep "$poll_seconds"
done
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi A10G

mkdir "$data_parent"
tar -xzf "$archive" -C "$data_parent"
echo "$manifest_sha256  $data_root/graph_context_relational_patch_manifest.json" \
  | sha256sum -c -
echo "$development_inventory_sha256  $development_inventory" | sha256sum -c -
cd "$workspace"
"$python_bin" -m py_compile \
  research/temporal_contrastive/graph_context_division_model.py \
  research/temporal_contrastive/graph_context_division_inference.py \
  research/temporal_contrastive/score_graph_context_division_development_probe.py \
  research/temporal_contrastive/train_graph_context_division_sweep.py
mkdir "$output_root"
set +e
CUDA_VISIBLE_DEVICES=0 "$python_bin" \
  research/temporal_contrastive/train_graph_context_division_sweep.py \
  --data-root "$data_root" \
  --initial-model "$initial_44" \
  --initial-model "$initial_6" \
  --output-root "$output_root/models" \
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
  >"$output_root/training.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" >"$output_root/training.exit-code"
probe_status=4
if test "$status" -eq 0; then
  set +e
  CUDA_VISIBLE_DEVICES=0 "$python_bin" \
    research/temporal_contrastive/score_graph_context_division_development_probe.py \
    --inventory "$development_inventory" \
    --cache-root "$probe_cache" \
    --results-root "$output_root/models" \
    --output "$output_root/graph_context_development_probe.json" \
    --batch-size 10 \
    >"$output_root/development-probe.log" 2>&1
  probe_status=$?
  set -e
fi
printf '%s\n' "$probe_status" >"$output_root/development-probe.exit-code"
cd /home/ubuntu/biohub-results
find competition-graph-context-division-sweep-v1 -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > /home/ubuntu/biohub-graph-context-division-sweep-v1.SHA256SUMS.partial
mv /home/ubuntu/biohub-graph-context-division-sweep-v1.SHA256SUMS.partial \
  "$output_root/SHA256SUMS"
tar -czf /home/ubuntu/biohub-graph-context-division-sweep-v1-results.tar.gz \
  -C /home/ubuntu/biohub-results competition-graph-context-division-sweep-v1
sha256sum /home/ubuntu/biohub-graph-context-division-sweep-v1-results.tar.gz \
  > /home/ubuntu/biohub-graph-context-division-sweep-v1-results.tar.gz.sha256
if test "$status" -ne 0; then
  exit "$status"
fi
exit "$probe_status"
