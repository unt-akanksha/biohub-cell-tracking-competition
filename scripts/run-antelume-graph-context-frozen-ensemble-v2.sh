#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/biohub
run_root=/home/ubuntu/biohub-graph-context-frozen-ensemble-v2
archive=/home/ubuntu/biohub-graph-context-relational-patches-v1.tar.gz
data_parent="$run_root/data"
data_root="$data_parent/biohub_graph_context_relational_patches_v1"
output_root=/home/ubuntu/biohub-results/competition-graph-context-division-frozen-ensemble-v2
prior_run_root=/home/ubuntu/biohub-peak-rank-detector-v1/hard-mined-temporal-snr-v27
prior_result_parent="$prior_run_root/results"
prior_archive=/home/ubuntu/biohub-peak-rank-hard-mined-temporal-snr-v27-results.tar.gz
initial_44=/home/ubuntu/biohub-results/competition-real-division-gate-head-v1/target_44b6/division_model.pt
initial_6=/home/ubuntu/biohub-results/competition-real-division-gate-head-v1/target_6bba/division_model.pt
python_bin=/home/ubuntu/venv/bin/python
development_inventory=/home/ubuntu/biohub-graph-context-development-inventory-v1.json
probe_cache=/home/ubuntu/biohub-probe-data/cache/competition-division-probe-frames-v1
result_archive=/home/ubuntu/biohub-graph-context-frozen-ensemble-v2-results.tar.gz

archive_sha256=__GRAPH_CONTEXT_ARCHIVE_SHA256__
manifest_sha256=__GRAPH_CONTEXT_MANIFEST_SHA256__
trainer_sha256=__GRAPH_CONTEXT_TRAINER_SHA256__
scorer_sha256=__GRAPH_CONTEXT_SCORER_SHA256__
model_sha256=__GRAPH_CONTEXT_MODEL_SHA256__
inference_sha256=__GRAPH_CONTEXT_INFERENCE_SHA256__
development_inventory_sha256=c8883e77abcf76c5a837c5fd0b21afdfefb2e51cb8e550e3a81f69d70562f311
initial_44_sha256=ad369d5c122a13a8763a92a94ac3e67548cd19628b0e3500fc8db5cae1201133
initial_6_sha256=cf2ba21a8b696216e4ae4bc59a2531c44f0ca47fc1d090d3fffc441f71607b75
minimum_free_bytes=5000000000

safe_remove_prior_results() {
  local resolved
  resolved="$(realpath -m -- "$prior_result_parent")"
  test "$resolved" = /home/ubuntu/biohub-peak-rank-detector-v1/hard-mined-temporal-snr-v27/results
  rm -rf -- "$resolved"
}

verified_prior_harvest() {
  test -s "$prior_archive"
  test -s "$prior_archive.sha256"
  test -s "$prior_run_root/harvest.verified"
  sha256sum -c "$prior_archive.sha256" >/dev/null
  test "$(awk '{print $1}' "$prior_run_root/harvest.verified")" = \
    "$(awk '{print $1}' "$prior_archive.sha256")"
}

exec 9>"$run_root/run.lock"
if ! flock -n 9; then
  echo "Another graph-context frozen-ensemble run holds the lock" >&2
  exit 3
fi
test ! -e "$output_root"
test ! -e "$data_parent"
test ! -e "$result_archive"
test ! -e "$result_archive.sha256"

echo "$archive_sha256  $archive" | sha256sum -c -
echo "$trainer_sha256  $workspace/research/temporal_contrastive/train_graph_context_division_sweep.py" | sha256sum -c -
echo "$scorer_sha256  $workspace/research/temporal_contrastive/score_graph_context_division_development_probe.py" | sha256sum -c -
echo "$model_sha256  $workspace/research/temporal_contrastive/graph_context_division_model.py" | sha256sum -c -
echo "$inference_sha256  $workspace/research/temporal_contrastive/graph_context_division_inference.py" | sha256sum -c -
echo "$development_inventory_sha256  $development_inventory" | sha256sum -c -
echo "$initial_44_sha256  $initial_44" | sha256sum -c -
echo "$initial_6_sha256  $initial_6" | sha256sum -c -

# The graph experiment enters the single-GPU queue only after V27 is locally
# hash-verified. This preserves sequential ownership and prevents early cleanup.
while test ! -f "$prior_run_root/run.complete" || \
  test ! -f "$prior_run_root/harvest.verified"; do
  sleep 30
done
verified_prior_harvest
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -q '[0-9]'; do
  sleep 30
done
safe_remove_prior_results
rm -f -- "$prior_archive" "$prior_archive.sha256"

available_bytes="$(df --output=avail -B1 "$run_root" | tail -n 1 | tr -d ' ')"
if test "$available_bytes" -lt "$minimum_free_bytes"; then
  echo "Insufficient disk for graph-context ensemble: $available_bytes bytes" >&2
  exit 9
fi
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi A10G

mkdir "$data_parent"
tar -xzf "$archive" -C "$data_parent"
echo "$manifest_sha256  $data_root/graph_context_relational_patch_manifest.json" | sha256sum -c -
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
  --seeds 1013131,1113137,1213139,1313141 \
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
  --policy-contract all-selection-admitted-equal-rank-ensemble-v2 \
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
find competition-graph-context-division-frozen-ensemble-v2 \
  -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
  >"$output_root/SHA256SUMS"
tar -czf "$result_archive" competition-graph-context-division-frozen-ensemble-v2
sha256sum "$result_archive" >"$result_archive.sha256"
printf '%s\n' completed >"$run_root/run.complete"
if test "$status" -ne 0; then
  exit "$status"
fi
exit "$probe_status"
