#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/biohub
run_root=/home/ubuntu/biohub-peak-rank-detector-v1/temporal-min-local-snr-balanced-v23
input_root="$run_root/input"
result_parent="$run_root/results"
result_name=synthetic256-expanded-real-temporal-min-balanced-pu-faint-local-shape-multiscale-blob-global-safe-rank-peak-rank-v23
result_root="$result_parent/$result_name"
synthetic_root=/home/ubuntu/biohub-temporal-localizer-v2/data/synthetic256
v7_run_root=/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-v7
real_root="$v7_run_root/data/competition_real_localization_expanded_shards_v2"
# Preserve the basename recognized by the Biohub-only yield guard.
trainer="$input_root/train_expanded_real_faint_detector.py"
stable_model="$input_root/model_temporal_stable.py"
balanced_trainer="$input_root/balanced_safe_rank_base.py"
safe_trainer="$input_root/safe_rank_base.py"
safe_model="$input_root/model_safe_rank.py"
multiscale_model="$input_root/model_multiscale.py"
global_model="$input_root/model_global.py"
blob_model="$input_root/model_blob.py"
local_shape_trainer="$input_root/local_shape_base.py"
expanded_trainer="$input_root/expanded_real_base.py"
faint_trainer="$input_root/train_faint_cell_pu_detector.py"
base_trainer="$workspace/research/peak_rank_detection/train_synthetic_real_detector.py"
python_bin=/home/ubuntu/venv/bin/python
result_archive=/home/ubuntu/biohub-peak-rank-temporal-min-local-snr-balanced-v23-results.tar.gz

v21_run_root=/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-xl-balanced-v21
v21_result_parent="$v21_run_root/results"
v21_archive=/home/ubuntu/biohub-peak-rank-expanded-real-xl-balanced-v21-results.tar.gz
v21_ack="$v21_run_root/harvest.verified"

trainer_sha256=57479c473929aad7f2fdaca5c1322ea88e5f051006faf32c96d066ee4f365a79
stable_model_sha256=dc2c6ed737dfaec2a225c9242343ec76f8aec9e4e5daee2de76f8c4fae501245
balanced_trainer_sha256=854a305143803d2b10116f83d7cf75483d6f91f5930bda8931d95323cdffc802
safe_trainer_sha256=5570ffcb9734623f28db006b5d85452cc0ff7372eee05a6a65231b0104540f93
safe_model_sha256=89a31ed36d172347b1e0f4e13979bbaa60701f7a86defdc2b4140f191cbbfdd8
multiscale_model_sha256=56cff859acd3c6e3554fb91bb9fb3b4f7c5dca64dbab00bc2a22e8aa5483b9bf
global_model_sha256=ac3f861d7693e760a4da14264ce22cfd55f964ecf5404d16f5ed44bc43acfa82
blob_model_sha256=6728c620a1fd10d4e192f9d1e6c9859971bc335b1bedb50ef54e3d85bc684b3d
local_shape_trainer_sha256=1c3f4fd526aa2126f7b14403d91b4493efbc3dc0d48903992c84bb3aecd8654a
expanded_trainer_sha256=6eb0f506204c1f30fbee3ad9859215826a07fe7c8d78a2be868371233fbd2ebe
faint_trainer_sha256=6ef8092a89c1c01c536c690da573a10b49ce99bf83d4d3ddd69403741f808b0c
base_trainer_sha256=f64dc9da4b8babab667d3ab216a98759fded2e806032f5c33decc30e31028880
model_sha256=da1eae6cb7dac7f3175c0a672403b590a19ea0ff2bbfb9e07dff17fd7347c637
objectives_sha256=9340af271c96512f58192917b2763c386b7f32d9560e24ce266ada006cbfc787
synthetic_data_sha256=8e108e0650f3565ea4c05f0e747fb04d3c9fdc09922b2ef077f753bc203160be
inventory_sha256=a80c9028b21fdba1746bc2686dc5c64f0852e7954d1e758ba1357bcca5aa0edc
minimum_free_bytes=1500000000

safe_remove_v21_results() {
  local resolved
  resolved="$(realpath -m -- "$v21_result_parent")"
  test "$resolved" = /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-xl-balanced-v21/results
  rm -rf -- "$resolved"
}

verified_v21_harvest() {
  test -s "$v21_archive"
  test -s "$v21_archive.sha256"
  test -s "$v21_ack"
  sha256sum -c "$v21_archive.sha256" >/dev/null
  test "$(awk '{print $1}' "$v21_ack")" = "$(awk '{print $1}' "$v21_archive.sha256")"
}

exec 9>"$run_root/run.lock"
if ! flock -n 9; then
  echo "Another temporal-minimum local-SNR detector run holds the lock" >&2
  exit 3
fi
test ! -e "$result_parent"
test ! -e "$result_archive"
test ! -e "$result_archive.sha256"
echo "$trainer_sha256  $trainer" | sha256sum -c -
echo "$stable_model_sha256  $stable_model" | sha256sum -c -
echo "$balanced_trainer_sha256  $balanced_trainer" | sha256sum -c -
echo "$safe_trainer_sha256  $safe_trainer" | sha256sum -c -
echo "$safe_model_sha256  $safe_model" | sha256sum -c -
echo "$multiscale_model_sha256  $multiscale_model" | sha256sum -c -
echo "$global_model_sha256  $global_model" | sha256sum -c -
echo "$blob_model_sha256  $blob_model" | sha256sum -c -
echo "$local_shape_trainer_sha256  $local_shape_trainer" | sha256sum -c -
echo "$expanded_trainer_sha256  $expanded_trainer" | sha256sum -c -
echo "$faint_trainer_sha256  $faint_trainer" | sha256sum -c -
echo "$base_trainer_sha256  $base_trainer" | sha256sum -c -
echo "$model_sha256  $workspace/research/peak_rank_detection/model.py" | sha256sum -c -
echo "$objectives_sha256  $workspace/research/peak_rank_detection/objectives.py" | sha256sum -c -
echo "$synthetic_data_sha256  $workspace/research/synthetic_pretrain/data.py" | sha256sum -c -

# Preserve sequential ownership: start only after v21 is completed, copied,
# hash-verified locally, and explicitly acknowledged on this host.
while test ! -f "$v21_run_root/run.complete" || test ! -f "$v21_ack"; do sleep 30; done
verified_v21_harvest
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -q '[0-9]'; do
  sleep 30
done
safe_remove_v21_results
rm -f -- "$v21_archive" "$v21_archive.sha256"

manifest="$real_root/real_localization_shard_manifest.json"
test -f "$manifest"
"$python_bin" - "$manifest" "$inventory_sha256" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
counts = {
    role: sum(row.get("role") == role for row in payload.get("files", []))
    for role in ("optimization", "selection", "sealed_audit")
}
embryos = {
    embryo: sum(
        row.get("role") == "optimization" and row.get("embryo") == embryo
        for row in payload.get("files", [])
    )
    for embryo in ("44b6", "6bba")
}
if not (
    payload.get("run_id") == "competition-real-localization-expanded-shards-v2"
    and payload.get("inventory_sha256") == sys.argv[2]
    and counts == {"optimization": 480, "selection": 17, "sealed_audit": 14}
    and embryos == {"44b6": 150, "6bba": 330}
    and payload.get("competition_test_data_read") is False
    and payload.get("public_leaderboard_used_for_selection") is False
):
    raise RuntimeError("expanded replay manifest or embryo inventory changed")
PY
real_manifest_sha256="$(sha256sum "$manifest" | awk '{print $1}')"
available_bytes="$(df --output=avail -B1 "$run_root" | tail -n 1 | tr -d ' ')"
if test "$available_bytes" -lt "$minimum_free_bytes"; then
  echo "Insufficient disk for temporal-minimum local-SNR run: $available_bytes bytes" >&2
  exit 9
fi
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi A10G

mkdir -p "$result_parent"
cd "$run_root"
"$python_bin" -m py_compile "$trainer" "$stable_model" "$balanced_trainer" \
  "$safe_trainer" "$safe_model" "$multiscale_model" "$global_model" \
  "$blob_model" "$local_shape_trainer" "$expanded_trainer" "$faint_trainer" \
  "$base_trainer" "$workspace/research/peak_rank_detection/model.py" \
  "$workspace/research/peak_rank_detection/objectives.py"
set +e
PYTHONPATH="$input_root:$workspace" CUDA_VISIBLE_DEVICES=0 "$python_bin" "$trainer" \
  --synthetic-root "$synthetic_root" \
  --real-root "$real_root" \
  --real-manifest-sha256 "$real_manifest_sha256" \
  --output-root "$result_root" \
  --steps 5000 \
  --validation-every 1000 \
  --log-every 50 \
  --learning-rate 2e-4 \
  --minimum-learning-rate 2e-6 \
  --weight-decay 2e-4 \
  --ema-decay 0.999 \
  --real-frequency 2 \
  --widths 128,256,512,1024 \
  --depths 3,3,9,3 \
  --seed 12568331 \
  --max-wall-seconds 72000 \
  >"$run_root/training.log" 2>&1
status=$?
set -e
mkdir -p "$result_root"
cp "$run_root/training.log" "$result_root/training.log"
printf '%s\n' "$status" >"$result_root/training.exit-code"
cd "$result_parent"
find "$result_name" -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum >"$result_root/SHA256SUMS"
tar -czf "$result_archive" "$result_name"
sha256sum "$result_archive" >"$result_archive.sha256"
printf '%s\n' completed >"$run_root/run.complete"
exit "$status"
