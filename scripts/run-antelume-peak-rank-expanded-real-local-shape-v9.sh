#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/biohub
run_root=/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-local-shape-v9
input_root="$run_root/input"
result_parent="$run_root/results"
result_root="$result_parent/synthetic256-expanded-real-pu-faint-local-shape-peak-rank-v9"
synthetic_root=/home/ubuntu/biohub-temporal-localizer-v2/data/synthetic256
v7_run_root=/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-v7
real_root="$v7_run_root/data/competition_real_localization_expanded_shards_v2"
# The executable copy keeps the existing guard-classified basename; its hash
# below binds it to the local-shape source. Supporting modules have distinct
# flat names and cannot shadow the executable.
trainer="$input_root/train_expanded_real_faint_detector.py"
expanded_trainer="$input_root/expanded_real_base.py"
faint_trainer="$input_root/train_faint_cell_pu_detector.py"
base_trainer="$workspace/research/peak_rank_detection/train_synthetic_real_detector.py"
python_bin=/home/ubuntu/venv/bin/python
result_archive=/home/ubuntu/biohub-peak-rank-expanded-real-local-shape-v9-results.tar.gz

v7_result_parent="$v7_run_root/results"
v7_archive=/home/ubuntu/biohub-peak-rank-expanded-real-faint-v7-results.tar.gz
v7_ack="$v7_run_root/harvest.verified"

trainer_sha256=1c3f4fd526aa2126f7b14403d91b4493efbc3dc0d48903992c84bb3aecd8654a
expanded_trainer_sha256=6eb0f506204c1f30fbee3ad9859215826a07fe7c8d78a2be868371233fbd2ebe
faint_trainer_sha256=6ef8092a89c1c01c536c690da573a10b49ce99bf83d4d3ddd69403741f808b0c
base_trainer_sha256=f64dc9da4b8babab667d3ab216a98759fded2e806032f5c33decc30e31028880
model_sha256=da1eae6cb7dac7f3175c0a672403b590a19ea0ff2bbfb9e07dff17fd7347c637
objectives_sha256=9340af271c96512f58192917b2763c386b7f32d9560e24ce266ada006cbfc787
synthetic_data_sha256=8e108e0650f3565ea4c05f0e747fb04d3c9fdc09922b2ef077f753bc203160be
inventory_sha256=a80c9028b21fdba1746bc2686dc5c64f0852e7954d1e758ba1357bcca5aa0edc
minimum_free_bytes=1300000000

safe_remove_v7_results() {
  local resolved
  resolved="$(realpath -m -- "$v7_result_parent")"
  test "$resolved" = /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-v7/results
  rm -rf -- "$resolved"
}

verified_v7_harvest() {
  test -s "$v7_archive"
  test -s "$v7_archive.sha256"
  test -s "$v7_ack"
  sha256sum -c "$v7_archive.sha256" >/dev/null
  test "$(awk '{print $1}' "$v7_ack")" = "$(awk '{print $1}' "$v7_archive.sha256")"
}

exec 9>"$run_root/run.lock"
if ! flock -n 9; then
  echo "Another expanded-real local-shape detector run holds the lock" >&2
  exit 3
fi
test ! -e "$result_parent"
test ! -e "$result_archive"
test ! -e "$result_archive.sha256"
echo "$trainer_sha256  $trainer" | sha256sum -c -
echo "$expanded_trainer_sha256  $expanded_trainer" | sha256sum -c -
echo "$faint_trainer_sha256  $faint_trainer" | sha256sum -c -
echo "$base_trainer_sha256  $base_trainer" | sha256sum -c -
echo "$model_sha256  $workspace/research/peak_rank_detection/model.py" | sha256sum -c -
echo "$objectives_sha256  $workspace/research/peak_rank_detection/objectives.py" | sha256sum -c -
echo "$synthetic_data_sha256  $workspace/research/synthetic_pretrain/data.py" | sha256sum -c -

# Reuse only v7's immutable expanded-real input after its result has been
# independently harvested.  The result copy may then be reclaimed; the input
# data remains read-only for this member.  Unrelated project paths are absent.
while test ! -f "$v7_run_root/run.complete" || test ! -f "$v7_ack"; do sleep 30; done
verified_v7_harvest
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -q '[0-9]'; do
  sleep 30
done
safe_remove_v7_results
rm -f -- "$v7_archive" "$v7_archive.sha256"

manifest="$real_root/real_localization_shard_manifest.json"
test -f "$manifest"
"$python_bin" - "$manifest" "$inventory_sha256" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
counts = {
    role: sum(row.get("role") == role for row in payload.get("files", []))
    for role in ("optimization", "selection", "sealed_audit")
}
if not (
    payload.get("run_id") == "competition-real-localization-expanded-shards-v2"
    and payload.get("inventory_sha256") == sys.argv[2]
    and counts == {"optimization": 480, "selection": 17, "sealed_audit": 14}
    and payload.get("competition_test_data_read") is False
    and payload.get("public_leaderboard_used_for_selection") is False
):
    raise RuntimeError("expanded replay manifest contract changed")
PY
real_manifest_sha256="$(sha256sum "$manifest" | awk '{print $1}')"
available_bytes="$(df --output=avail -B1 "$run_root" | tail -n 1 | tr -d ' ')"
if test "$available_bytes" -lt "$minimum_free_bytes"; then
  echo "Insufficient disk for local-shape capacity run: $available_bytes bytes" >&2
  exit 9
fi
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi A10G

mkdir -p "$result_parent"
cd "$run_root"
"$python_bin" -m py_compile "$trainer" "$expanded_trainer" "$faint_trainer" \
  "$base_trainer" "$workspace/research/peak_rank_detection/model.py" \
  "$workspace/research/peak_rank_detection/objectives.py"
set +e
PYTHONPATH="$input_root:$workspace" CUDA_VISIBLE_DEVICES=0 "$python_bin" "$trainer" \
  --synthetic-root "$synthetic_root" \
  --real-root "$real_root" \
  --real-manifest-sha256 "$real_manifest_sha256" \
  --output-root "$result_root" \
  --steps 3000 \
  --validation-every 1000 \
  --log-every 50 \
  --learning-rate 2e-4 \
  --minimum-learning-rate 2e-6 \
  --weight-decay 2e-4 \
  --ema-decay 0.999 \
  --real-frequency 2 \
  --widths 128,256,512,1024 \
  --depths 3,3,9,3 \
  --seed 5803219 \
  --max-wall-seconds 36000 \
  >"$run_root/training.log" 2>&1
status=$?
set -e
mkdir -p "$result_root"
cp "$run_root/training.log" "$result_root/training.log"
printf '%s\n' "$status" >"$result_root/training.exit-code"
cd "$result_parent"
find synthetic256-expanded-real-pu-faint-local-shape-peak-rank-v9 \
  -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum >"$result_root/SHA256SUMS"
tar -czf "$result_archive" synthetic256-expanded-real-pu-faint-local-shape-peak-rank-v9
sha256sum "$result_archive" >"$result_archive.sha256"
printf '%s\n' completed >"$run_root/run.complete"
exit "$status"
