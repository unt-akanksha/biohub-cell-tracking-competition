#!/usr/bin/env bash
set -euo pipefail

target=/home/ubuntu/biohub-peak-rank-detector-v1/temporal-min-local-snr-balanced-v23
stage="$target/scale-contract-repair-v1"
model="$target/input/model_temporal_stable.py"
runner="$target/run.sh"
receipt="$target/scale-contract-repair-v1.receipt"
old_model_sha256=dc2c6ed737dfaec2a225c9242343ec76f8aec9e4e5daee2de76f8c4fae501245
old_runner_sha256=7675d0f594fd24172d27d439576d97f46372761f4322de91396f2f05da2a4d5c
new_model_sha256=04f2c5d580279b7d035007ef0d5c422ad6ee43fec430f8f8a809d69858db172e
new_runner_sha256=c5948e48cfb0c07180cd1927eccaadc1540015a09d7d8543ec70b1f26ce88e7f

test ! -e "$receipt"
test -f "$stage/model_temporal_stable.py"
test -f "$stage/run.sh"
echo "$old_model_sha256  $model" | sha256sum -c -
echo "$old_runner_sha256  $runner" | sha256sum -c -
echo "$new_model_sha256  $stage/model_temporal_stable.py" | sha256sum -c -
echo "$new_runner_sha256  $stage/run.sh" | sha256sum -c -

# This repair is legal only while V23 is a dormant waiter.  Selection data and
# training output must still be absent, and its predecessor must be incomplete.
test ! -e "$target/results"
test ! -e "$target/training.log"
test ! -e "$target/run.complete"
test ! -e /home/ubuntu/biohub-peak-rank-temporal-min-local-snr-balanced-v23-results.tar.gz
test ! -e /home/ubuntu/biohub-peak-rank-temporal-min-local-snr-balanced-v23-results.tar.gz.sha256
test ! -e /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-xl-balanced-v21/run.complete

mapfile -t pids < <(pgrep -f "^bash $target/run.sh$")
test "${#pids[@]}" -eq 1
old_pid="${pids[0]}"
test "$(tr '\0' ' ' <"/proc/$old_pid/cmdline")" = "bash $target/run.sh "
kill -TERM "$old_pid"
for _ in $(seq 1 80); do
  if ! kill -0 "$old_pid" 2>/dev/null; then
    break
  fi
  sleep 0.5
done
! kill -0 "$old_pid" 2>/dev/null

# A child sleep can briefly inherit the lock after its parent exits.
for _ in $(seq 1 80); do
  if flock -n "$target/run.lock" true; then
    break
  fi
  sleep 0.5
done
flock -n "$target/run.lock" true

install -m 0644 "$stage/model_temporal_stable.py" "$model"
install -m 0755 "$stage/run.sh" "$runner"
echo "$new_model_sha256  $model" | sha256sum -c -
echo "$new_runner_sha256  $runner" | sha256sum -c -
bash -n "$runner"
cd "$target/input"
PYTHONPATH="$target/input:/home/ubuntu/biohub" /home/ubuntu/venv/bin/python - <<'PY'
from model_temporal_stable import (
    LOCAL_SNR_BANDS,
    TemporalMinimumLocalSnrSafeRankDetector,
)

assert LOCAL_SNR_BANDS == ((3, 9), (3, 11), (5, 13))
model = TemporalMinimumLocalSnrSafeRankDetector(
    widths=(128, 256, 512, 1024), depths=(3, 3, 9, 3)
)
assert sum(parameter.numel() for parameter in model.parameters()) == 83_812_614
PY

nohup bash "$runner" >>"$target/controller.log" 2>&1 < /dev/null &
new_pid=$!
sleep 2
kill -0 "$new_pid"
temporary="$receipt.partial"
printf '%s\n' \
  "status=scale_contract_repaired_waiting_for_v21" \
  "recorded_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "old_pid=$old_pid" \
  "new_pid=$new_pid" \
  "old_model_sha256=$old_model_sha256" \
  "new_model_sha256=$new_model_sha256" \
  "old_runner_sha256=$old_runner_sha256" \
  "new_runner_sha256=$new_runner_sha256" \
  "local_snr_bands=3-9,3-11,5-13" >"$temporary"
mv "$temporary" "$receipt"
cat "$receipt"
