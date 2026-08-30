#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/biohub
probe_inventory=/home/ubuntu/biohub-probe-data/analysis/competition-division-probe-inventory.json
probe_root=/home/ubuntu/biohub-probe-data/cache/competition-division-probe-frames-v1
sweep_root=/home/ubuntu/biohub-results/competition-real-division-seed-sweep-v1
ensemble_root=/home/ubuntu/biohub-results/competition-real-division-seed-ensemble-v1
output_root=/home/ubuntu/biohub-results/competition-real-division-seed-ensemble-probe-v1
python_bin=/home/ubuntu/venv/bin/python
ensemble_terminal="$ensemble_root/seed_ensemble_terminal.json"
poll_seconds=60
maximum_polls=720

test ! -e "$output_root"
for required in \
  "$workspace/research/temporal_contrastive/score_real_division_seed_ensemble_probe.py" \
  "$workspace/research/temporal_contrastive/evaluate_real_division_seed_ensemble.py" \
  "$probe_inventory" \
  "$probe_root/probe_cache_manifest.json" \
  "$python_bin"; do
  test -e "$required"
done

poll=0
while ! test -f "$ensemble_terminal"; do
  poll=$((poll + 1))
  if test "$poll" -ge "$maximum_polls"; then
    echo "Timed out waiting for the overnight seed ensemble" >&2
    exit 3
  fi
  sleep "$poll_seconds"
done

mkdir "$output_root"
set +e
"$python_bin" - "$ensemble_terminal" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text())
eligible = bool(
    payload.get("authorized_for_development_probe") is True
    and payload.get("status") in {
        "ensemble_eligible_for_development_probe",
        "stronger_individuals_eligible_for_development_probe",
    }
    and payload.get("competition_test_data_read") is False
    and payload.get("public_leaderboard_used_for_selection") is False
    and payload.get("submission_created") is False
)
raise SystemExit(0 if eligible else 4)
PY
eligible_status=$?
set -e

if test "$eligible_status" -eq 4; then
  "$python_bin" - "$ensemble_terminal" "$output_root/probe_controller_terminal.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

source = Path(sys.argv[1])
target = Path(sys.argv[2])
payload = {
    "schema_version": 1,
    "status": "skipped_after_selection_rejection",
    "run_id": "competition-real-division-seed-ensemble-probe-controller-v1",
    "selection_terminal_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "development_probe_opened": False,
    "competition_test_data_read": False,
    "submission_created": False,
    "authorized_for_submission": False,
}
target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  sha256sum "$output_root/probe_controller_terminal.json" >"$output_root/SHA256SUMS"
  exit 0
fi
test "$eligible_status" -eq 0

set +e
CUDA_VISIBLE_DEVICES=0 "$python_bin" \
  "$workspace/research/temporal_contrastive/score_real_division_seed_ensemble_probe.py" \
  --inventory "$probe_inventory" \
  --cache-root "$probe_root" \
  --sweep-root "$sweep_root" \
  --ensemble-terminal "$ensemble_terminal" \
  --output "$output_root/seed_ensemble_probe.json" \
  --batch-size 64 \
  >"$output_root/probe.log" 2>&1
probe_status=$?
set -e
printf '%s\n' "$probe_status" >"$output_root/probe.exit-code"

"$python_bin" - \
  "$ensemble_terminal" \
  "$output_root/seed_ensemble_probe.json" \
  "$output_root/probe_controller_terminal.json" \
  "$probe_status" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

selection = Path(sys.argv[1])
probe = Path(sys.argv[2])
target = Path(sys.argv[3])
exit_code = int(sys.argv[4])
payload = {
    "schema_version": 1,
    "status": "completed" if exit_code == 0 and probe.is_file() else "failed",
    "run_id": "competition-real-division-seed-ensemble-probe-controller-v1",
    "selection_terminal_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
    "development_probe_opened": probe.is_file(),
    "development_probe_sha256": (
        hashlib.sha256(probe.read_bytes()).hexdigest() if probe.is_file() else None
    ),
    "probe_exit_code": exit_code,
    "competition_test_data_read": False,
    "submission_created": False,
    "authorized_for_submission": False,
}
target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

sha256sum \
  "$output_root/probe_controller_terminal.json" \
  "$output_root/probe.exit-code" \
  "$output_root/probe.log" \
  "$output_root/seed_ensemble_probe.json" 2>/dev/null \
  >"$output_root/SHA256SUMS"
exit "$probe_status"
