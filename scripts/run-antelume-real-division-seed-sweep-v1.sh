#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/biohub
data_root=/home/ubuntu/biohub-real-division-data/biohub_real_division_patches_v1
initial_44=/home/ubuntu/biohub-results/competition-real-division-gate-head-v1/target_44b6/division_model.pt
initial_6=/home/ubuntu/biohub-results/competition-real-division-gate-head-v1/target_6bba/division_model.pt
output_root=/home/ubuntu/biohub-results/competition-real-division-seed-sweep-v1
python_bin=/home/ubuntu/venv/bin/python

initial_44_sha256=ad369d5c122a13a8763a92a94ac3e67548cd19628b0e3500fc8db5cae1201133
initial_6_sha256=cf2ba21a8b696216e4ae4bc59a2531c44f0ca47fc1d090d3fffc441f71607b75
manifest_sha256=943717472518b917175312bd4ada9e12660d31d3ebf0bf7afd5672ab40442e1e
seeds=(205043 305047 405049 505051 605057 705071 805073 905081)

exec 9>/home/ubuntu/biohub-real-division-seed-sweep-v1.lock
if ! flock -n 9; then
  echo "Another real-division seed sweep holds the lock" >&2
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
mkdir -p "$output_root"

cd "$workspace"
for seed in "${seeds[@]}"; do
  run_root="$output_root/seed-$seed"
  log_path="$output_root/seed-$seed.log"
  set +e
  CUDA_VISIBLE_DEVICES=0 "$python_bin" \
    research/temporal_contrastive/train_real_division_gate.py \
    --data-root "$data_root" \
    --target-44b6-initial-model "$initial_44" \
    --target-6bba-initial-model "$initial_6" \
    --output-root "$run_root" \
    --train-mode focused \
    --seed "$seed" \
    --steps 50000 \
    --batch-size 48 \
    --validation-batch-size 64 \
    --validation-every 500 \
    --log-every 250 \
    --learning-rate 2e-5 \
    --minimum-learning-rate 2e-7 \
    --weight-decay 1e-4 \
    --ema-decay 0.995 \
    >"$log_path" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" >"$output_root/seed-$seed.exit-code"
  find "$run_root" -type f -print0 2>/dev/null \
    | sort -z \
    | xargs -0 -r sha256sum \
    >"$output_root/seed-$seed.sha256"
done

"$python_bin" - "$output_root" "${seeds[@]}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
seeds = [int(value) for value in sys.argv[2:]]
runs = []
for seed in seeds:
    exit_code = int((root / f"seed-{seed}.exit-code").read_text().strip())
    terminal_path = root / f"seed-{seed}" / "real_division_gate_terminal.json"
    terminal = json.loads(terminal_path.read_text()) if terminal_path.is_file() else None
    runs.append(
        {
            "seed": seed,
            "exit_code": exit_code,
            "terminal_exists": terminal is not None,
            "status": terminal.get("status") if terminal else "missing",
            "selection_average_precision": (
                terminal.get("ensemble_selection", {}).get("average_precision")
                if terminal
                else None
            ),
            "final_probe_opened": terminal.get("final_probe_opened") if terminal else None,
        }
    )
payload = {
    "schema_version": 1,
    "status": "completed",
    "run_id": "competition-real-division-seed-sweep-v1",
    "architecture_parameters_per_model": 46_386_607,
    "trainable_parameters_per_model": 25_178_047,
    "planned_model_count": 16,
    "steps_per_model": 50_000,
    "seeds": seeds,
    "runs": runs,
    "competition_train_data_read": True,
    "competition_test_data_read": False,
    "final_probe_opened": False,
    "public_leaderboard_used_for_selection": False,
    "submission_created": False,
    "authorized_for_submission": False,
}
(root / "seed_sweep_terminal.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

sha256sum "$output_root/seed_sweep_terminal.json" >"$output_root/SHA256SUMS"
