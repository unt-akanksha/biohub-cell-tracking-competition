#!/usr/bin/env bash
set -euo pipefail

sweep_root=/home/ubuntu/biohub-results/competition-real-division-seed-sweep-v1
ensemble_root=/home/ubuntu/biohub-results/competition-real-division-seed-ensemble-v1
probe_root=/home/ubuntu/biohub-results/competition-real-division-seed-ensemble-probe-v1
controller_terminal="$probe_root/probe_controller_terminal.json"
poll_seconds=60
maximum_polls=720

poll=0
while ! test -f "$controller_terminal"; do
  poll=$((poll + 1))
  if test "$poll" -ge "$maximum_polls"; then
    echo "Timed out waiting for the seed-probe controller terminal" >&2
    exit 3
  fi
  sleep "$poll_seconds"
done

/home/ubuntu/venv/bin/python - "$sweep_root" "$ensemble_root" "$probe_root" <<'PY'
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import time

sweep_root = Path(sys.argv[1])
ensemble_root = Path(sys.argv[2])
probe_root = Path(sys.argv[3])
controller_path = probe_root / "probe_controller_terminal.json"
controller = json.loads(controller_path.read_text())
files: list[tuple[Path, str]] = []
included_names: set[str] = set()


def include(path: Path, archive_name: str) -> None:
    if path.is_file() and archive_name not in included_names:
        files.append((path, archive_name))
        included_names.add(archive_name)


include(controller_path, "probe/probe_controller_terminal.json")
for name in ("probe.exit-code", "probe.log", "seed_ensemble_probe.json", "SHA256SUMS"):
    include(probe_root / name, f"probe/{name}")
for name in ("seed_ensemble_terminal.json", "evaluation.exit-code", "evaluation.log", "SHA256SUMS"):
    include(ensemble_root / name, f"ensemble/{name}")
include(sweep_root / "seed_sweep_terminal.json", "sweep/seed_sweep_terminal.json")

members = []
probe_path = probe_root / "seed_ensemble_probe.json"
if probe_path.is_file():
    probe = json.loads(probe_path.read_text())
    members = probe.get("members", [])
    for member in members:
        seed = int(member["seed"])
        fold = str(member["fold"])
        prefix = f"sweep/seed-{seed}/{fold}"
        source = sweep_root / f"seed-{seed}" / fold
        include(source / "division_model.pt", f"{prefix}/division_model.pt")
        include(source / "training_config.json", f"{prefix}/training_config.json")
        include(
            sweep_root / f"seed-{seed}" / "real_division_gate_terminal.json",
            f"sweep/seed-{seed}/real_division_gate_terminal.json",
        )

records = []
for path, archive_name in files:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    records.append(
        {
            "path": archive_name,
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    )
manifest = {
    "schema_version": 1,
    "status": "complete",
    "run_id": "competition-real-division-seed-probe-harvest-v1",
    "probe_controller_status": controller.get("status"),
    "member_count": len(members),
    "members": members,
    "files": records,
    "competition_test_data_read": False,
    "public_leaderboard_used_for_selection": False,
    "submission_created": False,
    "authorized_for_submission": False,
    "created_unix_seconds": time.time(),
}
rendered = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
    info = tarfile.TarInfo("HARVEST_MANIFEST.json")
    info.size = len(rendered)
    info.mtime = int(time.time())
    archive.addfile(info, io.BytesIO(rendered))
    for path, archive_name in files:
        archive.add(path, arcname=archive_name, recursive=False)
PY
