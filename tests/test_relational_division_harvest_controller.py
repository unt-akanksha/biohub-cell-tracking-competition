from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "scripts/verify-antelume-relational-division-harvest.py"
SPEC = importlib.util.spec_from_file_location("relational_harvest_verifier", VERIFIER_PATH)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def build_harvest(path: Path, *, unbound: bool = False) -> None:
    aggregate = {
        "schema_version": 1,
        "status": "completed",
        "run_id": verifier.RUN_ID,
        "planned_model_count": 8,
        "completed_model_count": 8,
        "steps_per_model": 15_000,
        "selection_accepted_members": ["seed-1-init-1"],
        "independently_strong_members": ["seed-1-init-1"],
        "ensemble_eligible": False,
        "final_probe_opened": False,
        "competition_test_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }
    files = {
        f"{verifier.ROOT}/training.exit-code": b"0\n",
        f"{verifier.ROOT}/training.log": b"completed\n",
        f"{verifier.ROOT}/models/relational_division_sweep_terminal.json": (
            json.dumps(aggregate).encode()
        ),
    }
    sums = "".join(
        f"{hashlib.sha256(value).hexdigest()}  {name}\n"
        for name, value in sorted(files.items())
    ).encode()
    files[f"{verifier.ROOT}/SHA256SUMS"] = sums
    if unbound:
        files[f"{verifier.ROOT}/unbound.txt"] = b"no"
    with tarfile.open(path, "w:gz") as archive:
        for name, value in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(value)
            archive.addfile(info, io.BytesIO(value))


def test_remote_run_packages_relative_hash_bound_results() -> None:
    text = (ROOT / "scripts/run-antelume-relational-division-sweep-v1.sh").read_text()

    assert "cd /home/ubuntu/biohub-results" in text
    assert "! -name SHA256SUMS" in text
    assert "competition-relational-division-sweep-v1 -type f" in text
    assert "biohub-relational-division-sweep-v1-results.tar.gz.sha256" in text


def test_harvester_holds_one_ssh_session_and_does_not_submit() -> None:
    text = (ROOT / "scripts/wait-harvest-antelume-relational-division-sweep.ps1").read_text()

    assert "deployed_and_queued" in text
    assert "ServerAliveInterval=60" in text
    assert "cat \"$archive\"" in text
    assert "Start-Process -FilePath ssh.exe" in text
    assert "kaggle competitions submit" not in text
    assert 'competition_submission_performed"] = $false' in text


def test_harvest_verifier_recovers_terminal_evidence(tmp_path: Path) -> None:
    archive = tmp_path / "results.tar.gz"
    build_harvest(archive)

    result = verifier.verify_archive(archive)

    assert result["status"] == "harvest_verified"
    assert result["training_exit_code"] == 0
    assert result["completed_model_count"] == 8
    assert result["independently_strong_members"] == ["seed-1-init-1"]


def test_harvest_verifier_rejects_unbound_file(tmp_path: Path) -> None:
    archive = tmp_path / "results.tar.gz"
    build_harvest(archive, unbound=True)

    with pytest.raises(ValueError, match="unbound"):
        verifier.verify_archive(archive)
