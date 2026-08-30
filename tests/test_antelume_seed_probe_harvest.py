from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import runpy
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts/verify-antelume-seed-probe-harvest.py"
MODULE = runpy.run_path(str(VERIFIER))
REMOTE = ROOT / "scripts/wait-package-antelume-seed-probe-harvest-v1.sh"
LOCAL = ROOT / "scripts/wait-harvest-antelume-seed-probe.ps1"
POST_HARVEST = ROOT / "scripts/wait-package-strong-member-consensus-runtime.ps1"


def make_archive(path: Path, *, tamper: bool = False) -> None:
    payload = b"selection-evidence"
    record = {
        "path": "ensemble/seed_ensemble_terminal.json",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "run_id": MODULE["RUN_ID"],
        "probe_controller_status": "skipped_after_selection_rejection",
        "member_count": 0,
        "members": [],
        "files": [record],
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }
    rendered = json.dumps(manifest).encode()
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("HARVEST_MANIFEST.json")
        info.size = len(rendered)
        archive.addfile(info, io.BytesIO(rendered))
        member = tarfile.TarInfo(record["path"])
        observed = payload + (b"tamper" if tamper else b"")
        member.size = len(observed)
        archive.addfile(member, io.BytesIO(observed))


def test_harvest_verifier_checks_every_member_without_extracting(tmp_path: Path) -> None:
    archive = tmp_path / "harvest.tar.gz"
    make_archive(archive)

    result = MODULE["verify_harvest"](archive)

    assert result["status"] == "verified"
    assert result["member_count"] == 0
    assert result["file_count"] == 1


def test_harvest_extractor_writes_only_after_full_verification(tmp_path: Path) -> None:
    archive = tmp_path / "harvest.tar.gz"
    destination = tmp_path / "extracted"
    make_archive(archive)

    result = MODULE["extract_verified_harvest"](archive, destination)

    assert result["status"] == "verified"
    assert (destination / "ensemble/seed_ensemble_terminal.json").read_bytes() == (
        b"selection-evidence"
    )
    assert (destination / "HARVEST_MANIFEST.json").is_file()


def test_harvest_extractor_does_not_create_destination_for_tampered_archive(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "harvest.tar.gz"
    destination = tmp_path / "extracted"
    make_archive(archive, tamper=True)

    with pytest.raises(ValueError, match="changed"):
        MODULE["extract_verified_harvest"](archive, destination)

    assert not destination.exists()


def test_harvest_verifier_rejects_tampered_stream(tmp_path: Path) -> None:
    archive = tmp_path / "harvest.tar.gz"
    make_archive(archive, tamper=True)

    with pytest.raises(ValueError, match="changed"):
        MODULE["verify_harvest"](archive)


def test_remote_and_local_harvest_controllers_are_submission_free() -> None:
    remote = REMOTE.read_text(encoding="utf-8")
    local = LOCAL.read_text(encoding="utf-8")

    assert "while ! test -f \"$controller_terminal\"" in remote
    assert "tarfile.open(fileobj=sys.stdout.buffer" in remote
    assert "competition_test_data_read\": False" in remote
    assert "Start-Process -FilePath ssh.exe" in local
    assert "verify-antelume-seed-probe-harvest.py" in local
    assert "kaggle competitions submit" not in remote + local


def test_post_harvest_controller_is_evidence_gated_and_submission_free() -> None:
    controller = POST_HARVEST.read_text(encoding="utf-8")

    assert 'probe_controller_status -ne "completed"' in controller
    assert "verify-ranked-consensus-development-baseline.py" in controller
    assert "evaluate_ranked_consensus_division_recovery.py" in controller
    assert "build-strong-member-consensus-division-dataset.py" in controller
    assert 'status = "development_rejected"' in controller
    assert 'status = "runtime_packaged"' in controller
    assert "kaggle datasets" not in controller
    assert "kaggle kernels" not in controller
    assert "kaggle competitions submit" not in controller
