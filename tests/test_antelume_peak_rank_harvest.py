import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify-antelume-peak-rank-detector-harvest.py"
SPEC = importlib.util.spec_from_file_location("peak_rank_harvest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _archive(path: Path, *, accepted: bool = True, tamper: bool = False) -> None:
    root = "synthetic256-real-positive-temporal-peak-rank-v1"
    checkpoint = b"independent learned weights"
    last_checkpoint = b"recoverable validation weights"
    terminal = {
        "schema_version": 1,
        "run_id": root,
        "status": "accepted_at_audit" if accepted else "rejected_at_selection",
        "seed": 1_041_729,
        "completed_steps": 3_000,
        "best_step": 2_000 if accepted else 0,
        "parameter_count": 38_381_478,
        "widths": [96, 192, 384, 768],
        "depths": [3, 3, 9, 3],
        "selection_passed": accepted,
        "audit_opened": accepted,
        "audit_passed": accepted,
        "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest() if accepted else None,
        "last_checkpoint_sha256": hashlib.sha256(last_checkpoint).hexdigest(),
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_predictions_read": False,
        "public_notebook_weights_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }
    files = {
        f"{root}/terminal.json": json.dumps(terminal).encode(),
        f"{root}/training.exit-code": b"0\n" if accepted else b"1\n",
        f"{root}/training.log": b"training evidence\n",
        f"{root}/last_peak_rank_detector.pt": last_checkpoint,
    }
    if accepted:
        files[f"{root}/peak_rank_detector.pt"] = checkpoint
    sums = "".join(
        f"{hashlib.sha256(payload).hexdigest()}  {name}\n"
        for name, payload in sorted(files.items())
    ).encode()
    files[f"{root}/SHA256SUMS"] = sums
    if tamper:
        files[f"{root}/training.log"] = b"altered after checksums\n"
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_accepts_hash_bound_audited_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "results.tar.gz"
    _archive(path)

    report = MODULE.verify(path)
    assert report["status"] == "verified"
    assert report["accepted_for_kaggle_validation"] is True
    assert report["checkpoint_sha256"] is not None


def test_preserves_clean_selection_rejection(tmp_path: Path) -> None:
    path = tmp_path / "results.tar.gz"
    _archive(path, accepted=False)

    report = MODULE.verify(path)
    assert report["accepted_for_kaggle_validation"] is False
    assert report["audit_opened"] is False


def test_rejects_checksum_tampering(tmp_path: Path) -> None:
    path = tmp_path / "results.tar.gz"
    _archive(path, tamper=True)
    with pytest.raises(ValueError, match="checksum mismatch"):
        MODULE.verify(path)

    # An unsafe traversal member must also fail before
    # any extraction is attempted.
    bad = tmp_path / "unsafe.tar.gz"
    with tarfile.open(bad, "w:gz") as archive:
        payload = b"bad"
        info = tarfile.TarInfo("../escape")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="unsafe"):
        MODULE.verify(bad)


def test_harvest_controller_is_bounded_and_non_submitting() -> None:
    source = (
        ROOT / "scripts" / "wait-harvest-antelume-peak-rank-detector-v1.ps1"
    ).read_text(encoding="utf-8")
    assert "MaximumPolls" in source
    assert "ServerAliveInterval=60" in source
    assert "accepted_for_kaggle_validation" in source
    assert "competition_submission_performed" in source
    assert "harvest.verified" in source
    assert "Remote v1 harvest acknowledgement failed" in source
    assert "kaggle competitions submit" not in source
