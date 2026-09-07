from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts/verify-antelume-nucverse3d-compatibility-v1-harvest.py"
HARVESTER = ROOT / "scripts/wait-harvest-antelume-nucverse3d-compatibility-v1.ps1"
SPEC = importlib.util.spec_from_file_location("nucverse_harvest", VERIFIER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def receipt(phase: str, passed: bool) -> dict:
    return {
        "schema_version": 1,
        "run_id": MODULE.RUN_ID,
        "phase": phase,
        "status": "complete",
        "compatibility_passed": passed,
        "summary": {"examples": 16, "points": 32},
        "source_commit": MODULE.EXPECTED_SOURCE_COMMIT,
        "checkpoint_sha256": "1c4e288350b1a86d361359cdd02151744d418e9fcf7ebe588c1c77e2cd8bbd67",
        "onnx_sha256": MODULE.EXPECTED_ONNX_SHA256,
        "manifest_sha256": MODULE.EXPECTED_MANIFEST_SHA256,
        "model_parameters": 40_458_005,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }


def make_archive(path: Path, *, optimization_passed: bool, selection_passed: bool) -> None:
    files = {
        "results/optimization-screen.json": json.dumps(
            receipt("optimization", optimization_passed)
        ).encode(),
        "results/optimization.exit-code": b"0\n",
        "results/selection.exit-code": b"0\n",
        "results/screen.exit-code": b"0\n",
        "results/screen.log": b"clean compatibility evidence\n",
    }
    if optimization_passed:
        files["results/selection-screen.json"] = json.dumps(
            receipt("selection", selection_passed)
        ).encode()
    files["results/SHA256SUMS"] = "".join(
        f"{hashlib.sha256(payload).hexdigest()}  {name}\n"
        for name, payload in sorted(files.items())
    ).encode()
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_verifier_accepts_gated_selection(tmp_path: Path) -> None:
    archive = tmp_path / "nucverse.tar.gz"
    make_archive(archive, optimization_passed=True, selection_passed=True)
    report = MODULE.verify(archive)

    assert report["optimization_passed"] is True
    assert report["selection_opened"] is True
    assert report["selection_passed"] is True
    assert report["accepted_for_detector_integration"] is True
    assert report["authorized_for_submission"] is False


def test_verifier_keeps_selection_closed_after_optimization_failure(tmp_path: Path) -> None:
    archive = tmp_path / "nucverse.tar.gz"
    make_archive(archive, optimization_passed=False, selection_passed=False)
    report = MODULE.verify(archive)

    assert report["selection_opened"] is False
    assert report["accepted_for_detector_integration"] is False


def test_harvester_is_bounded_acknowledged_and_non_submitting() -> None:
    source = HARVESTER.read_text(encoding="utf-8")
    assert "MaximumPolls" in source
    assert "ServerAliveInterval=60" in source
    assert "harvest.verified" in source
    assert "accepted_for_detector_integration" in source
    assert "kaggle competitions submit" not in source

