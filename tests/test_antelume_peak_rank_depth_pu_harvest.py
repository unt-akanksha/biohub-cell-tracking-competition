from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify-antelume-peak-rank-depth-pu-v2-harvest.py"
SPEC = importlib.util.spec_from_file_location("depth_pu_harvest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_archive(path: Path) -> None:
    root = "synthetic256-real-conservative-pu-depth-robust-peak-rank-v2"
    checkpoint = b"depth robust independent learned weights"
    terminal = {
        "schema_version": 1,
        "run_id": root,
        "status": "accepted_at_audit",
        "seed": 1_407_733,
        "completed_steps": 3_000,
        "best_step": 2_000,
        "parameter_count": 38_381_478,
        "widths": [96, 192, 384, 768],
        "depths": [3, 3, 9, 3],
        "selection_passed": True,
        "audit_opened": True,
        "audit_passed": True,
        "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
        "last_checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
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
        f"{root}/training.exit-code": b"0\n",
        f"{root}/training.log": b"depth-pu training evidence\n",
        f"{root}/last_peak_rank_detector.pt": checkpoint,
        f"{root}/peak_rank_detector.pt": checkpoint,
    }
    files[f"{root}/SHA256SUMS"] = "".join(
        f"{hashlib.sha256(payload).hexdigest()}  {name}\n"
        for name, payload in sorted(files.items())
    ).encode()
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_depth_pu_archive_uses_distinct_root_and_seed(tmp_path: Path) -> None:
    archive = tmp_path / "depth-pu.tar.gz"
    make_archive(archive)
    report = MODULE.verify_depth_pu(archive)
    assert report["accepted_for_kaggle_validation"] is True
    assert report["variant"] == "conservative_positive_unlabeled_depth_attenuation"


def test_depth_pu_harvester_is_bounded_and_non_submitting() -> None:
    source = (
        ROOT / "scripts" / "wait-harvest-antelume-peak-rank-depth-pu-v2.ps1"
    ).read_text(encoding="utf-8")
    assert "MaximumPolls" in source
    assert "ServerAliveInterval=60" in source
    assert "accepted_for_kaggle_validation" in source
    assert "harvest.verified" in source
    assert "Remote v2 harvest acknowledgement failed" in source
    assert "kaggle competitions submit" not in source
