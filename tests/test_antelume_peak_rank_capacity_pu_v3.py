from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

from research.peak_rank_detection import train_capacity_depth_robust_pu_detector as trainer
from research.peak_rank_detection.model import TemporalPeakRankDetector, count_parameters


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run-antelume-peak-rank-capacity-pu-v3.sh"
VERIFIER = ROOT / "scripts" / "verify-antelume-peak-rank-capacity-pu-v3-harvest.py"
HARVESTER = ROOT / "scripts" / "wait-harvest-antelume-peak-rank-capacity-pu-v3.ps1"
SPEC = importlib.util.spec_from_file_location("capacity_pu_harvest", VERIFIER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_archive(path: Path) -> None:
    root = "synthetic256-real-conservative-pu-depth-robust-capacity-peak-rank-v3"
    checkpoint = b"capacity scaled independent learned weights"
    terminal = {
        "schema_version": 1,
        "run_id": root,
        "status": "accepted_at_audit",
        "seed": 2_607_157,
        "completed_steps": 2_000,
        "best_step": 2_000,
        "parameter_count": 66_977_670,
        "widths": [128, 256, 512, 1024],
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
        f"{root}/training.log": b"capacity-pu training evidence\n",
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


def test_capacity_model_and_training_contract_are_distinct() -> None:
    model = TemporalPeakRankDetector(
        widths=(128, 256, 512, 1024), depths=(3, 3, 9, 3)
    )
    assert count_parameters(model) == 66_977_670
    assert trainer.RUN_ID.endswith("capacity-peak-rank-v3")
    assert trainer.MINIMUM_DEPTH_FACTOR == 0.25


def test_capacity_runner_is_sequential_hash_bound_and_disk_guarded() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for required in (
        "depth-pu-v2/run.complete",
        "harvest.verified",
        "verified_local_harvest",
        "safe_remove_biohub_tree",
        "minimum_free_bytes=1300000000",
        "--steps 2000",
        "--validation-every 1000",
        "--widths 128,256,512,1024",
        "--seed 2607157",
        "while nvidia-smi --query-compute-apps=pid",
    ):
        assert required in source
    assert "/home/ubuntu/biohub-peak-rank-detector-v1/results" in source
    assert "/home/ubuntu/biohub-peak-rank-detector-v1/depth-pu-v2/results" in source
    assert "/home/ubuntu/rsna" not in source.lower()
    for forbidden in ("kill -", "pkill", "killall", "systemctl"):
        assert forbidden not in source


def test_capacity_archive_verifier_binds_size_schedule_and_seed(tmp_path: Path) -> None:
    archive = tmp_path / "capacity-pu.tar.gz"
    make_archive(archive)
    report = MODULE.verify_capacity_pu(archive)
    assert report["accepted_for_kaggle_validation"] is True
    assert report["parameter_count"] == 66_977_670
    assert report["variant"] == "capacity_scaled_conservative_pu_depth_attenuation"


def test_capacity_harvester_is_bounded_and_non_submitting() -> None:
    source = HARVESTER.read_text(encoding="utf-8")
    assert "MaximumPolls" in source
    assert "ServerAliveInterval=60" in source
    assert "accepted_for_kaggle_validation" in source
    assert "kaggle competitions submit" not in source
