from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

from research.peak_rank_detection import train_faint_cell_pu_detector as trainer


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run-antelume-peak-rank-faint-pu-v4.sh"
VERIFIER = ROOT / "scripts/verify-antelume-peak-rank-faint-pu-v4-harvest.py"
HARVESTER = ROOT / "scripts/wait-harvest-antelume-peak-rank-faint-pu-v4.ps1"
SPEC = importlib.util.spec_from_file_location("faint_pu_harvest", VERIFIER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_archive(path: Path) -> None:
    root = "synthetic256-real-conservative-pu-faint-temporal-peak-rank-v4"
    checkpoint = b"faint temporal capacity weights"
    terminal = {
        "schema_version": 1,
        "run_id": root,
        "status": "accepted_at_audit",
        "seed": 3_601_079,
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
        f"{root}/training.log": b"faint-PU evidence\n",
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


def test_runner_is_sequential_hash_bound_and_cleanup_guarded() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for required in (
        'nucverse_complete="$nucverse_run_root/run.complete"',
        'v3_ack="$v3_run_root/harvest.verified"',
        'nucverse_ack="$nucverse_run_root/harvest.verified"',
        "verified_local_harvest",
        "safe_remove_biohub_tree",
        "minimum_free_bytes=1300000000",
        "--steps 2000",
        "--widths 128,256,512,1024",
        "--seed 3601079",
        "while nvidia-smi --query-compute-apps=pid",
    ):
        assert required in source
    assert "6ef8092a89c1c01c536c690da573a10b49ce99bf83d4d3ddd69403741f808b0c" in source
    assert "systemctl" not in source
    assert "pkill" not in source
    assert "killall" not in source
    assert "rsna" not in source.lower()


def test_archive_verifier_binds_faint_capacity_contract(tmp_path: Path) -> None:
    archive = tmp_path / "faint-pu.tar.gz"
    make_archive(archive)
    report = MODULE.verify_faint_pu(archive)

    assert report["accepted_for_kaggle_validation"] is True
    assert report["parameter_count"] == 66_977_670
    assert report["variant"] == "capacity_conservative_pu_depth_and_temporal_fading"
    assert trainer.RUN_ID.endswith("faint-temporal-peak-rank-v4")


def test_harvester_is_bounded_and_non_submitting() -> None:
    source = HARVESTER.read_text(encoding="utf-8")
    assert "MaximumPolls" in source
    assert "ServerAliveInterval=60" in source
    assert "accepted_for_kaggle_validation" in source
    assert "kaggle competitions submit" not in source
