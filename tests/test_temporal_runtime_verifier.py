from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.temporal_contrastive.verify_runtime import verify_runtime


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_fixture(tmp_path: Path) -> Path:
    source = tmp_path / "model.py"
    source.write_text("value = 1\n", encoding="utf-8")
    manifest = {
        "files": {
            source.name: {"bytes": source.stat().st_size, "sha256": digest(source)}
        },
        "integrity": {
            "required_gpu_count": 2,
            "competition_submission_command_included": False,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "maximum_submission_inference_seconds": 36_000,
            "minimum_kaggle_finalization_reserve_seconds": 7_200,
            "required_kaggle_machine_shape": "NvidiaTeslaT4",
            "submission_internet_enabled": False,
            "timed_out_worker_termination_grace_seconds": 15,
            "predeclared_trackastra_control_allowed": True,
        },
    }
    (tmp_path / "SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return tmp_path


def test_runtime_verifier_checks_every_declared_file(tmp_path: Path) -> None:
    root = runtime_fixture(tmp_path)

    result = verify_runtime(root)

    assert result["status"] == "verified"
    assert result["files_checked"] == 1
    assert result["required_gpu_count"] == 2


def test_runtime_verifier_rejects_mutation(tmp_path: Path) -> None:
    root = runtime_fixture(tmp_path)
    (root / "model.py").write_text("value = 2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="hash changed"):
        verify_runtime(root)
