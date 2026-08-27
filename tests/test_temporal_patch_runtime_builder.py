from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-temporal-patch-runtime.py"
TARGET = ROOT / ".biohub" / "staging" / "biohub-temporal-patch-runtime-v1"


def test_runtime_builder_hashes_complete_two_gpu_appearance_pipeline() -> None:
    subprocess.run([sys.executable, str(BUILDER), "--replace"], check=True)
    manifest_path = TARGET / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["integrity"] == {
        "required_gpu_count": 2,
        "opened_processed_acceptance_stems_excluded_from_training": True,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "competition_submission_command_included": False,
    }
    required = {
        "train_dual_fold_patch.py",
        "calibrate_dual_fold_blend.py",
        "dual_fold_appearance_processed_acceptance.py",
        "dual_fold_appearance_submission.py",
        "submission_sharding.py",
        "trackastra_source/trackastra/model/model.py",
    }
    assert required.issubset(manifest["files"])
    for name, row in manifest["files"].items():
        path = TARGET / name
        assert path.stat().st_size == row["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    combined = "\n".join(
        (TARGET / name).read_text(encoding="utf-8")
        for name in required
        if name.endswith(".py")
    )
    assert "device_count() != 2" in combined
    assert "kaggle competitions submit" not in combined
