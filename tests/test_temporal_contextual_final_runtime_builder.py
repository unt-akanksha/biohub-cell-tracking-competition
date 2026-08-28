from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-temporal-contextual-final-runtime.py"
TARGET = (
    ROOT
    / ".biohub"
    / "staging"
    / "biohub-temporal-contextual-final-runtime-v1"
)


def test_builder_clones_frozen_runtime_and_changes_only_final_inference() -> None:
    subprocess.run([sys.executable, str(BUILDER), "--replace"], check=True)
    manifest = json.loads((TARGET / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    metadata = json.loads((TARGET / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert manifest["parent_runtime"] == {
        "dataset_id": "indarkarhana/biohub-temporal-contextual-transfer-runtime-v1",
        "dataset_version": 4,
        "manifest_sha256": "cbe5fe27639155746c95a98d91702d5fbe595172b058e0e9db330374ecfff25d",
        "files_changed": ["dual_fold_appearance_submission.py"],
    }
    assert manifest["integrity"]["transition_partitioned_inference"] is True
    assert manifest["integrity"]["each_consecutive_transition_processed_exactly_once"] is True
    assert metadata["id"] == "indarkarhana/biohub-temporal-contextual-final-runtime-v1"
    assert metadata["isPrivate"] is True
    for name, row in manifest["files"].items():
        path = TARGET / name
        assert path.stat().st_size == row["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    code = (TARGET / "dual_fold_appearance_submission.py").read_text(encoding="utf-8")
    assert "build_transition_work_plan" in code
    assert '"transition_partitioned_inference": True' in code
    assert "kaggle competitions submit" not in code.casefold()


def test_built_runtime_passes_its_frozen_verifier() -> None:
    subprocess.run(
        [
            sys.executable,
            str(TARGET / "verify_runtime.py"),
            "--root",
            str(TARGET),
        ],
        check=True,
    )
