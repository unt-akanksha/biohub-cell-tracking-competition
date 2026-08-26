from __future__ import annotations

import hashlib
import json
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-lsm-fm-ensemble-runtime.py"
TARGET = ROOT / ".biohub" / "staging" / "biohub-lsm-fm-ensemble-runtime-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_ensemble_runtime_is_private_hash_bound_and_has_fixed_weights(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(BUILDER), "--replace"])
    runpy.run_path(str(BUILDER), run_name="__main__")
    manifest = json.loads((TARGET / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    metadata = json.loads((TARGET / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "indarkarhana/biohub-lsm-fm-ensemble-runtime-v1"
    assert metadata["isPrivate"] is True
    assert manifest["candidate"]["candidate_count"] == 5
    assert manifest["candidate"]["ensemble_weight_selection"] is False
    assert manifest["candidate"]["ensemble_weights"] == [0.5, 0.5]
    assert manifest["candidate"]["public_leaderboard_used_for_selection"] is False
    assert manifest["candidate"]["competition_submission_performed"] is False
    assert len(manifest["ensemble_pretrained_models"]) == 2
    for name in (
        "evaluate_lsm_fm_ensemble.py",
        "lsm_fm_image_only_model.py",
        "lsm_fm_image_text_model.py",
        "lsm_fm_image_only_student.pt",
        "lsm_fm_image_text_student.pt",
    ):
        assert sha256_file(TARGET / name) == manifest["files"][name]["sha256"]
