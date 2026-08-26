from __future__ import annotations

import hashlib
import json
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-lsm-fm-image-text-refinement-runtime.py"
TARGET = (
    ROOT
    / ".biohub"
    / "staging"
    / "biohub-lsm-fm-image-text-refinement-runtime-v1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_refinement_runtime_is_private_hash_bound_and_submission_sealed(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(BUILDER), "--replace"])
    runpy.run_path(str(BUILDER), run_name="__main__")

    manifest = json.loads((TARGET / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    metadata = json.loads((TARGET / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == (
        "indarkarhana/biohub-lsm-fm-image-text-refinement-runtime-v1"
    )
    assert metadata["isPrivate"] is True
    assert manifest["candidate"]["parameters"] == 35_072_515
    assert manifest["candidate"]["selection_movies"] == 8
    assert manifest["candidate"]["sealed_acceptance_movies"] == 4
    assert manifest["candidate"]["public_leaderboard_used_for_selection"] is False
    assert manifest["candidate"]["competition_submission_performed"] is False
    assert "feature-36" in manifest["candidate"]["architecture"]
    for name in (
        "evaluate_localization_refinement.py",
        "localization_refinement.py",
        "lsm_fm_model.py",
        "lsm_fm_image_text_student.pt",
    ):
        assert sha256_file(TARGET / name) == manifest["files"][name]["sha256"]
