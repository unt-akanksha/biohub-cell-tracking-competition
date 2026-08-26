from __future__ import annotations

import hashlib
import json
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-lsm-fm-boundary-runtime.py"
TARGET = ROOT / ".biohub" / "staging" / "biohub-lsm-fm-boundary-runtime-v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_boundary_runtime_is_private_count_preserving_and_submission_sealed(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "argv", [str(BUILDER), "--replace"])
    runpy.run_path(str(BUILDER), run_name="__main__")
    manifest = json.loads((TARGET / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    metadata = json.loads((TARGET / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "indarkarhana/biohub-lsm-fm-boundary-runtime-v1"
    assert metadata["isPrivate"] is True
    candidate = manifest["candidate"]
    assert candidate["peak_count_preserved"] is True
    assert candidate["confidence_ranking_preserved"] is True
    assert len(candidate["boundary_candidates"]) == 6
    assert candidate["competition_submission_performed"] is False
    assert sha256_file(TARGET / "evaluate_boundary_refinement.py") == manifest["files"][
        "evaluate_boundary_refinement.py"
    ]["sha256"]
