from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build-lsm-fm-center-enhancement-runtime.py"
DATASET_NAME = "biohub-lsm-fm-boundary-runtime-v1"
TARGET = ROOT / ".biohub" / "staging" / DATASET_NAME
BOUNDARY_EVALUATOR = (
    ROOT / "research" / "lsm_fm_detection" / "evaluate_boundary_refinement.py"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    values = runpy.run_path(str(BASE_BUILDER))
    base_main = values["main"]
    base_main.__globals__.update({"DATASET_NAME": DATASET_NAME, "TARGET": TARGET})
    original_argv = sys.argv
    try:
        sys.argv = [str(BASE_BUILDER)] + (["--replace"] if args.replace else [])
        base_main()
    finally:
        sys.argv = original_argv

    evaluator_name = "evaluate_boundary_refinement.py"
    shutil.copy2(BOUNDARY_EVALUATOR, TARGET / evaluator_name)
    manifest_path = TARGET / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["purpose"] = (
        "Clean inference-only boundary sweep around the near-gate feature-36 "
        "probability_r2_p2 centroid; no submission"
    )
    manifest["candidate"].update(
        {
            "architecture": "frozen 35.1M-parameter LSM-FM feature-36 detector",
            "boundary_candidates": [
                "probability_r2_p2",
                "probability_r2_p1",
                "probability_r2_p1_5",
                "probability_r3_p1",
                "probability_r3_p1_5",
                "probability_r3_p2",
            ],
            "peak_count_preserved": True,
            "confidence_ranking_preserved": True,
            "selection_movies": 8,
            "sealed_acceptance_movies": 4,
            "public_leaderboard_used_for_selection": False,
            "competition_submission_performed": False,
        }
    )
    path = TARGET / evaluator_name
    manifest["files"][evaluator_name] = {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    write_json(manifest_path, manifest)

    metadata_path = TARGET / "dataset-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "title": "Biohub LSM-FM Boundary Runtime v1",
            "id": f"indarkarhana/{DATASET_NAME}",
            "description": (
                "Private hash-bound runtime for an inference-only near-gate LSM-FM "
                "centroid boundary sweep."
            ),
            "isPrivate": True,
        }
    )
    write_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "target": str(TARGET),
                "manifest_sha256": sha256_file(manifest_path),
                "files": sum(path.is_file() for path in TARGET.rglob("*")),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
