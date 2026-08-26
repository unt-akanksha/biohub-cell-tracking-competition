from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build-lsm-fm-image-text-pu-runtime.py"
DATASET_NAME = "biohub-lsm-fm-image-text-refinement-runtime-v1"
TARGET = ROOT / ".biohub" / "staging" / DATASET_NAME
REFINEMENT_SOURCES = {
    "evaluate_localization_refinement.py": ROOT
    / "research"
    / "lsm_fm_detection"
    / "evaluate_localization_refinement.py",
    "localization_refinement.py": ROOT
    / "research"
    / "lsm_fm_detection"
    / "localization_refinement.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    values = runpy.run_path(str(BASE_BUILDER))
    base_main = values["main"]
    base_main.__globals__.update(
        {
            "DATASET_NAME": DATASET_NAME,
            "TARGET": TARGET,
        }
    )
    import sys

    original_argv = sys.argv
    try:
        sys.argv = [str(BASE_BUILDER)] + (["--replace"] if args.replace else [])
        base_main()
    finally:
        sys.argv = original_argv

    for name, source in REFINEMENT_SOURCES.items():
        shutil.copy2(source, TARGET / name)

    manifest_path = TARGET / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["purpose"] = (
        "Selection-gated inference-only coordinate refinement of an independently "
        "trained multimodal LSM-FM detector; no submission"
    )
    manifest["candidate"].update(
        {
            "architecture": (
                "LSM-FM image-text SwinUNETR feature-36 with a Biohub-owned "
                "one-channel heatmap head"
            ),
            "refinement_scope": (
                "one global sub-voxel coordinate strategy; peak identities, "
                "confidence ranking, and density calibration remain frozen"
            ),
            "selection_movies": 8,
            "sealed_acceptance_movies": 4,
            "public_leaderboard_used_for_selection": False,
            "competition_submission_performed": False,
        }
    )
    for name in REFINEMENT_SOURCES:
        path = TARGET / name
        manifest["files"][name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    write_json(manifest_path, manifest)

    metadata_path = TARGET / "dataset-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "title": "Biohub LSM-FM Image-Text Refinement Runtime v1",
            "id": f"indarkarhana/{DATASET_NAME}",
            "description": (
                "Private feature-36 LSM-FM detector and inference-only Biohub "
                "localization refinement runtime; attribution and licenses included."
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
