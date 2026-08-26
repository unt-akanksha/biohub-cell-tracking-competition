from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build-lsm-fm-image-text-refinement-runtime.py"
DATASET_NAME = "biohub-lsm-fm-ensemble-runtime-v1"
TARGET = ROOT / ".biohub" / "staging" / DATASET_NAME
FEATURE24_CHECKPOINT = (
    ROOT / ".biohub" / "cache" / "models" / "lsm-fm" / "lsm_fm_image_only_student.pt"
)
EXPECTED_FEATURE24_SHA256 = (
    "d287049e5f86ad1db7350cdf30acf2309c9dca34be8570c398f330f346afcfb0"
)
ADDITIONAL_SOURCES = {
    "evaluate_lsm_fm_ensemble.py": ROOT
    / "research"
    / "lsm_fm_detection"
    / "evaluate_ensemble.py",
    "lsm_fm_image_only_model.py": ROOT / "research" / "lsm_fm_detection" / "model.py",
    "lsm_fm_image_text_model.py": ROOT
    / "research"
    / "lsm_fm_detection"
    / "image_text_model.py",
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
    if sha256_file(FEATURE24_CHECKPOINT) != EXPECTED_FEATURE24_SHA256:
        raise RuntimeError("stripped feature-24 LSM-FM checkpoint hash changed")

    values = runpy.run_path(str(BASE_BUILDER))
    base_main = values["main"]
    base_main.__globals__.update({"DATASET_NAME": DATASET_NAME, "TARGET": TARGET})
    original_argv = sys.argv
    try:
        sys.argv = [str(BASE_BUILDER)] + (["--replace"] if args.replace else [])
        base_main()
    finally:
        sys.argv = original_argv

    for name, source in ADDITIONAL_SOURCES.items():
        shutil.copy2(source, TARGET / name)
    shutil.copy2(FEATURE24_CHECKPOINT, TARGET / "lsm_fm_image_only_student.pt")

    manifest_path = TARGET / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["purpose"] = (
        "Clean held-out comparison of independently adapted feature-24 and "
        "feature-36 LSM-FM detectors plus a fixed equal-probability ensemble; no submission"
    )
    manifest["ensemble_pretrained_models"] = [
        {
            "name": "LSM-FM image-only SwinUNETR feature-24 student",
            "source_doi": "10.5281/zenodo.20146516",
            "stripped_checkpoint_sha256": EXPECTED_FEATURE24_SHA256,
            "license": "CC-BY-4.0",
            "detector_parameters": 15_702_979,
        },
        {
            "name": "LSM-FM image-text SwinUNETR feature-36 student",
            "source_doi": "10.5281/zenodo.20146516",
            "stripped_checkpoint_sha256": manifest["pretrained_model"][
                "stripped_checkpoint_sha256"
            ],
            "license": "CC-BY-4.0",
            "detector_parameters": 35_072_515,
        },
    ]
    manifest["candidate"].update(
        {
            "architecture": (
                "global selection among independently adapted LSM-FM feature-24, "
                "feature-36, and their fixed equal-probability ensemble"
            ),
            "candidate_count": 5,
            "ensemble_weight_selection": False,
            "ensemble_weights": [0.5, 0.5],
            "selection_movies": 8,
            "sealed_acceptance_movies": 4,
            "public_leaderboard_used_for_selection": False,
            "competition_submission_performed": False,
        }
    )
    staged = {
        **ADDITIONAL_SOURCES,
        "lsm_fm_image_only_student.pt": FEATURE24_CHECKPOINT,
    }
    for name in staged:
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
            "title": "Biohub LSM-FM Ensemble Runtime v1",
            "id": f"indarkarhana/{DATASET_NAME}",
            "description": (
                "Private feature-24/feature-36 LSM-FM ensemble validation runtime; "
                "attribution and licenses included."
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
