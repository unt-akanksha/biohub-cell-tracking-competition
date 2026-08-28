#!/usr/bin/env python
"""Build and verify the private offline localization training runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "division-localization-runtime-v1"
DATASET_ID = "indarkarhana/biohub-division-localization-runtime-v1"
SOURCE_NAMES = (
    "patch_model.py",
    "pair_fusion.py",
    "transition_context.py",
    "contextual_pair_fusion.py",
    "multiscale_contextual_pair_fusion.py",
    "multiscale_division_localization.py",
    "train_dual_fold_division_localization.py",
    "verify_division_localization_training_output.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_runtime(root: Path) -> dict[str, Any]:
    manifest_path = root / "RUNTIME_MANIFEST.json"
    metadata_path = root / "dataset-metadata.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("localization runtime manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {"id": DATASET_ID, "isPrivate": True}
    )
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("run_id") == RUN_ID
        and manifest.get("required_visible_gpu_count") == 2
        and manifest.get("internet_required") is False
        and manifest.get("competition_data_read") is False
        and manifest.get("submission_command_included") is False
        and metadata.get("id") == DATASET_ID
        and metadata.get("isPrivate") is True
    ):
        raise ValueError("localization runtime policy changed")
    files = manifest.get("files", {})
    if set(files) != set(SOURCE_NAMES):
        raise ValueError("localization runtime source inventory changed")
    for name in SOURCE_NAMES:
        path = root / name
        if not path.is_file() or sha256_file(path) != files[name]["sha256"]:
            raise ValueError(f"localization runtime source changed: {name}")
    return {
        "run_id": RUN_ID,
        "files": len(files),
        "manifest_sha256": sha256_file(manifest_path),
        "required_visible_gpu_count": 2,
        "internet_required": False,
        "submission_command_included": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / ".biohub/staging/biohub-division-localization-runtime-v1",
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    source_root = ROOT / "research/temporal_contrastive"
    for name in SOURCE_NAMES:
        shutil.copy2(source_root / name, args.output_root / name)
    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "required_visible_gpu_count": 2,
        "internet_required": False,
        "competition_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_command_included": False,
        "files": {
            name: {"path": name, "sha256": sha256_file(args.output_root / name)}
            for name in SOURCE_NAMES
        },
    }
    write_json(args.output_root / "RUNTIME_MANIFEST.json", manifest)
    write_json(
        args.output_root / "dataset-metadata.json",
        {
            "title": "Biohub Division Localization Runtime v1",
            "id": DATASET_ID,
            "licenses": [{"name": "other"}],
            "isPrivate": True,
        },
    )
    print(json.dumps(verify_runtime(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
