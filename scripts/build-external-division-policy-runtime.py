#!/usr/bin/env python
"""Build a private, competition-free runtime for division-policy calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "research/temporal_contrastive"
RUN_ID = "external-division-policy-runtime-v1"
DATASET_ID = "indarkarhana/biohub-external-division-policy-runtime-v1"
REQUIRED_ENTRYPOINTS = {
    "calibrate_division_recovery_policy.py",
    "verify_multiscale_pretraining_output.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def source_inventory() -> dict[str, Path]:
    sources = {path.name: path for path in sorted(SOURCE_ROOT.glob("*.py"))}
    if not REQUIRED_ENTRYPOINTS <= set(sources):
        raise RuntimeError("external division policy entrypoints are missing")
    return sources


def verify_runtime(root: Path) -> dict[str, Any]:
    manifest_path = root / "RUNTIME_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("run_id") == RUN_ID
        and manifest.get("competition_data_read") is False
        and manifest.get("competition_source_allowed") is False
        and manifest.get("public_predictions_copied") is False
        and manifest.get("public_leaderboard_used_for_selection") is False
        and manifest.get("submission_command_included") is False
        and REQUIRED_ENTRYPOINTS <= set(files)
    ):
        raise ValueError("external division policy runtime manifest is invalid")
    for name, record in files.items():
        path = root / name
        if (
            not path.is_file()
            or record.get("sha256") != sha256_file(path)
            or int(record.get("bytes", -1)) != path.stat().st_size
        ):
            raise ValueError(f"external division policy runtime changed: {name}")
    return {
        "run_id": RUN_ID,
        "file_count": len(files),
        "manifest_sha256": sha256_file(manifest_path),
        "competition_data_read": False,
        "submission_command_included": False,
        "authorized_for_kaggle_calibration": True,
        "authorized_for_submission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_runtime(args.output_root), indent=2, sort_keys=True))
        return
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    sources = source_inventory()
    for name, source in sources.items():
        shutil.copy2(source, args.output_root / name)
    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "source_family": "project temporal_contrastive Python modules",
        "competition_data_read": False,
        "competition_source_allowed": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_command_included": False,
        "files": {
            name: {
                "sha256": sha256_file(args.output_root / name),
                "bytes": (args.output_root / name).stat().st_size,
            }
            for name in sorted(sources)
        },
    }
    write_json(args.output_root / "RUNTIME_MANIFEST.json", manifest)
    write_json(
        args.output_root / "dataset-metadata.json",
        {
            "title": "Biohub External Division Policy Runtime v1",
            "id": DATASET_ID,
            "licenses": [{"name": "MIT"}],
            "isPrivate": True,
        },
    )
    print(json.dumps(verify_runtime(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
