#!/usr/bin/env python
"""Fail-closed integrity check for the portable temporal-patch runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_runtime(root: Path, *, require_gpus: bool = False) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    integrity = manifest.get("integrity")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("runtime manifest has no file inventory")
    if not (
        isinstance(integrity, dict)
        and integrity.get("required_gpu_count") == 2
        and integrity.get("competition_submission_command_included") is False
        and integrity.get("public_predictions_copied") is False
        and integrity.get("public_leaderboard_used_for_selection") is False
        and integrity.get("maximum_submission_inference_seconds") == 36_000
        and integrity.get("minimum_kaggle_finalization_reserve_seconds") == 7_200
    ):
        raise RuntimeError("runtime integrity policy changed")

    checked_bytes = 0
    for relative, expected in sorted(files.items()):
        if not isinstance(relative, str) or not isinstance(expected, dict):
            raise RuntimeError("runtime manifest contains an invalid file entry")
        path = (root / Path(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(f"runtime path escapes package root: {relative}") from error
        if not path.is_file():
            raise FileNotFoundError(f"runtime file is missing: {relative}")
        observed_bytes = path.stat().st_size
        observed_hash = sha256_file(path)
        if observed_bytes != int(expected.get("bytes", -1)):
            raise RuntimeError(f"runtime byte count changed: {relative}")
        if observed_hash != expected.get("sha256"):
            raise RuntimeError(f"runtime hash changed: {relative}")
        checked_bytes += observed_bytes

    detected_gpus: int | None = None
    if require_gpus:
        import torch

        detected_gpus = torch.cuda.device_count()
        if detected_gpus != 2:
            raise RuntimeError(
                f"temporal-patch execution requires exactly two GPUs, saw {detected_gpus}"
            )
    return {
        "status": "verified",
        "root": str(root),
        "manifest_sha256": sha256_file(manifest_path),
        "files_checked": len(files),
        "bytes_checked": checked_bytes,
        "required_gpu_count": 2,
        "detected_gpu_count": detected_gpus,
        "submission_command_included": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--require-gpus", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify_runtime(args.root, require_gpus=args.require_gpus),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
