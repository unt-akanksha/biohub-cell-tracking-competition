#!/usr/bin/env python
"""Verify the immutable train-only NucVerse3D compatibility archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


RUN_ID = "nucverse3d-generalized-physical-compatibility-v1"
EXPECTED_ONNX_SHA256 = (
    "ca16e1b26d21ae522d68aba384ee7121f5ba2de28a122c291d1e1e627601e871"
)
EXPECTED_MANIFEST_SHA256 = (
    "5ccd52b96a36db7dff5f4ae482bb7e7cdd28f49c024b047d55fa7c575504a697"
)
EXPECTED_SOURCE_COMMIT = "d809a2e6cf380342708b7a9107574b259e6b34eb"
REQUIRED = {
    "results/SHA256SUMS",
    "results/optimization-screen.json",
    "results/optimization.exit-code",
    "results/screen.exit-code",
    "results/screen.log",
    "results/selection.exit-code",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_receipt(payload: dict[str, Any], *, phase: str) -> None:
    if not (
        payload.get("schema_version") == 1
        and payload.get("run_id") == RUN_ID
        and payload.get("phase") == phase
        and payload.get("status") == "complete"
        and payload.get("source_commit") == EXPECTED_SOURCE_COMMIT
        and payload.get("onnx_sha256") == EXPECTED_ONNX_SHA256
        and payload.get("manifest_sha256") == EXPECTED_MANIFEST_SHA256
        and payload.get("model_parameters") == 40_458_005
        and payload.get("competition_train_data_read") is True
        and payload.get("competition_test_data_read") is False
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("submission_created") is False
        and payload.get("authorized_for_submission") is False
    ):
        raise ValueError(f"{phase} receipt violates the clean screen contract")
    summary = payload.get("summary", {})
    if not (
        isinstance(summary.get("examples"), int)
        and summary["examples"] > 0
        and isinstance(summary.get("points"), int)
        and summary["points"] > 0
    ):
        raise ValueError(f"{phase} receipt has invalid summary counts")


def verify(archive_path: Path) -> dict[str, Any]:
    archive_path = archive_path.resolve()
    archive_bytes = archive_path.stat().st_size
    if archive_bytes <= 0:
        raise ValueError("compatibility archive is empty")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("compatibility archive has duplicate paths")
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("compatibility archive contains an unsafe path")
            if member.issym() or member.islnk():
                raise ValueError("compatibility archive contains a link")
        files = {
            member.name: archive.extractfile(member).read()
            for member in members
            if member.isfile()
        }
    if not REQUIRED <= set(files):
        raise ValueError(f"compatibility archive is missing {sorted(REQUIRED - set(files))}")
    declared = {}
    for line in files["results/SHA256SUMS"].decode("utf-8").splitlines():
        digest, name = line.split("  ", 1)
        declared[f"results/{name.removeprefix('results/')}"] = digest
    hashed_files = set(files) - {"results/SHA256SUMS"}
    if set(declared) != hashed_files:
        raise ValueError("compatibility SHA256SUMS inventory differs from archive")
    for name, digest in declared.items():
        if sha256_bytes(files[name]) != digest:
            raise ValueError(f"compatibility member hash changed: {name}")

    optimization_status = int(files["results/optimization.exit-code"].decode().strip())
    selection_status = int(files["results/selection.exit-code"].decode().strip())
    final_status = int(files["results/screen.exit-code"].decode().strip())
    if optimization_status != 0 or final_status != selection_status:
        raise ValueError("compatibility phase exit codes are inconsistent")
    optimization = json.loads(files["results/optimization-screen.json"])
    validate_receipt(optimization, phase="optimization")
    optimization_passed = optimization.get("compatibility_passed") is True
    selection_name = "results/selection-screen.json"
    if optimization_passed:
        if selection_status != 0 or selection_name not in files:
            raise ValueError("passed optimization did not produce clean selection")
        selection = json.loads(files[selection_name])
        validate_receipt(selection, phase="selection")
        selection_passed = selection.get("compatibility_passed") is True
    else:
        if selection_status != 0 or selection_name in files:
            raise ValueError("failed optimization unexpectedly opened selection")
        selection = None
        selection_passed = False
    return {
        "schema_version": 1,
        "run_id": "antelume-nucverse3d-compatibility-v1-harvest-verification",
        "status": "verified",
        "archive_sha256": sha256_file(archive_path),
        "archive_bytes": archive_bytes,
        "optimization_passed": optimization_passed,
        "selection_opened": selection is not None,
        "selection_passed": selection_passed,
        "accepted_for_detector_integration": selection_passed,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
        "optimization_summary": optimization["summary"],
        "selection_summary": selection["summary"] if selection else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.archive)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".partial")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

