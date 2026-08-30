#!/usr/bin/env python
"""Verify a streamed Antelume seed-probe harvest without extracting it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import PurePosixPath, Path
import tarfile
from typing import Any


RUN_ID = "competition-real-division-seed-probe-harvest-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_harvest(path: Path) -> dict[str, Any]:
    archive_sha256 = sha256_file(path)
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or "HARVEST_MANIFEST.json" not in names:
            raise ValueError("harvest archive inventory is invalid")
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"unsafe harvest path: {name}")
        if any(not member.isreg() for member in members):
            raise ValueError("harvest archive contains a non-regular member")
        manifest_stream = archive.extractfile("HARVEST_MANIFEST.json")
        if manifest_stream is None:
            raise ValueError("harvest manifest is unreadable")
        manifest = json.loads(manifest_stream.read())
        records = manifest.get("files", [])
        expected_names = {"HARVEST_MANIFEST.json", *(row["path"] for row in records)}
        if not (
            manifest.get("schema_version") == 1
            and manifest.get("status") == "complete"
            and manifest.get("run_id") == RUN_ID
            and manifest.get("probe_controller_status")
            in {"completed", "skipped_after_selection_rejection", "failed"}
            and 0 <= int(manifest.get("member_count", -1)) <= 16
            and manifest.get("member_count") == len(manifest.get("members", []))
            and manifest.get("competition_test_data_read") is False
            and manifest.get("public_leaderboard_used_for_selection") is False
            and manifest.get("submission_created") is False
            and manifest.get("authorized_for_submission") is False
            and set(names) == expected_names
        ):
            raise ValueError("harvest manifest is ineligible")
        for record in records:
            stream = archive.extractfile(record["path"])
            if stream is None:
                raise ValueError(f"harvest member is unreadable: {record['path']}")
            digest = hashlib.sha256()
            observed_bytes = 0
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                observed_bytes += len(block)
                digest.update(block)
            if (
                observed_bytes != int(record["bytes"])
                or digest.hexdigest() != record["sha256"]
            ):
                raise ValueError(f"harvest member changed: {record['path']}")
    return {
        "schema_version": 1,
        "status": "verified",
        "run_id": RUN_ID,
        "archive_path": str(path.resolve()),
        "archive_sha256": archive_sha256,
        "probe_controller_status": manifest["probe_controller_status"],
        "member_count": manifest["member_count"],
        "file_count": len(records),
        "competition_test_data_read": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }


def extract_verified_harvest(path: Path, destination: Path) -> dict[str, Any]:
    """Verify first, then extract regular members without tar path handling."""

    verified = verify_harvest(path)
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive.getmembers():
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"harvest member is unreadable: {member.name}")
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    stream.write(block)
    verified["extracted_to"] = str(destination.resolve())
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--extract-to", type=Path)
    args = parser.parse_args()
    result = (
        extract_verified_harvest(args.archive, args.extract_to)
        if args.extract_to is not None
        else verify_harvest(args.archive)
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(args.report.suffix + ".partial")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.report)
    print(rendered, end="")


if __name__ == "__main__":
    main()
