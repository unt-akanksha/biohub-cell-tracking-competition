#!/usr/bin/env python
"""Verify the immutable expanded real-localization replay tar from Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile


RUN_ID = "competition-real-localization-expanded-shards-v2"
TERMINAL_RUN_ID = "competition-real-localization-expanded-kaggle-cache-v2"
INVENTORY_SHA256 = "a80c9028b21fdba1746bc2686dc5c64f0852e7954d1e758ba1357bcca5aa0edc"
ROOT_NAME = "competition_real_localization_expanded_shards_v2"
ROLE_COUNTS = {"optimization": 480, "selection": 17, "sealed_audit": 14}
FINAL_PROBES = {
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_member(archive: tarfile.TarFile, name: str) -> bytes:
    member = archive.getmember(name)
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"tar member is not readable: {name}")
    return stream.read()


def verify(archive_path: Path, sha_path: Path, terminal_path: Path) -> dict:
    archive_sha256 = sha256_file(archive_path)
    fields = sha_path.read_text(encoding="utf-8").strip().split()
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if not (
        fields == [archive_sha256, archive_path.name]
        and terminal.get("status") == "completed"
        and terminal.get("run_id") == TERMINAL_RUN_ID
        and terminal.get("archive_name") == archive_path.name
        and terminal.get("archive_bytes") == archive_path.stat().st_size
        and terminal.get("archive_sha256") == archive_sha256
        and terminal.get("accelerator") == "cpu"
        and terminal.get("gpu_used") is False
        and terminal.get("competition_test_data_read") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("authorized_for_submission") is False
    ):
        raise ValueError("expanded replay archive receipt is invalid")
    with tarfile.open(archive_path, mode="r") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("expanded replay tar has duplicate members")
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] != ROOT_NAME
                or not (member.isdir() or member.isfile())
            ):
                raise ValueError(f"unsafe expanded replay tar member: {member.name}")
        manifest_name = f"{ROOT_NAME}/real_localization_shard_manifest.json"
        manifest_bytes = read_member(archive, manifest_name)
        manifest = json.loads(manifest_bytes)
        records = manifest.get("files")
        if not (
            manifest.get("schema_version") == 1
            and manifest.get("status") == "complete"
            and manifest.get("run_id") == RUN_ID
            and manifest.get("inventory_sha256") == INVENTORY_SHA256
            and manifest.get("source_mode")
            == "kaggle_cpu_direct_competition_train_expanded_optimization_only"
            and isinstance(records, list)
            and len(records) == 511
            and manifest.get("competition_test_data_read") is False
            and manifest.get("public_leaderboard_used_for_selection") is False
            and manifest.get("authorized_for_submission") is False
            and set(manifest.get("excluded_final_probe_stems", ())) == FINAL_PROBES
        ):
            raise ValueError("expanded replay manifest is invalid")
        by_role = {
            role: sum(row.get("role") == role for row in records)
            for role in ROLE_COUNTS
        }
        if by_role != ROLE_COUNTS:
            raise ValueError("expanded replay role counts changed")
        if len({row.get("path") for row in records}) != len(records):
            raise ValueError("expanded replay contains duplicate shard paths")
        for row in records:
            relative = PurePosixPath(str(row["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("expanded replay shard path escaped root")
            name = f"{ROOT_NAME}/{relative.as_posix()}"
            member = archive.getmember(name)
            if not member.isfile() or member.size != int(row["bytes"]):
                raise ValueError(f"expanded replay shard size changed: {name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"expanded replay shard is unreadable: {name}")
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            if digest.hexdigest() != row["sha256"]:
                raise ValueError(f"expanded replay shard hash changed: {name}")
    return {
        "schema_version": 1,
        "status": "verified",
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_path.stat().st_size,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "shards": len(records),
        "role_counts": by_role,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "authorized_for_submission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256-file", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.archive, args.sha256_file, args.terminal), sort_keys=True))


if __name__ == "__main__":
    main()
