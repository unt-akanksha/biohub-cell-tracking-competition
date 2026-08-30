#!/usr/bin/env python
"""Stream-verify the exact Antelume relational sweep result archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Any


ROOT = "competition-relational-division-sweep-v1"
RUN_ID = "competition-relational-division-sweep-v1"
SHA_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


def sha256_stream(handle: Any) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return sha256_stream(handle)


def _safe(member: tarfile.TarInfo) -> bool:
    path = PurePosixPath(member.name)
    return bool(
        not member.issym()
        and not member.islnk()
        and not path.is_absolute()
        and ".." not in path.parts
        and path.parts
        and path.parts[0] == ROOT
    )


def _read_hash_manifest(path: Path, sums_name: str) -> dict[str, str]:
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            if not _safe(member):
                raise ValueError("relational harvest contains an unsafe member")
            if member.name != sums_name:
                continue
            if not member.isfile():
                raise ValueError("relational harvest hash manifest is not a file")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("cannot read relational harvest hashes")
            expected: dict[str, str] = {}
            for line in handle.read().decode("utf-8").splitlines():
                match = SHA_LINE.fullmatch(line)
                if match is None:
                    raise ValueError("relational harvest hash manifest changed")
                digest, name = match.groups()
                member_name = PurePosixPath(name).as_posix()
                if (
                    member_name in expected
                    or PurePosixPath(member_name).is_absolute()
                    or ".." in PurePosixPath(member_name).parts
                    or not member_name.startswith(f"{ROOT}/")
                    or member_name == sums_name
                ):
                    raise ValueError("relational harvest hash path is unsafe")
                expected[member_name] = digest
            if not expected:
                raise ValueError("relational harvest hash manifest is empty")
            return expected
    raise ValueError("relational harvest hash manifest is missing")


def verify_archive(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    sums_name = f"{ROOT}/SHA256SUMS"
    exit_name = f"{ROOT}/training.exit-code"
    log_name = f"{ROOT}/training.log"
    aggregate_name = f"{ROOT}/models/relational_division_sweep_terminal.json"
    expected = _read_hash_manifest(path, sums_name)
    if not all(name in expected for name in (exit_name, log_name)):
        raise ValueError("relational harvest control files are incomplete")
    observed: set[str] = set()
    exit_bytes = None
    aggregate_bytes = None
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            if not _safe(member):
                raise ValueError("relational harvest contains an unsafe member")
            if not member.isfile():
                continue
            if member.name in observed:
                raise ValueError("relational harvest contains duplicate file paths")
            observed.add(member.name)
            if member.name == sums_name:
                continue
            expected_hash = expected.get(member.name)
            if expected_hash is None:
                raise ValueError("relational harvest contains unbound files")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"cannot read relational harvest member: {member.name}")
            if member.name in {exit_name, aggregate_name}:
                value = handle.read()
                actual_hash = hashlib.sha256(value).hexdigest()
                if member.name == exit_name:
                    exit_bytes = value
                else:
                    aggregate_bytes = value
            else:
                actual_hash = sha256_stream(handle)
            if actual_hash != expected_hash:
                raise ValueError(f"relational harvest hash changed: {member.name}")
    if observed != set(expected) | {sums_name}:
        raise ValueError("relational harvest contains unbound files")
    if exit_bytes is None:
        raise ValueError("cannot read relational training exit code")
    exit_code = int(exit_bytes.decode("utf-8").strip())
    aggregate = json.loads(aggregate_bytes) if aggregate_bytes is not None else None
    if aggregate is not None:
        if not (
            aggregate.get("schema_version") == 1
            and aggregate.get("status") == "completed"
            and aggregate.get("run_id") == RUN_ID
            and aggregate.get("planned_model_count") == 8
            and aggregate.get("completed_model_count") == 8
            and aggregate.get("steps_per_model") == 15_000
            and aggregate.get("final_probe_opened") is False
            and aggregate.get("competition_test_data_read") is False
            and aggregate.get("public_code_copied") is False
            and aggregate.get("public_predictions_copied") is False
            and aggregate.get("public_leaderboard_used_for_selection") is False
            and aggregate.get("submission_created") is False
            and aggregate.get("authorized_for_submission") is False
        ):
            raise ValueError("relational aggregate terminal is ineligible")
    if exit_code in (0, 2) and aggregate is None:
        raise ValueError("completed relational run is missing its aggregate terminal")
    return {
        "schema_version": 1,
        "status": "harvest_verified",
        "run_id": RUN_ID,
        "archive": str(path.resolve()),
        "archive_bytes": path.stat().st_size,
        "archive_sha256": sha256_file(path),
        "training_exit_code": exit_code,
        "aggregate_terminal_present": aggregate is not None,
        "completed_model_count": (
            aggregate.get("completed_model_count") if aggregate is not None else None
        ),
        "selection_accepted_members": (
            aggregate.get("selection_accepted_members", []) if aggregate is not None else []
        ),
        "independently_strong_members": (
            aggregate.get("independently_strong_members", []) if aggregate is not None else []
        ),
        "ensemble_eligible": (
            aggregate.get("ensemble_eligible", False) if aggregate is not None else False
        ),
        "final_probe_opened": False,
        "competition_test_data_read": False,
        "authorized_for_submission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = verify_archive(args.archive)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".partial")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(args.report)
    print(rendered, end="")


if __name__ == "__main__":
    main()
