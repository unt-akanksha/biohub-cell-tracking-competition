#!/usr/bin/env python
"""Verify the immutable Antelume peak-ranking detector result archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = PurePosixPath("synthetic256-real-positive-temporal-peak-rank-v1")
EXPECTED_RUN_ID = "synthetic256-real-positive-temporal-peak-rank-v1"
EXPECTED_PARAMETER_COUNT = 38_381_478
EXPECTED_STEPS = 3_000
EXPECTED_SEED = 1_041_729
EXPECTED_WIDTHS = [96, 192, 384, 768]
EXPECTED_DEPTHS = [3, 3, 9, 3]
HASH_PATTERN = re.compile(r"^([0-9a-f]{64})  (.+)$")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    result = {}
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(f"unsafe archive member: {member.name}")
        if path.parts[0] != str(ROOT):
            raise ValueError(f"archive member outside frozen root: {member.name}")
        if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
            raise ValueError(f"unsupported archive member type: {member.name}")
        if member.name in result:
            raise ValueError(f"duplicate archive member: {member.name}")
        result[member.name] = member
    return result


def read_bytes(
    archive: tarfile.TarFile, members: dict[str, tarfile.TarInfo], relative: str
) -> bytes:
    name = (ROOT / relative).as_posix()
    member = members.get(name)
    if member is None or not member.isfile():
        raise ValueError(f"missing archive file: {name}")
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"could not read archive file: {name}")
    return stream.read()


def parse_sums(payload: bytes) -> dict[str, str]:
    result = {}
    for raw_line in payload.decode("utf-8").splitlines():
        match = HASH_PATTERN.fullmatch(raw_line)
        if match is None:
            raise ValueError(f"invalid SHA256SUMS line: {raw_line}")
        digest, name = match.groups()
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or path.parts[0] != str(ROOT):
            raise ValueError(f"unsafe checksum path: {name}")
        if name in result:
            raise ValueError(f"duplicate checksum path: {name}")
        result[name] = digest
    if not result:
        raise ValueError("SHA256SUMS is empty")
    return result


def validate_terminal(
    terminal: dict[str, Any], files: dict[str, bytes], exit_code: int
) -> bool:
    if not (
        terminal.get("schema_version") == 1
        and terminal.get("run_id") == EXPECTED_RUN_ID
        and terminal.get("completed_steps") == EXPECTED_STEPS
        and terminal.get("seed") == EXPECTED_SEED
        and terminal.get("parameter_count") == EXPECTED_PARAMETER_COUNT
        and terminal.get("widths") == EXPECTED_WIDTHS
        and terminal.get("depths") == EXPECTED_DEPTHS
        and terminal.get("competition_train_data_read") is True
        and terminal.get("competition_test_data_read") is False
        and terminal.get("public_predictions_read") is False
        and terminal.get("public_notebook_weights_read") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("submission_created") is False
        and terminal.get("authorized_for_submission") is False
    ):
        raise ValueError("peak-ranking terminal violates the frozen run contract")
    last_name = (ROOT / "last_peak_rank_detector.pt").as_posix()
    last_checkpoint = files.get(last_name)
    if last_checkpoint is None or sha256_bytes(last_checkpoint) != terminal.get(
        "last_checkpoint_sha256"
    ):
        raise ValueError("last validation checkpoint does not match terminal")
    status = terminal.get("status")
    accepted = status == "accepted_at_audit"
    if accepted:
        if not (
            exit_code == 0
            and terminal.get("selection_passed") is True
            and terminal.get("audit_opened") is True
            and terminal.get("audit_passed") is True
            and terminal.get("best_step", 0) > 0
        ):
            raise ValueError("accepted terminal is internally inconsistent")
        checkpoint_name = (ROOT / "peak_rank_detector.pt").as_posix()
        checkpoint = files.get(checkpoint_name)
        if checkpoint is None or sha256_bytes(checkpoint) != terminal.get(
            "checkpoint_sha256"
        ):
            raise ValueError("accepted checkpoint hash does not match terminal")
    elif status in {"rejected_at_selection", "rejected_at_audit"}:
        if exit_code != 1 or terminal.get("audit_passed") is not False:
            raise ValueError("rejected terminal is internally inconsistent")
        if status == "rejected_at_selection" and (
            terminal.get("selection_passed") is not False
            or terminal.get("audit_opened") is not False
        ):
            raise ValueError("selection rejection opened forbidden audit data")
    else:
        raise ValueError(f"unexpected peak-ranking terminal status: {status}")
    return accepted


def verify(archive_path: Path) -> dict[str, Any]:
    if not archive_path.is_file() or archive_path.stat().st_size <= 0:
        raise FileNotFoundError(archive_path)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = safe_members(archive)
        sums = parse_sums(read_bytes(archive, members, "SHA256SUMS"))
        files = {
            name: archive.extractfile(member).read()  # type: ignore[union-attr]
            for name, member in members.items()
            if member.isfile() and name != (ROOT / "SHA256SUMS").as_posix()
        }
        if set(sums) != set(files):
            raise ValueError("archive files do not exactly match SHA256SUMS")
        for name, payload in files.items():
            if sha256_bytes(payload) != sums[name]:
                raise ValueError(f"archive checksum mismatch: {name}")
        terminal = json.loads(files[(ROOT / "terminal.json").as_posix()])
        exit_code = int(
            files[(ROOT / "training.exit-code").as_posix()].decode("utf-8").strip()
        )
        accepted = validate_terminal(terminal, files, exit_code)
    return {
        "schema_version": 1,
        "run_id": "antelume-peak-rank-detector-harvest-verification-v1",
        "status": "verified",
        "archive_sha256": sha256_file(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "training_exit_code": exit_code,
        "model_status": terminal["status"],
        "selection_passed": terminal["selection_passed"],
        "audit_opened": terminal["audit_opened"],
        "audit_passed": terminal["audit_passed"],
        "accepted_for_kaggle_validation": accepted,
        "best_step": terminal["best_step"],
        "parameter_count": terminal["parameter_count"],
        "checkpoint_sha256": terminal.get("checkpoint_sha256"),
        "competition_submission_performed": False,
        "authorized_for_submission": False,
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
