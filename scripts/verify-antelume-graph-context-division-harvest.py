#!/usr/bin/env python
"""Stream-verify the exact Antelume graph-context sweep result archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Any


ROOT = "competition-graph-context-division-sweep-v1"
RUN_ID = "competition-graph-context-division-sweep-v1"
FAMILY = "temporal_multiscale_graph_context_division_v1"
PROBE_RUN_ID = "competition-graph-context-division-development-probe-v1"
EXPECTED_PARAMETER_COUNT = 74_732_308
SHA_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


def sha256_stream(handle: Any) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)


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
                raise ValueError("graph-context harvest contains an unsafe member")
            if member.name != sums_name:
                continue
            if not member.isfile():
                raise ValueError("graph-context hash manifest is not a file")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("cannot read graph-context harvest hashes")
            expected: dict[str, str] = {}
            for line in stream.read().decode("utf-8").splitlines():
                match = SHA_LINE.fullmatch(line)
                if match is None:
                    raise ValueError("graph-context hash manifest changed")
                digest, name = match.groups()
                normalized = PurePosixPath(name).as_posix()
                if (
                    normalized in expected
                    or PurePosixPath(normalized).is_absolute()
                    or ".." in PurePosixPath(normalized).parts
                    or not normalized.startswith(f"{ROOT}/")
                    or normalized == sums_name
                ):
                    raise ValueError("graph-context harvest hash path is unsafe")
                expected[normalized] = digest
            if not expected:
                raise ValueError("graph-context harvest hash manifest is empty")
            return expected
    raise ValueError("graph-context harvest hash manifest is missing")


def verify_archive(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    sums_name = f"{ROOT}/SHA256SUMS"
    exit_name = f"{ROOT}/training.exit-code"
    probe_exit_name = f"{ROOT}/development-probe.exit-code"
    log_name = f"{ROOT}/training.log"
    probe_name = f"{ROOT}/graph_context_development_probe.json"
    aggregate_name = f"{ROOT}/models/graph_context_division_sweep_terminal.json"
    expected = _read_hash_manifest(path, sums_name)
    if not all(name in expected for name in (exit_name, probe_exit_name, log_name)):
        raise ValueError("graph-context harvest control files are incomplete")
    observed: set[str] = set()
    captured: dict[str, bytes] = {}
    capture_names = {exit_name, probe_exit_name, aggregate_name, probe_name}
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            if not _safe(member):
                raise ValueError("graph-context harvest contains an unsafe member")
            if not member.isfile():
                continue
            if member.name in observed:
                raise ValueError("graph-context harvest contains duplicate paths")
            observed.add(member.name)
            if member.name == sums_name:
                continue
            expected_hash = expected.get(member.name)
            if expected_hash is None:
                raise ValueError("graph-context harvest contains unbound files")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read graph-context member: {member.name}")
            if member.name in capture_names:
                value = stream.read()
                captured[member.name] = value
                actual_hash = hashlib.sha256(value).hexdigest()
            else:
                actual_hash = sha256_stream(stream)
            if actual_hash != expected_hash:
                raise ValueError(f"graph-context harvest hash changed: {member.name}")
    if observed != set(expected) | {sums_name}:
        raise ValueError("graph-context harvest contains unbound files")
    if exit_name not in captured or probe_exit_name not in captured:
        raise ValueError("graph-context harvest exit evidence is missing")
    exit_code = int(captured[exit_name].decode().strip())
    probe_exit_code = int(captured[probe_exit_name].decode().strip())
    aggregate = (
        json.loads(captured[aggregate_name]) if aggregate_name in captured else None
    )
    probe = json.loads(captured[probe_name]) if probe_name in captured else None
    if aggregate is not None:
        if not (
            aggregate.get("schema_version") == 1
            and aggregate.get("status") == "completed"
            and aggregate.get("run_id") == RUN_ID
            and aggregate.get("family") == FAMILY
            and aggregate.get("parameter_count") == EXPECTED_PARAMETER_COUNT
            and aggregate.get("planned_model_count") == 8
            and aggregate.get("completed_model_count") == 8
            and aggregate.get("steps_per_model") == 20_000
            and aggregate.get("ensemble_members_precommitted_before_audit") is True
            and aggregate.get("absolute_threshold_used_for_deployment") is False
            and aggregate.get("model_subset_searched_on_audit") is False
            and aggregate.get("final_probe_opened") is False
            and aggregate.get("competition_test_data_read") is False
            and aggregate.get("public_leaderboard_used_for_selection") is False
            and aggregate.get("authorized_for_submission") is False
        ):
            raise ValueError("graph-context aggregate terminal is ineligible")
    if exit_code in (0, 2) and aggregate is None:
        raise ValueError("completed graph-context run lacks an aggregate terminal")
    if exit_code == 0:
        if not (
            aggregate is not None
            and aggregate.get("policy_audit_passed") is True
            and aggregate.get("deployment_policy")
            in {
                "equal_rank_selection_admitted_ensemble",
                "strongest_selection_individual",
            }
            and aggregate.get("deployment_members")
            and set(aggregate["deployment_members"])
            <= set(aggregate.get("independently_strong_members", []))
            and probe_exit_code == 0
            and probe is not None
            and probe.get("schema_version") == 1
            and probe.get("status") == "development_probe_complete"
            and probe.get("run_id") == PROBE_RUN_ID
            and probe.get("selection_policy") == aggregate.get("deployment_policy")
            and probe.get("member_count") == len(aggregate["deployment_members"])
            and probe.get("absolute_threshold_used") is False
            and probe.get("weights_searched_on_probe") is False
            and probe.get("model_subset_searched_on_probe") is False
            and probe.get("competition_test_data_read") is False
            and probe.get("public_leaderboard_used_for_selection") is False
            and probe.get("authorized_for_submission") is False
        ):
            raise ValueError("graph-context development probe is ineligible")
    elif exit_code == 2:
        if probe_exit_code != 4 or probe is not None:
            raise ValueError("rejected graph-context run unexpectedly opened development")
    else:
        raise ValueError(f"unsupported graph-context training exit code: {exit_code}")
    return {
        "schema_version": 1,
        "status": "harvest_verified",
        "run_id": RUN_ID,
        "archive": str(path.resolve()),
        "archive_bytes": path.stat().st_size,
        "archive_sha256": sha256_file(path),
        "training_exit_code": exit_code,
        "development_probe_exit_code": probe_exit_code,
        "development_probe_present": probe is not None,
        "development_probe_sha256": (
            hashlib.sha256(captured[probe_name]).hexdigest() if probe is not None else None
        ),
        "aggregate_terminal_present": aggregate is not None,
        "completed_model_count": aggregate.get("completed_model_count") if aggregate else None,
        "selection_accepted_members": aggregate.get("selection_accepted_members", []) if aggregate else [],
        "independently_strong_members": aggregate.get("independently_strong_members", []) if aggregate else [],
        "deployment_members": aggregate.get("deployment_members", []) if aggregate else [],
        "ensemble_eligible": aggregate.get("ensemble_eligible", False) if aggregate else False,
        "final_probe_opened": bool(probe is not None),
        "competition_test_data_read": False,
        "authorized_for_submission": False,
    }


def extract_verified_archive(path: Path, destination: Path) -> dict[str, Any]:
    result = verify_archive(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            if not _safe(member):
                raise ValueError("graph-context harvest contains an unsafe member")
            if not member.isfile():
                continue
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot extract graph-context member: {member.name}")
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as stream:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    stream.write(block)
    result["extracted_to"] = str(destination.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--extract-to", type=Path)
    args = parser.parse_args()
    result = (
        extract_verified_archive(args.archive, args.extract_to)
        if args.extract_to is not None
        else verify_archive(args.archive)
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".partial")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(args.report)
    print(rendered, end="")


if __name__ == "__main__":
    main()
