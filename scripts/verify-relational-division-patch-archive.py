#!/usr/bin/env python
"""Verify the exact Kaggle relational patch archive without extracting it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any


RUN_ID = "competition-relational-division-patches-v3"
ROOT = "biohub_relational_division_patches_v3"
INVENTORY_SHA256 = "94150632f5a80b2ef48a39743a425cbe1b8e57b1c131c19ef0bde3d97d1c783e"
EXPECTED_SUMMARY = {
    "rows": 3_013,
    "positives": 134,
    "hard_negatives": 2_879,
    "inference_eligible_positives": 55,
    "inference_eligible_hard_negatives": 165,
}


def sha256_stream(handle: Any) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return sha256_stream(handle)


def _safe_member(member: tarfile.TarInfo) -> bool:
    path = PurePosixPath(member.name)
    return bool(
        not member.issym()
        and not member.islnk()
        and not path.is_absolute()
        and ".." not in path.parts
        and path.parts
        and path.parts[0] == ROOT
    )


def verify_archive(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        if not members or any(not _safe_member(member) for member in members):
            raise ValueError("relational archive contains an unsafe member")
        files = {member.name: member for member in members if member.isfile()}
        manifest_name = f"{ROOT}/relational_division_patch_manifest.json"
        if manifest_name not in files:
            raise ValueError("relational archive manifest is missing")
        manifest_handle = archive.extractfile(files[manifest_name])
        if manifest_handle is None:
            raise ValueError("cannot read relational archive manifest")
        manifest_bytes = manifest_handle.read()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        summary = manifest.get("summary", {})
        observed = {key: summary.get(key) for key in EXPECTED_SUMMARY}
        if not (
            manifest.get("schema_version") == 1
            and manifest.get("status") == "complete"
            and manifest.get("run_id") == RUN_ID
            and manifest.get("inventory_sha256") == INVENTORY_SHA256
            and observed == EXPECTED_SUMMARY
            and manifest.get("final_probe_movies_extracted") is False
            and manifest.get("audit_labels_scored") is False
            and manifest.get("competition_test_data_read") is False
            and manifest.get("public_code_copied") is False
            and manifest.get("public_predictions_copied") is False
            and manifest.get("public_leaderboard_used_for_selection") is False
            and manifest.get("submission_created") is False
            and manifest.get("authorized_for_submission") is False
        ):
            raise ValueError("relational archive manifest is ineligible")
        records = manifest.get("records", [])
        record_summary = {
            key: sum(int(record.get(key, -10**9)) for record in records)
            for key in EXPECTED_SUMMARY
        }
        strata = {
            (embryo, role): [
                record
                for record in records
                if record.get("embryo") == embryo and record.get("role") == role
            ]
            for embryo in ("44b6", "6bba")
            for role in ("optimization", "selection", "audit")
        }
        if not (
            records
            and record_summary == EXPECTED_SUMMARY
            and len({record.get("stem") for record in records}) == 146
            and len({record.get("path") for record in records}) == len(records)
            and all(
                rows
                and sum(int(row.get("positives", 0)) for row in rows) > 0
                and sum(int(row.get("hard_negatives", 0)) for row in rows) > 0
                and sum(int(row.get("inference_eligible_positives", 0)) for row in rows) > 0
                and sum(int(row.get("inference_eligible_hard_negatives", 0)) for row in rows) > 0
                for rows in strata.values()
            )
        ):
            raise ValueError("relational archive has no shard records")
        for record in records:
            member_name = f"{ROOT}/{record['path']}"
            member = files.get(member_name)
            if member is None or member.size != int(record["bytes"]):
                raise ValueError(f"relational archive shard inventory changed: {member_name}")
            handle = archive.extractfile(member)
            if handle is None or sha256_stream(handle) != record["sha256"]:
                raise ValueError(f"relational archive shard hash changed: {member_name}")
        expected_files = {manifest_name} | {
            f"{ROOT}/{record['path']}" for record in records
        }
        if set(files) != expected_files:
            raise ValueError("relational archive contains unbound files")
    return {
        "schema_version": 1,
        "status": "verified",
        "run_id": RUN_ID,
        "archive": str(path.resolve()),
        "archive_bytes": path.stat().st_size,
        "archive_sha256": sha256_file(path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "record_count": len(records),
        "summary": observed,
        "audit_labels_scored": False,
        "final_probe_movies_extracted": False,
        "competition_test_data_read": False,
        "authorized_for_submission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify_archive(args.archive)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(args.report.suffix + ".partial")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.report)
    print(rendered, end="")


if __name__ == "__main__":
    main()
