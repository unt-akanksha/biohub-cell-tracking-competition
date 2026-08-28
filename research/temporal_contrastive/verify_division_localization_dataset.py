#!/usr/bin/env python
"""Fail-closed verification for frozen division-localization shards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from verify_zebrahub_contextual_dataset import validate_arrays
except ModuleNotFoundError:
    from research.temporal_contrastive.verify_zebrahub_contextual_dataset import (
        validate_arrays,
    )


RUN_ID = "zebrahub-division-localization-shards-v1"
DATASET_ID = "indarkarhana/biohub-division-localization-shards-v1"
SPLIT_CONTRACT = {
    "selection": (156, 157, 158, 159, 456, 457, 458, 459),
    "audit": (276, 277, 278, 279, 556, 557, 558, 559),
}
V3_V4_VALIDATION_TIMEPOINTS = frozenset(
    {
        96, 97, 98, 99,
        236, 237, 238, 239,
        376, 377, 378, 379,
        516, 517, 518, 519,
    }
)
PROHIBITED_RAW_SUFFIXES = {".blosc", ".csv", ".zarray", ".zarr"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_file(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"dataset path escapes its root: {relative}") from error
    if not candidate.is_file():
        raise FileNotFoundError(f"dataset file is missing: {relative}")
    return candidate


def verify_split(
    root: Path, name: str, payload: dict[str, Any]
) -> dict[str, Any]:
    expected_times = SPLIT_CONTRACT[name]
    records = payload.get("records")
    if not (
        payload.get("source") == "ZSNS005"
        and payload.get("split") == name
        and payload.get("source_role") == "external_validation"
        and payload.get("selected_timepoints") == list(expected_times)
        and isinstance(records, list)
        and len(records) == len(expected_times)
    ):
        raise ValueError(f"{name} localization split contract changed")
    if set(expected_times) & V3_V4_VALIDATION_TIMEPOINTS:
        raise ValueError(f"{name} overlaps v3/v4 external validation")
    summaries: list[dict[str, int]] = []
    for record, timepoint in zip(records, expected_times, strict=True):
        if record.get("csv_timepoint") != timepoint:
            raise ValueError(f"{name} localization timepoint order changed")
        shard = safe_file(root, str(record["path"]))
        evidence_path = safe_file(root, str(record["manifest_path"]))
        shard_hash = sha256_file(shard)
        evidence_hash = sha256_file(evidence_path)
        if not (
            shard.suffix == ".npz"
            and shard.stat().st_size == record.get("bytes")
            and shard_hash == record.get("sha256")
            and evidence_hash == record.get("manifest_sha256")
        ):
            raise ValueError(f"{name} localization record changed")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if not (
            evidence.get("source") == "ZSNS005"
            and evidence.get("source_role") == "external_validation"
            and evidence.get("csv_timepoint") == timepoint
            and evidence.get("patch_shape") == [17, 17, 17]
            and evidence.get("patch_half_extent_um") == [8.0, 8.0, 8.0]
            and evidence.get("organizer_declared_test_overlap") is False
            and evidence.get("competition_test_data_read") is False
            and evidence.get("public_competition_predictions_read") is False
            and evidence.get("leaderboard_used") is False
            and evidence.get("submission_created") is False
            and evidence.get("shard", {}).get("sha256") == shard_hash
        ):
            raise ValueError(f"{name} localization shard evidence changed")
        summary = validate_arrays(shard)
        if any(evidence.get(key) != value for key, value in summary.items()):
            raise ValueError(f"{name} localization shard counts changed")
        summaries.append(summary)
    expected_inventory = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if payload.get("inventory_sha256") != expected_inventory:
        raise ValueError(f"{name} localization inventory hash changed")
    return {
        "shards": len(records),
        "timepoints": list(expected_times),
        "source_nodes": sum(row["source_nodes"] for row in summaries),
        "target_nodes": sum(row["target_nodes"] for row in summaries),
        "division_sources": sum(row["division_sources"] for row in summaries),
        "inventory_sha256": expected_inventory,
    }


def verify_dataset(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "DATASET_MANIFEST.json"
    metadata_path = root / "dataset-metadata.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("localization dataset manifest is missing")
    # Kaggle consumes dataset-metadata.json during publication and omits it
    # from downloads. Verify it when present in staging, but let the immutable
    # manifest and per-file hashes authenticate a remote round trip.
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not (
            metadata.get("id") == DATASET_ID and metadata.get("isPrivate") is True
        ):
            raise ValueError("localization publication metadata changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("run_id") == RUN_ID
        and manifest.get("organizer_declared_test_overlap") is False
        and manifest.get("competition_test_data_read") is False
        and manifest.get("public_competition_predictions_read") is False
        and manifest.get("leaderboard_used") is False
        and manifest.get("submission_created") is False
        and manifest.get("raw_movie_files_included") is False
        and manifest.get("selection_and_audit_frozen_before_extraction") is True
    ):
        raise ValueError("localization dataset provenance changed")
    raw_files = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() in PROHIBITED_RAW_SUFFIXES
            or path.name in {".zarray", ".zattrs", ".zgroup"}
        )
    ]
    if raw_files:
        raise ValueError(f"localization dataset contains raw source files: {raw_files}")
    selection = verify_split(root, "selection", manifest["selection"])
    audit = verify_split(root, "audit", manifest["audit"])
    if set(selection["timepoints"]) & set(audit["timepoints"]):
        raise ValueError("localization selection and audit overlap")
    return {
        "run_id": RUN_ID,
        "selection": selection,
        "audit": audit,
        "manifest_sha256": sha256_file(manifest_path),
        "competition_test_data_read": False,
        "public_competition_predictions_read": False,
        "leaderboard_used": False,
        "submission_created": False,
    }
