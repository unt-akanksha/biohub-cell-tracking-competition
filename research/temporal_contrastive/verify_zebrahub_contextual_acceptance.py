#!/usr/bin/env python
"""Fail-closed verification for the untouched ZSNS001 acceptance shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from verify_zebrahub_contextual_dataset import (
        DIVISION_QUOTA_FRACTION,
        MAXIMUM_DIVISION_SOURCE_FRACTION,
        PROHIBITED_RAW_SUFFIXES,
        SAMPLING_POLICY,
        safe_file,
        sha256_file,
        validate_arrays,
    )
except ModuleNotFoundError:
    from research.temporal_contrastive.verify_zebrahub_contextual_dataset import (
        DIVISION_QUOTA_FRACTION,
        MAXIMUM_DIVISION_SOURCE_FRACTION,
        PROHIBITED_RAW_SUFFIXES,
        SAMPLING_POLICY,
        safe_file,
        sha256_file,
        validate_arrays,
    )


RUN_ID = "zebrahub-contextual-acceptance-v1"
DATASET_ID = "indarkarhana/biohub-zebrahub-contextual-acceptance-v1"
SOURCE = "ZSNS001"
SOURCE_ROLE = "external_acceptance"
ACCEPTANCE_COUNT = 16
REQUIRED_SOURCE_FILES = {
    "patch_model.py",
    "transition_context.py",
    "verify_zebrahub_contextual_acceptance.py",
    "verify_zebrahub_contextual_dataset.py",
    "zebrahub_external.py",
}


def verify_acceptance(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "DATASET_MANIFEST.json"
    metadata_path = root / "dataset-metadata.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("acceptance dataset manifest is missing")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not (
            metadata.get("id") == DATASET_ID and metadata.get("isPrivate") is True
        ):
            raise ValueError("acceptance publication metadata changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("run_id") == RUN_ID
        and manifest.get("source") == SOURCE
        and manifest.get("source_role") == SOURCE_ROLE
        and manifest.get("selection_policy")
        == "fixed_before_model_weights; one-shot post-selection audit only"
        and manifest.get("organizer_declared_test_overlap") is False
        and manifest.get("competition_test_data_read") is False
        and manifest.get("public_competition_predictions_read") is False
        and manifest.get("leaderboard_used") is False
        and manifest.get("model_predictions_read") is False
        and manifest.get("submission_created") is False
        and manifest.get("raw_movie_files_included") is False
        and manifest.get("sampling_policy") == SAMPLING_POLICY
        and manifest.get("division_quota_fraction") == DIVISION_QUOTA_FRACTION
        and manifest.get("maximum_division_source_fraction")
        == MAXIMUM_DIVISION_SOURCE_FRACTION
    ):
        raise ValueError("acceptance dataset global provenance changed")
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
        raise ValueError(f"acceptance dataset contains raw movie files: {raw_files}")

    inventory = manifest.get("acceptance")
    records = inventory.get("records") if isinstance(inventory, dict) else None
    if not (
        isinstance(inventory, dict)
        and inventory.get("source") == SOURCE
        and inventory.get("split") == "acceptance"
        and inventory.get("source_role") == SOURCE_ROLE
        and inventory.get("required_count") == ACCEPTANCE_COUNT
        and isinstance(records, list)
        and len(records) == ACCEPTANCE_COUNT
        and len(inventory.get("selected_timepoints", [])) == ACCEPTANCE_COUNT
        and len(set(inventory["selected_timepoints"])) == ACCEPTANCE_COUNT
    ):
        raise ValueError("acceptance split contract changed")
    expected_inventory = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if expected_inventory != inventory.get("inventory_sha256"):
        raise ValueError("acceptance inventory hash changed")

    summaries: list[dict[str, int]] = []
    shard_hashes: list[str] = []
    manifest_hashes: list[str] = []
    for record in records:
        shard = safe_file(root, str(record.get("path", "")))
        shard_manifest = safe_file(root, str(record.get("manifest_path", "")))
        actual_shard_hash = sha256_file(shard)
        actual_manifest_hash = sha256_file(shard_manifest)
        if not (
            shard.suffix == ".npz"
            and shard.stat().st_size == record.get("bytes")
            and actual_shard_hash == record.get("sha256")
            and actual_manifest_hash == record.get("manifest_sha256")
        ):
            raise ValueError("acceptance record hash or byte count changed")
        evidence = json.loads(shard_manifest.read_text(encoding="utf-8"))
        if not (
            evidence.get("source") == SOURCE
            and evidence.get("source_role") == SOURCE_ROLE
            and evidence.get("csv_timepoint") == record.get("csv_timepoint")
            and evidence.get("organizer_declared_test_overlap") is False
            and evidence.get("competition_test_data_read") is False
            and evidence.get("public_competition_predictions_read") is False
            and evidence.get("leaderboard_used") is False
            and evidence.get("submission_created") is False
            and evidence.get("candidate_context_width") == 18
            and evidence.get("sampling_policy") == SAMPLING_POLICY
            and evidence.get("division_quota_fraction") == DIVISION_QUOTA_FRACTION
            and evidence.get("maximum_division_source_fraction")
            == MAXIMUM_DIVISION_SOURCE_FRACTION
            and int(evidence.get("source_nodes", 0)) >= 32
            and evidence.get("shard", {}).get("path") == shard.name
            and evidence.get("shard", {}).get("bytes") == shard.stat().st_size
            and evidence.get("shard", {}).get("sha256") == actual_shard_hash
        ):
            raise ValueError(f"acceptance shard evidence changed: {shard.name}")
        summary = validate_arrays(shard)
        if any(evidence.get(key) != value for key, value in summary.items()):
            raise ValueError(f"acceptance shard counts changed: {shard.name}")
        summaries.append(summary)
        shard_hashes.append(actual_shard_hash)
        manifest_hashes.append(actual_manifest_hash)
    if len(set(shard_hashes)) != ACCEPTANCE_COUNT:
        raise ValueError("acceptance contains duplicate shard content")
    if len(list(root.rglob("*.npz"))) != ACCEPTANCE_COUNT:
        raise ValueError("acceptance dataset contains an unmanifested shard")

    generator = manifest.get("generator", {})
    generator_path = safe_file(root, str(generator.get("path", "")))
    if sha256_file(generator_path) != generator.get("sha256"):
        raise ValueError("acceptance generator source hash changed")
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or set(sources) != REQUIRED_SOURCE_FILES:
        raise ValueError("acceptance source inventory changed")
    for name, source in sources.items():
        path = safe_file(root, str(source.get("path", "")))
        if path.name != name or sha256_file(path) != source.get("sha256"):
            raise ValueError(f"acceptance source hash changed: {name}")
    return {
        "schema_version": 1,
        "status": "verified_unopened",
        "run_id": RUN_ID,
        "manifest_sha256": sha256_file(manifest_path),
        "source": SOURCE,
        "source_role": SOURCE_ROLE,
        "shards": ACCEPTANCE_COUNT,
        "source_nodes": sum(row["source_nodes"] for row in summaries),
        "target_nodes": sum(row["target_nodes"] for row in summaries),
        "candidate_edges": sum(row["candidate_edges"] for row in summaries),
        "positive_edges": sum(row["positive_edges"] for row in summaries),
        "division_sources": sum(row["division_sources"] for row in summaries),
        "inventory_sha256": expected_inventory,
        "model_predictions_read": False,
        "competition_test_data_read": False,
        "public_competition_predictions_read": False,
        "leaderboard_used": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_acceptance(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
