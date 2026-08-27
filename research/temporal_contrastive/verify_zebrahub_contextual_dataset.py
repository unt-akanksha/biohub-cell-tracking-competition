#!/usr/bin/env python
"""Fail-closed verification for the derived ZebraHub contextual dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


RUN_ID = "zebrahub-contextual-shards-v1"
SPLIT_CONTRACT = {
    "training": {
        "directory": "train",
        "source": "ZSNS004",
        "source_role": "external_pretraining",
        "count": 64,
    },
    "validation": {
        "directory": "validation",
        "source": "ZSNS005",
        "source_role": "external_validation",
        "count": 16,
    },
}
PROHIBITED_RAW_SUFFIXES = {".blosc", ".csv", ".zarray", ".zarr"}
TRANSITION_CONTEXT_WIDTH = 8
SAMPLING_POLICY = "seeded_local_neighborhood_balanced_v2"
DIVISION_QUOTA_FRACTION = 0.125
MAXIMUM_DIVISION_SOURCE_FRACTION = 0.25
REQUIRED_SOURCE_FILES = {
    "zebrahub_external.py",
    "transition_context.py",
    "patch_model.py",
    "verify_zebrahub_contextual_dataset.py",
}


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


def validate_arrays(path: Path) -> dict[str, int]:
    with np.load(path) as data:
        required = {
            "source_patches",
            "target_patches",
            "source_ids",
            "target_ids",
            "source_coords_um",
            "target_coords_um",
            "candidate_mask",
            "positive_mask",
            "division_target",
            "transition_context",
            "candidate_context",
        }
        if set(data.files) != required:
            raise ValueError(f"derived shard array inventory changed: {path.name}")
        source_count = len(data["source_ids"])
        target_count = len(data["target_ids"])
        candidates = data["candidate_mask"]
        positives = data["positive_mask"]
        division_target = data["division_target"]
        expected_shapes = {
            "source_patches": (source_count, 3, 17, 17, 17),
            "target_patches": (target_count, 3, 17, 17, 17),
            "source_coords_um": (source_count, 3),
            "target_coords_um": (target_count, 3),
            "candidate_mask": (source_count, target_count),
            "positive_mask": (source_count, target_count),
            "division_target": (source_count,),
            "transition_context": (TRANSITION_CONTEXT_WIDTH,),
            "candidate_context": (source_count, target_count, 18),
        }
        for name, shape in expected_shapes.items():
            if data[name].shape != shape:
                raise ValueError(f"derived shard {name} shape changed: {path.name}")
        if candidates.dtype != np.bool_ or positives.dtype != np.bool_:
            raise ValueError(f"derived shard masks are not boolean: {path.name}")
        if np.any(positives & ~candidates):
            raise ValueError(f"derived positives escape candidates: {path.name}")
        if np.any(candidates.sum(axis=1) <= positives.sum(axis=1)):
            raise ValueError(f"derived source lacks a hard negative: {path.name}")
        expected_divisions = (positives.sum(axis=1) >= 2).astype(np.float32)
        if not np.array_equal(division_target, expected_divisions):
            raise ValueError(f"derived division targets changed: {path.name}")
        division_count = int(division_target.sum())
        if division_count > max(
            1, int(np.floor(source_count * MAXIMUM_DIVISION_SOURCE_FRACTION))
        ):
            raise ValueError(f"derived division sampling is imbalanced: {path.name}")
        if not np.any(expected_divisions == 0.0):
            raise ValueError(f"derived shard contains no ordinary links: {path.name}")
        numeric = (
            data["source_patches"],
            data["target_patches"],
            data["source_coords_um"],
            data["target_coords_um"],
            division_target,
            data["transition_context"],
            data["candidate_context"],
        )
        if any(not np.isfinite(value).all() for value in numeric):
            raise ValueError(f"derived shard contains non-finite values: {path.name}")
        if len(np.unique(data["source_ids"])) != source_count:
            raise ValueError(f"derived source identifiers repeat: {path.name}")
        if len(np.unique(data["target_ids"])) != target_count:
            raise ValueError(f"derived target identifiers repeat: {path.name}")
        return {
            "source_nodes": source_count,
            "target_nodes": target_count,
            "candidate_edges": int(candidates.sum()),
            "positive_edges": int(positives.sum()),
            "division_sources": division_count,
        }


def verify_split(
    root: Path,
    name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    contract = SPLIT_CONTRACT[name]
    records = payload.get("records")
    if not (
        payload.get("source") == contract["source"]
        and payload.get("split") == contract["directory"]
        and payload.get("source_role") == contract["source_role"]
        and payload.get("required_count") == contract["count"]
        and isinstance(records, list)
        and len(records) == contract["count"]
        and len(payload.get("selected_timepoints", [])) == contract["count"]
    ):
        raise ValueError(f"{name} split contract changed")
    if len(set(payload["selected_timepoints"])) != contract["count"]:
        raise ValueError(f"{name} selected timepoints repeat")
    summaries: list[dict[str, int]] = []
    shard_hashes: list[str] = []
    manifest_hashes: list[str] = []
    for record in records:
        shard = safe_file(root, str(record["path"]))
        shard_manifest = safe_file(root, str(record["manifest_path"]))
        actual_shard_hash = sha256_file(shard)
        actual_manifest_hash = sha256_file(shard_manifest)
        if not (
            shard.suffix == ".npz"
            and shard.stat().st_size == record.get("bytes")
            and actual_shard_hash == record.get("sha256")
            and actual_manifest_hash == record.get("manifest_sha256")
        ):
            raise ValueError(f"{name} record hash or byte count changed")
        evidence = json.loads(shard_manifest.read_text(encoding="utf-8"))
        if not (
            evidence.get("source") == contract["source"]
            and evidence.get("source_role") == contract["source_role"]
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
            raise ValueError(f"{name} per-shard evidence changed: {shard.name}")
        summary = validate_arrays(shard)
        if any(evidence.get(key) != value for key, value in summary.items()):
            raise ValueError(f"{name} per-shard counts changed: {shard.name}")
        summaries.append(summary)
        shard_hashes.append(actual_shard_hash)
        manifest_hashes.append(actual_manifest_hash)
    expected_inventory = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if expected_inventory != payload.get("inventory_sha256"):
        raise ValueError(f"{name} inventory hash changed")
    if len(set(shard_hashes)) != len(shard_hashes):
        raise ValueError(f"{name} contains duplicate shard content")
    return {
        "source": contract["source"],
        "source_role": contract["source_role"],
        "shards": len(records),
        "shard_hashes": shard_hashes,
        "manifest_hashes": manifest_hashes,
        "source_nodes": sum(row["source_nodes"] for row in summaries),
        "target_nodes": sum(row["target_nodes"] for row in summaries),
        "candidate_edges": sum(row["candidate_edges"] for row in summaries),
        "positive_edges": sum(row["positive_edges"] for row in summaries),
        "division_sources": sum(row["division_sources"] for row in summaries),
        "inventory_sha256": expected_inventory,
    }


def verify_dataset(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "DATASET_MANIFEST.json"
    metadata_path = root / "dataset-metadata.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("derived dataset manifest is missing")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not (
            metadata.get("id")
            == "indarkarhana/biohub-zebrahub-contextual-shards-v1"
            and metadata.get("isPrivate") is True
        ):
            raise ValueError("derived dataset publication metadata changed")
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
        and manifest.get("sampling_policy") == SAMPLING_POLICY
        and manifest.get("division_quota_fraction") == DIVISION_QUOTA_FRACTION
        and manifest.get("maximum_division_source_fraction")
        == MAXIMUM_DIVISION_SOURCE_FRACTION
    ):
        raise ValueError("derived dataset global provenance changed")
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
        raise ValueError(f"derived dataset contains raw movie files: {raw_files}")
    training = verify_split(root, "training", manifest["training"])
    validation = verify_split(root, "validation", manifest["validation"])
    if set(training["shard_hashes"]) & set(validation["shard_hashes"]):
        raise ValueError("derived train and validation shard content overlaps")
    expected_npz = training["shards"] + validation["shards"]
    actual_npz = list(root.rglob("*.npz"))
    if len(actual_npz) != expected_npz:
        raise ValueError("derived dataset contains an unmanifested shard")
    generator = manifest.get("generator", {})
    generator_path = safe_file(root, str(generator.get("path", "")))
    if sha256_file(generator_path) != generator.get("sha256"):
        raise ValueError("derived dataset generator source hash changed")
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or set(sources) != REQUIRED_SOURCE_FILES:
        raise ValueError("derived dataset source inventory changed")
    for name, source in sources.items():
        path = safe_file(root, str(source.get("path", "")))
        if path.name != name or sha256_file(path) != source.get("sha256"):
            raise ValueError(f"derived dataset source hash changed: {name}")
    return {
        "schema_version": 1,
        "status": "verified",
        "run_id": RUN_ID,
        "manifest_sha256": sha256_file(manifest_path),
        "training": {key: value for key, value in training.items() if not key.endswith("hashes")},
        "validation": {key: value for key, value in validation.items() if not key.endswith("hashes")},
        "raw_movie_files_included": False,
        "competition_test_data_read": False,
        "public_competition_predictions_read": False,
        "leaderboard_used": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_dataset(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
