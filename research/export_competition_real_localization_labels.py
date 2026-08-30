#!/usr/bin/env python
"""Package train-only Biohub graph labels for CPU replay preprocessing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.build_competition_division_training_inventory import validate_geff_cache
from research.division_recovery_feasibility import graph_plain


RUN_ID = "competition-real-localization-labels-v1"
INVENTORY_RUN_ID = "competition-real-localization-inventory-v1"
EXPECTED_INVENTORY_SHA256 = (
    "55159ef0636d49fcc31eea6d5fe9c327be59c2813d6d0083d6cdfabc9f6112e1"
)
DATASET_ID = "indarkarhana/biohub-real-localization-labels-v1"
ARCHIVE_NAME = "biohub_real_localization_labels_v1.tar.gz"
FINAL_PROBE_STEMS = {
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_inventory(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_INVENTORY_SHA256:
        raise ValueError("real localization inventory hash changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    movies = payload.get("movies")
    if not (
        payload.get("schema_version") == 1
        and payload.get("status") == "complete"
        and payload.get("run_id") == INVENTORY_RUN_ID
        and payload.get("excluded_final_probe_stems") == sorted(FINAL_PROBE_STEMS)
        and payload.get("competition_train_data_read") is True
        and payload.get("competition_test_data_read") is False
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("submission_created") is False
        and payload.get("authorized_for_submission") is False
        and isinstance(movies, list)
        and len(movies) == 115
        and not ({str(row.get("stem")) for row in movies} & FINAL_PROBE_STEMS)
        and {str(row.get("role")) for row in movies}
        == {"optimization", "selection", "sealed_audit"}
    ):
        raise ValueError("real localization inventory is ineligible")
    return payload


def export_labels(
    inventory: dict[str, Any], geff_root: Path, staging_root: Path
) -> dict[str, Any]:
    labels_root = staging_root / "labels"
    labels_root.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    for movie in inventory["movies"]:
        stem = str(movie["stem"])
        nodes, edges = graph_plain(geff_root / "train" / f"{stem}.geff")
        node_ids = np.asarray(sorted(nodes), dtype=np.int64)
        times = np.asarray([int(nodes[int(node_id)][0]) for node_id in node_ids], dtype=np.int32)
        coords = np.asarray(
            [nodes[int(node_id)][1:] for node_id in node_ids], dtype=np.float32
        )
        edge_rows = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        if (
            len(node_ids) == 0
            or coords.shape != (len(node_ids), 3)
            or len(node_ids) != len(np.unique(node_ids))
            or any(int(frame) not in set(times.tolist()) for frame in movie["center_frames"])
        ):
            raise RuntimeError(f"invalid graph labels for {stem}")
        destination = labels_root / f"{stem}.npz"
        np.savez_compressed(
            destination,
            node_ids=node_ids,
            times=times,
            coords_voxel=coords,
            edges=edge_rows,
        )
        records.append(
            {
                "path": destination.relative_to(staging_root).as_posix(),
                "stem": stem,
                "nodes": len(node_ids),
                "edges": len(edge_rows),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    if len(records) != 115:
        raise RuntimeError("label export lost movies")
    return {
        "schema_version": 1,
        "status": "complete",
        "run_id": RUN_ID,
        "inventory_sha256": EXPECTED_INVENTORY_SHA256,
        "movies": inventory["movies"],
        "files": records,
        "summary": {
            "movies": len(records),
            "nodes": sum(row["nodes"] for row in records),
            "edges": sum(row["edges"] for row in records),
            "bytes": sum(row["bytes"] for row in records),
        },
        "excluded_final_probe_stems": sorted(FINAL_PROBE_STEMS),
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--geff-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    inventory = validate_inventory(args.inventory)
    geff_manifest_path = args.geff_cache_root / "train_geff_cache_manifest.json"
    geff_manifest = json.loads(geff_manifest_path.read_text(encoding="utf-8"))
    validate_geff_cache(args.geff_cache_root, geff_manifest)
    if args.output_root.exists():
        raise FileExistsError(f"label upload root exists: {args.output_root}")
    args.output_root.mkdir(parents=True)

    with tempfile.TemporaryDirectory(dir=args.output_root.parent) as temporary:
        staging_root = Path(temporary)
        manifest = export_labels(inventory, args.geff_cache_root, staging_root)
        manifest["geff_cache_manifest_sha256"] = sha256_file(geff_manifest_path)
        atomic_json(staging_root / "labels_manifest.json", manifest)
        (staging_root / "inventory.json").write_bytes(args.inventory.read_bytes())
        labels_manifest_sha256 = sha256_file(staging_root / "labels_manifest.json")
        archive_path = args.output_root / ARCHIVE_NAME
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(staging_root / "inventory.json", arcname="inventory.json")
            archive.add(staging_root / "labels_manifest.json", arcname="labels_manifest.json")
            archive.add(staging_root / "labels", arcname="labels", recursive=True)

    metadata = {
        "title": "Biohub Real Localization Labels v1",
        "id": DATASET_ID,
        "licenses": [{"name": "other"}],
        "isPrivate": True,
    }
    atomic_json(args.output_root / "dataset-metadata.json", metadata)
    upload_manifest = {
        "schema_version": 1,
        "status": "complete",
        "run_id": RUN_ID,
        "dataset_id": DATASET_ID,
        "archive": {
            "path": ARCHIVE_NAME,
            "bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
        },
        "labels_manifest_sha256": labels_manifest_sha256,
        "inventory_sha256": EXPECTED_INVENTORY_SHA256,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }
    atomic_json(args.output_root / "upload_manifest.json", upload_manifest)
    print(json.dumps(upload_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
