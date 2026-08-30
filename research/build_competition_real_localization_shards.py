#!/usr/bin/env python
"""Convert hash-bound Biohub train frames and GEFFs into pooled replay triplets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.build_competition_division_training_inventory import validate_geff_cache
from research.division_recovery_feasibility import graph_plain
from research.synthetic_pretrain.data import NATIVE_VOXEL_UM, POOLED_VOXEL_UM


RUN_ID = "competition-real-localization-shards-v1"
FRAME_CACHE_RUN_ID = "competition-real-localization-frame-cache-v1"
INVENTORY_RUN_ID = "competition-real-localization-inventory-v1"
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_inputs(
    inventory_path: Path,
    inventory_sha256: str,
    frame_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(inventory_path) != inventory_sha256.lower():
        raise ValueError("real localization inventory hash changed")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    frame_manifest_path = frame_root / "real_localization_frame_cache_manifest.json"
    frame_manifest = json.loads(frame_manifest_path.read_text(encoding="utf-8"))
    movies = inventory.get("movies")
    if not (
        inventory.get("schema_version") == 1
        and inventory.get("status") == "complete"
        and inventory.get("run_id") == INVENTORY_RUN_ID
        and inventory.get("competition_test_data_read") is False
        and inventory.get("public_leaderboard_used_for_selection") is False
        and isinstance(movies, list)
        and movies
        and not (set(movie["stem"] for movie in movies) & FINAL_PROBE_STEMS)
        and frame_manifest.get("schema_version") == 1
        and frame_manifest.get("status") == "complete"
        and frame_manifest.get("run_id") == FRAME_CACHE_RUN_ID
        and frame_manifest.get("inventory_sha256") == inventory_sha256.lower()
        and frame_manifest.get("movies") == movies
        and frame_manifest.get("competition_train_data_read") is True
        and frame_manifest.get("competition_test_data_read") is False
        and frame_manifest.get("public_leaderboard_used_for_selection") is False
        and frame_manifest.get("submission_created") is False
        and frame_manifest.get("authorized_for_submission") is False
    ):
        raise ValueError("real localization frame cache is ineligible")
    for record in frame_manifest["files"]:
        path = frame_root / record["local_relative_path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"real localization frame changed: {path}")
    return inventory, frame_manifest


def triplet_payload(
    nodes: dict[int, tuple[float, ...]],
    edges: list[tuple[int, int]],
    center: int,
    frame_reader: Callable[[int], np.ndarray],
) -> dict[str, np.ndarray]:
    frames = (center - 1, center, center + 1)
    volumes = np.stack([np.asarray(frame_reader(frame)) for frame in frames])
    if volumes.shape[1:] != (64, 256, 256):
        raise ValueError(f"unexpected real localization frame shape: {volumes.shape}")
    selected_ids = sorted(
        node_id for node_id, values in nodes.items() if int(values[0]) in frames
    )
    center_ids = [
        node_id for node_id in selected_ids if int(nodes[node_id][0]) == center
    ]
    if not center_ids:
        raise ValueError("real localization triplet has no center-frame labels")
    by_id = {node_id: row for row, node_id in enumerate(selected_ids)}
    node_rows = np.asarray(
        [
            (
                int(nodes[node_id][0]) - center + 1,
                float(nodes[node_id][1]),
                float(nodes[node_id][2]),
                float(nodes[node_id][3]),
                int(node_id),
            )
            for node_id in selected_ids
        ],
        dtype=np.float32,
    )
    edge_rows = np.asarray(
        [
            (by_id[int(source)], by_id[int(target)])
            for source, target in edges
            if int(source) in by_id and int(target) in by_id
        ],
        dtype=np.int64,
    ).reshape(-1, 2)
    outgoing: dict[int, list[int]] = {}
    for source, target in edge_rows.tolist():
        outgoing.setdefault(int(source), []).append(int(target))
    divisions = np.asarray(
        sorted(source for source, targets in outgoing.items() if len(targets) >= 2),
        dtype=np.int64,
    )
    pooled = np.ascontiguousarray(volumes[:, :, ::4, ::4])
    corrected = node_rows[:, 1:4].copy()
    corrected[:, 1:] /= 4.0
    if (
        pooled.shape != (3, 64, 64, 64)
        or np.any(node_rows[:, 0] < 0)
        or np.any(node_rows[:, 0] > 2)
        or np.any(corrected < 0)
        or np.any(corrected >= np.asarray(pooled.shape[1:])[None])
        or not np.allclose(NATIVE_VOXEL_UM * np.asarray((1, 4, 4)), POOLED_VOXEL_UM)
    ):
        raise ValueError("real localization pooling or coordinates are invalid")
    return {
        "volumes": pooled,
        "nodes": node_rows,
        "edges": edge_rows,
        "divisions": divisions,
        "voxel_um_pooled": POOLED_VOXEL_UM.copy(),
    }


def division_critical_center_rows(payload: dict[str, np.ndarray]) -> int:
    nodes = payload["nodes"]
    outgoing: dict[int, list[int]] = {}
    for source, target in payload["edges"].tolist():
        outgoing.setdefault(int(source), []).append(int(target))
    critical: set[int] = set()
    for source, targets in outgoing.items():
        if len(targets) >= 2:
            critical.add(source)
            critical.update(targets)
    return sum(int(nodes[row, 0]) == 1 for row in critical)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--geff-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    inventory, frame_manifest = validate_inputs(
        args.inventory, args.inventory_sha256, args.frame_root
    )
    geff_manifest_path = args.geff_cache_root / "train_geff_cache_manifest.json"
    geff_manifest = json.loads(geff_manifest_path.read_text(encoding="utf-8"))
    validate_geff_cache(args.geff_cache_root, geff_manifest)
    if args.output_root.exists():
        raise FileExistsError(f"real localization shard output exists: {args.output_root}")
    args.output_root.mkdir(parents=True)

    import zarr

    records: list[dict[str, Any]] = []
    for movie_index, movie in enumerate(inventory["movies"], start=1):
        stem = movie["stem"]
        role = movie["role"]
        nodes, edges = graph_plain(
            args.geff_cache_root / "train" / f"{stem}.geff"
        )
        array = zarr.open_array(
            args.frame_root / "train" / f"{stem}.zarr" / "0", mode="r"
        )
        for center in movie["center_frames"]:
            payload = triplet_payload(nodes, edges, int(center), lambda frame: array[frame])
            destination = args.output_root / role / f"{stem}__t{int(center):04d}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, **payload)
            center_nodes = int(np.sum(payload["nodes"][:, 0] == 1))
            critical = division_critical_center_rows(payload)
            if role in {"selection", "sealed_audit"} and critical <= 0:
                raise RuntimeError(f"real localization gate shard lost division: {destination}")
            records.append(
                {
                    "path": destination.relative_to(args.output_root).as_posix(),
                    "stem": stem,
                    "embryo": movie["embryo"],
                    "role": role,
                    "center_frame": int(center),
                    "nodes": len(payload["nodes"]),
                    "center_nodes": center_nodes,
                    "division_critical_center_nodes": critical,
                    "bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                }
            )
        if movie_index % 20 == 0:
            print(f"built {movie_index}/{len(inventory['movies'])} real movies", flush=True)
    counts = {
        role: {
            "shards": sum(row["role"] == role for row in records),
            "center_nodes": sum(
                row["center_nodes"] for row in records if row["role"] == role
            ),
            "division_critical_center_nodes": sum(
                row["division_critical_center_nodes"]
                for row in records
                if row["role"] == role
            ),
        }
        for role in ("optimization", "selection", "sealed_audit")
    }
    if (
        counts["optimization"]["shards"] != 146
        or counts["selection"]["shards"] != 17
        or counts["sealed_audit"]["shards"] != 14
        or any(counts[role]["center_nodes"] <= 0 for role in counts)
        or any(
            counts[role]["division_critical_center_nodes"] <= 0
            for role in ("selection", "sealed_audit")
        )
    ):
        raise RuntimeError("real localization shard inventory changed")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "run_id": RUN_ID,
        "inventory_sha256": args.inventory_sha256.lower(),
        "frame_cache_manifest_sha256": sha256_file(
            args.frame_root / "real_localization_frame_cache_manifest.json"
        ),
        "geff_cache_manifest_sha256": sha256_file(geff_manifest_path),
        "files": records,
        "summary": {
            "shards": len(records),
            "bytes": sum(row["bytes"] for row in records),
            "by_role": counts,
        },
        "excluded_final_probe_stems": sorted(FINAL_PROBE_STEMS),
        "geometry": {
            "source_shape_zyx": [64, 256, 256],
            "pooled_shape_zyx": [64, 64, 64],
            "xy_stride": 4,
            "pooled_voxel_um": POOLED_VOXEL_UM.tolist(),
        },
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }
    atomic_json(args.output_root / "real_localization_shard_manifest.json", manifest)
    print(json.dumps(manifest["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
