#!/usr/bin/env python
"""Add Trackastra-style detection context to sealed relational patch shards.

Only GEFF node ID/time/coordinate arrays are opened. Edge arrays, which encode
the division labels, are never read by this transformation. The source role
split and labels are copied byte-for-value from the already sealed v3 archive.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile
from typing import Any

import numpy as np
import zarr


RUN_ID = "competition-graph-context-relational-patches-v1"
SOURCE_RUN_ID = "competition-relational-division-patches-v3"
INVENTORY_RUN_ID = "competition-relational-division-inventory-v3"
SOURCE_ARCHIVE_SHA256 = "66a822bce0c60d06f6a2b60ada313f0d4d55062de1f84fb60bded4ae456266c2"
SOURCE_MANIFEST_SHA256 = "3ba3f95f5e1cd22044c4022d94bda71280941877ec214130ab98c9e0944f5ab3"
INVENTORY_SHA256 = "94150632f5a80b2ef48a39743a425cbe1b8e57b1c131c19ef0bde3d97d1c783e"
GEFF_MANIFEST_SHA256 = "744f06f75388a7c9199179e5a90bfae4fb3aac835d4e5181893f3745601a9cb9"
SOURCE_ROOT = "biohub_relational_division_patches_v3"
OUTPUT_ROOT_NAME = "biohub_graph_context_relational_patches_v1"
VOXEL_SIZE_ZYX_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float32)
CONTEXT_TIME_OFFSETS = (-2, -1, 0, 1, 2)
CONTEXT_NEIGHBORS_PER_TIME = 8
CONTEXT_RADIUS_UM = 30.0
CONTEXT_SPATIAL_SCALE_UM = 20.0
CONTEXT_TOKEN_COUNT = 3 + len(CONTEXT_TIME_OFFSETS) * CONTEXT_NEIGHBORS_PER_TIME
CONTEXT_FEATURE_WIDTH = 8
EXPECTED_SUMMARY = {
    "rows": 3_013,
    "positives": 134,
    "hard_negatives": 2_879,
    "inference_eligible_positives": 55,
    "inference_eligible_hard_negatives": 165,
}
NODE_ARRAY_PATHS = (
    "nodes/ids",
    "nodes/props/t/values",
    "nodes/props/z/values",
    "nodes/props/y/values",
    "nodes/props/x/values",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_detection_nodes(path: Path) -> dict[int, tuple[int, np.ndarray]]:
    """Read only GEFF node arrays; do not instantiate or inspect edge groups."""

    group = zarr.open_group(str(path), mode="r")
    arrays = [np.asarray(group[name][:]) for name in NODE_ARRAY_PATHS]
    if not arrays or len({len(array) for array in arrays}) != 1:
        raise ValueError(f"inconsistent GEFF node arrays: {path}")
    ids, times, z_values, y_values, x_values = arrays
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"duplicate GEFF node IDs: {path}")
    return {
        int(node_id): (
            int(timepoint),
            np.asarray((z, y, x), dtype=np.float32) * VOXEL_SIZE_ZYX_UM,
        )
        for node_id, timepoint, z, y, x in zip(
            ids, times, z_values, y_values, x_values, strict=True
        )
    }


def _token(
    *,
    relative_time: int,
    delta_um: np.ndarray,
    parent_anchor: bool,
    daughter_anchor: bool,
    context_node: bool,
) -> np.ndarray:
    distance = float(np.linalg.norm(delta_um))
    return np.asarray(
        (
            relative_time / max(abs(value) for value in CONTEXT_TIME_OFFSETS),
            *(np.clip(delta_um / CONTEXT_SPATIAL_SCALE_UM, -2.0, 2.0)),
            math.log1p(distance) / math.log1p(CONTEXT_RADIUS_UM),
            float(parent_anchor),
            float(daughter_anchor),
            float(context_node),
        ),
        dtype=np.float32,
    )


def context_tokens(
    nodes: dict[int, tuple[int, np.ndarray]],
    *,
    parent_id: int,
    existing_child_id: int,
    proposed_child_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    anchor_ids = {int(parent_id), int(existing_child_id), int(proposed_child_id)}
    if len(anchor_ids) != 3 or not anchor_ids.issubset(nodes):
        raise ValueError("graph-context candidate anchors are invalid")
    parent_time, parent_position = nodes[int(parent_id)]
    daughter_rows = [nodes[int(existing_child_id)], nodes[int(proposed_child_id)]]
    if any(timepoint != parent_time + 1 for timepoint, _position in daughter_rows):
        raise ValueError("graph-context daughters must follow the parent")

    features = np.zeros((CONTEXT_TOKEN_COUNT, CONTEXT_FEATURE_WIDTH), dtype=np.float32)
    mask = np.zeros(CONTEXT_TOKEN_COUNT, dtype=np.bool_)
    features[0] = _token(
        relative_time=0,
        delta_um=np.zeros(3, dtype=np.float32),
        parent_anchor=True,
        daughter_anchor=False,
        context_node=False,
    )
    mask[0] = True
    ordered_daughters = sorted(
        (
            (tuple(float(value) for value in position), node_id, position)
            for node_id, (_timepoint, position) in (
                (int(existing_child_id), nodes[int(existing_child_id)]),
                (int(proposed_child_id), nodes[int(proposed_child_id)]),
            )
        ),
        key=lambda row: (row[0], row[1]),
    )
    for token_index, (_key, _node_id, position) in enumerate(
        ordered_daughters, start=1
    ):
        features[token_index] = _token(
            relative_time=1,
            delta_um=position - parent_position,
            parent_anchor=False,
            daughter_anchor=True,
            context_node=False,
        )
        mask[token_index] = True

    by_time: dict[int, list[tuple[int, np.ndarray]]] = {}
    for node_id, (timepoint, position) in nodes.items():
        if node_id not in anchor_ids:
            by_time.setdefault(timepoint, []).append((node_id, position))
    token_index = 3
    for relative_time in CONTEXT_TIME_OFFSETS:
        rows = []
        for node_id, position in by_time.get(parent_time + relative_time, ()):
            delta = position - parent_position
            distance = float(np.linalg.norm(delta))
            if distance <= CONTEXT_RADIUS_UM:
                rows.append((distance, int(node_id), delta))
        rows.sort(key=lambda row: (row[0], row[1]))
        for distance, _node_id, delta in rows[:CONTEXT_NEIGHBORS_PER_TIME]:
            if not math.isfinite(distance):
                raise ValueError("non-finite graph-context node distance")
            features[token_index] = _token(
                relative_time=relative_time,
                delta_um=delta,
                parent_anchor=False,
                daughter_anchor=False,
                context_node=True,
            )
            mask[token_index] = True
            token_index += 1
        token_index += CONTEXT_NEIGHBORS_PER_TIME - min(
            len(rows), CONTEXT_NEIGHBORS_PER_TIME
        )
    if token_index != CONTEXT_TOKEN_COUNT:
        raise RuntimeError("graph-context token layout changed")
    return features, mask


def validate_inputs(
    archive: Path, inventory_path: Path, geff_root: Path, geff_manifest: Path
) -> tuple[dict[str, Any], dict[str, Any], bytes, dict[str, tarfile.TarInfo]]:
    if sha256_file(archive) != SOURCE_ARCHIVE_SHA256:
        raise ValueError("source relational archive changed")
    if sha256_file(inventory_path) != INVENTORY_SHA256:
        raise ValueError("source relational inventory changed")
    if sha256_file(geff_manifest) != GEFF_MANIFEST_SHA256:
        raise ValueError("source GEFF cache manifest changed")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if not (
        inventory.get("run_id") == INVENTORY_RUN_ID
        and inventory.get("status") == "complete"
        and inventory.get("source_geff_manifest_sha256") == GEFF_MANIFEST_SHA256
        and inventory.get("summary", {}).get("rows") == EXPECTED_SUMMARY["rows"]
        and inventory.get("audit_opened") is False
        and inventory.get("competition_test_data_read") is False
        and inventory.get("public_leaderboard_used_for_selection") is False
    ):
        raise ValueError("source relational inventory is ineligible")
    if len(list(geff_root.glob("*.geff"))) != 199:
        raise ValueError("complete competition-train GEFF cache is required")
    with tarfile.open(archive, "r:gz") as source:
        members = {member.name: member for member in source.getmembers() if member.isfile()}
        manifest_names = [
            name for name in members if name.endswith("/relational_division_patch_manifest.json")
        ]
        if len(manifest_names) != 1:
            raise ValueError("source relational archive manifest is ambiguous")
        stream = source.extractfile(members[manifest_names[0]])
        if stream is None:
            raise ValueError("source relational manifest cannot be read")
        manifest_bytes = stream.read()
    if sha256_bytes(manifest_bytes) != SOURCE_MANIFEST_SHA256:
        raise ValueError("source relational manifest changed")
    manifest = json.loads(manifest_bytes)
    if not (
        manifest.get("run_id") == SOURCE_RUN_ID
        and manifest.get("status") == "complete"
        and len(manifest.get("records", [])) == 2_274
        and all(manifest.get("summary", {}).get(key) == value for key, value in EXPECTED_SUMMARY.items())
        and manifest.get("audit_labels_scored") is False
        and manifest.get("final_probe_movies_extracted") is False
    ):
        raise ValueError("source relational manifest is ineligible")
    return inventory, manifest, manifest_bytes, members


def write_enriched_shard(
    source_bytes: bytes,
    output_path: Path,
    nodes: dict[int, tuple[int, np.ndarray]],
) -> dict[str, int]:
    with np.load(io.BytesIO(source_bytes), allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    row_count = len(arrays["parent_ids"])
    contexts = np.zeros(
        (row_count, CONTEXT_TOKEN_COUNT, CONTEXT_FEATURE_WIDTH), dtype=np.float16
    )
    masks = np.zeros((row_count, CONTEXT_TOKEN_COUNT), dtype=np.bool_)
    for index, (parent, existing, proposed) in enumerate(
        zip(
            arrays["parent_ids"],
            arrays["existing_child_ids"],
            arrays["proposed_child_ids"],
            strict=True,
        )
    ):
        context, mask = context_tokens(
            nodes,
            parent_id=int(parent),
            existing_child_id=int(existing),
            proposed_child_id=int(proposed),
        )
        contexts[index] = context.astype(np.float16)
        masks[index] = mask
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata.update(
        {
            "run_id": RUN_ID,
            "source_run_id": SOURCE_RUN_ID,
            "context_layout": "parent,two-symmetric-daughters,five-times-eight-nearest-detections",
            "context_token_count": CONTEXT_TOKEN_COUNT,
            "context_feature_width": CONTEXT_FEATURE_WIDTH,
            "context_radius_um": CONTEXT_RADIUS_UM,
            "context_edges_read": False,
            "context_labels_used": False,
        }
    )
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".npz.partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            **arrays,
            graph_context_features=contexts,
            graph_context_mask=masks,
        )
    temporary.replace(output_path)
    return {"rows": row_count, "valid_tokens": int(masks.sum())}


def deterministic_archive(root: Path, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".partial")
    with temporary.open("wb") as raw_stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
                    info = archive.gettarinfo(
                        str(path), arcname=path.relative_to(root.parent).as_posix()
                    )
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    if path.is_file():
                        with path.open("rb") as stream:
                            archive.addfile(info, stream)
                    else:
                        archive.addfile(info)
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--geff-root", type=Path, required=True)
    parser.add_argument("--geff-manifest", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_parent / OUTPUT_ROOT_NAME
    output_archive = args.output_parent / f"{OUTPUT_ROOT_NAME}.tar.gz"
    if output_root.exists() or output_archive.exists():
        raise FileExistsError("graph-context output already exists")

    inventory, source_manifest, _manifest_bytes, members = validate_inputs(
        args.source_archive, args.inventory, args.geff_root, args.geff_manifest
    )
    examples = inventory["examples"]
    example_keys = {
        (
            str(row["stem"]),
            int(row["parent_id"]),
            int(row["existing_child_id"]),
            int(row["proposed_child_id"]),
        )
        for row in examples
    }
    if len(example_keys) != EXPECTED_SUMMARY["rows"]:
        raise ValueError("source relational example identities are not unique")

    output_root.mkdir(parents=True)
    graph_cache: dict[str, dict[int, tuple[int, np.ndarray]]] = {}
    records: list[dict[str, Any]] = []
    total_valid_tokens = 0
    with tarfile.open(args.source_archive, "r:gz") as source:
        for index, record in enumerate(source_manifest["records"], start=1):
            source_name = f"{SOURCE_ROOT}/{record['path']}"
            member = members.get(source_name)
            if member is None:
                raise FileNotFoundError(f"source relational shard is missing: {source_name}")
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError(f"source relational shard cannot be read: {source_name}")
            source_bytes = stream.read()
            if len(source_bytes) != int(record["bytes"]) or sha256_bytes(source_bytes) != record["sha256"]:
                raise ValueError(f"source relational shard changed: {source_name}")
            stem = str(record["stem"])
            if stem not in graph_cache:
                graph_path = args.geff_root / f"{stem}.geff"
                if not graph_path.is_dir():
                    raise FileNotFoundError(f"source train GEFF is missing: {stem}")
                graph_cache[stem] = load_detection_nodes(graph_path)
            output_path = output_root / str(record["path"])
            counts = write_enriched_shard(source_bytes, output_path, graph_cache[stem])
            total_valid_tokens += counts["valid_tokens"]
            records.append(
                {
                    **record,
                    "bytes": output_path.stat().st_size,
                    "sha256": sha256_file(output_path),
                    "source_sha256": record["sha256"],
                    "context_token_count": CONTEXT_TOKEN_COUNT,
                    "context_feature_width": CONTEXT_FEATURE_WIDTH,
                    "context_valid_tokens": counts["valid_tokens"],
                }
            )
            if index % 250 == 0:
                print(f"enriched {index}/{len(source_manifest['records'])} shards", flush=True)

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "run_id": RUN_ID,
        "source_run_id": SOURCE_RUN_ID,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "inventory_sha256": INVENTORY_SHA256,
        "source_geff_manifest_sha256": GEFF_MANIFEST_SHA256,
        "records": records,
        "summary": {
            **source_manifest["summary"],
            "context_valid_tokens": total_valid_tokens,
        },
        "context_time_offsets": list(CONTEXT_TIME_OFFSETS),
        "context_neighbors_per_time": CONTEXT_NEIGHBORS_PER_TIME,
        "context_radius_um": CONTEXT_RADIUS_UM,
        "context_token_count": CONTEXT_TOKEN_COUNT,
        "context_feature_width": CONTEXT_FEATURE_WIDTH,
        "context_node_arrays_read": list(NODE_ARRAY_PATHS),
        "context_edge_arrays_read": [],
        "context_edges_read": False,
        "context_labels_used": False,
        "daughter_order_invariant": True,
        "audit_features_extracted": True,
        "audit_labels_scored": False,
        "final_probe_stems": source_manifest["final_probe_stems"],
        "final_probe_movies_extracted": False,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }
    manifest_path = output_root / "graph_context_relational_patch_manifest.json"
    atomic_json(manifest_path, manifest)
    deterministic_archive(output_root, output_archive)
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "archive": str(output_archive),
                "archive_sha256": sha256_file(output_archive),
                "records": len(records),
                "summary": manifest["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
