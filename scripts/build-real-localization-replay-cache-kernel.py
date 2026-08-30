#!/usr/bin/env python
"""Build the CPU-only Kaggle fallback for train replay materialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL_ID = "biohub-real-localization-replay-cache-v1"
KERNEL_REF = f"indarkarhana/{KERNEL_ID}"
DATASET_REF = "indarkarhana/biohub-real-localization-labels-v1"
ARCHIVE_NAME = "biohub_real_localization_labels_v1.tar.gz"
EXPECTED_INVENTORY_SHA256 = (
    "55159ef0636d49fcc31eea6d5fe9c327be59c2813d6d0083d6cdfabc9f6112e1"
)


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def kernel_source(*, archive_sha256: str, labels_manifest_sha256: str) -> str:
    return f'''import hashlib
import json
from pathlib import Path
import tarfile
import time

import numpy as np
import zarr

RUN_ID = "competition-real-localization-shards-v1"
LABEL_RUN_ID = "competition-real-localization-labels-v1"
EXPECTED_ARCHIVE_SHA256 = "{archive_sha256}"
EXPECTED_LABELS_MANIFEST_SHA256 = "{labels_manifest_sha256}"
EXPECTED_INVENTORY_SHA256 = "{EXPECTED_INVENTORY_SHA256}"
FINAL_PROBE_STEMS = {{
    "44b6_12dfb391", "44b6_267148e4", "6bba_062c8d37", "6bba_07e24132"
}}
STARTED = time.monotonic()
INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
OUTPUT_ROOT = WORKING / "competition_real_localization_shards_v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
    temporary.replace(path)


archives = sorted(INPUT_ROOT.glob("*/{ARCHIVE_NAME}"))
if len(archives) != 1 or sha256_file(archives[0]) != EXPECTED_ARCHIVE_SHA256:
    raise RuntimeError({{"eligible_label_archives": [str(path) for path in archives]}})
label_root = WORKING / "real_localization_labels_v1"
label_root.mkdir()
with tarfile.open(archives[0], "r:gz") as archive:
    members = archive.getmembers()
    if any(
        member.name.startswith(("/", "\\\\"))
        or ".." in Path(member.name).parts
        for member in members
    ):
        raise RuntimeError("unsafe label archive member")
    archive.extractall(label_root)
labels_manifest_path = label_root / "labels_manifest.json"
inventory_path = label_root / "inventory.json"
if (
    sha256_file(labels_manifest_path) != EXPECTED_LABELS_MANIFEST_SHA256
    or sha256_file(inventory_path) != EXPECTED_INVENTORY_SHA256
):
    raise RuntimeError("label package provenance changed")
labels_manifest = json.loads(labels_manifest_path.read_text())
inventory = json.loads(inventory_path.read_text())
movies = inventory.get("movies")
if not (
    labels_manifest.get("status") == "complete"
    and labels_manifest.get("run_id") == LABEL_RUN_ID
    and labels_manifest.get("inventory_sha256") == EXPECTED_INVENTORY_SHA256
    and labels_manifest.get("competition_test_data_read") is False
    and labels_manifest.get("public_leaderboard_used_for_selection") is False
    and labels_manifest.get("authorized_for_submission") is False
    and inventory.get("status") == "complete"
    and inventory.get("competition_train_data_read") is True
    and inventory.get("competition_test_data_read") is False
    and inventory.get("public_leaderboard_used_for_selection") is False
    and inventory.get("authorized_for_submission") is False
    and isinstance(movies, list)
    and len(movies) == 115
    and not ({{row["stem"] for row in movies}} & FINAL_PROBE_STEMS)
):
    raise RuntimeError("ineligible train-only replay contract")
label_records = {{row["stem"]: row for row in labels_manifest["files"]}}
if set(label_records) != {{row["stem"] for row in movies}}:
    raise RuntimeError("label movie inventory changed")
for stem, row in label_records.items():
    path = label_root / row["path"]
    if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
        raise RuntimeError(f"label file changed: {{stem}}")

candidate_paths = [
    Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train"),
    Path("/kaggle/input/biohub-cell-tracking-during-development/train"),
]
candidate_paths.extend(path / "train" for path in INPUT_ROOT.iterdir() if path.is_dir())
train_roots = []
for path in candidate_paths:
    if path.is_dir() and len(list(path.glob("*.geff"))) == 199:
        resolved = path.resolve()
        if resolved not in train_roots:
            train_roots.append(resolved)
if len(train_roots) != 1:
    raise RuntimeError({{"eligible_train_roots": [str(path) for path in train_roots]}})
train_root = train_roots[0]
if train_root.name != "train" or "test" in train_root.as_posix().lower():
    raise RuntimeError("competition train root boundary failed")

OUTPUT_ROOT.mkdir()
records = []
frame_records = []
for movie_index, movie in enumerate(movies, start=1):
    stem = movie["stem"]
    role = movie["role"]
    with np.load(label_root / label_records[stem]["path"], allow_pickle=False) as data:
        node_ids = data["node_ids"].astype(np.int64, copy=False)
        times = data["times"].astype(np.int32, copy=False)
        coords = data["coords_voxel"].astype(np.float32, copy=False)
        edges_by_id = data["edges"].astype(np.int64, copy=False).reshape(-1, 2)
    array = zarr.open_array(train_root / f"{{stem}}.zarr" / "0", mode="r")
    if tuple(array.shape[1:]) != (64, 256, 256):
        raise RuntimeError(f"unexpected image shape: {{stem}} {{array.shape}}")
    frames = {{}}
    for frame in movie["required_frames"]:
        volume = np.asarray(array[int(frame)])
        if volume.shape != (64, 256, 256):
            raise RuntimeError(f"unexpected frame shape: {{stem}} t={{frame}}")
        frames[int(frame)] = volume
        frame_records.append({{
            "stem": stem,
            "frame": int(frame),
            "shape": list(volume.shape),
            "dtype": str(volume.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(volume).tobytes()).hexdigest(),
        }})
    for center in movie["center_frames"]:
        center = int(center)
        triplet_times = (center - 1, center, center + 1)
        selected_mask = np.isin(times, triplet_times)
        selected_ids = node_ids[selected_mask]
        selected_times = times[selected_mask]
        selected_coords = coords[selected_mask]
        if not np.any(selected_times == center):
            raise RuntimeError(f"center labels missing: {{stem}} t={{center}}")
        by_id = {{int(node_id): row for row, node_id in enumerate(selected_ids)}}
        edge_rows = np.asarray([
            (by_id[int(source)], by_id[int(target)])
            for source, target in edges_by_id.tolist()
            if int(source) in by_id and int(target) in by_id
        ], dtype=np.int64).reshape(-1, 2)
        outgoing = {{}}
        for source, target in edge_rows.tolist():
            outgoing.setdefault(int(source), []).append(int(target))
        divisions = np.asarray(sorted(
            source for source, targets in outgoing.items() if len(targets) >= 2
        ), dtype=np.int64)
        volumes = np.stack([frames[frame] for frame in triplet_times])
        pooled = np.ascontiguousarray(volumes[:, :, ::4, ::4])
        node_rows = np.column_stack((
            selected_times - center + 1,
            selected_coords,
            selected_ids,
        )).astype(np.float32)
        corrected = node_rows[:, 1:4].copy()
        corrected[:, 1:] /= 4.0
        if (
            pooled.shape != (3, 64, 64, 64)
            or np.any(corrected < 0)
            or np.any(corrected >= np.asarray(pooled.shape[1:])[None])
        ):
            raise RuntimeError(f"pooling or coordinates invalid: {{stem}} t={{center}}")
        destination = OUTPUT_ROOT / role / f"{{stem}}__t{{center:04d}}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            volumes=pooled,
            nodes=node_rows,
            edges=edge_rows,
            divisions=divisions,
            voxel_um_pooled=np.asarray((1.625, 1.625, 1.625), dtype=np.float32),
        )
        critical = set()
        for source, targets in outgoing.items():
            if len(targets) >= 2:
                critical.add(source)
                critical.update(targets)
        critical_center = sum(int(node_rows[row, 0]) == 1 for row in critical)
        if role in {{"selection", "sealed_audit"}} and critical_center <= 0:
            raise RuntimeError(f"gate shard lost division: {{destination.name}}")
        records.append({{
            "path": destination.relative_to(OUTPUT_ROOT).as_posix(),
            "stem": stem,
            "embryo": movie["embryo"],
            "role": role,
            "center_frame": center,
            "nodes": len(node_rows),
            "center_nodes": int(np.sum(node_rows[:, 0] == 1)),
            "division_critical_center_nodes": critical_center,
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }})
    if movie_index % 20 == 0:
        print(f"built {{movie_index}}/{{len(movies)}} real movies", flush=True)

frame_digest_manifest = {{
    "schema_version": 1,
    "status": "complete",
    "run_id": "competition-real-localization-kaggle-source-digests-v1",
    "inventory_sha256": EXPECTED_INVENTORY_SHA256,
    "files": frame_records,
    "summary": {{"frames": len(frame_records)}},
    "competition_train_data_read": True,
    "competition_test_data_read": False,
    "public_leaderboard_used_for_selection": False,
    "submission_created": False,
    "authorized_for_submission": False,
}}
frame_manifest_path = OUTPUT_ROOT / "source_frame_digest_manifest.json"
atomic_json(frame_manifest_path, frame_digest_manifest)
counts = {{
    role: {{
        "shards": sum(row["role"] == role for row in records),
        "center_nodes": sum(row["center_nodes"] for row in records if row["role"] == role),
        "division_critical_center_nodes": sum(
            row["division_critical_center_nodes"] for row in records if row["role"] == role
        ),
    }}
    for role in ("optimization", "selection", "sealed_audit")
}}
if (
    len(records) != 177
    or len(frame_records) != 525
    or counts["optimization"]["shards"] != 146
    or counts["selection"]["shards"] != 17
    or counts["sealed_audit"]["shards"] != 14
    or any(counts[role]["center_nodes"] <= 0 for role in counts)
    or any(counts[role]["division_critical_center_nodes"] <= 0 for role in ("selection", "sealed_audit"))
):
    raise RuntimeError("real localization output inventory changed")
manifest = {{
    "schema_version": 1,
    "status": "complete",
    "run_id": RUN_ID,
    "source_mode": "kaggle_cpu_direct_competition_train",
    "inventory_sha256": EXPECTED_INVENTORY_SHA256,
    "labels_manifest_sha256": EXPECTED_LABELS_MANIFEST_SHA256,
    "frame_cache_manifest_sha256": sha256_file(frame_manifest_path),
    "geff_cache_manifest_sha256": labels_manifest["geff_cache_manifest_sha256"],
    "files": records,
    "summary": {{
        "shards": len(records),
        "bytes": sum(row["bytes"] for row in records),
        "source_frames": len(frame_records),
        "by_role": counts,
    }},
    "excluded_final_probe_stems": sorted(FINAL_PROBE_STEMS),
    "geometry": {{
        "source_shape_zyx": [64, 256, 256],
        "pooled_shape_zyx": [64, 64, 64],
        "xy_stride": 4,
        "pooled_voxel_um": [1.625, 1.625, 1.625],
    }},
    "competition_train_data_read": True,
    "competition_test_data_read": False,
    "public_leaderboard_used_for_selection": False,
    "submission_created": False,
    "authorized_for_submission": False,
}}
manifest_path = OUTPUT_ROOT / "real_localization_shard_manifest.json"
atomic_json(manifest_path, manifest)
terminal = {{
    "schema_version": 1,
    "status": "completed",
    "run_id": "competition-real-localization-kaggle-cache-v1",
    "elapsed_seconds": round(time.monotonic() - STARTED, 3),
    "accelerator": "cpu",
    "gpu_used": False,
    "shard_manifest_sha256": sha256_file(manifest_path),
    "shards": len(records),
    "competition_train_data_read": True,
    "competition_test_data_read": False,
    "public_leaderboard_used_for_selection": False,
    "submission_created": False,
    "authorized_for_submission": False,
}}
atomic_json(WORKING / "launcher_terminal.json", terminal)
print(json.dumps(terminal, indent=2, sort_keys=True))
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-upload-root", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "kaggle" / KERNEL_ID
    )
    args = parser.parse_args()

    upload_manifest_path = args.label_upload_root / "upload_manifest.json"
    upload_manifest = json.loads(upload_manifest_path.read_text(encoding="utf-8"))
    archive = upload_manifest.get("archive", {})
    if not (
        upload_manifest.get("status") == "complete"
        and upload_manifest.get("dataset_id") == DATASET_REF
        and upload_manifest.get("inventory_sha256") == EXPECTED_INVENTORY_SHA256
        and upload_manifest.get("competition_test_data_read") is False
        and upload_manifest.get("authorized_for_submission") is False
        and archive.get("path") == ARCHIVE_NAME
        and len(str(archive.get("sha256", ""))) == 64
        and len(str(upload_manifest.get("labels_manifest_sha256", ""))) == 64
    ):
        raise ValueError("label upload manifest is ineligible")
    if args.output_root.exists():
        raise FileExistsError(f"kernel output root exists: {args.output_root}")
    args.output_root.mkdir(parents=True)

    metadata = {
        "id": KERNEL_REF,
        "title": "Biohub Real Localization Replay Cache v1",
        "code_file": f"{KERNEL_ID}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["cpu", "cell-tracking", "training-data", "replay"],
        "dataset_sources": [DATASET_REF],
        "kernel_sources": [],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "model_sources": [],
    }
    notebook = {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {
                "accelerator": "none",
                "dataSources": [
                    {"sourceId": DATASET_REF, "sourceType": "dataset"},
                    {
                        "sourceId": "biohub-cell-tracking-during-development",
                        "sourceType": "competition",
                    },
                ],
                "isInternetEnabled": False,
                "language": "python",
                "sourceType": "notebook",
                "isGpuEnabled": False,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": [
            code_cell(
                kernel_source(
                    archive_sha256=str(archive["sha256"]),
                    labels_manifest_sha256=str(
                        upload_manifest["labels_manifest_sha256"]
                    ),
                )
            )
        ],
    }
    (args.output_root / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_root / f"{KERNEL_ID}.ipynb").write_text(
        json.dumps(notebook, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output_root)


if __name__ == "__main__":
    main()
