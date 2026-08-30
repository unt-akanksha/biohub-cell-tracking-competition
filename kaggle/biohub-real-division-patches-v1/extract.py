#!/usr/bin/env python
"""Extract reciprocal real-domain division patches on Kaggle CPU only."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tarfile
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


RUN_ID = "competition-real-division-patches-v1"
VOXEL_SIZE_ZYX_UM = (1.625, 0.40625, 0.40625)
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


def stratified_selection_stems(examples_by_stem: dict[str, list[dict[str, Any]]]) -> set[str]:
    selected: set[str] = set()
    for embryo in ("44b6", "6bba"):
        eligible = [
            stem
            for stem, rows in examples_by_stem.items()
            if stem.startswith(embryo + "_")
            and stem not in FINAL_PROBE_STEMS
            and sum(row["division_target"] for row in rows) > 0
        ]
        target = int(
            math.ceil(
                sum(
                    sum(row["division_target"] for row in examples_by_stem[stem])
                    for stem in eligible
                )
                * 0.20
            )
        )
        accumulated = 0
        for stem in sorted(
            eligible, key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
        ):
            selected.add(stem)
            accumulated += sum(
                row["division_target"] for row in examples_by_stem[stem]
            )
            if accumulated >= target:
                break
        if accumulated < target:
            raise RuntimeError(f"division-stratified selection failed for {embryo}")
    return selected


def read_geff(path: Path) -> tuple[dict[int, tuple[float, ...]], list[tuple[int, int]]]:
    import zarr

    group = zarr.open_group(str(path), mode="r")
    node_ids = np.asarray(group["nodes/ids"][:], dtype=np.int64)
    times = np.asarray(group["nodes/props/t/values"][:], dtype=np.int64)
    coordinates = np.stack(
        [
            np.asarray(group[f"nodes/props/{axis}/values"][:], dtype=np.float64)
            for axis in ("z", "y", "x")
        ],
        axis=1,
    )
    edge_ids = np.asarray(group["edges/ids"][:], dtype=np.int64).reshape(-1, 2)
    if len(node_ids) != len(times) or coordinates.shape != (len(node_ids), 3):
        raise ValueError(f"GEFF node arrays disagree: {path.name}")
    nodes = {
        int(node_id): (int(timepoint), *(float(value) for value in coordinate))
        for node_id, timepoint, coordinate in zip(
            node_ids, times, coordinates, strict=True
        )
    }
    edges = [tuple(int(value) for value in edge) for edge in edge_ids]
    return nodes, edges


def movie_examples(
    nodes: dict[int, tuple[float, ...]], edges: list[tuple[int, int]]
) -> list[dict[str, Any]]:
    outgoing: dict[int, set[int]] = {}
    for source, target in edges:
        if source not in nodes or target not in nodes:
            raise ValueError("train GEFF contains a dangling edge")
        if int(nodes[target][0]) != int(nodes[source][0]) + 1:
            raise ValueError("train GEFF contains a non-adjacent edge")
        outgoing.setdefault(source, set()).add(target)
    division_times = {
        int(nodes[node_id][0])
        for node_id, targets in outgoing.items()
        if len(targets) == 2
    }
    rows = []
    for node_id, targets in sorted(outgoing.items()):
        timepoint = int(nodes[node_id][0])
        if timepoint not in division_times or len(targets) not in (1, 2):
            continue
        rows.append(
            {
                "node_id": int(node_id),
                "timepoint": timepoint,
                "center_zyx_voxel": [float(value) for value in nodes[node_id][1:]],
                "division_target": len(targets) == 2,
            }
        )
    return rows


def sample_physical_patches(
    volume: np.ndarray,
    centers_zyx_voxel: np.ndarray,
    *,
    output_shape_zyx: tuple[int, int, int] = (17, 17, 17),
    half_extent_zyx_um: tuple[float, float, float] = (8.0, 8.0, 8.0),
    chunk_size: int = 64,
) -> torch.Tensor:
    image = torch.as_tensor(volume, dtype=torch.float32)
    centers = torch.as_tensor(centers_zyx_voxel, dtype=torch.float32)
    if image.ndim != 4 or image.shape[0] != 3:
        raise ValueError("temporal volume must have shape (3, Z, Y, X)")
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape (N, 3)")
    offsets = [
        torch.linspace(-extent, extent, count) / spacing
        for extent, count, spacing in zip(
            half_extent_zyx_um,
            output_shape_zyx,
            VOXEL_SIZE_ZYX_UM,
            strict=True,
        )
    ]
    offset_grid = torch.stack(torch.meshgrid(*offsets, indexing="ij"), dim=-1)
    spatial_shape = tuple(int(value) for value in image.shape[-3:])
    patches = []
    for start in range(0, len(centers), chunk_size):
        batch_centers = centers[start : start + chunk_size]
        coordinates = batch_centers[:, None, None, None, :] + offset_grid[None]
        normalized = [
            2.0 * coordinates[..., axis] / float(size - 1) - 1.0
            for axis, size in enumerate(spatial_shape)
        ]
        grid = torch.stack((normalized[2], normalized[1], normalized[0]), dim=-1)
        patches.append(
            F.grid_sample(
                image[None].expand(len(batch_centers), -1, -1, -1, -1),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
        )
    result = torch.cat(patches)
    means = result.mean(dim=(2, 3, 4), keepdim=True)
    scales = result.std(dim=(2, 3, 4), keepdim=True).clamp_min(1e-4)
    return ((result - means) / scales).clamp(-6.0, 6.0)


def discover_train_root(input_root: Path) -> Path:
    candidates = [
        Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train"),
        Path("/kaggle/input/biohub-cell-tracking-during-development/train"),
    ]
    candidates.extend(path / "train" for path in input_root.iterdir() if path.is_dir())
    eligible = [
        path
        for path in dict.fromkeys(candidates)
        if path.is_dir()
        and len(list(path.glob("*.geff"))) == 199
        and len(list(path.glob("*.zarr"))) == 199
    ]
    if len(eligible) != 1:
        raise RuntimeError({"eligible_train_roots": [str(path) for path in eligible]})
    return eligible[0]


def write_shard(
    output_root: Path,
    *,
    stem: str,
    embryo: str,
    role: str,
    timepoint: int,
    examples: list[dict[str, Any]],
    patches: torch.Tensor,
) -> dict[str, Any]:
    target = output_root / role / embryo / f"{stem}-t{timepoint:03d}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(
        [example["division_target"] for example in examples], dtype=np.float32
    )
    label_weights = np.where(labels > 0.5, 1.0, 0.25).astype(np.float32)
    metadata = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "stem": stem,
        "embryo": embryo,
        "role": role,
        "timepoint": timepoint,
        "rows": len(examples),
        "division_positives": int(labels.sum()),
        "ordinary_unlabeled_controls": int((labels < 0.5).sum()),
        "negative_label_weight": 0.25,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
    }
    temporary = target.with_suffix(".npz.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            source_patches=patches.numpy().astype(np.float16),
            division_target=labels,
            label_weight=label_weights,
            node_ids=np.asarray([example["node_id"] for example in examples]),
            centers_zyx_voxel=np.asarray(
                [example["center_zyx_voxel"] for example in examples],
                dtype=np.float32,
            ),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(target)
    return {
        **metadata,
        "path": target.relative_to(output_root).as_posix(),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def main() -> None:
    import zarr

    if torch.cuda.is_available():
        raise RuntimeError("real division patch extraction must use Kaggle CPU only")
    started = time.monotonic()
    input_root = Path("/kaggle/input")
    working = Path("/kaggle/working")
    train_root = discover_train_root(input_root)
    output_root = working / "biohub_real_division_patches_v1"
    output_root.mkdir(parents=True, exist_ok=False)
    geffs = sorted(path for path in train_root.glob("*.geff") if path.is_dir())
    stems = [path.stem for path in geffs]
    embryo_counts = {
        prefix: sum(stem.startswith(prefix + "_") for stem in stems)
        for prefix in ("44b6", "6bba")
    }
    if embryo_counts != {"44b6": 71, "6bba": 128}:
        raise RuntimeError(f"competition train embryo inventory changed: {embryo_counts}")

    examples_by_stem: dict[str, list[dict[str, Any]]] = {}
    for geff_index, geff_path in enumerate(geffs, start=1):
        nodes, edges = read_geff(geff_path)
        examples_by_stem[geff_path.stem] = movie_examples(nodes, edges)
        if geff_index % 25 == 0:
            print(f"read {geff_index}/{len(geffs)} train GEFFs", flush=True)
    selection_stems = stratified_selection_stems(examples_by_stem)

    records: list[dict[str, Any]] = []
    movie_summaries: list[dict[str, Any]] = []
    for movie_index, geff_path in enumerate(geffs, start=1):
        stem = geff_path.stem
        embryo = stem.split("_", 1)[0]
        if stem in FINAL_PROBE_STEMS:
            movie_summaries.append(
                {"stem": stem, "embryo": embryo, "role": "final_probe", "excluded": True}
            )
            continue
        examples = examples_by_stem[stem]
        role = "selection" if stem in selection_stems else "optimization"
        image = None
        movie_records = []
        for timepoint in sorted({row["timepoint"] for row in examples}):
            selected = [row for row in examples if row["timepoint"] == timepoint]
            if image is None:
                image = zarr.open_group(str(train_root / f"{stem}.zarr"), mode="r")["0"]
            frame_indices = [
                min(max(timepoint + offset, 0), int(image.shape[0]) - 1)
                for offset in (-1, 0, 1)
            ]
            context = np.stack([np.asarray(image[index]) for index in frame_indices])
            centers = np.asarray(
                [row["center_zyx_voxel"] for row in selected], dtype=np.float32
            )
            patches = sample_physical_patches(context, centers)
            movie_records.append(
                write_shard(
                    output_root,
                    stem=stem,
                    embryo=embryo,
                    role=role,
                    timepoint=timepoint,
                    examples=selected,
                    patches=patches,
                )
            )
        records.extend(movie_records)
        movie_summaries.append(
            {
                "stem": stem,
                "embryo": embryo,
                "role": role,
                "excluded": False,
                "shards": len(movie_records),
                "division_positives": sum(row["division_positives"] for row in movie_records),
                "ordinary_unlabeled_controls": sum(
                    row["ordinary_unlabeled_controls"] for row in movie_records
                ),
            }
        )
        if movie_index % 20 == 0:
            print(f"processed {movie_index}/{len(geffs)} train movies", flush=True)

    by_embryo_role: dict[str, dict[str, dict[str, int]]] = {}
    for embryo in ("44b6", "6bba"):
        by_embryo_role[embryo] = {}
        for role in ("optimization", "selection"):
            selected = [
                record
                for record in records
                if record["embryo"] == embryo and record["role"] == role
            ]
            by_embryo_role[embryo][role] = {
                "shards": len(selected),
                "rows": sum(record["rows"] for record in selected),
                "division_positives": sum(record["division_positives"] for record in selected),
                "ordinary_unlabeled_controls": sum(
                    record["ordinary_unlabeled_controls"] for record in selected
                ),
            }
            if by_embryo_role[embryo][role]["division_positives"] <= 0:
                raise RuntimeError(f"division patch split lost positives: {embryo}/{role}")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "run_id": RUN_ID,
        "split_policy": (
            "four complete final-probe movies excluded; division-positive movies "
            "ordered by sha256(stem) within each embryo until at least 20 percent "
            "of division positives enter selection"
        ),
        "negative_label_policy": (
            "annotated one-child parents in true-division frames are PU controls "
            "with loss weight 0.25, never certain background"
        ),
        "final_probe_stems": sorted(FINAL_PROBE_STEMS),
        "movies": movie_summaries,
        "shards": records,
        "summary": {
            "movies": len(geffs),
            "excluded_final_probe_movies": 4,
            "shards": len(records),
            "rows": sum(record["rows"] for record in records),
            "division_positives": sum(record["division_positives"] for record in records),
            "ordinary_unlabeled_controls": sum(
                record["ordinary_unlabeled_controls"] for record in records
            ),
            "bytes": sum(record["bytes"] for record in records),
            "by_embryo_role": by_embryo_role,
        },
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }
    manifest_path = output_root / "real_division_patch_manifest.json"
    atomic_json(manifest_path, manifest)
    archive_path = working / "biohub_real_division_patches_v1.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(output_root, arcname=output_root.name, recursive=True)
    terminal = {
        "schema_version": 1,
        "status": "completed",
        "run_id": RUN_ID,
        "accelerator": "cpu",
        "gpu_used": False,
        "elapsed_seconds": time.monotonic() - started,
        "manifest_sha256": sha256_file(manifest_path),
        "archive_sha256": sha256_file(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "summary": manifest["summary"],
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "submission_created": False,
    }
    atomic_json(working / "launcher_terminal.json", terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
