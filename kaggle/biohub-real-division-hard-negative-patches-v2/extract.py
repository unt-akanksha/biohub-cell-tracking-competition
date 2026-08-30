#!/usr/bin/env python
"""Extract real division patches plus deterministic no-division hard negatives."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tarfile
import time
from typing import Any

import numpy as np
import torch


RUN_ID = "competition-real-division-hard-negative-patches-v2"
FINAL_PROBE_STEMS = {
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
}
ROLE_FRACTION = 0.20
MAXIMUM_NEGATIVE_FRAME_ROWS = 32


def _load_base_module() -> Any:
    candidates = [
        Path("/kaggle/input/datasets/indarkarhana/biohub-real-division-extractor-runtime-v1/extract_base.py"),
        Path("/kaggle/input/biohub-real-division-extractor-runtime-v1/extract_base.py"),
        Path(__file__).resolve().parents[1]
        / "biohub-real-division-patches-v1"
        / "extract.py",
    ]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError({"eligible_extractor_bases": [str(path) for path in matches]})
    spec = importlib.util.spec_from_file_location("biohub_real_division_extract_base", matches[0])
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the pinned real-division extractor base")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_module()


def stable_order(values: list[str], namespace: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{RUN_ID}|{namespace}|{value}".encode("utf-8")
        ).hexdigest(),
    )


def allocate_roles(
    examples_by_stem: dict[str, dict[int, dict[str, Any]]]
) -> dict[str, str]:
    roles: dict[str, str] = {}
    for embryo in ("44b6", "6bba"):
        eligible = [
            stem
            for stem, frames in examples_by_stem.items()
            if stem.startswith(embryo + "_")
            and stem not in FINAL_PROBE_STEMS
            and sum(
                sum(row["division_target"] for row in frame["rows"])
                for frame in frames.values()
            )
            > 0
        ]
        positives = {
            stem: sum(
                sum(row["division_target"] for row in frame["rows"])
                for frame in examples_by_stem[stem].values()
            )
            for stem in eligible
        }
        target = int(math.ceil(sum(positives.values()) * ROLE_FRACTION))
        audit: set[str] = set()
        accumulated = 0
        for stem in stable_order(eligible, f"{embryo}-audit"):
            audit.add(stem)
            accumulated += positives[stem]
            if accumulated >= target:
                break
        remaining = [stem for stem in eligible if stem not in audit]
        selection: set[str] = set()
        accumulated = 0
        for stem in stable_order(remaining, f"{embryo}-selection"):
            selection.add(stem)
            accumulated += positives[stem]
            if accumulated >= target:
                break
        if not audit or not selection or audit & selection:
            raise RuntimeError(f"hard-negative split failed for {embryo}")
        for stem in eligible:
            roles[stem] = (
                "audit"
                if stem in audit
                else "selection"
                if stem in selection
                else "optimization"
            )
    return roles


def movie_frames(
    stem: str,
    nodes: dict[int, tuple[float, ...]],
    edges: list[tuple[int, int]],
) -> dict[int, dict[str, Any]]:
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
    frames: dict[int, dict[str, Any]] = {}
    for timepoint in sorted(division_times):
        rows = [
            {
                "node_id": int(node_id),
                "timepoint": timepoint,
                "center_zyx_voxel": [float(value) for value in nodes[node_id][1:]],
                "division_target": len(targets) == 2,
            }
            for node_id, targets in sorted(outgoing.items())
            if int(nodes[node_id][0]) == timepoint and len(targets) in (1, 2)
        ]
        frames[timepoint] = {"frame_role": "division", "rows": rows}

    negative_by_time: dict[int, list[int]] = {}
    for node_id, targets in outgoing.items():
        timepoint = int(nodes[node_id][0])
        if timepoint not in division_times and len(targets) == 1:
            negative_by_time.setdefault(timepoint, []).append(int(node_id))
    eligible_times = [
        timepoint for timepoint, node_ids in negative_by_time.items() if len(node_ids) >= 4
    ]
    if eligible_times:
        negative_time = int(
            stable_order([str(value) for value in eligible_times], f"{stem}-negative")[0]
        )
        node_ids = sorted(
            negative_by_time[negative_time],
            key=lambda node_id: hashlib.sha256(
                f"{RUN_ID}|{stem}|{negative_time}|{node_id}".encode("utf-8")
            ).hexdigest(),
        )[:MAXIMUM_NEGATIVE_FRAME_ROWS]
        frames[negative_time] = {
            "frame_role": "no_division_hard_negative",
            "rows": [
                {
                    "node_id": node_id,
                    "timepoint": negative_time,
                    "center_zyx_voxel": [float(value) for value in nodes[node_id][1:]],
                    "division_target": False,
                }
                for node_id in node_ids
            ],
        }
    return frames


def write_shard(
    output_root: Path,
    *,
    stem: str,
    embryo: str,
    role: str,
    timepoint: int,
    frame_role: str,
    examples: list[dict[str, Any]],
    patches: torch.Tensor,
) -> dict[str, Any]:
    target = output_root / role / embryo / f"{stem}-t{timepoint:03d}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(
        [example["division_target"] for example in examples], dtype=np.float32
    )
    negative_weight = 0.50 if frame_role == "no_division_hard_negative" else 0.25
    label_weights = np.where(labels > 0.5, 1.0, negative_weight).astype(np.float32)
    metadata = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "stem": stem,
        "embryo": embryo,
        "role": role,
        "frame_role": frame_role,
        "timepoint": timepoint,
        "rows": len(examples),
        "division_positives": int(labels.sum()),
        "ordinary_controls": int((labels < 0.5).sum()),
        "negative_label_weight": negative_weight,
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
        "sha256": base.sha256_file(target),
    }


def summarize(records: list[dict[str, Any]], embryo: str, role: str) -> dict[str, int]:
    selected = [
        record
        for record in records
        if record["embryo"] == embryo and record["role"] == role
    ]
    return {
        "shards": len(selected),
        "movies": len({record["stem"] for record in selected}),
        "rows": sum(record["rows"] for record in selected),
        "division_positives": sum(record["division_positives"] for record in selected),
        "ordinary_controls": sum(record["ordinary_controls"] for record in selected),
        "negative_frames": sum(
            record["frame_role"] == "no_division_hard_negative" for record in selected
        ),
    }


def main() -> None:
    if torch.cuda.is_available():
        raise RuntimeError("hard-negative division extraction must use Kaggle CPU only")
    started = time.monotonic()
    input_root = Path("/kaggle/input")
    working = Path("/kaggle/working")
    base.install_numcodecs(input_root, working)
    train_root = base.discover_train_root(input_root)
    output_root = working / "biohub_real_division_hard_negative_patches_v2"
    output_root.mkdir(parents=True, exist_ok=False)
    geffs = sorted(path for path in train_root.glob("*.geff") if path.is_dir())
    if len(geffs) != 199:
        raise RuntimeError("competition train GEFF inventory changed")

    examples_by_stem: dict[str, dict[int, dict[str, Any]]] = {}
    for index, geff_path in enumerate(geffs, start=1):
        nodes, edges = base.read_geff(geff_path)
        examples_by_stem[geff_path.stem] = movie_frames(geff_path.stem, nodes, edges)
        if index % 25 == 0:
            print(f"read {index}/{len(geffs)} train GEFFs", flush=True)
    roles = allocate_roles(examples_by_stem)

    records: list[dict[str, Any]] = []
    for movie_index, geff_path in enumerate(geffs, start=1):
        stem = geff_path.stem
        if stem in FINAL_PROBE_STEMS:
            continue
        role = roles.get(stem, "optimization")
        embryo = stem.split("_", 1)[0]
        image_path = train_root / f"{stem}.zarr" / "0"
        image_metadata = json.loads(
            (image_path / "zarr.json").read_text(encoding="utf-8")
        )
        frame_count = int(image_metadata["shape"][0])
        for timepoint, frame in sorted(examples_by_stem[stem].items()):
            context = np.stack(
                [
                    base.read_v3_frame(
                        image_path, min(max(timepoint + offset, 0), frame_count - 1)
                    )
                    for offset in (-1, 0, 1)
                ]
            )
            centers = np.asarray(
                [row["center_zyx_voxel"] for row in frame["rows"]], dtype=np.float32
            )
            patches = base.sample_physical_patches(context, centers)
            records.append(
                write_shard(
                    output_root,
                    stem=stem,
                    embryo=embryo,
                    role=role,
                    timepoint=timepoint,
                    frame_role=frame["frame_role"],
                    examples=frame["rows"],
                    patches=patches,
                )
            )
        if movie_index % 20 == 0:
            print(f"processed {movie_index}/{len(geffs)} train movies", flush=True)

    by_embryo_role = {
        embryo: {
            role: summarize(records, embryo, role)
            for role in ("optimization", "selection", "audit")
        }
        for embryo in ("44b6", "6bba")
    }
    if any(
        by_embryo_role[embryo][role]["division_positives"] <= 0
        for embryo in by_embryo_role
        for role in by_embryo_role[embryo]
    ):
        raise RuntimeError("hard-negative split lost a division-positive stratum")
    summary = {
        "movies": len(geffs),
        "excluded_final_probe_movies": len(FINAL_PROBE_STEMS),
        "shards": len(records),
        "rows": sum(record["rows"] for record in records),
        "division_positives": sum(record["division_positives"] for record in records),
        "ordinary_controls": sum(record["ordinary_controls"] for record in records),
        "negative_frames": sum(
            record["frame_role"] == "no_division_hard_negative" for record in records
        ),
        "bytes": sum(record["bytes"] for record in records),
        "by_embryo_role": by_embryo_role,
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "run_id": RUN_ID,
        "split_policy": (
            "four historical complete probes excluded; deterministic 20 percent "
            "division-positive movie groups reserved independently for selection "
            "and sealed audit within each embryo"
        ),
        "negative_frame_policy": (
            "one deterministic no-division frame per eligible movie, capped at "
            "32 annotated one-child parents and weighted 0.50"
        ),
        "final_probe_stems": sorted(FINAL_PROBE_STEMS),
        "shards": records,
        "summary": summary,
        "audit_opened": False,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }
    manifest_path = output_root / "real_division_hard_negative_manifest.json"
    base.atomic_json(manifest_path, manifest)
    archive_path = working / "biohub_real_division_hard_negative_patches_v2.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(output_root, arcname=output_root.name, recursive=True)
    terminal = {
        "schema_version": 1,
        "status": "completed",
        "run_id": RUN_ID,
        "accelerator": "cpu",
        "gpu_used": False,
        "elapsed_seconds": time.monotonic() - started,
        "manifest_sha256": base.sha256_file(manifest_path),
        "archive_sha256": base.sha256_file(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "summary": summary,
        "audit_opened": False,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "submission_created": False,
    }
    base.atomic_json(working / "launcher_terminal.json", terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
